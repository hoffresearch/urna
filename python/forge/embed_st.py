"""Sentence-transformers multimodal backend: WeMM-Embedding and
jina-embeddings-v5-omni (RFC-3).

Asymmetric contract: queries and documents go through DIFFERENT encode
routes when the model provides them (`encode_query` / `encode_document`);
the roles and the image prompt come from the preset's usage fields, spec-
overridable, and all of it lives in the embedding_recipe_hash — never
hidden in free-form kwargs.

model_hash identifies THE MODEL: weights/tokenizer/processor fingerprint
(model_fingerprint convention) + sha256 of every remote-code file that
affects inference + pooling/normalize/dtype policy. Changing any of those
changes the hash and the gate fails loudly.

Offline: when the model dir resolves locally the HF offline env vars are
forced before any hub-capable import; a hub download needs
URNA_ALLOW_DOWNLOAD=1 explicitly (repo convention).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

# files beyond the weights that change inference when they change: the
# remote code itself and the chat/processor configs it reads.
REMOTE_CODE_GLOBS = ("modeling_*.py", "*chat_template*.jinja", "processor_config.json")


def _default_device() -> str:
    if os.environ.get("URNA_ST_DEVICE"):
        return os.environ["URNA_ST_DEVICE"]
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError, RuntimeError):
        pass
    return "cpu"


def _default_dtype(device: str) -> str:
    if os.environ.get("URNA_ST_DTYPE"):
        return os.environ["URNA_ST_DTYPE"]
    # bf16 where it is first-class; fp16 on mps (measured this session:
    # wemm-2b image embeds 0.5s vs 23s in fp32, cosines agree to ~1e-2);
    # fp32 on cpu.
    if device == "cuda":
        return "bfloat16"
    return "float16" if device == "mps" else "float32"


class STMultimodalEmbedder:
    def __init__(
        self,
        preset,
        *,
        model_dir: Path | None = None,
        device: str | None = None,
        batch_size: int = 8,
        dtype: str = "",
        usage: dict | None = None,
    ):
        self.preset = preset
        self.embedding_model = preset.embedding_model
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device or _default_device()
        self.batch_size = batch_size
        self.dtype = dtype or _default_dtype(self.device)
        u = usage or {}
        self.text_corpus_mode = u.get("text_corpus_mode") or preset.text_corpus_mode
        self.text_query_mode = u.get("text_query_mode") or preset.text_query_mode
        self.image_prompt = u.get("image_prompt") or preset.image_prompt
        self.normalize = u.get("normalize", preset.normalize)
        self.image_max_side = int(u.get("image_max_side") or getattr(preset, "image_max_side", 0))
        self.image_doc_format = u.get("image_doc_format") or getattr(
            preset, "image_doc_format", "dict"
        )
        self.encode_kwargs = {
            **dict(getattr(preset, "encode_kwargs", ()) or ()),
            **(u.get("encode_kwargs") or {}),
        }
        self._model = None
        self._dim: int | None = None
        self._model_hash: str | None = None

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if self._model is not None:
            return
        allow_download = self.model_dir is None and os.environ.get("URNA_ALLOW_DOWNLOAD") == "1"
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            if allow_download:
                # the explicit opt-in wins over the blanket offline default
                # that importing forge (via embed_potion) installs.
                os.environ.pop(k, None)
            else:
                os.environ.setdefault(k, "1")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                f"preset '{self.preset.name}' needs the sentence-transformers stack: "
                'pip install torch "sentence-transformers>=5.7" "transformers==5.2.0" '
                '"qwen-vl-utils==0.0.14" "accelerate>=1.1.0"'
            ) from e
        target = str(self.model_dir) if self.model_dir else self.preset.model_id
        torch_dtype = getattr(torch, self.dtype)
        self._model = SentenceTransformer(
            target,
            trust_remote_code=self.preset.trust_remote_code,
            device=self.device,
            model_kwargs={"dtype": torch_dtype},
        )
        self._model.eval()

    # ---------------------------------------------------------------- encode
    def _encode(self, items: list, role: str) -> np.ndarray:
        self._load()
        kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize,
            "show_progress_bar": False,
            **self.encode_kwargs,
        }
        if role == "query" and hasattr(self._model, "encode_query"):
            out = self._model.encode_query(items, **kwargs)
        elif role == "document" and hasattr(self._model, "encode_document"):
            out = self._model.encode_document(items, **kwargs)
        else:
            out = self._model.encode(items, **kwargs)
        out = np.asarray(out, dtype=np.float32)
        if self.normalize:  # belt over suspenders: the gate depends on unit norm
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
        return out

    def embed_texts(self, texts, role: str = "document") -> np.ndarray:
        mode = self.text_query_mode if role == "query" else self.text_corpus_mode
        return self._encode(list(texts), mode if mode != "plain" else "plain")

    def _pil(self, obj):
        from PIL import Image

        img = Image.open(obj).convert("RGB") if isinstance(obj, (str, Path)) else obj
        if self.image_max_side and max(img.size) > self.image_max_side:
            img = img.copy()
            img.thumbnail((self.image_max_side, self.image_max_side))
        return img

    def _image_doc(self, img):
        if self.image_doc_format == "bare":
            return img
        return {"image": img, "text": self.image_prompt}

    def embed_paths(self, paths) -> np.ndarray:
        docs = [self._image_doc(self._pil(str(p))) for p in paths]
        return self._encode(docs, "document")

    def embed_arrays(self, frames) -> np.ndarray:
        from PIL import Image

        docs = [
            self._image_doc(self._pil(Image.fromarray(np.asarray(f, dtype=np.uint8))))
            for f in frames
        ]
        return self._encode(docs, "document")

    # ------------------------------------------------------------ identities
    @property
    def dim(self) -> int:
        if self._dim is None:
            if self.preset.default_dim:
                self._dim = self.preset.default_dim
            else:
                self._dim = int(self.embed_texts(["dim probe"], role="document").shape[1])
        return self._dim

    def _resolved_dir(self) -> Path:
        if self.model_dir is not None:
            return self.model_dir
        from model_fingerprint import hf_cache_snapshot

        return hf_cache_snapshot(self.preset.model_id)

    def fingerprint(self) -> dict:
        return fingerprint_for(self.preset, self._resolved_dir(), self.normalize, self.dtype)

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            blob = json.dumps(self.fingerprint(), sort_keys=True, separators=(",", ":"))
            self._model_hash = "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
        return self._model_hash


def fingerprint_for(preset, model_dir: Path, normalize: bool, dtype_policy: str) -> dict:
    """Model identity WITHOUT loading the model: weights/tokenizer/processor
    fingerprint + remote-code hashes + pooling/normalize/dtype policy. Shared
    by the in-process embedder and the subprocess adapter (the parent computes
    the triad without paying a model load when the cache is warm)."""
    from model_fingerprint import compute_model_fingerprint

    fp = compute_model_fingerprint(model_dir, model_id=preset.model_id).to_dict()
    code_hashes = {}
    for pattern in REMOTE_CODE_GLOBS:
        for f in sorted(Path(model_dir).glob(pattern)):
            code_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return {
        "embedder": "st_multimodal",
        "model_fingerprint": fp,
        "remote_code_sha256": code_hashes,
        "pooling": "model-native",
        "normalize": "l2" if normalize else "none",
        "dtype_policy": dtype_policy,
    }
