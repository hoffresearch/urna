"""Real model2vec/potion static embedder for forge: offline, no torch, no network.

the #04 default is a quality FLOOR: each token gets a fixed pseudo-random vector,
so cosine reflects shared LITERAL tokens only (carro vs automovel ~ carro vs
banana). this module is the SEMANTIC default: a vendored model2vec/potion static
table (minishlab/potion-base-8M, mit, dim 256) whose token rows ALREADY carry
meaning distilled from a teacher model, so synonyms land close (car ~ automobile
>> car ~ banana) with no torch, no model download, and no network at runtime.

inference is the model2vec recipe, reproduced with numpy + tokenizers only:
tokenize (add_special_tokens=False) -> gather the token rows from the vendored
embeddings table -> mean pool -> l2 normalize. it reproduces the published model
vector-for-vector (verified against the model2vec reference), but ships none of
its torch-adjacent dependency surface.

it is offline by construction: the table and tokenizer are read from local files
under model/potionb8m/, the hugging face offline flags are forced on, and
no code path here ever opens a socket. the same StaticEmbedder interface and the
same model_hash convention as the floor apply, so it is a drop-in swap; the
brought sentence-transformers model stays the quality ceiling.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

# force offline before tokenizers/hf libs are imported, so even a misconfigured
# environment can never trigger a hub round-trip. runtime is fail-closed.
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "minishlab/potion-base-8M"
POTION_VERSION = "1"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "potionb8m"
# the files that actually affect inference output; the model_hash fingerprints
# their bytes, mirroring modelhash.RELEVANT_FILES.
RELEVANT_FILES: tuple[str, ...] = ("config.json", "tokenizer.json", "model.safetensors")


def _read_safetensors_f32(path: Path, name: str = "embeddings"):
    """lRead one float32 tensor from a .safetensors file with the stdlib only:
    8-byte LE header length, json header, then the raw little-endian f32 payload.
    returns an (rows, dim) numpy array. no safetensors lib, no torch."""
    import numpy as np

    raw = path.read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + hlen])
    meta = header[name]
    if meta["dtype"] != "F32":
        raise ValueError(f"expected F32 table, got {meta['dtype']}")
    start, end = meta["data_offsets"]
    payload = raw[8 + hlen + start : 8 + hlen + end]
    return np.frombuffer(payload, dtype="<f4").reshape(meta["shape"]).copy()


@lru_cache(maxsize=4)
def _load_table(model_dir: str):
    """lLoad (tokenizer, embeddings, dim) once per model dir. cached so repeated
    embedders share the mmap-backed table. raises a clear, actionable error if
    the vendored asset is missing (it ships via git-lfs)."""
    md = Path(model_dir)
    st = md / "model.safetensors"
    tj = md / "tokenizer.json"
    if not st.is_file() or st.stat().st_size < 1024:
        raise FileNotFoundError(
            f"vendored potion table missing or not pulled from git-lfs: {st}. "
            f"run `git lfs pull`, or fetch minishlab/potion-base-8M into {md}."
        )
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tj))
    emb = _read_safetensors_f32(st)
    cfg = json.loads((md / "config.json").read_text())
    dim = int(cfg.get("hidden_dim") or emb.shape[1])
    return tok, emb, dim


def _files_hash(model_dir: Path) -> str:
    h = hashlib.sha256()
    for rel in RELEVANT_FILES:
        p = model_dir / rel
        if not p.is_file():
            continue
        fh = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                fh.update(chunk)
        h.update(rel.encode())
        h.update(b"\0")
        h.update(fh.hexdigest().encode())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _tokenizer_hash(model_dir: Path) -> str:
    p = model_dir / "tokenizer.json"
    if not p.is_file():
        return ""
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


class PotionEmbedder:
    """lThe offline SEMANTIC default embedder. same surface as the floor's
    StaticEmbedder (embedding_model/embedding_dim/fingerprint/model_hash/
    embed_texts/__call__), so it drops straight into builder.Pipeline."""

    def __init__(self, model_dir: Path | str = MODEL_DIR, normalize: bool = True):
        self.model_dir = Path(model_dir)
        self.normalize = normalize
        self._tok = None
        self._emb = None
        self._dim = None
        self._hash: str | None = None

    def _ensure(self) -> None:
        if self._tok is None:
            self._tok, self._emb, self._dim = _load_table(str(self.model_dir))

    @property
    def embedding_model(self) -> str:
        return f"{MODEL_ID}/v{POTION_VERSION}"

    @property
    def embedding_dim(self) -> int:
        self._ensure()
        return int(self._dim)

    def fingerprint(self) -> dict:
        """lThe inference-relevant config, hashed into the model_hash. files_hash
        + tokenizer_hash pin the exact vendored table, so a different table (or a
        truncated lfs pull) produces a different hash and fails the manifest gate
        loudly instead of returning cosine-valid garbage."""
        self._ensure()
        return {
            "embedder": MODEL_ID,
            "version": POTION_VERSION,
            "files_hash": _files_hash(self.model_dir),
            "tokenizer_hash": _tokenizer_hash(self.model_dir),
            "embedding_dim": int(self._dim),
            "tokenizer": "bert-wordpiece-lower",
            "add_special_tokens": False,
            "pooling": "mean",
            "normalize": "l2" if self.normalize else "none",
            "dtype": "float32-stable",
        }

    def model_hash(self) -> str:
        """l`sha256:<hex>` of the canonical-json fingerprint, the SAME convention
        as modelhash.fingerprint_to_model_hash and the floor."""
        if self._hash is None:
            canonical = json.dumps(self.fingerprint(), sort_keys=True, separators=(",", ":"))
            self._hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        return self._hash

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        self._ensure()
        emb = self._emb
        unk = self._tok.token_to_id("[UNK]") or 1
        out: list[list[float]] = []
        for enc in self._tok.encode_batch(list(texts), add_special_tokens=False):
            ids = enc.ids or [unk]  # empty text -> deterministic non-zero (unk row)
            v = emb[ids].mean(axis=0)
            if self.normalize:
                n = float(np.linalg.norm(v))
                if n > 0.0:
                    v = v / n
            # f32-stable: cast to float32 so a cache-backed rebuild (float32 cache)
            # is byte-identical; .tolist() yields python floats equal to the f32 value.
            out.append(v.astype(np.float32).tolist())
        return out

    def __call__(self, specs: Sequence[object]) -> list[list[float]]:
        texts = [getattr(s, "canonical_text", s) for s in specs]
        return self.embed_texts(texts)  # type: ignore[arg-type]


def potion_embedder(model_dir: Path | str = MODEL_DIR) -> PotionEmbedder:
    """lThe offline semantic default: vendored model2vec/potion static table."""
    return PotionEmbedder(model_dir)


# forge's DEFAULT embedder is now the real semantic table. the lexical floor in
# lexifloor.py stays available as the zero-dependency fallback.
default_embedder = potion_embedder
