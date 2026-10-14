"""Vision embedder for image corpora.

This lives in the forge tooling layer because it pulls heavy optional
dependencies (torch, open_clip, Pillow). It never enters the sovereign
runtime, and no `.urna` reader needs it.

Models are addressed the way open_clip addresses them:
`hf-hub:redlessone/DermLIP_ViT-B-16` for dermatology, or a plain
architecture name plus a pretrained tag (`ViT-B-32` + `openai`) for the
general-image case. A preset pinned to a hub revision hands over that
snapshot's directory instead (`snapshot` plus `weights`): the weights and
the tokenizer are then read from those files, never resolved by name, so a
query runs offline against exactly the revision that was verified.

`model_hash` fingerprints the weights that were actually loaded, not the
model name. The manifest gate only means something if a different
checkpoint produces a different hash, and a name plus a constant string
cannot do that.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# offline by default, the guarantee the forge package gave the image path
# before the move: open_clip loads here without going through the registry,
# so the same defaults as presetmap.py are set here, unless the caller opts
# in with URNA_ALLOW_DOWNLOAD=1.
if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")


class ImageEmbedder:
    """Embed images or decoded frames with an open_clip vision tower.

    The model loads lazily, so callers that only need metadata do not pay
    the import and checkpoint cost.
    """

    def __init__(
        self,
        model_id: str = "hf-hub:redlessone/DermLIP_ViT-B-16",
        *,
        pretrained: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        snapshot: Path | None = None,
        weights: str | None = None,
    ):
        self.model_id = model_id
        self.pretrained = pretrained
        self.snapshot = Path(snapshot) if snapshot is not None else None
        self.weights = weights
        self._tokenizer = None
        self.device = device or self._default_device()
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None
        self._dim: int | None = None
        self._model_hash: str | None = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except (AttributeError, RuntimeError):
            pass
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        from PIL import Image

        # hf-hub: ids carry their own weights; a bare architecture name does
        # not, and open_clip silently returns RANDOM weights when the tag is
        # missing, so the tag is required rather than defaulted.
        if self.model_id.startswith("hf-hub:"):
            model, _, preprocess = open_clip.create_model_and_transforms(self.model_id)
        else:
            if not self.pretrained:
                raise ValueError(
                    f"model '{self.model_id}' needs an explicit pretrained tag "
                    "(for example --pretrained openai); without it open_clip "
                    "initializes random weights and search silently returns noise"
                )
            model, _, preprocess = self._create(open_clip)
        self._model_hash = self._fingerprint(model, preprocess)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._image = Image
        self._dim = self._resolve_dim(model)

    def _create(self, open_clip):
        """The model and preprocess for an architecture name plus a pretrained tag.

        With a snapshot, the checkpoint file is loaded into the built-in
        architecture and the preprocess is the tag's own (mean, std,
        interpolation, resize mode), so the fingerprint below sees the same
        weights and the same transform as a load by tag, and model_hash does
        not move. `local-dir:` is not used for the weights: it reads the
        preprocess back from json, and the lists it builds repr differently
        from the tag's tuples.
        """
        if self.snapshot is None:
            return open_clip.create_model_and_transforms(self.model_id, pretrained=self.pretrained)
        cfg = open_clip.get_pretrained_cfg(self.model_id, self.pretrained)
        if not cfg:
            raise ValueError(
                f"open_clip has no pretrained tag '{self.pretrained}' for {self.model_id}"
            )
        return open_clip.create_model_and_transforms(
            self.model_id,
            pretrained=str(self.snapshot / self.weights),
            image_mean=cfg.get("mean"),
            image_std=cfg.get("std"),
            image_interpolation=cfg.get("interpolation"),
            image_resize_mode=cfg.get("resize_mode"),
        )

    def _fingerprint(self, model, preprocess) -> str:
        """Hash the loaded weights plus the preprocess transform.

        Two checkpoints under the same name must not collide, and the same
        weights read through a different resize/normalize must not either:
        the preprocess is part of what produced the vectors.
        """
        digest = hashlib.sha256()
        state = model.state_dict()
        for key in sorted(state):
            tensor = state[key]
            digest.update(key.encode("utf-8"))
            digest.update(f"{tuple(tensor.shape)}|{tensor.dtype}".encode())
            digest.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
        payload = "\n".join(
            [self.model_id, self.pretrained or "", repr(preprocess), digest.hexdigest(), "l2"]
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_dim(self, model) -> int:
        for holder in (model, getattr(model, "visual", None)):
            dim = getattr(holder, "output_dim", None)
            if isinstance(dim, int) and dim > 0:
                return dim
        # last resort: ask the model, rather than guess from an unrelated
        # weight matrix (the text tower's width is not the joint dim).
        probe = np.zeros((8, 8, 3), dtype=np.uint8)
        return int(self.embed_arrays([probe]).shape[1])

    @property
    def dim(self) -> int:
        self._load()
        return self._dim  # type: ignore[return-value]

    @property
    def model_hash(self) -> str:
        self._load()
        return self._model_hash  # type: ignore[return-value]

    def _encode(self, images: list) -> np.ndarray:
        import torch

        tensors = [self._preprocess(img) for img in images]
        with torch.no_grad():
            feats = self._model.encode_image(torch.stack(tensors).to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def embed_paths(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """Return an (n, dim) L2-normalized float32 array for files on disk."""
        self._load()
        paths = [Path(p) for p in image_paths]
        out = []
        for start in range(0, len(paths), self.batch_size):
            batch = []
            for path in paths[start : start + self.batch_size]:
                with self._image.open(path) as img:
                    batch.append(img.convert("RGB"))
            out.append(self._encode(batch))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_arrays(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Return an (n, dim) array for already-decoded RGB frames.

        This is the path used when the index must represent the compressed
        frames rather than the source pixels.
        """
        self._load()
        out = []
        for start in range(0, len(frames), self.batch_size):
            batch = [
                self._image.fromarray(arr).convert("RGB")
                for arr in frames[start : start + self.batch_size]
            ]
            out.append(self._encode(batch))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_one(self, image_path: str | Path) -> np.ndarray:
        return self.embed_paths([image_path])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts through the model's TEXT tower, same joint space.

        The space is an image-text set: a clinical description searches the
        corpus directly, and the `model_hash` gate applies unchanged because
        the weights behind both towers are the ones fingerprinted. The
        tokenizer is not in the fingerprint; with a snapshot it is read from
        the snapshot's files (`local-dir:`), the revision the weights came
        from. By name, an hf tokenizer (siglip2) resolves through the hub and
        fails offline on a repo without config.json, even fully cached.
        """
        self._load()
        import open_clip
        import torch

        if self._tokenizer is None:
            source = f"local-dir:{self.snapshot}" if self.snapshot is not None else self.model_id
            self._tokenizer = open_clip.get_tokenizer(source)
        with torch.no_grad():
            feats = self._model.encode_text(self._tokenizer(list(texts)).to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def list_images(root: Path, extensions: Sequence[str] | None = None) -> list[Path]:
    """Every image under `root`, in a stable sorted order.

    The order is the corpus ordinal, so it has to be deterministic across
    machines: `rglob` alone is filesystem-ordered.
    """
    ext_set = {e.lower() for e in (extensions or IMAGE_EXTENSIONS)}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    paths.sort()
    return paths
