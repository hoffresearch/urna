"""Vision embedder for image corpora.

This lives in the forge tooling layer because it pulls heavy optional
dependencies (torch, open_clip, Pillow). It never enters the sovereign
runtime, and no `.urna` reader needs it.

Models are addressed the way open_clip addresses them:
`hf-hub:redlessone/DermLIP_ViT-B-16` for dermatology, or a plain
architecture name plus a pretrained tag (`ViT-B-32` + `openai`) for the
general-image case.

`model_hash` fingerprints the weights that were actually loaded, not the
model name. The manifest gate only means something if a different
checkpoint produces a different hash, and a name plus a constant string
cannot do that.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

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
    ):
        self.model_id = model_id
        self.pretrained = pretrained
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
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_id, pretrained=self.pretrained
            )
        self._model_hash = self._fingerprint(model, preprocess)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._image = Image
        self._dim = self._resolve_dim(model)

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
        the weights behind both towers are the ones fingerprinted.
        """
        self._load()
        import open_clip
        import torch

        tokenizer = open_clip.get_tokenizer(self.model_id)
        with torch.no_grad():
            feats = self._model.encode_text(tokenizer(list(texts)).to(self.device))
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
