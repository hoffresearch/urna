"""Adapter classes behind `model_registry.create_embedder`.

One duck-typed surface over four backends: embed_texts/embed_paths/
embed_arrays -> (n, dim) float32 L2, plus `dim`, `model_hash` and
`fingerprint()`. Potion/open_clip wrap the existing embedders untouched;
the fake adapter is the deterministic no-ML test double; the st adapter
runs each sentence-transformers model in its own persistent worker
process (two trust_remote_code models in one process collide in
transformers' dynamic-module machinery; measured, see embed_st_worker).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import numpy as np

from forge.model_registry import CapabilityError, ModelPreset, RegistryError


class _PotionAdapter:
    def __init__(self, preset: ModelPreset, inner):
        self.preset, self._inner = preset, inner
        self.embedding_model = inner.embedding_model
        self.batch_size = 32

    @property
    def dim(self) -> int:
        return int(self._inner.embedding_dim)

    @property
    def model_hash(self) -> str:
        return self._inner.model_hash()

    def fingerprint(self) -> dict:
        return self._inner.fingerprint()

    def embed_texts(self, texts, role: str = "document") -> np.ndarray:
        return np.asarray(self._inner.embed_texts(list(texts)), dtype=np.float32)

    def embed_paths(self, paths) -> np.ndarray:
        raise CapabilityError(f"preset '{self.preset.name}' is text-only")

    embed_arrays = embed_paths


class _OpenClipAdapter:
    def __init__(self, preset: ModelPreset, inner):
        self.preset, self._inner = preset, inner
        self.embedding_model = preset.embedding_model
        self.batch_size = inner.batch_size

    @property
    def dim(self) -> int:
        return int(self._inner.dim)

    @property
    def model_hash(self) -> str:
        return self._inner.model_hash

    def fingerprint(self) -> dict:
        return {
            "embedder": "open_clip",
            "model_id": self.preset.model_id,
            "pretrained": self.preset.pretrained or "",
            "embedding_dim": self.dim,
            "normalize": "l2",
            "model_hash": self.model_hash,
        }

    def embed_texts(self, texts, role: str = "document") -> np.ndarray:
        return np.asarray(self._inner.embed_texts(list(texts)), dtype=np.float32)

    def embed_paths(self, paths) -> np.ndarray:
        return self._inner.embed_paths(list(paths))

    def embed_arrays(self, frames) -> np.ndarray:
        return self._inner.embed_arrays(list(frames))


class _FakeAdapter:
    """Deterministic no-ML adapter: sha256 of the input seeds a unit vector."""

    def __init__(self, preset: ModelPreset):
        self.preset = preset
        self.embedding_model = preset.embedding_model
        self.batch_size = 32

    @property
    def dim(self) -> int:
        return self.preset.default_dim

    @property
    def model_hash(self) -> str:
        return "sha256:" + hashlib.sha256(b"urna-forge-fake-test/v1").hexdigest()

    def fingerprint(self) -> dict:
        return {"embedder": "fake", "embedding_dim": self.dim, "normalize": "l2"}

    def _vec(self, payload: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_texts(self, texts, role: str = "document") -> np.ndarray:
        return np.stack([self._vec(t.encode()) for t in texts])

    def embed_paths(self, paths) -> np.ndarray:
        return np.stack([self._vec(Path(p).read_bytes()) for p in paths])

    def embed_arrays(self, frames) -> np.ndarray:
        return np.stack([self._vec(np.ascontiguousarray(f).tobytes()) for f in frames])


class _SubprocessSTAdapter:
    """st_multimodal adapter over a persistent one-model worker process.

    Two trust_remote_code ST models in one process collide in transformers'
    dynamic-module machinery (measured: jina's custom_st ends up wrapping
    wemm's weights); a worker per model is the isolation that actually holds,
    and closing the adapter returns the model's memory to the OS. Identity
    (fingerprint/model_hash) is computed IN-PROCESS from the files alone —
    no model load needed for a warm-cache triad check.
    """

    def __init__(self, preset, *, model_dir, model_path, device, batch_size, usage):
        self.preset = preset
        self.embedding_model = preset.embedding_model
        self.model_dir = model_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.usage = dict(usage)
        if device:
            self.usage["device_class"] = device
        self._proc = None
        self._tmp = None
        self._model_hash = None
        self._seq = 0

    @property
    def dim(self) -> int:
        if self.preset.default_dim:
            return self.preset.default_dim
        return int(self.embed_texts(["dim probe"]).shape[1])

    def _dtype_policy(self) -> str:
        from forge import embed_st

        device = self.usage.get("device_class")
        if device in (None, "", "auto"):
            device = embed_st._default_device()  # what the worker will actually use
        return self.usage.get("model_dtype") or embed_st._default_dtype(device)

    def fingerprint(self) -> dict:
        from forge import embed_st

        model_dir = self.model_dir
        if model_dir is None:
            from model_fingerprint import hf_cache_snapshot

            model_dir = hf_cache_snapshot(self.preset.model_id)
        normalize = self.usage.get("normalize", self.preset.normalize)
        return embed_st.fingerprint_for(self.preset, model_dir, normalize, self._dtype_policy())

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            import json as _json

            blob = _json.dumps(self.fingerprint(), sort_keys=True, separators=(",", ":"))
            self._model_hash = "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
        return self._model_hash

    def _ensure_worker(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        import json as _json
        import subprocess
        import sys as _sys
        import tempfile

        self._tmp = self._tmp or tempfile.mkdtemp(prefix=f"urna-st-{self.preset.name}-")
        worker = Path(__file__).parent / "embed_st_worker.py"
        cmd = [
            _sys.executable,
            str(worker),
            "--preset",
            self.preset.name,
            "--batch-size",
            str(self.batch_size),
            "--usage-json",
            _json.dumps(self.usage),
        ]
        if self.model_path:
            cmd += ["--model-path", str(self.model_path)]
        env = dict(
            os.environ, URNA_ENABLE_FAKE_PRESET=os.environ.get("URNA_ENABLE_FAKE_PRESET", "")
        )
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env
        )

    def _request(self, task: dict):
        import json as _json

        import numpy as np

        self._ensure_worker()
        self._seq += 1
        task_path = Path(self._tmp) / f"task-{self._seq}.json"
        task["out"] = str(Path(self._tmp) / f"out-{self._seq}.npz")
        task_path.write_text(_json.dumps(task))
        self._proc.stdin.write(str(task_path) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline().strip()
        if not line.startswith("ok "):
            raise RegistryError(
                f"preset '{self.preset.name}' worker failed: {line or 'worker died'}"
            )
        with np.load(line[3:]) as z:
            vecs = z["vectors"]
        task_path.unlink(missing_ok=True)
        Path(line[3:]).unlink(missing_ok=True)
        return vecs

    def embed_texts(self, texts, role: str = "document"):
        return self._request({"op": "texts", "texts": list(texts), "role": role})

    def embed_paths(self, paths):
        return self._request({"op": "paths", "paths": [str(p) for p in paths]})

    def embed_arrays(self, frames):
        import numpy as np

        self._ensure_worker()
        frames_npz = Path(self._tmp) / f"frames-{self._seq + 1}.npz"
        np.savez(frames_npz, *[np.asarray(f, dtype=np.uint8) for f in frames])
        try:
            return self._request({"op": "arrays", "frames_npz": str(frames_npz)})
        finally:
            frames_npz.unlink(missing_ok=True)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=30)
        self._proc = None

    def __del__(self):  # best-effort; close() is the real contract
        with contextlib.suppress(Exception):
            self.close()
