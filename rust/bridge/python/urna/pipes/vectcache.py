"""Embedding cache with the RFC-0 N2 key triad and N8 concurrency rules.

A cache entry is valid only when (model_hash, embedding_recipe_hash,
corpus_input_hash) all match; row keys alone are never a key. Entries are
content-addressed: the file name is a hash of the triad plus the arrays
flags, under a shared root (`cache_root`), so two specs that need the same
vectors read one file and a changed knob adds a sibling instead of
overwriting. Writes go to a temp file + atomic rename under an flock'd
lockfile; a sha256 sidecar guards against torn writes: any mismatch means
recompute, never reuse.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

TRIAD_KEYS = ("model_hash", "embedding_recipe_hash", "corpus_input_hash")


def cache_root(override: str = "") -> Path:
    """Where embed caches and model probes live, shared across specs and
    output dirs: explicit override (spec `[output] cache_dir`) > env
    `URNA_CACHE_DIR` > `${XDG_CACHE_HOME:-~/.cache}/urna`."""
    if override:
        return Path(os.path.expanduser(override))
    env = os.environ.get("URNA_CACHE_DIR", "")
    if env:
        return Path(os.path.expanduser(env))
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "urna"


@contextmanager
def locked(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(path, json.dumps(payload, sort_keys=True, indent=1).encode())


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def triad_hash(triad: dict) -> str:
    """16 hex chars of the canonical hash over the WHOLE triad dict (the
    three RFC-0 keys plus the arrays flags), the entry's file stem."""
    return canonical_hash(triad).removeprefix("sha256:")[:16]


class EmbedCache:
    """One content-addressed .npz per (preset, triad) under `<root>/embed/`,
    with its .sha256 sidecar and .lock beside it. The stored meta repeats
    the three triad keys and load() re-checks them as a second guard."""

    def __init__(self, root: Path, preset: str, triad: dict):
        self.triad = triad
        self.path = Path(root) / "embed" / preset / f"{triad_hash(triad)}.npz"

    def load(self) -> dict[str, np.ndarray] | None:
        """Return cached arrays iff the triad and the checksum both match."""
        with locked(self.path):
            if not (self.path.is_file() and _sidecar(self.path).is_file()):
                return None
            digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if _sidecar(self.path).read_text().strip() != digest:
                return None  # torn write: recompute
            with np.load(self.path, allow_pickle=False) as z:
                meta = json.loads(bytes(z["meta"]).decode())
                if any(meta.get(k) != self.triad[k] for k in TRIAD_KEYS):
                    return None
                return {k: z[k] for k in z.files if k != "meta"}

    def store(self, arrays: dict[str, np.ndarray]) -> None:
        meta = np.frombuffer(
            json.dumps({k: self.triad[k] for k in TRIAD_KEYS}, sort_keys=True).encode(),
            dtype=np.uint8,
        )
        with locked(self.path):
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".npz")
            os.close(fd)
            try:
                np.savez(tmp, meta=meta, **arrays)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
            atomic_write_bytes(_sidecar(self.path), digest.encode())


def canonical_hash(payload: dict) -> str:
    """sha256 over canonical JSON, the same convention as model fingerprints."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
