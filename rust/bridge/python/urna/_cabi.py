"""_cabi.py - the urna C ABI (rust/cbinds, include/urna.h) through ctypes.

the first part of #643's bridge: the potion embedder of `urna-neural`, so
the forge tokenizes and embeds with our WordPiece and our kernels instead of
`tokenizers` and `numpy`. stdlib only (ctypes), no build step of its own.

the library is the `urna-cbinds` cdylib (`cargo build --release -p
urna-cbinds`), found at, in order:

1. $URNA_CBINDS, a path to the library (an explicit path that does not
   exist is an error, not a fallback);
2. beside this file, as a package would ship it;
3. target/release/ then target/debug/ of the source tree this file sits in.

`library()` is None when none exists; callers keep their reference path.
"""

from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_NAME = {"darwin": "liburna_cbinds.dylib", "win32": "urna_cbinds.dll"}.get(
    sys.platform, "liburna_cbinds.so"
)
SIZE_MAX = ctypes.c_size_t(-1).value


class CabiError(RuntimeError):
    """A call into the library failed; the message is urna_last_error()."""


def _candidates() -> list[Path]:
    explicit = os.environ.get("URNA_CBINDS")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"URNA_CBINDS={explicit}: no such library")
        return [Path(explicit)]
    found = [HERE / LIB_NAME]
    # rust/bridge/python/urna/ -> the repo root, when this is a source tree
    root = HERE.parents[3] if len(HERE.parents) > 3 else None
    if root is not None and (root / "rust" / "cbinds").is_dir():
        found += [root / "target" / "release" / LIB_NAME, root / "target" / "debug" / LIB_NAME]
    return found


def _declare(lib: ctypes.CDLL) -> None:
    p, s = ctypes.c_void_p, ctypes.c_char_p
    u32p, f32p = ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float)
    sigs = {
        "urna_version": ([], s),
        "urna_last_error": ([], s),
        "urna_embedder_open": ([s], p),
        "urna_embedder_close": ([p], None),
        "urna_embedder_dim": ([p], ctypes.c_size_t),
        "urna_embedder_model_hash": ([p], s),
        "urna_embedder_tokenize": ([p, s, u32p, ctypes.c_size_t], ctypes.c_size_t),
        "urna_embedder_embed": ([p, s, f32p], ctypes.c_int32),
    }
    for name, (args, res) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = res


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL | None:
    for path in _candidates():
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            _declare(lib)
            return lib
    return None


def last_error(lib: ctypes.CDLL) -> str:
    return (lib.urna_last_error() or b"").decode("utf-8", "replace")


def _c_text(text: str) -> bytes:
    # a C string ends at its first NUL; the BERT normalizer drops NUL as a
    # control character anyway, so removing it keeps the tokens the same.
    return text.replace("\0", "").encode("utf-8")


class Embedder:
    """The potion embedder of `model_dir`, loaded once in Rust."""

    def __init__(self, model_dir: str | os.PathLike, lib: ctypes.CDLL | None = None):
        lib = lib or library()
        if lib is None:
            raise FileNotFoundError(f"{LIB_NAME} not found (build it: cargo build -p urna-cbinds)")
        handle = lib.urna_embedder_open(os.fsencode(model_dir))
        if not handle:
            raise CabiError(last_error(lib))
        self._lib = lib
        self._handle = handle
        self.dim = int(lib.urna_embedder_dim(handle))
        self.model_hash = lib.urna_embedder_model_hash(handle).decode()

    def tokenize(self, text: str) -> list[int]:
        """The WordPiece ids of `text`, no special tokens."""
        raw = _c_text(text)
        cap = max(16, len(raw))
        while True:
            buf = (ctypes.c_uint32 * cap)()
            n = self._lib.urna_embedder_tokenize(self._handle, raw, buf, cap)
            if n == SIZE_MAX:
                raise CabiError(last_error(self._lib))
            if n <= cap:
                return list(buf[:n])
            cap = n

    def embed(self, text: str) -> list[float]:
        """The unit-norm vector of `text`, `dim` float32 values."""
        out = (ctypes.c_float * self.dim)()
        if self._lib.urna_embedder_embed(self._handle, _c_text(text), out) != 0:
            raise CabiError(last_error(self._lib))
        return list(out)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.urna_embedder_close(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()
