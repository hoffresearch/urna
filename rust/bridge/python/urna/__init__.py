"""Python entry point for the .urna binary format.

Loads the PyO3 extension `_urna` (built from the `urna-bridge` Rust crate)
and re-exports a stable surface:

  - urna.open(path)                     -> UrnaFile
  - UrnaFile.search(query, k)           -> list[SearchHit] (exact, recall=1.0)
  - UrnaFile.search_ann(query, k, ef)   -> list[SearchHit] (HNSW + exact rerank)
  - UrnaFile.search_hybrid(query, query_text, k, candidates) -> list[SearchHit]
  - UrnaFile.retrieve(query, k, ..., query_text=None) -> list[RetrieveHit]
        (agent-native: routes by what the file carries, hybrid when it has a
        bm25 section and `query_text` is given, hnsw when it has an hnsw
        section, exact otherwise; score IS the exact-cosine rerank value, each
        hit carries the tier-1 stored canonical text + verifying hashes + the
        urna:// citation_id + the rerank_source precision marker. embed the
        query OFFLINE first; see rust/bridge/python/urna/reads/retrieval.py for the potion path.)
  - UrnaFile.embedding_dim
  - UrnaFile.n_embeddings
  - UrnaFile.dtype                       ("float32" | "float16" | "int8")
  - UrnaFile.simd_backend                ("scalar" | "avx2" | "neon")
  - UrnaFile.has_ann / has_bm25
  - UrnaFile.file_hash / content_hash
  - SearchHit fields: chunk_id, score, score_type, source_uri,
    offset_start, offset_end, embedding_model, index_type, reranked,
    file_hash, content_hash, citation_id
  - urna.build(..., preset=...)         -> path
  - urna.chunk_id(text, source_uri, byte_start, byte_end, chunker_version)
"""

import importlib.util
import os


def _load_extension():
    """Load the `_urna` PyO3 extension.

    `_urna` is a submodule of the package in both layouts: the installed
    wheel names it `_urna.abi3.so`, the dev repo has the cargo build copied
    to `rust/bridge/python/urna/_urna.so` (see README > install). a build
    left under another name (`lib_urna.dylib`) is loaded from its file.
    """
    try:
        from . import _urna

        return _urna
    except ImportError:
        pass
    base = os.path.dirname(os.path.abspath(__file__))
    for name in ("_urna.so", "_urna.abi3.so", "_urna.dylib", "lib_urna.dylib"):
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            spec = importlib.util.spec_from_file_location("_urna", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "Cannot find _urna extension. Run "
        "`cargo build --release -p urna-bridge && "
        "cp target/release/lib_urna.dylib rust/bridge/python/urna/_urna.so` "
        "from the repo root."
    )


# the extension loads on first use, not on `import urna`: the query embedders
# under urna/embed and urna/model run in the release payload, which carries
# no extension, and importing them imports this package first.
_EXPORTS = {
    "UrnaFile": "UrnaFile",
    "SearchHit": "SearchHitPy",
    "RetrieveHit": "RetrieveHitPy",
    "build": "build",
    "chunk_id": "chunk_id",
}
_ext = None


def _extension():
    global _ext
    if _ext is None:
        _ext = _load_extension()
    return _ext


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(_extension(), _EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def open(path: str):
    """Open a .urna file for read-only mmap-backed search."""
    return _extension().UrnaFile.open(path)


def potion_model_path() -> str | None:
    """Path to the bundled potion-base-8M model dir, or None.

    The wheel bundles the offline potion static table under
    `urna/model/potionb8m/` so `ask`/`retrieve`-style embedding works
    with no network after install. the dev repo keeps it at the same place
    in the package (git-lfs), so a checkout returns its path too.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base, "model", "potionb8m")
    return candidate if os.path.isdir(candidate) else None


__all__ = [
    "UrnaFile",
    "SearchHit",
    "RetrieveHit",
    "open",
    "build",
    "chunk_id",
    "potion_model_path",
]
