"""Python entry point for the .urna binary format.

Loads the PyO3 extension `_urna` (built from the `urna-python` Rust crate)
and re-exports a stable surface:

  - urna.open(path)                     -> UrnaFile
  - UrnaFile.search(query, k)           -> list[SearchHit] (exact, recall=1.0)
  - UrnaFile.search_ann(query, k, ef)   -> list[SearchHit] (HNSW + exact rerank)
  - UrnaFile.search_hybrid(query, query_text, k, candidates) -> list[SearchHit]
  - UrnaFile.retrieve(query, k, ...)    -> list[RetrieveHit] (agent-native:
        routes by manifest capability, score IS the exact-cosine rerank value,
        each hit carries the tier-1 stored canonical text + verifying hashes +
        the urna:// citation_id + the rerank_source precision marker. embed the
        query OFFLINE first; see python/forge/retrieve.py for the potion path.)
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

    Two layouts share this file:
      - installed wheel: `urna` is a package and `_urna` is a proper
        submodule (`urna._urna`, named `_urna.abi3.so`), so a relative
        import resolves it.
      - dev repo: `urna.py` is a top-level module under `python/` and the
        extension sits next to it as `_urna.so` (see README > install);
        the relative import fails and we fall back to file-based loading.
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
        "`cargo build --release -p urna-python && "
        "cp target/release/lib_urna.dylib python/_urna.so` "
        "from the repo root."
    )


_mod = _load_extension()

UrnaFile = _mod.UrnaFile
SearchHit = _mod.SearchHitPy
RetrieveHit = _mod.RetrieveHitPy
build = _mod.build
chunk_id = _mod.chunk_id


def open(path: str):
    """Open a .urna file for read-only mmap-backed search."""
    return UrnaFile.open(path)


def potion_model_path() -> str | None:
    """Path to the bundled potion-base-8M model dir, or None.

    The wheel bundles the offline potion static table under
    `urna/models/potion-base-8M/` so `ask`/`retrieve`-style embedding works
    with no network after install. the dev repo keeps the table at
    `python/forge/models/potion-base-8M/` (git-lfs) and this returns None.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base, "models", "potion-base-8M")
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
