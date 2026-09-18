"""Reusable Python pipeline for building deterministic .urna files.

The pipeline stages are:

  1. ingest documents (caller-provided)
  2. chunk text with byte-accurate spans
  3. compute or attach embeddings (caller-provided callable)
  4. cache (chunk_id -> embedding) in a SQLite scratch DB so re-runs
     skip the embedding step
  5. emit a .urna file via urna.build()
  6. invoke `urna validate` on the result

The chunker, embedder and emitter are deliberately decoupled so a caller
can swap the embedding model or the chunking strategy without touching
the rest of the pipeline.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urna


@dataclass(frozen=True)
class ChunkSpec:
    """A pre-embedding chunk with the spans used to derive chunk_id."""

    canonical_text: str
    source_uri: str
    byte_start: int
    byte_end: int

    def chunk_id(self, chunker_version: str) -> str:
        return urna.chunk_id(
            self.canonical_text,
            self.source_uri,
            self.byte_start,
            self.byte_end,
            chunker_version,
        )


def chunk_text(
    text: str,
    source_uri: str,
    *,
    max_chars: int = 512,
    overlap: int = 0,
) -> list[ChunkSpec]:
    """Greedy character-window chunker. Splits on a hard character budget,
    with optional overlap. Returns chunks whose byte spans index into the
    UTF-8 encoding of `text`, so the spans round-trip through `urna cite`.

    The simplest possible thing that's still useful — production callers
    will want a sentence-aware splitter, but the chunk_id contract is
    independent of the splitter so it's easy to swap.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    encoded = text.encode("utf-8")
    chunks: list[ChunkSpec] = []
    char_idx = 0
    text_len = len(text)
    while char_idx < text_len:
        end = min(char_idx + max_chars, text_len)
        piece = text[char_idx:end]
        prefix_bytes = len(text[:char_idx].encode("utf-8"))
        piece_bytes = len(piece.encode("utf-8"))
        chunks.append(
            ChunkSpec(
                canonical_text=piece,
                source_uri=source_uri,
                byte_start=prefix_bytes,
                byte_end=prefix_bytes + piece_bytes,
            )
        )
        if end == text_len:
            break
        char_idx = end - overlap
    # sanity: spans must cover bytes within the original encoding.
    for c in chunks:
        assert c.byte_end <= len(encoded), "byte span overshoots source"
    return chunks


class EmbeddingCache:
    """SQLite scratch keyed by (chunk_id, model). Stores raw float32 embeddings.

    Intended to be reused across runs so re-builds skip the expensive
    embedding step for chunks that have already been computed.

    The key includes `model_key` (embedding_model + model_hash). A chunk_id is
    only a function of the text/spans/chunker, NOT the embedding model, so a
    cache keyed on chunk_id alone would silently hand back a PREVIOUS model's
    vectors when the same corpus is re-embedded with a different model —
    shipping a .urna whose vectors do not match its declared model_hash. Keying
    on the model closes that.

    Uses table `embeddings_v2`: an old chunk-id-only `embeddings` table (from a
    prior urna version) is simply not reused (chunks are re-embedded), never
    misread — no in-place migration, no stale-model hit.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS embeddings_v2 (
            chunk_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            data BLOB NOT NULL,
            PRIMARY KEY (chunk_id, model)
        );
    """

    def __init__(self, path: str, model_key: str = ""):
        self.path = path
        self.model_key = model_key
        self.conn = sqlite3.connect(path)
        self.conn.executescript(self.SCHEMA)

    def get(self, chunk_id: str, dim: int) -> list[float] | None:
        row = self.conn.execute(
            "SELECT dim, data FROM embeddings_v2 WHERE chunk_id=? AND model=?",
            (chunk_id, self.model_key),
        ).fetchone()
        if row is None:
            return None
        if row[0] != dim:
            raise ValueError(f"chunk {chunk_id} cached with dim={row[0]}, requested dim={dim}")
        return list(struct.unpack(f"<{dim}f", row[1]))

    def put(self, chunk_id: str, embedding: Sequence[float]) -> None:
        data = struct.pack(f"<{len(embedding)}f", *embedding)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings_v2(chunk_id, model, dim, data) VALUES(?,?,?,?)",
            (chunk_id, self.model_key, len(embedding), data),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass
class BuildConfig:
    output_path: str
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    model_hash: str
    title: str | None = None
    version: str | None = None
    description: str | None = None
    license: str | None = None
    reproducible: bool = True
    # encoding preset (see urna.build docstring for the full table): exact /
    # compressed / tiny / nano (int4, dim %64==0, stored-prec) / hybrid.
    preset: str = "exact"
    # per-knob overrides (None = inherit from preset)
    text_encoding: str | None = None  # "raw" | "zstd"
    dtype: str | None = None  # "float32" | "float16" | "int8" | "int4"
    # matryoshka prefix dim (None = no truncation). When set, each vector is
    # sliced to the first mrl_dim components and re-L2-normalized at build time
    # (before quantization); embedding_dim becomes mrl_dim, source dim recorded
    # as full_dim. int4 needs mrl_dim divisible by 64.
    mrl_dim: int | None = None
    with_hnsw: bool | None = None
    with_bm25: bool | None = None
    # G1 chunk-to-chunk graph: emit the additive graph_adjacency (0x0C) csr
    # (NEXT_CHUNK + top-graph_top_m SEMANTIC), excluded from content_hash.
    with_graph: bool = False
    graph_top_m: int = 8
    # opt-in chunk_overlap drop (text reclaim): implies with_graph (read-side
    # graph_context.neighbor_context rebuilds context). gate on a recall@10-vs-
    # baseline check (graph_recall_gate.py) before shipping a dropped corpus.
    drop_overlap: bool = False
    hnsw_m: int = 16
    hnsw_ef_construction: int = 400
    hnsw_seed: int = 42


class Pipeline:
    """Glue between chunking, embedding, caching and the urna writer.

    Usage::

        pipe = Pipeline(cfg, embedder=embed_fn, scratch_db="cache.sqlite")
        for source_uri, text in documents:
            for spec in chunk_text(text, source_uri):
                pipe.add(spec)
        pipe.emit()
    """

    def __init__(
        self,
        cfg: BuildConfig,
        *,
        embedder: Callable[[Sequence[ChunkSpec]], list[Sequence[float]]],
        scratch_db: str | None = None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        # key the cache by the model too, so re-embedding the same corpus with
        # a different model never reuses stale vectors under a new model_hash.
        model_key = f"{cfg.embedding_model}\x00{cfg.model_hash}"
        self.cache = EmbeddingCache(scratch_db, model_key=model_key) if scratch_db else None
        self._specs: list[ChunkSpec] = []

    def add(self, spec: ChunkSpec) -> None:
        self._specs.append(spec)

    def add_many(self, specs: Iterable[ChunkSpec]) -> None:
        self._specs.extend(specs)

    def _embeddings(self) -> list[list[float]]:
        # resolve from cache where possible; embed only the misses.
        cached: list[list[float] | None] = [None] * len(self._specs)
        misses_idx: list[int] = []
        misses: list[ChunkSpec] = []

        for i, spec in enumerate(self._specs):
            if self.cache is None:
                misses_idx.append(i)
                misses.append(spec)
                continue
            cid = spec.chunk_id(self.cfg.chunker_version)
            hit = self.cache.get(cid, self.cfg.embedding_dim)
            if hit is None:
                misses_idx.append(i)
                misses.append(spec)
            else:
                cached[i] = hit

        if misses:
            new_embs = self.embedder(misses)
            if len(new_embs) != len(misses):
                raise RuntimeError(
                    f"embedder returned {len(new_embs)} embeddings for {len(misses)} chunks"
                )
            for spec, idx, emb in zip(misses, misses_idx, new_embs, strict=False):
                if len(emb) != self.cfg.embedding_dim:
                    raise RuntimeError(
                        f"embedder produced dim={len(emb)}, expected {self.cfg.embedding_dim}"
                    )
                cached[idx] = list(emb)
                if self.cache is not None:
                    self.cache.put(spec.chunk_id(self.cfg.chunker_version), emb)

        # cached is fully populated by construction.
        return [c for c in cached if c is not None]  # type: ignore[misc]

    def emit(self, *, provenance: dict | None = None) -> str:
        if not self._specs:
            raise RuntimeError("pipeline has no chunks")
        embeddings = self._embeddings()

        chunks = [
            dict(
                canonical_text=s.canonical_text,
                source_uri=s.source_uri,
                byte_start=s.byte_start,
                byte_end=s.byte_end,
                embedding=emb,
            )
            for s, emb in zip(self._specs, embeddings, strict=False)
        ]

        # drop_overlap implies with_graph (read-side neighbor reconstruction
        # needs the NEXT_CHUNK edges); the caller still owns the recall@10 gate.
        want_graph = self.cfg.with_graph or self.cfg.drop_overlap

        if os.path.exists(self.cfg.output_path):
            os.unlink(self.cfg.output_path)
        urna.build(
            output_path=self.cfg.output_path,
            embedding_model=self.cfg.embedding_model,
            embedding_dim=self.cfg.embedding_dim,
            chunker_version=self.cfg.chunker_version,
            model_hash=self.cfg.model_hash,
            chunks=chunks,
            title=self.cfg.title,
            version=self.cfg.version,
            description=self.cfg.description,
            license=self.cfg.license,
            provenance=provenance,
            reproducible=self.cfg.reproducible,
            preset=self.cfg.preset,
            text_encoding=self.cfg.text_encoding,
            dtype=self.cfg.dtype,
            mrl_dim=self.cfg.mrl_dim,
            with_hnsw=self.cfg.with_hnsw,
            with_bm25=self.cfg.with_bm25,
            with_graph=want_graph,
            graph_top_m=self.cfg.graph_top_m,
            hnsw_m=self.cfg.hnsw_m,
            hnsw_ef_construction=self.cfg.hnsw_ef_construction,
            hnsw_seed=self.cfg.hnsw_seed,
        )

        # final integrity check via the in-process reader (PyO3 path).
        # no CLI subprocess: one Python entry point only.
        db = urna.open(self.cfg.output_path)
        db.validate()
        return self.cfg.output_path

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()
