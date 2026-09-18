"""Convert a legacy SQLite-based truw_ptbr.urna into the new v1 binary format.

Legacy layout:
  - articles(id, block_id, pos_in_block, source, label)
  - text_blocks(block_id, data: zstd[ u32 n_texts | n_texts * u32 offset | body ])
  - blobs(name, data) with name in {manifest, faiss_index, embeddings}
    - embeddings: zstd[ float16 (N, D) ]

New v1 .urna:
  - manifest.embedding_model from legacy manifest
  - one chunk per article, canonical_text = article body
  - synthetic source_uri = `legacy://truw_ptbr/<id>`, byte span = [0, len(utf8(text))]
  - provenance carries the legacy labels/sources so nothing is lost
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import time

# Force HF/sentence-transformers OFFLINE by default (opt in with
# URNA_ALLOW_DOWNLOAD=1) before any hub access. model_fingerprint also sets
# this on import; kept here too so the guarantee is explicit at the entry point.
if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_k, "1")

import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urna
from model_fingerprint import (
    PLACEHOLDER_HASH,
    compute_model_fingerprint,
    fingerprint_to_model_hash,
    resolve_model_dir,
)


def load_blocks(conn: sqlite3.Connection):
    """Yield (block_id, [text, ...]) for every text_block, in block_id order."""
    dec = zstd.ZstdDecompressor()
    for block_id, data in conn.execute("SELECT block_id, data FROM text_blocks ORDER BY block_id"):
        raw = dec.decompress(data)
        n = struct.unpack_from("<I", raw, 0)[0]
        offsets = struct.unpack_from(f"<{n}I", raw, 4)
        body = raw[4 + n * 4 :]
        texts: list[str] = []
        for i in range(n):
            start = offsets[i]
            end = offsets[i + 1] if i + 1 < n else len(body)
            texts.append(body[start:end].decode("utf-8"))
        yield block_id, texts


def load_embeddings(conn: sqlite3.Connection, n: int, dim: int) -> np.ndarray:
    dec = zstd.ZstdDecompressor()
    row = conn.execute("SELECT data FROM blobs WHERE name='embeddings'").fetchone()
    if row is None:
        raise SystemExit("legacy file has no `embeddings` blob")
    raw = dec.decompress(row[0])
    arr = np.frombuffer(raw, dtype=np.float16).reshape(n, dim).astype(np.float32)
    # the legacy builder stored unit-normalized vectors but float16 round-trip
    # can drift the norm by ~1e-3. Re-normalize so the runtime's dot product
    # is exactly cosine.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def convert(src: str, dst: str, *, reproducible: bool) -> None:
    if not os.path.exists(src):
        raise SystemExit(f"source not found: {src}")

    t0 = time.time()
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    legacy_manifest = json.loads(
        conn.execute("SELECT data FROM blobs WHERE name='manifest'").fetchone()["data"]
    )
    n = legacy_manifest["n_articles"]
    dim = legacy_manifest["embedding_dim"]
    model = legacy_manifest["embedding_model"]
    print(f"legacy: {n} articles, dim={dim}, model={model}")

    # load all texts into memory, indexed by article id.
    texts: list[str | None] = [None] * n
    for block_id, block_texts in load_blocks(conn):
        for pos, text in enumerate(block_texts):
            aid = block_id * 256 + pos
            if aid < n:
                texts[aid] = text
    missing = [i for i, t in enumerate(texts) if t is None]
    if missing:
        raise SystemExit(f"{len(missing)} articles have no text (e.g. id={missing[:5]})")

    # per-article metadata.
    rows = list(conn.execute("SELECT id, source, label FROM articles ORDER BY id"))
    if len(rows) != n:
        raise SystemExit(f"articles table has {len(rows)} rows but manifest says {n}")

    print("loading embeddings...")
    embs = load_embeddings(conn, n, dim)

    chunks = []
    labels = []
    sources = []
    for r in rows:
        aid = r["id"]
        text = texts[aid]
        ub = text.encode("utf-8")
        chunks.append(
            dict(
                canonical_text=text,
                source_uri=f"legacy://truw_ptbr/{aid}",
                byte_start=0,
                byte_end=len(ub),
                embedding=embs[aid].tolist(),
            )
        )
        labels.append(r["label"])
        sources.append(r["source"])

    provenance = {
        "legacy_source": "truw_ptbr.urna (SQLite)",
        "legacy_version": legacy_manifest.get("version", "unknown"),
        "legacy_created": legacy_manifest.get("created"),
        "labels": labels,
        "sources": sources,
        "note": (
            "synthetic source_uri = legacy://truw_ptbr/<id> since the legacy "
            "format did not preserve original URIs"
        ),
    }

    # resolve the model snapshot and compute a real fingerprint. We
    # require the snapshot to be locally available — if it isn't, the
    # caller can re-download it via `python -c "from sentence_transformers
    # import SentenceTransformer; SentenceTransformer('<id>')"`. The
    # placeholder zero-hash is no longer accepted by the manifest.
    model_dir = resolve_model_dir(model)
    fp = compute_model_fingerprint(model_dir, model_id=model)
    model_hash = fingerprint_to_model_hash(fp)
    if model_hash == PLACEHOLDER_HASH:
        raise SystemExit(
            f"refusing to write placeholder model_hash for model={model}; "
            f"snapshot at {model_dir} appears to be missing relevant files"
        )
    print(f"computed model_hash: {model_hash}")
    print(f"  fingerprint: {fp.to_dict()}")

    if os.path.exists(dst):
        os.unlink(dst)
    urna.build(
        output_path=dst,
        embedding_model=model,
        embedding_dim=dim,
        chunker_version="legacy/truw_ptbr_v0.1.0",
        model_hash=model_hash,
        chunks=chunks,
        title="truw_ptbr",
        version="v1-from-legacy",
        description="Portuguese fake-news corpus, converted from the legacy SQLite-based .urna",
        license=legacy_manifest.get("license"),
        provenance=provenance,
        reproducible=reproducible,
    )
    elapsed = time.time() - t0
    size = os.path.getsize(dst)
    print(f"wrote {dst}: {size / 1e6:.2f} MB in {elapsed:.1f}s")

    # final integrity check through the same Rust reader the runtime uses.
    db = urna.open(dst)
    db.validate()
    print(
        f"validated: {dst}\n"
        f"  file_hash:    {db.file_hash}\n"
        f"  content_hash: {db.content_hash}\n"
        f"  chunks:       {db.n_embeddings}\n"
        f"  dim:          {db.embedding_dim}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="path to a legacy SQLite-based truw_ptbr.urna")
    p.add_argument("--dst", required=True, help="output path for the converted v1 .urna")
    p.add_argument("--reproducible", action="store_true")
    args = p.parse_args()
    convert(args.src, args.dst, reproducible=args.reproducible)


if __name__ == "__main__":
    main()
