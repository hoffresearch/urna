"""Pull (chunks, manifest_meta) out of a baseline `.urna` so we can
rebuild it under a different preset without re-embedding.

This decoder reaches inside the binary container directly (parses the
section table by hand) instead of using the PyO3 reader because we
want the raw f32 vectors, the canonical text strings, and the byte
spans — values the runtime intentionally hides behind its public
search API.

Internal to `python/tools/`. Not a public Python module.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
import urna  # noqa: E402  (sys.path inserted above)


def decode_baseline(path: Path):
    """Return `(chunks, meta)` ready to feed into `urna.build()`.

    `chunks` is a list of dicts with `canonical_text`, `source_uri`,
    `byte_start`, `byte_end`, `embedding`. `meta` is the manifest fields
    we need to rebuild (model, dim, chunker_version, model_hash).

    Raises `SystemExit` if the file is not raw-encoded.
    """
    db = urna.open(str(path))
    info = db.inspect()
    n = db.n_embeddings
    dim = db.embedding_dim
    print(f"baseline: {path}  n={n} dim={dim} dtype={db.dtype}", file=sys.stderr)

    raw = path.read_bytes()
    section_table_offset = int.from_bytes(raw[40:48], "little")
    section_table_count = int.from_bytes(raw[48:56], "little")

    embeddings_payload = None
    canonical_payload = None
    spans_payload = None
    chunk_ids_payload = None
    for i in range(section_table_count):
        eoff = section_table_offset + i * 32
        sid = int.from_bytes(raw[eoff : eoff + 4], "little")
        enc = int.from_bytes(raw[eoff + 4 : eoff + 8], "little")
        off = int.from_bytes(raw[eoff + 8 : eoff + 16], "little")
        size = int.from_bytes(raw[eoff + 16 : eoff + 24], "little")
        payload = raw[off : off + size]
        if enc != 0:
            raise SystemExit(f"baseline must be raw-encoded; section 0x{sid:02x} encoding={enc}")
        if sid == 0x04:
            embeddings_payload = payload
        elif sid == 0x02:
            canonical_payload = payload
        elif sid == 0x03:
            spans_payload = payload
        elif sid == 0x01:
            chunk_ids_payload = payload
    assert embeddings_payload and canonical_payload and spans_payload and chunk_ids_payload

    embs = list(struct.iter_unpack("<f", embeddings_payload))
    embs = [e[0] for e in embs]

    texts = _decode_strings(canonical_payload, n)
    spans = _decode_spans(spans_payload, n)

    chunks = []
    for i in range(n):
        chunks.append(
            dict(
                canonical_text=texts[i],
                source_uri=spans[i][0],
                byte_start=spans[i][1],
                byte_end=spans[i][2],
                embedding=embs[i * dim : (i + 1) * dim],
            )
        )
    meta = dict(
        embedding_model=info["manifest"]["embedding_model"],
        embedding_dim=dim,
        chunker_version=info["manifest"]["chunker_version"],
        model_hash=info["manifest"]["model_hash"],
    )
    return chunks, meta


def _decode_strings(buf: bytes, expected: int) -> list[str]:
    pos = 0
    ver = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    cnt = struct.unpack_from("<Q", buf, pos)[0]
    pos += 8
    assert ver == 1 and cnt == expected, (ver, cnt, expected)
    out: list[str] = []
    for _ in range(cnt):
        slen = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        out.append(buf[pos : pos + slen].decode("utf-8"))
        pos += slen
    return out


def _decode_spans(buf: bytes, expected: int) -> list[tuple[str, int, int]]:
    pos = 0
    ver = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    cnt = struct.unpack_from("<Q", buf, pos)[0]
    pos += 8
    assert ver == 1 and cnt == expected
    out: list[tuple[str, int, int]] = []
    for _ in range(expected):
        slen = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        uri = buf[pos : pos + slen].decode("utf-8")
        pos += slen
        start = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        end = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        out.append((uri, start, end))
    return out


# Re-export the path constant some callers want.
DEFAULT_BASELINE = REPO / "dat" / "corpus_next.v1.urna"
OUT_DIR = REPO / "dat" / "measure"

# Silence the "imported but unused" warning when this module is loaded
# for its side effects (sys.path insertion).
_ = os
