"""console entry point shipped by the `urna` wheel (issue #75).

`uvx --from urna urna ...` / `pipx run` get a working `urna` command from
the python package alone. this is a THIN read-only shim over the library
api: validate / inspect / stats / search. the full verb set (ask, retrieve,
search-text, benchmark, cite, doctor) lives in the rust `urna` binary,
installed by scripts/install.sh.

dev repo usage:  python3 python/urna_cli.py validate dat/corpus_next.v1.urna
installed usage: urna validate corpus.urna
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # dev layout
import urna  # noqa: E402


def _cmd_validate(args) -> int:
    db = urna.open(args.file)
    db.validate()
    print(f"ok: {args.file}")
    print(f"  file_hash:    {db.file_hash}")
    print(f"  content_hash: {db.content_hash}")
    return 0


def _cmd_inspect(args) -> int:
    db = urna.open(args.file)
    print(json.dumps(db.inspect(), indent=2, default=str))
    return 0


def _cmd_stats(args) -> int:
    db = urna.open(args.file)
    print(f"file:           {args.file}")
    print(f"embedding_dim:  {db.embedding_dim}")
    print(f"n_embeddings:   {db.n_embeddings}")
    print(f"dtype:          {db.dtype}")
    print(f"simd_backend:   {db.simd_backend}")
    print(f"has_ann:        {db.has_ann}")
    print(f"has_bm25:       {db.has_bm25}")
    print(f"has_graph:      {db.has_graph}")
    print(f"model_hash:     {db.model_hash}")
    print(f"file_hash:      {db.file_hash}")
    print(f"content_hash:   {db.content_hash}")
    return 0


def _cmd_search(args) -> int:
    qvec = json.loads(args.query)
    db = urna.open(args.file)
    hits = db.search(qvec, args.k)
    for i, h in enumerate(hits):
        print(f"[{i + 1}] score={h.score:.6f} chunk_id={h.chunk_id} citation={h.citation_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="urna",
        description="read-only urna verbs from the urna wheel; "
        "the full cli is the rust binary (scripts/install.sh)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="full integrity check")
    p.add_argument("file")
    p.set_defaults(fn=_cmd_validate)

    p = sub.add_parser("inspect", help="header, manifest, hashes as json")
    p.add_argument("file")
    p.set_defaults(fn=_cmd_inspect)

    p = sub.add_parser("stats", help="sizes, dtype, simd backend, hashes")
    p.add_argument("file")
    p.set_defaults(fn=_cmd_stats)

    p = sub.add_parser("search", help="exact top-k; query is a json array of f32")
    p.add_argument("file")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(fn=_cmd_search)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except Exception as e:  # the pyo3 layer raises typed errors; print + exit
        print(f"urna: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
