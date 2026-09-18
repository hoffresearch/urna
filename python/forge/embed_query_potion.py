"""embed a single text query OFFLINE with the default potion static table.

the offline twin of python/embed_query.py: that one loads sentence-transformers
(network on first use, breaks offline-by-construction), this one uses the
vendored model2vec/potion-base-8M table (numpy + tokenizers only, no torch, no
socket). the flagship verbs `urna ask` / `urna retrieve` shell out to THIS
script so an offline corpus built with the default embedder gets a cited answer
with no network.

output: a single-line json document on stdout with the SAME shape
python/embed_query.py emits, so the rust caller can reuse the search_text.rs
model_hash gate verbatim:

    {
      "model_hash":      "sha256:...",   # potion self-fingerprint, matches the manifest
      "fingerprint":     {...},          # the inference-relevant config dict
      "embedding_model": "<name>",       # potion's embedding_model name (echoed)
      "embedding_dim":   256,
      "vector":          [<f32, ...>]    # l2-normalized
    }

the rust caller passes the manifest's embedding_model name as the positional
`model` arg; this script ignores it for inference (the table is fixed) but
echoes potion's own name back so the name check in the caller is meaningful.
errors go to stderr with a non-zero exit code; never opens a socket.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/
from forge.embed_potion import potion_embedder  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default=None,
        help="local path to the vendored potion table dir (default: bundled).",
    )
    p.add_argument(
        "--mrl-dim",
        type=int,
        default=0,
        help="slice+renormalize the query to this dim (corpora built with build.mrl_dim)",
    )
    p.add_argument("model", help="manifest embedding_model name; echoed back, not for inference")
    p.add_argument("query", nargs="?", default="")
    args = p.parse_args()

    if not args.query:
        print("error: query required", file=sys.stderr)
        return 2

    emb = potion_embedder(args.model_path) if args.model_path else potion_embedder()
    try:
        vector = emb.embed_texts([args.query])[0]
    except FileNotFoundError as e:
        # vendored table missing / not pulled from git-lfs: fail loudly, offline.
        print(f"error: {e}", file=sys.stderr)
        return 3

    dim = emb.embedding_dim
    if args.mrl_dim:
        import numpy as np

        v = np.asarray(vector, dtype=np.float32)[: args.mrl_dim]
        norm = float(np.linalg.norm(v))
        vector = (v / norm if norm > 0 else v).tolist()
        dim = args.mrl_dim

    payload = {
        "model_hash": emb.model_hash(),
        "fingerprint": emb.fingerprint(),
        "embedding_model": emb.embedding_model,
        "embedding_dim": dim,
        "vector": vector,
    }
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
