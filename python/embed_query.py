"""Embed a single text query using the same sentence-transformers model
the corpus was built with.

Invoked by the Rust CLI's `urna search-text` subcommand. Stays in
Python because (a) sentence-transformers is the same toolchain used at
build time, so vectors are bit-identical (modulo float ops), and (b)
keeping the CLI binary lean — no candle/onnxruntime dependency.

Output: a single-line JSON document on stdout with the structured
shape the CLI expects:

    {
      "model_hash":      "sha256:...",          # compact hash for manifest match
      "fingerprint":     {...},                 # full ModelFingerprint dict
      "embedding_model": "<name as passed>",
      "embedding_dim":   384,
      "vector":          [<f32, ...>]
    }

The CLI cross-checks `embedding_model` and `model_hash` against the
manifest before running the search. A mismatch fails with a typed
error rather than silently returning cosine-valid garbage.

Vectors are L2-normalized so the runtime's cosine assumption holds.
Errors go to stderr with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Force HuggingFace/sentence-transformers OFFLINE by default, BEFORE the
# (lazy) sentence_transformers import ever runs. A hostile or misconfigured
# corpus model name must never trigger a hub download mid-run — especially
# while the box is handling PHI (see the data governance section of
# docs/SECURITY.md). Opt into the
# first-time model fetch explicitly with URNA_ALLOW_DOWNLOAD=1.
if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_k, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_fingerprint import (  # noqa: E402
    compute_model_fingerprint,
    fingerprint_to_model_hash,
    hf_cache_snapshot,
)


def _embed(model_name_or_path: str, query: str) -> tuple[list[float], int, str]:
    """Return (vector, dim, resolved_local_path) for `query`."""
    from sentence_transformers import SentenceTransformer  # local import: heavy

    model = SentenceTransformer(model_name_or_path)
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    dim = int(model.get_sentence_embedding_dimension())
    # defensive re-normalize (some sentence-transformers versions skip it
    # on certain backbones).
    n = math.sqrt(sum(float(x) * float(x) for x in vec))
    vec = [float(x) / n for x in vec] if n > 0 else [float(x) for x in vec]

    # locate the actual snapshot directory the model was loaded from.
    local_path = _resolve_local_path(model, model_name_or_path)
    return vec, dim, local_path


def _resolve_local_path(model, fallback: str) -> str:
    """Best-effort resolution of the SentenceTransformer's on-disk dir.

    Strategy:
    1. If `fallback` is already a local directory, use it.
    2. Inspect `_modules` for an `auto_model.config._name_or_path` that
       points to a real directory (older sentence-transformers).
    3. Resolve the HF cache path
       (`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>`)
       — works for sentence-transformers v3+ which only stores the HF
       id in the config.
    """
    p = Path(fallback).expanduser()
    if p.is_dir():
        return str(p.resolve())
    for mod in model._modules.values():
        auto_model = getattr(mod, "auto_model", None)
        if auto_model is None:
            continue
        cfg = getattr(auto_model, "config", None)
        nop = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        if nop and Path(nop).is_dir():
            return str(Path(nop).resolve())
    # hF cache fallback: resolve the revision refs/main points to (the one
    # actually loaded), NOT an arbitrary alphabetical snapshot. A bare name
    # with no org ("all-MiniLM-L6-v2") is loaded by sentence-transformers
    # from its own org, so retry with that prefix before giving up.
    snap = hf_cache_snapshot(fallback)
    if snap is None and "/" not in fallback:
        snap = hf_cache_snapshot(f"sentence-transformers/{fallback}")
    if snap is not None:
        return str(snap)
    return fallback


def _embed_dim(model_name_or_path: str) -> int:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name_or_path)
    return int(model.get_sentence_embedding_dimension())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embed-dim",
        action="store_true",
        help="print the model's embedding dim and exit (legacy, no fingerprint)",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help=(
            "Local path to the model snapshot. Overrides the default "
            "sentence-transformers cache resolution. Use this for fully "
            "offline operation: copy the model directory once, point "
            "--model-path at it forever."
        ),
    )
    p.add_argument("model", help="HF id or local path; --model-path overrides")
    p.add_argument("query", nargs="?", default="")
    args = p.parse_args()

    model_arg = args.model_path or args.model

    if args.embed_dim:
        print(_embed_dim(model_arg))
        return 0

    if not args.query:
        print("error: query required", file=sys.stderr)
        return 2

    vec, dim, local_path = _embed(model_arg, args.query)
    fp = compute_model_fingerprint(local_path, model_id=args.model)
    model_hash = fingerprint_to_model_hash(fp)
    payload = {
        "model_hash": model_hash,
        "fingerprint": fp.to_dict(),
        "embedding_model": args.model,
        "embedding_dim": dim,
        "vector": vec,
    }
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
