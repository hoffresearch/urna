"""Embed a single text query using the same sentence-transformers model
the corpus was built with.

Invoked by the Rust CLI's `urna search-text` subcommand. Stays in
Python because (a) sentence-transformers is the same toolchain used at
build time, so vectors are bit-identical (modulo float ops), and (b)
keeping the CLI binary lean - no candle/onnxruntime dependency.

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
Errors go to stderr with a non-zero exit code. Two of them carry one stable
line the CLI reads to name the fix: `urna-needs: sentence-transformers`
(exit 4) when the package does not import, `urna-fetch: <model>` (exit 3)
when the model is not in the local cache and downloads are off.
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
# corpus model name must never trigger a hub download mid-run - especially
# while the box is handling PHI (see the data governance section of
# docs/SECURITY.md). Opt into the
# first-time model fetch explicitly with URNA_ALLOW_DOWNLOAD=1.
if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_k, "1")

# run as a file (rust/bridge/python/ or the payload python/), the folder that
# holds urna/ goes first; imported as a module, the package is already there.
if not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from urna.model.modelhash import (
    compute_model_fingerprint,
    fingerprint_to_model_hash,
    hf_cache_snapshot,
)


def slice_renorm(vec: list[float], n: int) -> list[float]:
    """The builder's matryoshka truncation on one query vector: keep the first
    `n` components and re-normalize, so a truncated corpus (urna.build
    mrl_dim=n) is queried at its own dim with the same geometry. the model
    fingerprint is untouched: it names the model, not the slice."""
    head = [float(x) for x in vec[:n]]
    norm = math.sqrt(sum(x * x for x in head))
    return [x / norm for x in head] if norm > 0 else head


def load(model_name_or_path: str):
    """Return (model, resolved_local_path): the SentenceTransformer and the
    snapshot dir it was loaded from, which the fingerprint reads."""
    from sentence_transformers import SentenceTransformer  # local import: heavy

    model = SentenceTransformer(model_name_or_path)
    return model, _resolve_local_path(model, model_name_or_path)


def encode(model, texts: list[str]) -> list[list[float]]:
    """L2-normalized vectors, the encode every MiniLM-era corpus was built
    with; `urna.model.embedders._STTextAdapter` calls this same function."""
    out = []
    for vec in model.encode(texts, normalize_embeddings=True, convert_to_numpy=True):
        # defensive re-normalize (some sentence-transformers versions skip it
        # on certain backbones).
        n = math.sqrt(sum(float(x) * float(x) for x in vec))
        out.append([float(x) / n for x in vec] if n > 0 else [float(x) for x in vec])
    return out


def dimension(model) -> int:
    """the model's output dim. sentence-transformers 6 renamed
    `get_sentence_embedding_dimension` to `get_embedding_dimension` and warns
    on the old name; the supported floor (5.7) may predate the new one."""
    get = getattr(model, "get_embedding_dimension", None)
    if get is None:
        get = model.get_sentence_embedding_dimension
    return int(get())


def fingerprint(local_path: str, model_id: str):
    """The model fingerprint over the snapshot, keyed by the manifest name."""
    return compute_model_fingerprint(local_path, model_id=model_id)


def _embed(model_name_or_path: str, query: str) -> tuple[list[float], int, str]:
    """Return (vector, dim, resolved_local_path) for `query`."""
    model, local_path = load(model_name_or_path)
    dim = dimension(model)
    return encode(model, [query])[0], dim, local_path


def _resolve_local_path(model, fallback: str) -> str:
    """Best-effort resolution of the SentenceTransformer's on-disk dir.

    Strategy:
    1. If `fallback` is already a local directory, use it.
    2. Inspect `_modules` for an `auto_model.config._name_or_path` that
       points to a real directory (older sentence-transformers).
    3. Resolve the HF cache path
       (`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>`)
       - works for sentence-transformers v3+ which only stores the HF
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


NEEDS = "urna-needs:"
FETCH = "urna-fetch:"


def _not_cached(err: BaseException) -> bool:
    """True when the hub refused an offline lookup somewhere in the chain."""
    seen: BaseException | None = err
    while seen is not None:
        if type(seen).__name__ == "LocalEntryNotFoundError":
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _embed_dim(model_name_or_path: str) -> int:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name_or_path)
    return dimension(model)


def main(argv: list[str] | None = None) -> int:
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
    p.add_argument(
        "--mrl-dim",
        type=int,
        default=None,
        help=(
            "slice the vector to its first N components and re-normalize, for a "
            "corpus whose default space was built with mrl_dim (the cli passes the "
            "manifest dim when full_dim is recorded); the model_hash is unchanged"
        ),
    )
    p.add_argument("model", help="HF id or local path; --model-path overrides")
    p.add_argument("query", nargs="?", default="")
    args = p.parse_args(argv)

    model_arg = args.model_path or args.model

    if args.embed_dim:
        print(_embed_dim(model_arg))
        return 0

    if not args.query:
        print("error: query required", file=sys.stderr)
        return 2

    try:
        vec, dim, local_path = _embed(model_arg, args.query)
    except ModuleNotFoundError as e:
        if e.name != "sentence_transformers":
            raise
        print(
            f"error: '{args.model}' is a sentence-transformers model and this python "
            f'cannot import it. install with: pip install "sentence-transformers"',
            file=sys.stderr,
        )
        print(f"{NEEDS} sentence-transformers", file=sys.stderr)
        return 4
    except OSError as e:
        if not _not_cached(e):
            raise
        print(
            f"error: model '{args.model}' is not in the local cache and the embedder "
            "runs offline; URNA_ALLOW_DOWNLOAD=1 fetches it once",
            file=sys.stderr,
        )
        print(f"{FETCH} {args.model}", file=sys.stderr)
        return 3
    if args.mrl_dim:
        if not 0 < args.mrl_dim <= dim:
            print(f"error: --mrl-dim must be in 1..={dim}, got {args.mrl_dim}", file=sys.stderr)
            return 2
        vec = slice_renorm(vec, args.mrl_dim)
        dim = args.mrl_dim
    fp = fingerprint(local_path, args.model)
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
