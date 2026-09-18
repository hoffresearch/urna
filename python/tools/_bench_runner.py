"""Latency-bench helpers used by `measure_presets.py`.

Pure functions over a `urna.UrnaFile` plus a list of `(qvec, qtext)`
queries — no `.urna` I/O, no result formatting. Internal to
`python/tools/`.
"""

from __future__ import annotations

import time
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((len(s) - 1) * p)))
    return s[idx]


def run_bench(
    db,
    queries,
    k: int,
    mode: str,
    ef: int = 100,
    candidates: int = 200,
):
    """Time `len(queries)` invocations of the requested search mode.

    Returns `(times_ms, hits_per_query)`. Caller computes recall against
    a baseline `hits_per_query` and percentiles over `times_ms`.
    """
    times: list[float] = []
    results = []
    for qvec, qtext in queries:
        t0 = time.time()
        if mode == "exact":
            hits = db.search(qvec, k)
        elif mode == "ann":
            hits = db.search_ann(qvec, k, ef)
        elif mode == "hybrid":
            hits = db.search_hybrid(qvec, qtext, k, candidates)
        else:
            raise ValueError(mode)
        dt = (time.time() - t0) * 1000.0
        times.append(dt)
        results.append(hits)
    return times, results


def parse_variant(name: str):
    """Resolve a variant name into urna.build kwargs.

    Three forms:
      - a plain preset: "exact" | "compressed" | "tiny" | "nano" | "hybrid"
        (built via `preset=`, default base-dim).
      - a named ladder point: "micro" | "nano". These are the two explicit,
        published sub-int8 ladder rungs. `nano` is the existing preset
        (int4 full-dim, ~0.209 ratio, ~0.913 recall@10). `micro` is the
        matryoshka lever mapped to the documented honest point mrl256-int8
        (mrl_dim=256 + int8, ~0.223 ratio, ~0.810 recall@10 on the non-mrl
        MiniLM baseline). `micro` is an ALIAS for mrl256-int8 kwargs, labelled
        "micro" so the published ladder has a stable name; both surface dtype
        and (for micro) mrl_dim/full_dim, so the stored precision is disclosed.
      - a matryoshka ladder point: "mrl<DIM>-<dtype>", where <dtype> is one
        of f32|f16|int8|int4 (e.g. "mrl256-int8", "mrl128-int4"). Built with
        `mrl_dim=<DIM>` + the explicit dtype on a zstd-text/hnsw base so it is
        comparable to the existing tiny/nano presets. The truncation +
        L2-renorm happens build-time in the rust builder before quantization.

    Returns `(label, kwargs)` where kwargs feed straight into urna.build.
    """
    dtype_alias = {
        "f32": "float32",
        "f16": "float16",
        "int8": "int8",
        "int4": "int4",
        "float32": "float32",
        "float16": "float16",
    }
    # `micro` is the named matryoshka rung of the published ladder: it reuses
    # the mrl256-int8 build path (mrl_dim=256, dtype=int8) but keeps the stable
    # public label "micro". `nano` falls through to the plain-preset path.
    if name == "micro":
        return name, dict(
            text_encoding="zstd",
            dtype="int8",
            mrl_dim=256,
            with_hnsw=True,
            with_bm25=False,
        )
    if name.startswith("mrl") and "-" in name:
        dim_part, dtype_part = name[3:].split("-", 1)
        mrl_dim = int(dim_part)
        dtype = dtype_alias.get(dtype_part)
        if dtype is None:
            raise ValueError(f"unknown dtype in variant {name!r}: {dtype_part}")
        kwargs = dict(
            text_encoding="zstd",
            dtype=dtype,
            mrl_dim=mrl_dim,
            with_hnsw=True,
            with_bm25=False,
        )
        return name, kwargs
    return name, dict(preset=name)


def build_variant(chunks, meta, preset: str, out_path: Path):
    """Build `out_path` with the given preset or mrl ladder point; return
    seconds elapsed.

    Imports `urna` lazily because `_bench_runner` is meant to be cheap
    to import (unlike the PyO3 extension load, which pulls a 1.6 MB .so).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import urna

    _label, variant_kwargs = parse_variant(preset)

    if out_path.exists():
        out_path.unlink()
    t0 = time.time()
    urna.build(
        output_path=str(out_path),
        embedding_model=meta["embedding_model"],
        embedding_dim=meta["embedding_dim"],
        chunker_version=meta["chunker_version"],
        model_hash=meta["model_hash"],
        chunks=chunks,
        reproducible=True,
        **variant_kwargs,
    )
    return time.time() - t0
