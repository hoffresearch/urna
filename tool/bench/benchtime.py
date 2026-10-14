"""Latency-bench helpers used by `presetrun.py`.

Pure functions over a `urna.UrnaFile` plus a list of `(qvec, qtext)`
queries - no `.urna` I/O, no result formatting. Internal to
`tool/bench/`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((len(s) - 1) * p)))
    return s[idx]


# timed runs per query in run_bench; the fastest is kept.
REPEATS = 5


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

    Every query runs once untimed first (the warm-up; its hits are the ones
    returned, so recall does not depend on the timing), then is timed
    `REPEATS` times and keeps its fastest: a system pause lands in some
    repeats, a real regression slows every one. The repeats are rounds over
    all the queries, so one query's repeats are spread out and a pause of a
    few milliseconds cannot cover all of them.
    """

    def search(qvec, qtext):
        if mode == "exact":
            return db.search(qvec, k)
        if mode == "ann":
            return db.search_ann(qvec, k, ef)
        if mode == "hybrid":
            return db.search_hybrid(qvec, qtext, k, candidates)
        raise ValueError(mode)

    results = [search(qvec, qtext) for qvec, qtext in queries]
    best = [float("inf")] * len(queries)
    for _ in range(REPEATS):
        for i, (qvec, qtext) in enumerate(queries):
            t0 = time.perf_counter()
            search(qvec, qtext)
            best[i] = min(best[i], time.perf_counter() - t0)
    times = [b * 1000.0 for b in best]
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


def _tmp_path(out_path: Path) -> Path:
    """The build target beside `out_path`: same directory (so the final
    rename is atomic), hidden, and still `*.urna` (so git ignores it)."""
    return out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.urna")


def _alive(pid: int) -> bool:
    """Whether `pid` names a running process. signal 0 probes without
    touching it on posix; on windows os.kill terminates instead, so there
    every other pid counts as alive and its temporary is left alone."""
    if pid == os.getpid():
        return False
    if os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # it exists but belongs to someone else
    return True


def _clear_dead_temporaries(out_path: Path) -> None:
    """Remove the temporaries of builds of `out_path` whose process is gone
    (a run that was killed); a temporary of a run still building is kept."""
    for tmp in out_path.parent.glob(f".{out_path.stem}.*.tmp.urna"):
        pid = tmp.name[len(out_path.stem) + 2 : -len(".tmp.urna")]
        if pid.isdigit() and not _alive(int(pid)):
            tmp.unlink(missing_ok=True)


def build_variant(chunks, meta, preset: str, out_path: Path):
    """Build `out_path` with the given preset or mrl ladder point; return
    seconds elapsed.

    The file is built under a temporary name in the same directory, opened
    and validated, and only then renamed over `out_path`. a build that
    fails, or a run interrupted halfway, never deletes or truncates the
    corpus already there; temporaries a killed run left behind are removed
    on the next build of the same preset, and the temporary of a run that
    is still building is left alone.

    Imports `urna` lazily because `benchtime` is meant to be cheap
    to import (unlike the PyO3 extension load, which pulls a 1.6 MB .so).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rust" / "bridge" / "python"))
    import urna

    _label, variant_kwargs = parse_variant(preset)

    _clear_dead_temporaries(out_path)
    tmp = _tmp_path(out_path)
    t0 = time.time()
    try:
        urna.build(
            output_path=str(tmp),
            embedding_model=meta["embedding_model"],
            embedding_dim=meta["embedding_dim"],
            chunker_version=meta["chunker_version"],
            model_hash=meta["model_hash"],
            chunks=chunks,
            reproducible=True,
            **variant_kwargs,
        )
        elapsed = time.time() - t0
        if urna.open(str(tmp)).validate() is not True:
            raise RuntimeError(f"{tmp} did not validate; {out_path} left as it was")
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return elapsed
