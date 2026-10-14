"""Compare two `presetrun.py --json` dumps against the project's
regression gates and exit non-zero on any failure.

Gates (all required to pass):

  compressed.size_ratio   <= 0.40
  tiny.size_ratio         <= 0.30
  tiny.recall_at_k        >= 0.95
  compressed.recall_at_k  >= 0.9999    (treated as "= 1.0")
  hybrid.recall_at_k      >= 0.99
  exact.p95_ms            <= baseline.exact.p95_ms * 1.50  (--p95-headroom)

Conditional sub-int8 gates (fire only when the rung is present in the run):

  nano.size_ratio         <= 0.25   nano.recall_at_k         >= 0.85
  micro.size_ratio        <= 0.25   micro.recall_at_k        >= 0.78
  mrl256-int8.size_ratio  <= 0.25   mrl256-int8.recall_at_k  >= 0.78

`micro` is the named matryoshka rung of the published ladder, an alias for the
mrl256-int8 build path; its recall floor sits below nano's because the shipped
MiniLM baseline is not matryoshka-trained, so prefix truncation costs real
recall. these are real cosine AT THE STORED int4/int8 precision (the 0x09
embeddings_fp source is not wired), disclosed via dtype + mrl_dim/full_dim.

Tolerant of the legacy first-line print bug that lived in
`presetrun.py` between Phase 0 and Phase 2: we strip any leading
non-JSON lines before parsing.

Usage:
    python tool/bench/benchgate.py BASELINE.json POST.json
    echo $?   # 0 = all gates pass, 1 = any gate failed

The "baseline" file is also used as the reference for exact.p95_ms.
That makes the latency gate adaptive - production-realistic builds get
the headroom they earned in the baseline run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    """Load a presetrun JSON dump.

    Tolerates the Phase-0 bug where the script printed `baseline: ...`
    to stdout before the JSON document, which corrupted the output.
    Strips anything before the first `{`.
    """
    raw = path.read_text()
    brace = raw.find("{")
    if brace == -1:
        raise SystemExit(f"{path}: no JSON object found")
    return json.loads(raw[brace:])


def by_preset(doc: dict) -> dict[str, dict]:
    return {p["preset"]: p for p in doc.get("presets", [])}


def fmt(val: float | int | None, width: int = 8, prec: int = 4) -> str:
    if val is None:
        return "-".rjust(width)
    return f"{val:>{width}.{prec}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate presetrun output against regression gates.")
    ap.add_argument("baseline", type=Path, help="reference presetrun JSON")
    ap.add_argument("post", type=Path, help="new presetrun JSON to validate")
    ap.add_argument(
        "--p95-headroom",
        type=float,
        default=1.50,
        help=(
            "multiplier on baseline.exact.p95_ms (default 1.50 = +50%%). "
            "Wider than you'd expect because p95 over 50–100 queries is "
            "noisy: a single OS hiccup pushes it 30%% in either direction. "
            "The gate exists to catch real regressions (e.g. SIMD broken, "
            "10× slowdown), not jitter."
        ),
    )
    args = ap.parse_args()

    base = load_metrics(args.baseline)
    post = load_metrics(args.post)
    # metrics from two corpora say nothing about a regression.
    if base.get("baseline_file_hash") != post.get("baseline_file_hash"):
        print(
            f"ERROR: the baseline was measured on {base.get('baseline_file_hash')} and the "
            f"post on {post.get('baseline_file_hash')}; compare runs of the same corpus",
            file=sys.stderr,
        )
        return 1

    base_p = by_preset(base)
    post_p = by_preset(post)

    needed = ["exact", "compressed", "tiny", "hybrid"]
    for name in needed:
        if name not in post_p:
            print(f"ERROR: post is missing preset '{name}'", file=sys.stderr)
            return 1

    base_exact_p95 = base_p.get("exact", {}).get("p95_ms")
    if base_exact_p95 is None:
        print("ERROR: baseline.exact.p95_ms missing", file=sys.stderr)
        return 1
    p95_limit = base_exact_p95 * args.p95_headroom

    gates = [
        (
            "compressed.size_ratio  ≤ 0.40",
            post_p["compressed"]["size_ratio"] <= 0.40,
            post_p["compressed"]["size_ratio"],
        ),
        (
            "tiny.size_ratio        ≤ 0.30",
            post_p["tiny"]["size_ratio"] <= 0.30,
            post_p["tiny"]["size_ratio"],
        ),
        (
            "tiny.recall_at_k       ≥ 0.95",
            post_p["tiny"]["recall_at_k"] >= 0.95,
            post_p["tiny"]["recall_at_k"],
        ),
        (
            "compressed.recall_at_k = 1.000",
            post_p["compressed"]["recall_at_k"] >= 0.9999,
            post_p["compressed"]["recall_at_k"],
        ),
        (
            "hybrid.recall_at_k     ≥ 0.99",
            post_p["hybrid"]["recall_at_k"] >= 0.99,
            post_p["hybrid"]["recall_at_k"],
        ),
        # nano (int4, stored precision): the first sub-int8 size lever.
        # gates only fire when the run includes nano. the recall floor sits
        # below tiny's 0.95 because int4 is the coarser stored precision
        # (real cosine AT THAT precision, disclosed via dtype); the headroom
        # absorbs query-sample noise while still catching a real regression.
        *(
            [
                (
                    "nano.size_ratio        ≤ 0.25",
                    post_p["nano"]["size_ratio"] <= 0.25,
                    post_p["nano"]["size_ratio"],
                ),
                (
                    "nano.recall_at_k       ≥ 0.85",
                    post_p["nano"]["recall_at_k"] >= 0.85,
                    post_p["nano"]["recall_at_k"],
                ),
            ]
            if "nano" in post_p
            else []
        ),
        # micro (the named matryoshka rung of the published ladder, aliased to
        # mrl256-int8): gates only fire when the run includes micro. same floors
        # as the underlying mrl256-int8 point. the recall floor is BELOW nano's
        # 0.85 because the shipped baseline is plain MiniLM, which is NOT
        # matryoshka-trained, so prefix truncation costs real recall
        # (front-loading is not guaranteed): micro measures ~0.81 recall@10 at
        # ~0.223 size_ratio over 100 queries. honestly disclosed as the cost of
        # the dimension lever on a non-MRL model, not a cherry-pick.
        *(
            [
                (
                    "micro.size_ratio       ≤ 0.25",
                    post_p["micro"]["size_ratio"] <= 0.25,
                    post_p["micro"]["size_ratio"],
                ),
                (
                    "micro.recall@k         ≥ 0.78",
                    post_p["micro"]["recall_at_k"] >= 0.78,
                    post_p["micro"]["recall_at_k"],
                ),
            ]
            if "micro" in post_p
            else []
        ),
        # mrl256-int8 (the same point as micro, surfaced under its raw curve
        # label): gates only fire when that curve point is present. kept for the
        # standalone mrl-curve runs that do not name a micro rung.
        *(
            [
                (
                    "mrl256-int8.size_ratio ≤ 0.25",
                    post_p["mrl256-int8"]["size_ratio"] <= 0.25,
                    post_p["mrl256-int8"]["size_ratio"],
                ),
                (
                    "mrl256-int8.recall@k   ≥ 0.78",
                    post_p["mrl256-int8"]["recall_at_k"] >= 0.78,
                    post_p["mrl256-int8"]["recall_at_k"],
                ),
            ]
            if "mrl256-int8" in post_p
            else []
        ),
        (
            (
                f"exact.p95_ms           ≤ {p95_limit:.3f} ms "
                f"(baseline {base_exact_p95:.3f} × {args.p95_headroom})"
            ),
            post_p["exact"]["p95_ms"] <= p95_limit,
            post_p["exact"]["p95_ms"],
        ),
    ]

    # pretty print the metrics side-by-side.
    print(f"# baseline ({args.baseline.name}) → post ({args.post.name})")
    print(f"  baseline n_queries={base.get('n_queries')}  post n_queries={post.get('n_queries')}")
    print()
    print(
        f"{'preset':12s} {'size_MB':>10} {'size_ratio':>11} "
        f"{'recall@10':>10} {'p50_ms':>8} {'p95_ms':>8} {'drift_max':>10}"
    )
    for name in needed:
        p = post_p[name]
        print(
            f"{name:12s} {fmt(p['size_mb'], 10, 2)} {fmt(p['size_ratio'], 11, 3)} "
            f"{fmt(p['recall_at_k'], 10, 4)} {fmt(p['p50_ms'], 8, 2)} "
            f"{fmt(p['p95_ms'], 8, 2)} {fmt(p['drift_max'], 10, 6)}"
        )
    print()
    print("## regression gates")
    failed = 0
    for label, ok, val in gates:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{status}] {label} (got {val:.4f})")

    print()
    if failed == 0:
        print("ALL GATES PASS")
        return 0
    print(f"FAIL: {failed} gate(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
