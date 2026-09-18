"""Acceptance harness: measure file size, recall@k, score drift, and
latency for the four presets against a baseline `dat/corpus_next.v1.urna`.

Pipeline:

  1. Open the baseline (`exact` preset) — this is the recall=1.0 ground
     truth for both ranking and score.
  2. Decode embeddings + canonical texts + chunk_ids out of the baseline.
  3. For each variant in {compressed, tiny, hybrid}, rebuild a .urna with
     the same corpus and the variant preset.
  4. For N random queries (sampled from the corpus' own embeddings —
     deterministic given a seed):
       - exact (baseline) top-k
       - variant top-k (exact / ann / hybrid as appropriate)
       - recall@k = |overlap| / k
       - score drift = |variant_score[0] - exact_score[0]| (top-hit)
  5. Latency: per-variant p50/p95/p99 over the same queries.

Output: a markdown table to stdout. With `--json`, the table goes to
stderr and a structured dump goes to stdout (used by
`python/tools/compare_measure.py` for regression gates).

WEAK RULER: queries are corpus vectors plus tiny noise (self-perturbation),
so recall@10 here measures rank-stability under quantization, NOT real-query
retrieval. the JSON dump carries a `ruler` provenance block saying so. the
real-query (mteb-style) ruler is gate-zero; see the ruler note in
doc/CHANGELOG.

Helpers live in private siblings:
  `_baseline_decoder.py`  — section-table parser
  `_bench_runner.py`      — percentile / build_variant / run_bench
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "python"))

import urna  # noqa: E402
from _baseline_decoder import DEFAULT_BASELINE, OUT_DIR, decode_baseline  # noqa: E402
from _bench_runner import build_variant, percentile, run_bench  # noqa: E402

# RULER PROVENANCE (machine-readable, emitted into the JSON dump). every recall
# number this harness reports comes from the self-perturbation ruler below: the
# query is a corpus vector plus tiny deterministic noise, so the figure measures
# rank-stability under quantization, NOT real-query retrieval. the real-query
# (mteb-style) ruler is gate-zero; until it exists these numbers are likely
# inflated. see the ruler note in doc/CHANGELOG.
_RULER_PROVENANCE = {
    "kind": "self-perturbation",
    "query": (
        "each query is a corpus chunk's own embedding plus deterministic per-dim "
        "noise (up to 8e-5 per dim), re-l2-normalized; it is a near-duplicate of an "
        "corpus point"
    ),
    "ground_truth": "the same float32 baseline's own top-k on that perturbed query",
    "measures": "rank-stability of a preset under quantization, NOT real-query retrieval quality",
    "caveat": (
        "this is NOT a real-query (mteb-style) labeled query-to-doc ruler; "
        "recall@10 here is easier than real retrieval and is likely inflated"
    ),
    "real_ruler": (
        "pending gate-zero (real-query labeled harness); see the ruler note in doc/CHANGELOG"
    ),
}


def _sample_queries(chunks, n_queries: int, seed: int):
    """Pick `n_queries` corpus chunks at random; perturb each chunk's
    own embedding by tiny deterministic noise; return `(qvec, qtext)`
    pairs ready for run_bench.

    WEAK RULER: the query is a corpus vector plus <=8e-5 per-dim noise, i.e. a
    near-duplicate of an existing point, so recall@10 here measures
    rank-stability under quantization, NOT real-query retrieval quality.
    See _RULER_PROVENANCE and the gate-zero real-query harness."""
    n = len(chunks)
    dim = len(chunks[0]["embedding"])
    rng = random.Random(seed)
    pool = rng.sample(range(n), n_queries)
    queries: list[tuple[list[float], str]] = []
    for i in pool:
        qvec = list(chunks[i]["embedding"])
        for j in range(dim):
            qvec[j] += ((j * 7 + i) % 17 - 8) * 1e-5
        norm = sum(x * x for x in qvec) ** 0.5 or 1.0
        qvec = [x / norm for x in qvec]
        qtext = chunks[i]["canonical_text"][:200]
        queries.append((qvec, qtext))
    return queries


def _baseline_row(base_size: int, t_exact: list[float], db_exact):
    row = (
        "exact (baseline)",
        f"{base_size / 1e6:.2f}",
        "1.000",
        "—",
        "1.0000",
        "0.000000",
        "0.000000",
        f"{percentile(t_exact, 0.50):.3f}",
        f"{percentile(t_exact, 0.95):.3f}",
        f"{percentile(t_exact, 0.99):.3f}",
    )
    measurement = {
        "preset": "exact",
        "is_baseline": True,
        "size_mb": round(base_size / 1e6, 4),
        "size_ratio": 1.0,
        "build_s": None,
        "recall_at_k": 1.0,
        "drift_max": 0.0,
        "drift_mean": 0.0,
        "p50_ms": round(percentile(t_exact, 0.50), 4),
        "p95_ms": round(percentile(t_exact, 0.95), 4),
        "p99_ms": round(percentile(t_exact, 0.99), 4),
        "dtype": db_exact.dtype,
        "has_ann": db_exact.has_ann,
        "has_bm25": db_exact.has_bm25,
        "search_mode": "exact",
        "file_hash": db_exact.file_hash,
        "content_hash": db_exact.content_hash,
    }
    return row, measurement


# dtypes that store BELOW int8 precision (int8 codes or coarser): for these the
# 0x09 embeddings_fp source is NOT wired (no SECTION_EMBEDDINGS_FP/FpSlab in the
# writer), so net-of-fp ratio == stored ratio. recall@10 is real cosine AT THE
# STORED PRECISION, disclosed via dtype (+ mrl_dim/full_dim where matryoshka).
_SUB_INT8_DTYPES = {"int8", "int4"}


def _variant_row(
    preset, size, base_size, build_time, db_v, t_v, recall, drift_max, drift_mean, mode, full_dim
):
    size_ratio = size / base_size
    row = (
        preset,
        f"{size / 1e6:.2f}",
        f"{size_ratio:.3f}",
        f"{build_time:.1f}",
        f"{recall:.4f}",
        f"{drift_max:.6f}",
        f"{drift_mean:.6f}",
        f"{percentile(t_v, 0.50):.3f}",
        f"{percentile(t_v, 0.95):.3f}",
        f"{percentile(t_v, 0.99):.3f}",
    )
    measurement = {
        "preset": preset,
        "is_baseline": False,
        "size_mb": round(size / 1e6, 4),
        "size_ratio": round(size_ratio, 4),
        "build_s": round(build_time, 2),
        "recall_at_k": round(recall, 4),
        "drift_max": round(drift_max, 6),
        "drift_mean": round(drift_mean, 6),
        "p50_ms": round(percentile(t_v, 0.50), 4),
        "p95_ms": round(percentile(t_v, 0.95), 4),
        "p99_ms": round(percentile(t_v, 0.99), 4),
        "dtype": db_v.dtype,
        "has_ann": db_v.has_ann,
        "has_bm25": db_v.has_bm25,
        "search_mode": mode,
        "file_hash": db_v.file_hash,
        "content_hash": db_v.content_hash,
    }
    # honest net-of-fp disclosure (machine-checkable). every sub-int8 row is
    # STORED-PRECISION: the 0x09 embeddings_fp writer is not wired, so the
    # net-of-fp ratio equals the stored ratio and the recall is real cosine at
    # the stored precision. surface that explicitly plus the precision axis.
    if db_v.dtype in _SUB_INT8_DTYPES:
        measurement["stored_precision"] = True
        measurement["has_fp_source"] = False  # 0x09 embeddings_fp slab not emitted
        measurement["net_of_fp_ratio"] = round(size_ratio, 4)
    # matryoshka disclosure: when the stored dim is shorter than the source
    # dim, record both so the dimension lever is visible and a citation is
    # honestly tied to a given mrl_dim. derived from the built file's stride,
    # so it is machine-checkable straight off the published ladder JSON.
    if db_v.embedding_dim != full_dim:
        measurement["mrl_dim"] = db_v.embedding_dim
        measurement["full_dim"] = full_dim
    return row, measurement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--variants",
        default=(
            # the published ladder: named rungs first (compressed -> tiny ->
            # micro -> nano -> hybrid), then the matryoshka curve points so the
            # full honest size/recall tradeoff is emitted, not a cherry-pick.
            # micro = mrl256-int8 (aliased); the curve below reports the rest of
            # the dimension axis at int8 and int4. mrl96-int8 is the smallest
            # int8 rung (int4 at 96 is blocked: 96%64!=0).
            "compressed,tiny,micro,nano,hybrid,"
            "mrl256-int8,mrl192-int8,mrl128-int8,mrl96-int8,"
            "mrl256-int4,mrl192-int4,mrl128-int4"
        ),
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON to stdout (table to stderr) for regression gates.",
    )
    ap.add_argument(
        "--reuse",
        action="store_true",
        help=(
            "Reuse an existing variant build when the .urna opens and validates "
            "cleanly (builds are deterministic, so the bytes are identical). "
            "build_s is recorded as 0.0 and the log line says reused=true. "
            "Default off: every variant is rebuilt."
        ),
    )
    args = ap.parse_args()

    base_path = Path(args.baseline)
    if not base_path.exists():
        raise SystemExit(f"baseline not found: {base_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks, meta = decode_baseline(base_path)
    queries = _sample_queries(chunks, args.n_queries, args.seed)

    db_exact = urna.open(str(base_path))
    base_size = base_path.stat().st_size
    t_exact, hits_exact = run_bench(db_exact, queries, args.k, mode="exact")
    base_top_score = [h[0].score for h in hits_exact]

    log = sys.stderr if args.json else sys.stdout
    print(file=log)
    print(
        f"## acceptance harness: {len(chunks)} chunks, dim={meta['embedding_dim']}, "
        f"{args.n_queries} queries, k={args.k}",
        file=log,
    )
    print(f"baseline:           {base_path}", file=log)
    print(f"baseline file_hash: {db_exact.file_hash}", file=log)
    print(f"baseline simd:      {db_exact.simd_backend}", file=log)
    print(file=log)

    rows = [
        (
            "preset",
            "size_MB",
            "size_ratio",
            "build_s",
            "recall@k",
            "drift_max",
            "drift_mean",
            "p50_ms",
            "p95_ms",
            "p99_ms",
        ),
    ]
    base_row, base_meas = _baseline_row(base_size, t_exact, db_exact)
    rows.append(base_row)
    measurements = [base_meas]

    for preset in args.variants.split(","):
        preset = preset.strip()
        if not preset:
            continue
        out_path = OUT_DIR / f"corpus_{preset}.urna"
        print(f"\n→ building preset={preset} → {out_path}", file=log)
        reused = False
        build_time = 0.0
        if args.reuse and out_path.exists():
            try:
                probe = urna.open(str(out_path))
                probe.validate()
                reused = True
            except Exception:
                reused = False  # partial or stale build: fall through to rebuild
        if not reused:
            build_time = build_variant(chunks, meta, preset, out_path)
        size = out_path.stat().st_size
        db_v = urna.open(str(out_path))
        db_v.validate()

        if db_v.has_ann and db_v.has_bm25 and preset == "hybrid":
            mode = "hybrid"
        elif db_v.has_ann and (preset in ("tiny", "nano", "micro") or preset.startswith("mrl")):
            mode = "ann"
        else:
            mode = "exact"
        print(
            f"  size={size / 1e6:.2f} MB ({size / base_size:.3f}× baseline)  "
            f"dtype={db_v.dtype}  has_ann={db_v.has_ann}  has_bm25={db_v.has_bm25}  "
            f"search_mode={mode}  build={build_time:.1f}s  reused={reused}",
            file=log,
        )
        # matryoshka: the stored vectors are truncated to mrl_dim, so the
        # query must be sliced to the same prefix and re-L2-normalized to
        # match the build-time truncate-then-renormalize. Recall is still
        # measured against the FULL-dim baseline top-k, so the curve is the
        # honest cost of truncation (MiniLM is not matryoshka-trained).
        v_queries = queries
        if db_v.embedding_dim != meta["embedding_dim"]:
            mrl = db_v.embedding_dim
            v_queries = []
            for qvec, qtext in queries:
                prefix = list(qvec[:mrl])
                norm = sum(x * x for x in prefix) ** 0.5 or 1.0
                v_queries.append(([x / norm for x in prefix], qtext))
        t_v, hits_v = run_bench(db_v, v_queries, args.k, mode=mode)

        recalls, drifts = [], []
        for h_exact, h_v, base_score in zip(hits_exact, hits_v, base_top_score, strict=False):
            ex_ids = {h.chunk_id for h in h_exact}
            v_ids = {h.chunk_id for h in h_v}
            recalls.append(len(ex_ids & v_ids) / args.k)
            if h_v:
                drifts.append(abs(h_v[0].score - base_score))

        recall = mean(recalls) if recalls else 0.0
        drift_max = max(drifts) if drifts else 0.0
        drift_mean = mean(drifts) if drifts else 0.0
        row, m = _variant_row(
            preset,
            size,
            base_size,
            build_time,
            db_v,
            t_v,
            recall,
            drift_max,
            drift_mean,
            mode,
            meta["embedding_dim"],
        )
        rows.append(row)
        measurements.append(m)

    print(file=log)
    print("## results", file=log)
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "  ".join("-" * w for w in widths)
    for i, row in enumerate(rows):
        line = "  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=False))
        print(line, file=log)
        if i == 0:
            print(sep, file=log)

    if args.json:
        doc = {
            "schema_version": 1,
            "n_chunks": len(chunks),
            "embedding_dim": meta["embedding_dim"],
            "n_queries": args.n_queries,
            "k": args.k,
            "seed": args.seed,
            "baseline_path": str(base_path),
            "baseline_file_hash": db_exact.file_hash,
            "baseline_content_hash": db_exact.content_hash,
            "simd_backend": db_exact.simd_backend,
            "ruler": _RULER_PROVENANCE,
            "presets": measurements,
        }
        json.dump(doc, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
