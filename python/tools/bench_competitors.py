"""urna vs the embedded vector stores it gets compared to, one table.

same rows, same queries, same machine, same process; every number is
measured here, including the ones that make urna look ordinary. synthetic
l2-normalized rows (seeded) so anyone can reproduce it without a dataset;
the recall ruler is brute-force top-k over the same rows.

    .venv/bin/python python/tools/bench_competitors.py --n 100000 --dim 384 \
        --queries 200 --out doc/benchmarks.md

columns: build time, bytes on disk, cold open + first query in a fresh
process (python startup subtracted), warm p50/p99 latency, recall@10 (exact
systems are 1.0 by construction and asserted), byte-identical rebuild
(measured by building twice), built-in integrity check. every hnsw path runs
m=16, ef_construction=200, ef_search=100 so the ann rows compare like for like.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from importlib.metadata import version as pkg_version

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import _bench_systems as systems  # noqa: E402

# one note per line; wrapped here only for the 100-column rule.
NOTES = (
    "\n".join(
        [
            "- `cold open + 1st query`: wall time of a fresh interpreter that opens the store and"
            " answers one query, minus an interpreter doing nothing (3 runs, min). urna's number"
            " is dominated by `open` verifying every section checksum and the footer hash over"
            " the whole file before serving anything; the other stores trust their bytes.",
            "- `build (s)`: single-threaded everywhere (hnswlib and usearch are told threads=1);"
            " urna's hnsw build is the slow row.",
            "- `p50 / p99`: warm, single-threaded, one query at a time, from python. python call"
            " overhead is inside every number.",
            "- `recall@k` is against brute force over the same rows; exact paths are asserted"
            " at 1.0.",
            "- `rebuild byte-identical`: two builds from the same rows compared by sha256 over the"
            " artefact (a directory is hashed file by file).",
            "- `integrity check`: whether the store can prove its own bytes. urna verifies sha256"
            " per section, per file and over the decoded content on `validate()`.",
            "- the same rows written with raw text and with zstd text share one `content_hash`:"
            " {same_citation}. re-encoding never moves a `urna://content_hash/chunk_id` citation;"
            " the other stores have no equivalent notion.",
        ]
    )
    + "\n"
)


def make_rows(n: int, dim: int, seed: int) -> np.ndarray:
    """clustered rows, not i.i.d. gaussians: real embeddings live near
    topics. at 384 dims i.i.d. gaussian rows have no neighbourhood structure
    (every distance concentrates on the same value) and EVERY hnsw
    implementation collapses to ~0.3 recall@10, which says nothing about the
    stores. one of `n // 50` centers plus 35% isotropic noise, l2-normalized."""
    rng = np.random.default_rng(seed)
    n_centers = max(8, n // 50)
    centers = rng.standard_normal((n_centers, dim), dtype=np.float32)
    assign = rng.integers(0, n_centers, size=n)
    rows = centers[assign] + 0.35 * rng.standard_normal((n, dim), dtype=np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return rows


def make_queries(rows: np.ndarray, n_q: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(rows.shape[0], n_q, replace=False)
    q = rows[idx] + 0.05 * rng.standard_normal((n_q, rows.shape[1]), dtype=np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def brute_force(rows: np.ndarray, queries: np.ndarray, k: int) -> list[set[int]]:
    sims = queries @ rows.T
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    return [set(int(x) for x in t) for t in top]


def bench_one(sys_obj, rows, queries, truth, k, path, python):
    t0 = time.perf_counter()
    sys_obj.build(rows, path)
    build_s = time.perf_counter() - t0
    size = systems.dir_or_file_bytes(path)
    first = systems.sha256_tree(path)
    sys_obj.build(rows, path + ".rebuild")
    identical = systems.sha256_tree(path + ".rebuild") == first
    systems.rm(path + ".rebuild")

    q0 = queries[0].tolist()
    cold = systems.cold_open_ms(sys_obj.reopen_snippet, path, q0, rows.shape[1], python)

    sys_obj.open(path)
    sys_obj.search(queries[0], k)  # warm
    lat = []
    hits = 0
    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        got = sys_obj.search(q, k)
        lat.append((time.perf_counter() - t0) * 1e3)
        hits += len(set(got) & truth[qi])
    recall = hits / (k * len(queries))
    if not sys_obj.has_ann:
        assert recall > 0.999, f"{sys_obj.name}: exact system below 1.0 recall ({recall})"
    row = {
        "system": sys_obj.name,
        "path": "ann (hnsw)" if sys_obj.has_ann else "exact",
        "build_s": round(build_s, 2),
        "bytes": size,
        "cold_open_ms": round(cold, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p99_ms": round(float(np.percentile(lat, 99)), 3),
        "recall_at_k": round(recall, 4),
        "byte_identical_rebuild": "yes" if identical else "no",
        "integrity_check": sys_obj.validate(),
    }
    if hasattr(sys_obj, "content_hash"):
        row["content_hash"] = sys_obj.content_hash()
    return row


def markdown(rows: list[dict], meta: dict) -> str:
    cols = [
        ("system", "system"),
        ("path", "path"),
        ("build_s", "build (s)"),
        ("bytes", "bytes on disk"),
        ("cold_open_ms", "cold open + 1st query (ms)"),
        ("p50_ms", "p50 (ms)"),
        ("p99_ms", "p99 (ms)"),
        ("recall_at_k", f"recall@{meta['k']}"),
        ("byte_identical_rebuild", "rebuild byte-identical"),
        ("integrity_check", "integrity check"),
    ]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r[key]
            cells.append(f"{v:,}" if key == "bytes" else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--work", default=os.path.join(os.getcwd(), ".bench-competitors"))
    ap.add_argument("--out", help="write the markdown report here (stdout otherwise)")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    rows = make_rows(args.n, args.dim, args.seed)
    queries = make_queries(rows, args.queries, args.seed)
    truth = brute_force(rows, queries, args.k)

    candidates = [
        ("urna_exact.urna", lambda: systems.UrnaSystem("exact", ann=False)),
        ("urna_hybrid.urna", lambda: systems.UrnaSystem("hybrid", ann=True)),
        ("usearch.usearch", systems.UsearchSystem),
        ("hnswlib.bin", systems.HnswlibSystem),
        ("sqlite_vec.db", systems.SqliteVecSystem),
        ("lancedb.lance", systems.LanceDbSystem),
    ]
    results, skipped = [], []
    for fname, ctor in candidates:
        try:
            sys_obj = ctor()
        except Exception as e:  # missing optional dependency: say so, do not fake a row
            skipped.append(f"{fname}: {e}")
            continue
        print(f"== {sys_obj.name}", file=sys.stderr, flush=True)
        store = os.path.join(args.work, fname)
        results.append(bench_one(sys_obj, rows, queries, truth, args.k, store, systems.PYTHON))

    versions = {}
    for pkg in ["usearch", "hnswlib", "sqlite-vec", "lancedb", "numpy"]:
        try:
            versions[pkg] = pkg_version(pkg)
        except Exception:
            versions[pkg] = "absent"
    meta = {
        "n": args.n,
        "dim": args.dim,
        "queries": args.queries,
        "k": args.k,
        "seed": args.seed,
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "versions": versions,
        "skipped": skipped,
        "date": time.strftime("%Y-%m-%d"),
    }
    # the citation claim, measured: the same rows written raw and with zstd
    # text must decode to the same canonical bytes, hence one content_hash.
    # (index type is part of the canonical search_contract, so exact vs
    # hybrid legitimately differ; that is not what the claim is about.)
    same_citation = None
    exact_rows = [r for r in results if r["system"] == "urna (exact)"]
    if exact_rows:
        z = systems.UrnaSystem("exact", ann=False, text_encoding="zstd")
        zpath = os.path.join(args.work, "urna_exact_zstd.urna")
        z.build(rows, zpath)
        z.open(zpath)
        same_citation = z.content_hash() == exact_rows[0]["content_hash"]

    header = (
        f"measured {meta['date']} on {meta['machine']}, python {meta['python']}, single "
        f"thread, n={args.n:,} synthetic clustered l2-normalized rows x {args.dim} dims "
        f"({max(8, args.n // 50)} centers), "
        f"{args.queries} queries, k={args.k}, seed {args.seed}. reproduce: "
        f"`.venv/bin/python python/tools/bench_competitors.py --n {args.n} --dim {args.dim} "
        f"--queries {args.queries}`."
    )
    notes = NOTES.format(same_citation=same_citation).strip("\n").splitlines()
    limits = (
        "what urna does NOT do that some of these do: in-place updates or deletes, metadata "
        "filtering, concurrent writers, a query language. it is a build-once, ship-and-query "
        "file; the table says nothing about workloads that need those."
    )
    tail = f"versions: {json.dumps(versions)}" + (f"; skipped: {skipped}" if skipped else "")
    report = ["# benchmarks", "", header, "", markdown(results, meta), "", "how to read it:", ""]
    report += notes + ["", limits, "", tail]
    text = "\n".join(report) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    print(text)


if __name__ == "__main__":
    main()
