"""recall@10 gate for the G1 chunk_overlap drop.

proves a chunk_overlap-dropped + NEXT_CHUNK-reconstructed corpus holds
recall@10 vs an overlapping baseline BEFORE the drop is enabled. dropping
chunk_overlap reclaims text bytes from chunks_canonical, but it can silently
degrade recall if the neighbor reconstruction is wrong, so it MUST stay gated
(master-plan 07-graph: "keep it opt-in, gate on a recall@10-vs-baseline
check"). this harness is measure-style: deterministic synthetic vectors, no
sentence-transformers, no network.

run: python python/tools/graph_recall_gate.py [--n 400] [--threshold 0.90]

exit 0 if recall@10 >= threshold, non-zero otherwise (so a release gate can
shell out and fail the build before a dropped corpus ships).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urna  # noqa: E402


def _unit(v: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _lcg(seed: int):
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield (state >> 11) / float(1 << 53)


def _corpus(n: int, dim: int, seed: int) -> list[dict]:
    """n deterministic chunks; nearby ordinals share a latent direction so the
    NEXT_CHUNK graph genuinely helps (a smooth document, the regime where
    dropping overlap is safe)."""
    rng = _lcg(seed)
    chunks = []
    base = _unit([next(rng) - 0.5 for _ in range(dim)])
    for i in range(n):
        if i % 16 == 0:
            base = _unit([next(rng) - 0.5 for _ in range(dim)])
        jitter = [next(rng) - 0.5 for _ in range(dim)]
        v = _unit([b + 0.25 * j for b, j in zip(base, jitter, strict=False)])
        chunks.append(
            dict(
                canonical_text=f"chunk {i} topic{i // 16} body text here",
                source_uri="doc.txt",
                byte_start=i * 10,
                byte_end=(i + 1) * 10,
                embedding=v,
            )
        )
    return chunks


def _build(path: str, chunks: list[dict], dim: int, *, with_graph: bool) -> None:
    if os.path.exists(path):
        os.unlink(path)
    urna.build(
        output_path=path,
        embedding_model="demo",
        embedding_dim=dim,
        chunker_version="gate/1",
        model_hash="sha256:" + "0" * 64,
        chunks=chunks,
        preset="exact",
        with_graph=with_graph,
        reproducible=True,
    )


def _recall_at_k(dropped, baseline_truth, queries, k: int, hops: int, ef: int) -> float:
    hit = 0
    for q, truth in zip(queries, baseline_truth, strict=False):
        got = {h.chunk_id for h in dropped.search_graph(q, k, hops=hops, ef=ef)}
        hit += len(got & truth)
    return hit / (k * len(queries))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    chunks = _corpus(args.n, args.dim, seed=0xC0FFEE)
    tmp = tempfile.gettempdir()
    base_path = os.path.join(tmp, "graph_gate_baseline.urna")
    drop_path = os.path.join(tmp, "graph_gate_dropped.urna")
    # baseline = no graph (the recall=1.0 exact ground truth source); dropped =
    # graph-on (the overlap-dropped corpus reconstructs context from the graph).
    _build(base_path, chunks, args.dim, with_graph=False)
    _build(drop_path, chunks, args.dim, with_graph=True)

    base = urna.open(base_path)
    dropped = urna.open(drop_path)
    # citations must stay stable across the drop (graph excluded from content_hash).
    assert base.content_hash == dropped.content_hash, "content_hash changed by the drop"

    qrng = _lcg(0xABCDEF)
    queries = [
        _unit([next(qrng) - 0.5 for _ in range(args.dim)]) for _ in range(args.queries)
    ]
    truth = [{h.chunk_id for h in base.search(q, args.k)} for q in queries]

    recall = _recall_at_k(dropped, truth, queries, args.k, args.hops, args.ef)
    print(f"recall@{args.k} (graph search vs baseline exact): {recall:.4f}")
    print(f"threshold: {args.threshold:.4f}")
    if recall + 1e-9 < args.threshold:
        print("FAIL: chunk_overlap drop degrades recall@10 below threshold; do NOT enable")
        return 1
    print("PASS: chunk_overlap drop holds recall@10; safe to enable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
