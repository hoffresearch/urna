"""The task-utility leg of the crf="auto" gate (RFC-2b): text-to-image hit@1.

Drift (cosine between source and decoded embeddings) is a stability
signal: it falls monotonically with crf whether or not retrieval still
answers. A retrieval-only corpus needs the floor retrieval serves: one
text query per sampled item, rendered from `utility_query_template` with
the item label, embedded once by the gate model's text tower, and at each
ladder rung hit@1 against the decoded frames of the same stratified
sample (argmax cosine == the item's own index). The same hit@1 against
the source frames is the lossless reference; a rung passes when
hit1_decoded >= max(utility_floor_hit1, hit1_source - utility_tol).

Measured 2026-09-12 on 38627 cards (experiment 13): drift p10 0.932 at
crf40 -> 0.829 at crf60 vetoes every rung, hit@1 on 100 queries does not
move up to crf50.
"""

from __future__ import annotations

import numpy as np

from urna.specs.specparse import SpecError

KEY = "media.quality.utility_floor_hit1"


def hit_at_1(queries: np.ndarray, gallery: np.ndarray, targets: list[int]) -> float:
    """Fraction of queries whose top cosine over the gallery is their target."""
    if len(targets) == 0:
        return 0.0
    scores = queries @ gallery.T
    top = np.argmax(scores, axis=1)
    return float(np.mean(top == np.asarray(targets)))


def query_subset(n_sample: int, n_queries: int) -> list[int]:
    """Sample-local indices of the queried items: every item when n_queries
    is 0, else an evenly spaced subset so every stratum keeps a share."""
    if n_queries <= 0 or n_queries >= n_sample:
        return list(range(n_sample))
    step = max(1, n_sample // n_queries)
    return list(range(n_sample))[::step][:n_queries]


class UtilityGate:
    """Prepared once per gate run: query vectors and the source hit@1.
    `check(dec_emb)` scores one ladder rung."""

    def __init__(self, q, adapter, labels: list[str], src_emb: np.ndarray):
        if not callable(getattr(adapter, "embed_texts", None)):
            raise SpecError(f"{KEY}: gate model has no text tower (cannot embed the queries)")
        self.floor = float(q.utility_floor_hit1)
        self.tol = float(q.utility_tol)
        self.template = q.utility_query_template
        self.targets = query_subset(len(labels), q.utility_queries)
        try:
            texts = [self.template.format(label=labels[i]) for i in self.targets]
        except (KeyError, IndexError, ValueError) as e:
            raise SpecError(
                f"media.quality.utility_query_template: only {{label}} is a placeholder ({e!r})"
            ) from e
        try:
            emb = adapter.embed_texts(texts, role="query")
        except Exception as e:  # CapabilityError and friends: name the key
            raise SpecError(f"{KEY}: gate model cannot embed text ({e})") from e
        self.queries = np.asarray(emb, dtype=np.float32)
        if self.queries.shape != (len(texts), src_emb.shape[1]):
            raise SpecError(
                f"{KEY}: text tower dim {self.queries.shape[-1]} != image dim {src_emb.shape[1]}"
            )
        self.hit1_source = hit_at_1(self.queries, src_emb, self.targets)
        self.threshold = max(self.floor, self.hit1_source - self.tol)

    def check(self, dec_emb: np.ndarray) -> tuple[float, bool]:
        hit1 = hit_at_1(self.queries, dec_emb, self.targets)
        return hit1, hit1 >= self.threshold

    def report(self) -> dict:
        return {
            "hit1_source": round(self.hit1_source, 4),
            "floor_hit1": self.floor,
            "tol": self.tol,
            "threshold": round(self.threshold, 4),
            "n_queries": len(self.targets),
            "query_template": self.template,
        }
