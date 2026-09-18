"""Measurement battery for image corpora, grau publicacao.

One home for the statistics every published image-corpus number must carry,
so a claim cannot be written down without its interval again (the failure
the 2026-08-07 changelog note retracts). Imported by `urna_image_eval.py`
(the two-ruler eval) and `urna_image_sweep.py` (the variant matrix).

- `bootstrap_delta`: paired percentile interval on a per-query difference.
- `sign_test`: assumption-free paired check, "how often", next to the
  bootstrap's "how big".
- `cosine_drift`: per-image drift between source-pixel and decoded-frame
  embeddings, as a distribution, never only a mean.
- `ranking_agreement`: overlap@k and kendall tau-b, which is what search
  actually feels.
- `per_class_delta` + `class_floor_ok`: the CP-0.6 floor. a mean can hide
  one destroyed class (melanoma on ph2 dropped 18.5 points while both nevi
  stayed flat); the gate is per class, not per mean.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# bootstrap resamples. fixed rather than a flag so two runs of the harness are
# comparable, and published intervals are reproducible from the shipped tool.
RESAMPLES = 5000


def bootstrap_delta(sample: np.ndarray, control: np.ndarray, *, seed: int = 42) -> dict:
    """Percentile bootstrap on a paired per-query difference.

    Paired because both indexes answer the SAME queries: resampling the pairs
    removes the query-difficulty variance that dominates the raw spread.
    """
    diff = np.asarray(sample, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    n = len(diff)
    if n == 0:
        raise ValueError("bootstrap_delta needs at least one paired observation")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, n, (RESAMPLES, n))].mean(axis=1)
    lo, hi = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    return {
        "mean": round(float(diff.mean()), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "significant": not (lo <= 0.0 <= hi),
        "n": n,
    }


def sign_test(sample: np.ndarray, control: np.ndarray) -> dict:
    """Two-sided paired sign test, exact binomial.

    Assumption-free companion to the bootstrap: it cannot say how large an
    effect is, only whether wins outnumber losses more than chance allows.
    """
    diff = np.asarray(sample, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(pos, neg) + 1))
    p_value = min(1.0, 2.0 * tail / (2**n))
    return {"pos": pos, "neg": neg, "p_value": p_value}


def cosine_drift(source: np.ndarray, decoded: np.ndarray) -> dict:
    """Per-image cosine between the source-pixel vector and the vector of
    what the corpus actually serves back, as a distribution."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(decoded, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    drift = (a * b).sum(axis=1)
    return {
        "min": round(float(drift.min()), 4),
        "p05": round(float(np.percentile(drift, 5)), 4),
        "p25": round(float(np.percentile(drift, 25)), 4),
        "median": round(float(np.median(drift)), 4),
        "p75": round(float(np.percentile(drift, 75)), 4),
        "p95": round(float(np.percentile(drift, 95)), 4),
        "max": round(float(drift.max()), 4),
    }


def ranking_agreement(
    ranks_a: Sequence[Sequence[int]], ranks_b: Sequence[Sequence[int]], *, k: int
) -> dict:
    """overlap@k plus kendall tau-b over the shared items of each top-k.

    overlap says whether the same neighbours come back; tau-b says whether
    they come back in the same order. both are per query, then averaged.
    """
    overlaps, taus = [], []
    for list_a, list_b in zip(ranks_a, ranks_b, strict=True):
        top_a, top_b = list(list_a)[:k], list(list_b)[:k]
        if not top_a or not top_b:
            continue
        overlaps.append(len(set(top_a) & set(top_b)) / k)
        shared = sorted(set(top_a) & set(top_b))
        if len(shared) < 2:
            continue
        pos_a = {item: i for i, item in enumerate(top_a)}
        pos_b = {item: i for i, item in enumerate(top_b)}
        concordant = discordant = 0
        for i in range(len(shared)):
            for j in range(i + 1, len(shared)):
                da = pos_a[shared[i]] - pos_a[shared[j]]
                db = pos_b[shared[i]] - pos_b[shared[j]]
                if da * db > 0:
                    concordant += 1
                elif da * db < 0:
                    discordant += 1
        pairs = concordant + discordant
        taus.append((concordant - discordant) / pairs if pairs else 0.0)
    return {
        f"overlap@{k}": round(float(np.mean(overlaps)), 4) if overlaps else None,
        "kendall_tau_b": round(float(np.mean(taus)), 4) if taus else None,
    }


def per_class_delta(
    sample: np.ndarray, control: np.ndarray, labels: Sequence[str], *, seed: int = 42
) -> dict:
    """The bootstrap delta computed per class, not only on the mean."""
    sample = np.asarray(sample, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    out = {}
    for klass in sorted(set(labels)):
        mask = [i for i, label in enumerate(labels) if label == klass]
        out[klass] = bootstrap_delta(sample[mask], control[mask], seed=seed)
    return out


def class_floor_ok(per_class: dict, *, floor: float = 0.05) -> bool:
    """The CP-0.6 gate: no class may fall past the floor with significance.

    A corpus whose mean is fine while one class is destroyed fails here,
    which is the failure mode that matters clinically.
    """
    for stats in per_class.values():
        if stats["significant"] and stats["mean"] <= -floor:
            return False
    return True
