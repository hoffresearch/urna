# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""Recall harness: the real potion table vs the lexical floor on demo/corpora/intro.

run: python tool/bench/recallcmp.py

each query is a PARAPHRASE of one demo doc that avoids the doc's literal
keywords, so a lexical embedder (shared-token cosine) has little to grab while a
semantic embedder retrieves by meaning. we split the four cc0 docs into
paragraph passages, embed them with both embedders, rank passages per query, and
report recall@1 / recall@3 / mrr side by side, plus the top hit each picks. the
point is not a leaderboard, it is the contrast: potion finds the right doc by
meaning where the floor cannot.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rust", "bridge", "python"))

from urna.embed.lexifloor import default_embedder as lexical_embedder
from urna.embed.potiontab import potion_embedder

CORPUS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "demo", "corpora", "intro"
)

# (query, gold doc number 1..4). the queries deliberately use synonyms and
# rephrasings, not the gold doc's own words.
QUERIES: list[tuple[str, int]] = [
    ("can it respond with no internet connection, everything staying on my own computer", 4),
    ("the tool must never phone home or contact a remote host to serve a request", 4),
    ("evidence showing precisely which original snippet a result was lifted from", 3),
    ("the match strength returned is a genuine recalculated value, not a cheap estimate", 3),
    ("everything packed in a lone movable bundle i can clone like a small embedded db", 1),
    ("a stable archival container whose layout will not change so old data keeps loading", 1),
    ("the stage that converts assorted raw uploads into a tidy uniform internal form", 2),
    ("normalizing varied source material so two builds come out exactly the same", 2),
]


def _passages() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for fn in sorted(os.listdir(CORPUS)):
        if not fn.endswith(".md") or fn == "README.md":
            continue
        doc = int(fn.split("-", 1)[0])
        with open(os.path.join(CORPUS, fn), encoding="utf-8") as fh:
            text = fh.read()
        for para in text.split("\n\n"):
            para = " ".join(para.split())
            if len(para) > 30:
                out.append((doc, para))
    return out


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _rank(emb, qvec, passages, pvecs) -> list[tuple[int, float]]:
    scored = [(passages[i][0], _cos(qvec, pvecs[i])) for i in range(len(passages))]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def _eval(name: str, emb, passages) -> tuple[float, float, float]:
    pvecs = emb.embed_texts([p for _, p in passages])
    qvecs = emb.embed_texts([q for q, _ in QUERIES])
    r1 = r3 = mrr = 0.0
    print(f"\n--- {name} ---")
    for (q, gold), qv in zip(QUERIES, qvecs, strict=False):
        ranked = _rank(emb, qv, passages, pvecs)
        top_docs = [d for d, _ in ranked]
        rank_of_gold = next((i for i, d in enumerate(top_docs) if d == gold), None)
        hit1 = rank_of_gold == 0
        hit3 = rank_of_gold is not None and rank_of_gold < 3
        r1 += hit1
        r3 += hit3
        mrr += 1.0 / (rank_of_gold + 1) if rank_of_gold is not None else 0.0
        mark = "ok " if hit1 else ("~  " if hit3 else "MISS")
        print(f"  [{mark}] gold=doc{gold} top=doc{top_docs[0]}  '{q[:54]}'")
    n = len(QUERIES)
    return r1 / n, r3 / n, mrr / n


def main() -> None:
    passages = _passages()
    print(f"intro: {len(passages)} passages from 4 docs, {len(QUERIES)} paraphrase queries")
    p1, p3, pm = _eval("potion (semantic, real table)", potion_embedder(), passages)
    f1, f3, fm = _eval("floor (lexical bag-of-words)", lexical_embedder(), passages)
    print("\n=== summary (recall@1 / recall@3 / mrr) ===")
    print(f"  potion : {p1:.3f} / {p3:.3f} / {pm:.3f}")
    print(f"  floor  : {f1:.3f} / {f3:.3f} / {fm:.3f}")
    print(f"  delta  : {p1 - f1:+.3f} / {p3 - f3:+.3f} / {pm - fm:+.3f}  (positive = potion wins)")


if __name__ == "__main__":
    main()
