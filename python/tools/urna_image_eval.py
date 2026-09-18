"""Measure retrieval quality of an image corpus, on two rulers.

Compression claims are only worth what the ruler behind them is worth, so
this reports two different things and never blends them:

identity
    the query is the source image of an indexed frame, and a hit means the
    frame came back. this measures RANK STABILITY UNDER THE CODEC, not
    retrieval quality. it is a self-retrieval ruler and it is inflated by
    construction: the corpus contains the answer, lightly perturbed. it is
    the same class of ruler the repo already flags in `dat/measure/*.json`.

label
    the query's own frame is REMOVED from its results, and the score is how
    many of the remaining neighbours share its label. nothing in the corpus
    is the answer, so this measures whether the space still groups by
    meaning after compression. reported against the random-pick baseline
    for the same label distribution, which is what makes a number readable.

Neither ruler says anything on its own. Pass `--baseline` with the
uncompressed control index to get the delta, which is the actual finding:
what the codec cost.

Usage:
    python/tools/urna_image_eval.py --index tmp/ph2/ph2.urna \\
        --baseline tmp/ph2-raw/ph2-raw.urna -k 1 5 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import urna
from _image_metrics import bootstrap_delta
from forge import embed_image, image_items

__all__ = ["bootstrap_delta"]


def load_manifest(index_path: Path) -> dict:
    manifest_path = Path(index_path).with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found; it is written by urna_build_image_corpus.py "
            "and carries the ordinals and labels this harness measures against"
        )
    return json.loads(manifest_path.read_text())


def pick_queries(items: list[dict], count: int | None, seed: int) -> list[dict]:
    if count is None or count >= len(items):
        return items
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(len(items), size=count, replace=False).tolist())
    return [items[i] for i in picked]


def random_label_precision(items: list[dict]) -> float | None:
    """Expected same-label rate when neighbours are drawn at random.

    Without this, a label precision of 0.6 is unreadable: it is excellent on
    a 10-class corpus and worthless on one where 60 percent of items share
    a label.
    """
    labels = [i["label"] for i in items if i.get("label")]
    if len(labels) < 2:
        return None
    total = len(labels)
    counts = Counter(labels)
    return sum((n / total) * ((n - 1) / (total - 1)) for n in counts.values())


def evaluate(index_path: Path, queries: list[dict], ks: list[int]) -> dict:
    """Run both rulers over one index."""
    db = urna.open(str(index_path))
    n_indexed = db.n_embeddings
    max_k = min(max(ks), n_indexed)
    labelled = [q for q in queries if q.get("label")]

    identity_hits = dict.fromkeys(ks, 0)
    label_matched = dict.fromkeys(ks, 0.0)
    label_any = dict.fromkeys(ks, 0)
    # kept per query, in query order, so a paired bootstrap against another
    # index can resample the same queries rather than two aggregates.
    per_query: dict[int, list[float]] = {k: [] for k in ks}
    # the raw top-max_k ordinals per query, so ranking agreement (overlap@k,
    # kendall tau-b) can be computed against another index's rankings.
    rankings: list[list[int]] = []

    for query in queries:
        # search one extra so removing the query's own frame still leaves k
        # neighbours for the label ruler.
        hits = db.search(query["_vector"], min(max_k + 1, n_indexed))
        ordinals = [h.offset_start for h in hits]
        rankings.append(ordinals[:max_k])
        others = [o for o in ordinals if o != query["ordinal"]]
        for k in ks:
            if query["ordinal"] in ordinals[:k]:
                identity_hits[k] += 1
        if not query.get("label"):
            continue
        for k in ks:
            window = others[:k]
            if not window:
                continue
            same = sum(1 for o in window if query["_by_ordinal"].get(o) == query["label"])
            label_matched[k] += same / len(window)
            label_any[k] += 1 if same else 0
            per_query[k].append(same / len(window))

    n_q, n_l = len(queries), len(labelled)
    report = {
        "index": str(index_path),
        "n_indexed": n_indexed,
        "n_queries": n_q,
        "n_labelled_queries": n_l,
        "identity": {f"recall@{k}": round(identity_hits[k] / n_q, 4) for k in ks},
    }
    if n_l:
        report["label"] = {
            **{f"precision@{k}": round(label_matched[k] / n_l, 4) for k in ks},
            **{f"hit@{k}": round(label_any[k] / n_l, 4) for k in ks},
        }
        report["_per_query"] = per_query
    report["_rankings"] = rankings
    return report


def query_image(item: dict, manifest: dict, scratch: Path) -> Path:
    """The original pixels behind one corpus entry.

    For images that is the source file. For pdf pages the build-time render
    is long gone, so the page is re-rendered from the pdf that produced it,
    which is deterministic and needs no extra storage.
    """
    root = Path(manifest.get("input_dir", ""))
    source = root / item["origin"]
    if item.get("page") is None:
        if source.exists():
            return source
        fallback = Path(item.get("render_path", ""))
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"source image gone: {source}")
    if not source.exists():
        raise FileNotFoundError(f"source pdf gone: {source}")
    return image_items.render_page(source, item["page"], scratch)


def attach_vectors(embedder, queries: list[dict], items: list[dict], manifest: dict) -> None:
    """Embed each query from its ORIGINAL pixels, never from a frame.

    A real query is a fresh image, not something already inside the corpus,
    so embedding the decoded frame back would measure the codec against
    itself and inflate every number.
    """
    by_ordinal = {i["ordinal"]: i.get("label", "") for i in items}
    with TemporaryDirectory(prefix="urna-eval-") as scratch:
        paths = [query_image(q, manifest, Path(scratch)) for q in queries]
        vectors = embedder.embed_paths(paths)
    for query, vector in zip(queries, vectors, strict=True):
        query["_vector"] = vector.tolist()
        query["_by_ordinal"] = by_ordinal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="uncompressed control index")
    parser.add_argument("-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--queries", type=int, help="query sample size (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="hf-hub:redlessone/DermLIP_ViT-B-16")
    parser.add_argument("--pretrained")
    parser.add_argument("--device")
    parser.add_argument("--out", type=Path, help="write the report json here")
    args = parser.parse_args()

    manifest = load_manifest(args.index)
    items = manifest["items"]
    queries = pick_queries(items, args.queries, args.seed)
    embedder = embed_image.ImageEmbedder(
        model_id=args.model, pretrained=args.pretrained, device=args.device
    )
    attach_vectors(embedder, queries, items, manifest)

    report = {
        "dataset": manifest["dataset"],
        "ruler": {
            "identity": (
                "self-retrieval: the query IS the source of an indexed frame. "
                "measures rank stability under the codec, not retrieval quality; "
                "inflated by construction"
            ),
            "label": (
                "leave-one-out: the query's own frame is excluded, score is the "
                "share of remaining neighbours with the same label"
            ),
            "queries_embedded_from": "original images",
            "seed": args.seed,
        },
        "random_label_precision": random_label_precision(items),
        "compressed": evaluate(args.index, queries, args.k),
    }
    if manifest.get("media"):
        report["media"] = manifest["media"]
    if args.baseline:
        base_manifest = load_manifest(args.baseline)
        if len(base_manifest["items"]) != len(items):
            raise ValueError("baseline and index cover different item counts")
        report["uncompressed"] = evaluate(args.baseline, queries, args.k)
        report["delta"] = {
            ruler: {
                metric: round(report["compressed"][ruler][metric] - value, 4)
                for metric, value in report["uncompressed"].get(ruler, {}).items()
            }
            for ruler in ("identity", "label")
            if ruler in report["compressed"] and ruler in report["uncompressed"]
        }
        # a delta is only readable next to its interval; without one, a
        # difference smaller than the sample can resolve reads as a finding.
        sample, control = report["compressed"], report["uncompressed"]
        if "_per_query" in sample and "_per_query" in control:
            report["delta"]["label_ci"] = {
                f"precision@{k}": bootstrap_delta(
                    np.array(sample["_per_query"][k]),
                    np.array(control["_per_query"][k]),
                    seed=args.seed,
                )
                for k in args.k
                if sample["_per_query"].get(k) and control["_per_query"].get(k)
            }
    for section in ("compressed", "uncompressed"):
        report.get(section, {}).pop("_per_query", None)
        report.get(section, {}).pop("_rankings", None)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
