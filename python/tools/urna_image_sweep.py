"""Rate-distortion sweep for image corpora: one model load, the whole matrix.

Builds the letterbox-lossless control once, embeds the queries once, then
builds and measures every requested variant in-process. Every number the
sweep reports carries the full battery from `_image_metrics.py`: paired
bootstrap interval, sign test, per-class floor (CP-0.6: the mean is not the
gate, the worst class is), cosine drift distribution, and ranking agreement.

Variant spec: `av1-inter:30,35;av1-intra:35;avif:25,35;avif444:35;dtype:f16,int8,int4`

`dtype` variants ride on the sweep's `--preset` base and override only the
vector dtype, so the ladder isolates quantization. Note (fase 5, F5.3):
DermLIP is not mrl-trained, so prefix truncation degrades more than on an
mrl model; pca is the fallback lever if int4/int8 collapse.

Usage:
    python/tools/urna_image_sweep.py --input-dir dat/demo/derm/ph2/images \
        --labels dat/demo/derm/ph2/PH2_simple_dataset.csv --dataset ph2 \
        --out-dir tmp/sweep-ph2 --variants "av1-inter:30,35,40;avif:35"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge import embed_image, image_items  # noqa: E402
from tools import urna_build_image_corpus as builder  # noqa: E402
from tools import urna_image_eval as ev  # noqa: E402
from tools._image_metrics import (  # noqa: E402
    bootstrap_delta,
    class_floor_ok,
    cosine_drift,
    per_class_delta,
    ranking_agreement,
    sign_test,
)

_DTYPES = {"f32": "float32", "f16": "float16", "int8": "int8", "int4": "int4"}


def parse_variants(spec: str) -> list[dict]:
    """`av1-inter:30,35;avif:25;dtype:int8` -> per-variant build kwargs."""
    variants = []
    for chunk in spec.split(";"):
        kind, _, values = chunk.partition(":")
        for value in values.split(","):
            if kind == "av1-inter":
                variants.append(
                    {
                        "name": f"{kind}-{value}",
                        "backend": "av1",
                        "crf": int(value),
                        "gop_policy": "inter",
                    }
                )
            elif kind == "av1-intra":
                variants.append(
                    {
                        "name": f"{kind}-{value}",
                        "backend": "av1",
                        "crf": int(value),
                        "gop_policy": "intra",
                    }
                )
            elif kind == "av1-order":
                # ordering is only visible to inter prediction (all-intra
                # encodes every frame alone, so permuting them changes
                # nothing), so the kind pins the gop to inter: it is the
                # measurement cell fase 0 left open for wsi tiles.
                variants.append(
                    {
                        "name": f"{kind}-{value}",
                        "backend": "av1",
                        "crf": int(value),
                        "gop_policy": "inter",
                        "order_similarity": True,
                    }
                )
            elif kind == "avif":
                variants.append(
                    {"name": f"{kind}-{value}", "backend": "avif", "avif_quality": int(value)}
                )
            elif kind == "avif444":
                variants.append(
                    {
                        "name": f"{kind}-{value}",
                        "backend": "avif",
                        "avif_quality": int(value),
                        "pix_fmt": "yuv444p",
                    }
                )
            elif kind == "av1-tune":
                # SVT-AV1 Still Picture tune (probed; unsupported = recorded
                # fallback to default) at the given crf, all-intra.
                variants.append(
                    {
                        "name": f"{kind}-{value}",
                        "backend": "av1",
                        "crf": int(value),
                        "gop_policy": "intra",
                        "tune": "still",
                    }
                )
            elif kind in ("jxl", "jxl-transcode"):
                # lossless rungs: source-pixel lossless / reversible jpeg
                # repack. the value is unused (lossless has no quality knob)
                # but kept for the kind:value grammar.
                variants.append({"name": kind, "backend": kind})
            elif kind == "dtype":
                if value not in _DTYPES:
                    raise ValueError(f"unknown dtype: {value} (one of {sorted(_DTYPES)})")
                variants.append({"name": f"{kind}-{value}", "dtype": _DTYPES[value]})
            else:
                raise ValueError(f"unknown variant kind: {kind}")
    return variants


def battery(sample: dict, control: dict, labels: list[str], ks: list[int], floor: float) -> dict:
    """The full paired comparison of one variant against the control."""
    k_max = max(ks)
    sample_pq = np.array(sample["_per_query"][k_max])
    control_pq = np.array(control["_per_query"][k_max])
    per_class = per_class_delta(sample_pq, control_pq, labels)
    return {
        f"precision@{k_max}": {
            "bootstrap": bootstrap_delta(sample_pq, control_pq),
            "sign_test": sign_test(sample_pq, control_pq),
        },
        "per_class": per_class,
        "class_floor_ok": class_floor_ok(per_class, floor=floor),
        "ranking": ranking_agreement(sample["_rankings"], control["_rankings"], k=k_max),
    }


def run_sweep(args: argparse.Namespace, embedder) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "input_dir": args.input_dir,
        "dataset_name": args.dataset,
        "embedder": embedder,
        "is_pdf": args.pdf,
        "labels": image_items.load_labels(args.labels),
        "sample": args.sample,
        "seed": args.seed,
        "width": args.width,
        "preset": args.preset,
        "return_embeddings": True,
    }

    control = builder.build_corpus(
        output_path=out_dir / "control" / "control.urna", control=True, **common
    )
    manifest = ev.load_manifest(Path(control["urna"]))
    items = manifest["items"]
    queries = ev.pick_queries(items, args.queries, args.seed)
    ev.attach_vectors(embedder, queries, items, manifest)
    labels = [q.get("label", "") for q in queries]
    control_eval = ev.evaluate(Path(control["urna"]), queries, args.k)
    query_vectors = np.array([q["_vector"] for q in queries], dtype=np.float32)

    report: dict = {
        "dataset": args.dataset,
        "model": {"id": embedder.model_id, "hash": embedder.model_hash},
        "n_items": len(items),
        "n_queries": len(queries),
        "canvas": manifest["media"]["canvas"],
        "random_label_precision": ev.random_label_precision(items),
        "class_floor": args.class_floor,
        "control": {
            "label": control_eval.get("label"),
            "identity": control_eval["identity"],
            # the one ruler every variant is compared against: the
            # letterbox-lossless corpus size, and the control index size.
            "media_bytes": (control["media"] or {}).get("output_bytes"),
            "urna_bytes": Path(control["urna"]).stat().st_size,
        },
        "variants": {},
    }
    for variant in parse_variants(args.variants):
        name = variant.pop("name")
        result = builder.build_corpus(
            output_path=out_dir / name / f"{name}.urna", **variant, **common
        )
        variant_eval = ev.evaluate(Path(result["urna"]), queries, args.k)
        drift = cosine_drift(query_vectors, result["embeddings"][[q["ordinal"] for q in queries]])
        media = result["media"] or {}
        entry: dict = {
            "build": variant,
            "media_bytes": media.get("output_bytes"),
            "urna_bytes": Path(result["urna"]).stat().st_size,
            "compression_ratio": media.get("compression_ratio"),
            "label": variant_eval.get("label"),
            "identity": variant_eval["identity"],
            "drift": drift,
        }
        # the label battery only exists when the corpus carries labels; an
        # unlabelled corpus still gets drift and the identity ruler.
        if "_per_query" in variant_eval and "_per_query" in control_eval:
            entry["delta_vs_control"] = battery(
                variant_eval, control_eval, labels, args.k, args.class_floor
            )
        report["variants"][name] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--model", default="hf-hub:redlessone/DermLIP_ViT-B-16")
    parser.add_argument("--pretrained")
    parser.add_argument("--device")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--queries", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--preset", default="compressed")
    parser.add_argument("--class-floor", type=float, default=0.05)
    parser.add_argument("-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    embedder = embed_image.ImageEmbedder(
        model_id=args.model, pretrained=args.pretrained, device=args.device
    )
    report = run_sweep(args, embedder)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
