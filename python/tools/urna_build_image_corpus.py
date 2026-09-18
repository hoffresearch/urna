"""CLI for `forge.image_corpus.build_corpus` (re-exported here for callers).

    images (or rendered pdf pages)
      -> one fixed canvas
      -> one AV1 stream (optionally sharded, optionally similarity-ordered)
      -> vision embeddings of the encoded frames
      -> .urna, one chunk per image or page

Usage:
    python/tools/urna_build_image_corpus.py \\
        --input-dir dat/demo/derm/ph2/images --dataset ph2 \\
        --output tmp/ph2/ph2.urna --labels tmp/ph2/labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge import embed_image, image_items  # noqa: E402
from forge.image_corpus import build_corpus  # noqa: E402,F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pdf", action="store_true", help="render pdf pages as the images")
    parser.add_argument("--no-compress", action="store_true", help="build the control index")
    parser.add_argument("--model", default="hf-hub:redlessone/DermLIP_ViT-B-16")
    parser.add_argument("--pretrained", help="required for bare architecture names")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--crf", type=int, default=35)
    parser.add_argument("--speed", type=int, default=8, help="svt-av1 preset, 0 best to 13 fast")
    parser.add_argument(
        "--gop-policy",
        choices=["auto", "intra", "inter"],
        default="auto",
        help="auto probes a spaced sample and lets the bytes decide (default)",
    )
    parser.add_argument(
        "--all-intra",
        action="store_true",
        help="force every frame a keyframe; overrides --gop-policy",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        help="split the stream into consecutive segments of about this many frames",
    )
    parser.add_argument(
        "--order-similarity",
        action="store_true",
        help="greedy nearest-neighbour frame order before encode; measured on wsi tiles",
    )
    parser.add_argument(
        "--backend",
        choices=["av1", "avif"],
        default="av1",
        help="av1 stream (default, won the size-matched matrix) or one avif per image",
    )
    parser.add_argument(
        "--pix-fmt",
        choices=["yuv420p", "yuv444p"],
        default="yuv420p",
        help="probed after encode; 444 only works on the avif backend",
    )
    parser.add_argument("--avif-quality", type=int, default=35, help="avifenc -q, 0 worst to 100")
    parser.add_argument(
        "--control",
        action="store_true",
        help="letterbox-lossless control corpus (png), the ruler codec cost is measured against",
    )
    parser.add_argument("--preset", default="compressed", help="urna encoding preset")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "int8", "int4"],
        help="override the preset's vector dtype (int4 needs dim divisible by 64)",
    )
    parser.add_argument("--scratch-db")
    parser.add_argument("--labels", type=Path, help="json map or csv of image id to label")
    args = parser.parse_args()

    embedder = embed_image.ImageEmbedder(
        model_id=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            build_corpus(
                input_dir=args.input_dir,
                output_path=args.output,
                dataset_name=args.dataset,
                embedder=embedder,
                is_pdf=args.pdf,
                compress=not args.no_compress,
                labels=image_items.load_labels(args.labels),
                sample=args.sample,
                seed=args.seed,
                width=args.width,
                crf=args.crf,
                speed=args.speed,
                all_intra=args.all_intra,
                backend=args.backend,
                pix_fmt=args.pix_fmt,
                avif_quality=args.avif_quality,
                control=args.control,
                gop_policy=args.gop_policy,
                shard_size=args.shard_size,
                order_similarity=args.order_similarity,
                scratch_db=args.scratch_db,
                preset=args.preset,
                dtype=args.dtype,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
