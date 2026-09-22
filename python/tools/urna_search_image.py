"""Search an image corpus with a query image or a clinical description.

Queries go through `retrieve`, so the embedder's `model_hash` is checked
against the manifest before anything is scored: an index built with one
checkpoint cannot be silently queried with another. Hits are resolved back
through the corpus manifest, so a result names the original file (and pdf
page), not only a frame ordinal.

`--query-text` embeds through the model's TEXT tower into the same joint
space (DermLIP is a CLIP model: a description like "dark irregular lesion"
searches the images directly). The flag exists because the tower was
shipped all along and never wired.

`--letterbox-query` normalizes the query onto the corpus canvas before
embedding, the symmetric counterpart of the build-side letterbox. Off by
default: measured on PH2 the effect is within noise for this dataset, so it
stays a flag until a corpus shows otherwise.

`--save-frames` decodes the matched frames out of the corpus media and
writes them as PNGs. That is also the end-to-end check that the media and
the index still agree.

Usage:
    python/tools/urna_search_image.py --index tmp/ph2/ph2.urna \
        --query-image lesion.jpg -k 5 --save-frames tmp/hits
    python/tools/urna_search_image.py --index tmp/ph2/ph2.urna \
        --query-text "dark irregular lesion with blue-white veil" -k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import urna
from forge import embed_image, image_decode, image_media


def load_manifest(index_path: Path) -> dict | None:
    manifest_path = Path(index_path).with_suffix(".manifest.json")
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else None


def save_frame(index_path: Path, manifest: dict, source_uri: str, out_dir: Path) -> str | None:
    """Materialize one hit's pixels out of the corpus media, any backend."""
    from PIL import Image

    media = manifest.get("media")
    if not media or not source_uri.startswith("media://"):
        return None
    name, ordinal = image_media.parse_media_uri(source_uri)
    media_path = image_media.media_dir_for(index_path) / name
    if not media_path.exists():
        raise FileNotFoundError(f"corpus media missing: {media_path}")
    if name.endswith(".avif"):
        frame = image_decode.decode_avif(media_path)
    elif name.endswith(".png"):
        with Image.open(media_path) as img:
            frame = np.asarray(img.convert("RGB"))
    else:
        frame = image_decode.decode_frame(
            media_path, tuple(media["canvas"]), ordinal, fps=int(media.get("fps", 1))
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ordinal if ordinal is not None else media_path.stem}.png"
    Image.fromarray(frame).save(out_path)
    return str(out_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--query-image", type=Path)
    query.add_argument("--query-text", help="search with a clinical description instead")
    parser.add_argument("--model", default="hf-hub:redlessone/DermLIP_ViT-B-16")
    parser.add_argument("--pretrained")
    parser.add_argument("--device")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--letterbox-query",
        action="store_true",
        help="normalize the query onto the corpus canvas before embedding",
    )
    parser.add_argument("--save-frames", type=Path, help="write matched frames here as png")
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="query with a different embedder than the corpus was built with",
    )
    return parser.parse_args(argv)


def embed_query(args: argparse.Namespace, embedder, manifest: dict | None):
    """The query vector, from pixels or from the text tower."""
    if args.query_text:
        return embedder.embed_texts([args.query_text])[0].tolist()
    if not args.query_image.exists():
        raise FileNotFoundError(args.query_image)
    if args.letterbox_query:
        media = (manifest or {}).get("media") or {}
        if not media.get("canvas"):
            raise ValueError("--letterbox-query needs a canvas in the corpus manifest")
        from PIL import Image

        with Image.open(args.query_image) as img:
            boxed = image_media.letterbox(img, tuple(media["canvas"]))
        return embedder.embed_arrays([np.asarray(boxed)])[0].tolist()
    return embedder.embed_one(args.query_image).tolist()


def main() -> int:
    args = parse_args()

    embedder = embed_image.ImageEmbedder(
        model_id=args.model, pretrained=args.pretrained, device=args.device
    )
    manifest = load_manifest(args.index)
    qvec = embed_query(args, embedder, manifest)

    db = urna.open(str(args.index))
    expected = None if args.skip_model_check else embedder.model_hash
    hits = db.retrieve(qvec, args.k, expected_model_hash=expected)

    by_ordinal = {i["ordinal"]: i for i in manifest["items"]} if manifest else {}

    results = []
    for hit in hits:
        item = by_ordinal.get(hit.offset_start, {})
        entry = {
            "score": round(float(hit.score), 6),
            "score_type": hit.score_type,
            "ordinal": hit.offset_start,
            "text": hit.text,
            "origin": item.get("origin"),
            "label": item.get("label") or None,
            "page": item.get("page"),
            "source_uri": hit.source_uri,
            "citation_id": hit.citation_id,
        }
        if args.save_frames and manifest:
            entry["frame_png"] = save_frame(args.index, manifest, hit.source_uri, args.save_frames)
        results.append({k: v for k, v in entry.items() if v is not None})

    query = args.query_text or str(args.query_image)
    print(json.dumps({"query": query, "hits": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
