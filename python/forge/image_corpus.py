"""Build a searchable `.urna` corpus from an image directory or from PDFs.

    images (or rendered pdf pages)
      -> one fixed canvas
      -> one AV1 stream (optionally sharded, optionally similarity-ordered)
      -> vision embeddings of the encoded frames
      -> .urna, one chunk per image or page

The index describes what a reader can actually get back, so when the corpus
is compressed the vectors are taken from the DECODED frames, not from the
source pixels. Building from the source pixels and shipping the compressed
stream would report a quality the corpus does not have.

A corpus is a directory: `corpus.urna` next to `corpus.media/`. Frame URIs
are relative to that pair, so the corpus can be copied elsewhere and still
resolve. `corpus.manifest.json` records what went in, for audit and for
`urna_image_eval.py`.

The CLI wrapper is `python/tools/urna_build_image_corpus.py`.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from builder import BuildConfig, ChunkSpec, Pipeline

from . import embed_image, image_backends, image_items, image_media, image_order
from .image_decode import frame_sha256

CHUNKER_VERSION = "image-v1"


def _embed_compressed(embedder, frames_fn, expected: int) -> tuple[np.ndarray, list[str]]:
    """Embed the DECODED frames and hash them in the same pass.

    The decode already walks every frame, so the per-frame sha256 costs
    nothing here; recorded in the manifest, the hashes are what catch a
    REORDERED stream, which the frame-count guard cannot see.
    """
    batches: list[np.ndarray] = []
    hashes: list[str] = []
    for batch in frames_fn(batch_size=max(1, embedder.batch_size)):
        batches.append(embedder.embed_arrays(batch))
        hashes.extend(frame_sha256(frame) for frame in batch)
    vectors = np.vstack(batches) if batches else np.zeros((0, embedder.dim), dtype=np.float32)
    if len(vectors) != expected:
        raise RuntimeError(f"decoded {len(vectors)} frames for {expected} items")
    return vectors, hashes


def build_corpus(
    input_dir: Path,
    output_path: Path,
    dataset_name: str,
    *,
    embedder: embed_image.ImageEmbedder,
    is_pdf: bool = False,
    compress: bool = True,
    labels: dict[str, str] | None = None,
    sample: int | None = None,
    seed: int = 42,
    width: int = 1024,
    crf: int = 35,
    speed: int = 8,
    all_intra: bool = False,
    backend: str = "av1",
    pix_fmt: str = "yuv420p",
    avif_quality: int = 35,
    control: bool = False,
    gop_policy: str = "auto",
    shard_size: int | None = None,
    order_similarity: bool = False,
    scratch_db: str | None = None,
    preset: str = "compressed",
    dtype: str | None = None,
    tune: str = "default",
    return_embeddings: bool = False,
) -> dict:
    input_dir, output_path = Path(input_dir), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        if is_pdf:
            # the rendered pages must outlive the encode and the embed, so the
            # temp dir is bound to the whole build, not to the render call.
            tmp_dir = Path(stack.enter_context(TemporaryDirectory(prefix="urna-pdf-")))
            items = image_items.render_pdf_pages(input_dir, tmp_dir)
        else:
            items = image_items.collect_images(input_dir, labels)
        items = image_items.subsample(items, sample, seed)
        if not items:
            raise RuntimeError(f"no input found under {input_dir}")

        render_paths = [Path(item.render_path) for item in items]
        media_info = None
        frame_hashes: list[str] = []
        if compress:
            canvas = image_media.canvas_size(render_paths, width)
            order = None
            if order_similarity and backend == "av1" and not control:
                # ordering buys little on unrelated images (fase 0: +6.6%),
                # so it is opt-in; wsi tiles are where fase 6 measures it.
                order = image_order.similarity_order(embedder.embed_paths(render_paths))
            built = image_backends.build_media(
                render_paths,
                output_path,
                dataset_name,
                backend=backend,
                canvas=canvas,
                crf=crf,
                speed=speed,
                all_intra=all_intra,
                pix_fmt=pix_fmt,
                avif_quality=avif_quality,
                control=control,
                gop_policy=gop_policy,
                order=order,
                shard_size=shard_size,
                tune=tune,
            )
            media_info, uris = built["media"], built["uris"]
            embeddings, frame_hashes = _embed_compressed(embedder, built["frames"], len(items))
            if built.get("order") is not None:
                # vectors/hashes came back in stream order; the uris are
                # already in item order, so un-permute to match them.
                inv = np.argsort(built["order"])
                embeddings = embeddings[inv]
                frame_hashes = [frame_hashes[i] for i in inv]
        else:
            uris = [Path(p).resolve().as_uri() for p in render_paths]
            embeddings = embedder.embed_paths(render_paths)

        cfg = BuildConfig(
            output_path=str(output_path),
            embedding_model=embedder.model_id,
            embedding_dim=embedder.dim,
            chunker_version=CHUNKER_VERSION,
            model_hash=embedder.model_hash,
            title=f"{dataset_name} image corpus",
            version="0.1.0",
            description=f"vision embeddings for {dataset_name}",
            preset=preset,
            dtype=dtype,
        )
        specs = [
            ChunkSpec(
                canonical_text=item.canonical_text(),
                source_uri=uri,
                byte_start=item.ordinal,
                byte_end=item.ordinal + 1,
            )
            for item, uri in zip(items, uris, strict=True)
        ]

        # keyed by chunk_id, never by position: Pipeline hands the embedder
        # only the chunks the scratch cache missed, so a positional lookup
        # returns a DIFFERENT image's vector on any partially warm run.
        by_id = {
            spec.chunk_id(CHUNKER_VERSION): vec.tolist()
            for spec, vec in zip(specs, embeddings, strict=True)
        }
        pipe = Pipeline(
            cfg,
            embedder=lambda missed: [by_id[s.chunk_id(CHUNKER_VERSION)] for s in missed],
            scratch_db=scratch_db,
        )
        pipe.add_many(specs)
        provenance = {
            "dataset": dataset_name,
            "n_items": len(items),
            "compressed": compress,
            "media": media_info,
            "sample": {"size": sample, "seed": seed} if sample else None,
            "input_kind": "pdf" if is_pdf else "images",
        }
        pipe.emit(provenance=provenance)
        pipe.close()

    manifest = {
        "dataset": dataset_name,
        "urna": output_path.name,
        "compressed": compress,
        # the durable way back to a query image. for pdfs the rendered pages
        # are build-time temporaries, so `input_dir` + `origin` + `page` is
        # what the eval harness re-renders from.
        "input_dir": str(Path(input_dir).resolve()),
        "input_kind": "pdf" if is_pdf else "images",
        "media": media_info,
        "model": {"id": embedder.model_id, "dim": embedder.dim, "hash": embedder.model_hash},
        # one content hash per decoded frame. the count guard catches a lost
        # frame; only these catch a reordered one. verify with
        # forge.image_decode.verify_frame_hashes.
        "frame_sha256": frame_hashes,
        "items": [
            {
                "ordinal": item.ordinal,
                "origin": item.origin,
                "label": item.label,
                "page": item.page,
                "source_uri": uri,
                # absolute path of what was embedded, for the eval harness;
                # never part of the corpus itself.
                "render_path": item.render_path,
            }
            for item, uri in zip(items, uris, strict=True)
        ],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    result = {
        "dataset": dataset_name,
        "urna": str(output_path),
        "manifest": str(manifest_path),
        "n_items": len(items),
        "compressed": compress,
        "media": media_info,
    }
    if return_embeddings:
        # programmatic callers only (the sweep): the vectors the index was
        # built from, so drift against source-pixel vectors costs no re-embed.
        result["embeddings"] = embeddings
    return result
