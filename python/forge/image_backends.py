"""Backend orchestration for image corpus builds.

One contract over three backends: the builder gets `media` (the manifest
block), `uris` (one per item), and `frames()`, an iterator of decoded RGB
batches for the embed pass. The index always describes the DECODED pixels,
never the sources, or it reports a quality the corpus does not have.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from . import image_media
from .image_backends_av1 import build_av1
from .image_decode import decode_avif, decode_frames, decode_jxl
from .image_encode_still import encode_avif, encode_jxl_dir


def _png_frames(png_dir: Path, batch_size: int = 32) -> Iterator[list[np.ndarray]]:
    from PIL import Image

    batch: list[np.ndarray] = []
    for out in sorted(png_dir.glob("*.png")):
        with Image.open(out) as img:
            batch.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _letterbox_all(
    render_paths: Sequence[Path], canvas: tuple[int, int], out_dir: Path
) -> list[Path]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, path in enumerate(render_paths):
        out = out_dir / f"{i:06d}.png"
        if not out.exists():
            with Image.open(path) as img:
                image_media.letterbox(img, canvas).save(out)
        written.append(out)
    return written


def _control(render_paths, output_path, dataset_name, canvas) -> dict:
    """Letterbox-lossless control corpus: the ruler the codec cost is
    measured against (fase 0, CP-0.1). Its byte size is recorded so every
    variant can be reported against the SAME ruler; the control itself
    records `output_bytes` only. every codec backend (av1, avif, jxl,
    jxl-transcode) records `source_bytes` as the sum of the ORIGINAL source
    files (the avif path keeps its letterboxed png sum apart as
    `letterboxed_input_bytes`), so `compression_ratio` is the same quantity
    across them."""
    png_dir = image_media.media_dir_for(output_path) / f"{dataset_name}-png"
    written = _letterbox_all(render_paths, canvas, png_dir)
    media = {
        "backend": "png-lossless",
        "canvas": [canvas[0], canvas[1]],
        "frame_count": len(render_paths),
        "output_bytes": sum(p.stat().st_size for p in written),
    }
    uris = [f"media://{dataset_name}-png/{i:06d}.png" for i in range(len(render_paths))]
    frames = lambda batch_size=32: _png_frames(png_dir, batch_size)  # noqa: E731
    return {"media": media, "uris": uris, "frames": frames}


def _avif(render_paths, output_path, dataset_name, canvas, pix_fmt, avif_quality, speed) -> dict:
    """One avif per image. The value is per-image O(1) semantics and real
    yuv444 (CP-0.6 asks for it on medical corpora), not compression: the
    stream won the size-matched matrix. Letterboxed onto the same canvas,
    so the corpus contract holds across backends."""
    avif_dir = image_media.media_dir_for(output_path) / f"{dataset_name}-avif"
    # the encoder reads letterboxed pngs from a tempdir; the manifest must
    # still account the ORIGINAL files, or the ratio is against a lossless
    # re-encode of the decoded canvas (5.4x inflated on the 38k card corpus).
    source_bytes = sum(p.stat().st_size for p in render_paths)
    with tempfile.TemporaryDirectory(prefix="urna-avif-src-") as tmp:
        tmp_pngs = _letterbox_all(render_paths, canvas, Path(tmp))
        yuv = {"yuv420p": "420", "yuv444p": "444"}[pix_fmt]
        media = encode_avif(
            tmp_pngs,
            avif_dir,
            quality=avif_quality,
            yuv=yuv,
            speed=speed,
            source_bytes=source_bytes,
        )
    media["canvas"] = [canvas[0], canvas[1]]
    uris = [f"media://{dataset_name}-avif/{i:06d}.avif" for i in range(len(render_paths))]

    def avif_frames(batch_size: int = 32) -> Iterator[list[np.ndarray]]:
        batch: list[np.ndarray] = []
        for frame in sorted(avif_dir.glob("*.avif")):
            batch.append(decode_avif(frame))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    return {"media": media, "uris": uris, "frames": avif_frames}


def build_media(
    render_paths: Sequence[Path],
    output_path: Path,
    dataset_name: str,
    *,
    backend: str,
    canvas: tuple[int, int] | None,
    crf: int,
    speed: int,
    all_intra: bool,
    pix_fmt: str,
    avif_quality: int,
    control: bool,
    gop_policy: str = "auto",
    order: Sequence[int] | None = None,
    shard_size: int | None = None,
    tune: str = "default",
    fps: int = 1,
    jxl_transcode=None,
) -> dict:
    """Build the media side of a corpus and return its manifest record."""
    assert canvas is not None
    if control:
        return _control(render_paths, output_path, dataset_name, canvas)
    if backend == "avif":
        return _avif(render_paths, output_path, dataset_name, canvas, pix_fmt, avif_quality, speed)
    if backend in ("jxl", "jxl-transcode"):
        return _jxl(
            render_paths, output_path, dataset_name, backend == "jxl-transcode", jxl_transcode
        )
    return build_av1(
        render_paths,
        output_path,
        dataset_name,
        canvas,
        crf,
        speed,
        all_intra,
        pix_fmt,
        gop_policy,
        order,
        shard_size,
        tune=tune,
        fps=fps,
    )


def _jxl(render_paths, output_path, dataset_name, transcode: bool, policy) -> dict:
    """One .jxl per SOURCE image, no letterbox: the whole point is losslessness.

    `jxl` mode is lossless of the source pixels; `jxl-transcode` is the
    bit-exact reversible JPEG repack. Decoded frames vary in size, which the
    embed pass handles; the uniform-canvas contract belongs to the lossy
    stream backends.
    """
    jxl_dir = image_media.media_dir_for(output_path) / f"{dataset_name}-jxl"
    media = encode_jxl_dir(
        render_paths,
        jxl_dir,
        transcode=transcode,
        on_unsupported_jpeg=getattr(policy, "on_unsupported_jpeg", "copy-source"),
        verify_roundtrip=getattr(policy, "verify_roundtrip", True),
    )
    uris = [f"media://{dataset_name}-jxl/{name}" for name in media["files"]]

    def jxl_frames(batch_size: int = 32) -> Iterator[list[np.ndarray]]:
        batch: list[np.ndarray] = []
        for name in media["files"]:
            batch.append(decode_jxl(jxl_dir / name))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    return {"media": media, "uris": uris, "frames": jxl_frames}


def decoded_frames_fn(media_dir: Path, media: dict, frame_uris: Sequence[str]):
    """Rebuild the decoded-frames iterator for an EXISTING media set (resume
    path): what `build_media` hands back at encode time, reconstructed from
    the manifest record. av1 yields STREAM order (the caller un-permutes
    with `order_permutation`); per-image backends yield item order."""
    backend = media.get("backend", "")
    if backend == "av1":
        canvas = tuple(media["canvas"])
        seg_names = [s["uri"] for s in media["segments"]]

        def frames(batch_size: int = 32) -> Iterator[list[np.ndarray]]:
            for name in seg_names:
                yield from decode_frames(media_dir / name, canvas, batch_size=batch_size)

        return frames

    rel_paths = [u.removeprefix("media://") for u in frame_uris]

    def _load(rel: str) -> np.ndarray:
        from PIL import Image

        path = media_dir / rel
        if path.suffix == ".jxl":
            return decode_jxl(path)
        if path.suffix == ".avif":
            return decode_avif(path)
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def per_image_frames(batch_size: int = 32) -> Iterator[list[np.ndarray]]:
        # per-image decode is one subprocess per file: fan it out on a
        # bounded window (order preserved, ~window frames in flight):
        # sequential djxl over a 38k corpus is hours, this is minutes.
        # decoding is deterministic, so bytes are unchanged.
        import os
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        from itertools import islice

        window = 32
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            it = iter(rel_paths)
            futs = deque(pool.submit(_load, rel) for rel in islice(it, window))
            batch: list[np.ndarray] = []
            while futs:
                arr = futs.popleft().result()
                nxt = next(it, None)
                if nxt is not None:
                    futs.append(pool.submit(_load, nxt))
                batch.append(arr)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    return per_image_frames
