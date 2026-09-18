"""Shared media primitives for image corpora: canvas, uri, probes, hashes.

The corpus contract: every image is normalized onto one fixed canvas before
encoding, so `frame[i]` stays bound to `image[i]` by construction. Encode
lives in `forge/image_encode.py`, decode in `forge/image_decode.py`; this
module keeps only what both sides (and the read-side tooling) share.

Letterboxing (fit inside the canvas, pad the remainder) is used instead of
a plain rescale so a dataset with mixed aspect ratios is not distorted.
The canvas is derived from the dataset's median aspect ratio and median
width, so a homogeneous dataset wastes no pixels on padding and is never
upscaled.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# yuv420p subsamples chroma 2x in both axes, so both canvas dimensions must
# be even or ffmpeg silently rounds and the decoded size stops matching.
_EVEN = 2


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _round_even(value: int) -> int:
    return max(_EVEN, int(value) - (int(value) % _EVEN))


def canvas_size(paths: Sequence[Path], width: int, *, probe_cap: int = 128) -> tuple[int, int]:
    """Pick one canvas for the whole dataset from its median aspect and width.

    `width` is a CEILING, not a target. Upscaling invents no detail, so every
    interpolated pixel is a bit the encoder spends on nothing and a decoder
    pays for on every read: measured on PH2, a 1024 canvas over a 765-wide
    source cost 33 percent of the file and 26 percent of the single-frame
    decode, for no information. Both the aspect and the cap come from the
    MEDIAN so one outsized image cannot set the geometry for the corpus.

    Only `probe_cap` evenly spaced images are opened: reading every file in a
    100k-image dataset costs minutes and the median is stable well before
    that. Evenly spaced (not random) keeps the choice reproducible.
    """
    from PIL import Image

    if not paths:
        raise ValueError("canvas_size needs at least one image")
    step = max(1, len(paths) // probe_cap)
    ratios, widths = [], []
    for path in list(paths)[::step][:probe_cap]:
        with Image.open(path) as img:
            w, h = img.size
        if w > 0 and h > 0:
            ratios.append(w / h)
            widths.append(w)
    if not ratios:
        raise RuntimeError("no readable images while choosing the canvas")
    aspect = float(np.median(ratios))
    target = min(int(width), int(np.median(widths)))
    return _round_even(target), _round_even(round(target / aspect))


def letterbox(image, canvas: tuple[int, int]):
    """Fit `image` inside `canvas` preserving aspect, pad the rest black."""
    from PIL import Image, ImageOps

    return ImageOps.pad(
        image.convert("RGB"), canvas, method=Image.Resampling.BICUBIC, color=(0, 0, 0)
    )


def _read_exact(stream, size: int) -> bytes:
    """Pipes return short reads; loop until `size` bytes or a clean EOF."""
    buf = bytearray()
    while len(buf) < size:
        block = stream.read(size - len(buf))
        if not block:
            break
        buf.extend(block)
    return bytes(buf)


def probe_frame_count(path: Path) -> int:
    """Count decoded frames. `nb_frames` is a container hint and is often
    absent or wrong for AV1 in mp4, so the frames are actually counted."""
    # fmt: off
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "json", str(path),
    ]
    # fmt: on
    out = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(out.stdout).get("streams") or [{}]
    return int(streams[0].get("nb_read_frames") or 0)


def probe_pix_fmt(path: Path) -> str:
    """The pixel format ACTUALLY written, not the one requested.

    Encoders fall back silently: this ffmpeg's libsvtav1 converts yuv444p to
    yuv420p and writes a file byte-identical to the 420 encode. A pix_fmt
    flag that is never probed is a lie waiting to be published.
    """
    # fmt: off
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=pix_fmt", "-of", "json", str(path),
    ]
    # fmt: on
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout).get("streams") or [{}]
    fmt = streams[0].get("pix_fmt")
    if not fmt:
        raise RuntimeError(f"could not probe pix_fmt of {path}")
    return str(fmt)


def media_dir_for(urna_path: Path) -> Path:
    """Where a corpus keeps its media, as a sibling of the `.urna`.

    A corpus is `corpus.urna` plus `corpus.media/`. Keeping media beside the
    file (never at an absolute path baked into a URI) is what lets a corpus
    be copied to another machine and still resolve.
    """
    urna_path = Path(urna_path)
    return urna_path.parent / f"{urna_path.stem}.media"


def parse_media_uri(uri: str) -> tuple[str, int | None]:
    """Split a media uri into its relative path and optional frame ordinal.

    Two shapes exist: `media://<file>#frame=<n>` for the av1 stream backend,
    and `media://<dir>/<file>` (no fragment) for per-image backends like
    avif, where the ordinal lives in the manifest, not the uri.
    """
    if not uri.startswith("media://"):
        raise ValueError(f"not a media uri: {uri}")
    body, _, fragment = uri[len("media://") :].partition("#")
    if not fragment:
        return body, None
    if not fragment.startswith("frame="):
        raise ValueError(f"media uri fragment is not a frame ordinal: {uri}")
    return body, int(fragment[len("frame=") :])
