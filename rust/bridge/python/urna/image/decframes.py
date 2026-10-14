"""Decode side of image corpus media: sequential, seek, batch, avif.

Random access uses input seeking (`-ss` before `-i`), never the legacy
select filter. Measured on PH2 (fase 0, CP-0.4, medians of 5 runs): the
select scan decodes every frame up to N and discards them (gop161 ordinal
199: 189 ms; all-intra 100/199: 97-154 ms), while seek lands on the nearest
preceding keyframe and decodes forward (67 ms and ~42 ms respectively), and
the two paths return byte-identical frames. An earlier docstring claimed
the select filter avoided decoding the preceding frames; it did not, the
filter runs after the decoder in the ffmpeg graph.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from urna.image.mediabase import _read_exact


def decode_frames(
    video_path: Path, canvas: tuple[int, int], *, batch_size: int = 32
) -> Iterator[list[np.ndarray]]:
    """Yield decoded RGB frames in order, in batches.

    Sequential decode, not per-frame seeking: seeking an AV1 stream costs a
    keyframe re-decode per call and turns embedding a corpus into an O(n^2)
    walk. Batching keeps peak memory at `batch_size` frames.
    """
    width, height = canvas
    frame_bytes = width * height * 3
    # fmt: off
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    # fmt: on
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    batch: list[np.ndarray] = []
    abandoned = True
    try:
        while True:
            raw = _read_exact(proc.stdout, frame_bytes)
            if len(raw) < frame_bytes:
                break
            batch.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
        abandoned = False
    finally:
        proc.stdout.close()
        if abandoned:
            # the consumer stopped early, so ffmpeg dies on a broken pipe.
            # that is not a decode failure and must not be reported as one.
            proc.terminate()
            proc.wait()
            proc.stderr.close()
        else:
            code = proc.wait()
            stderr = proc.stderr.read().decode()
            proc.stderr.close()
            if code != 0:
                raise RuntimeError(f"ffmpeg decode failed: {stderr}")


def _read_one_frame(video_path: Path, canvas: tuple[int, int], second: int | float) -> bytes:
    width, height = canvas
    # fmt: off
    cmd = [
        "ffmpeg", "-v", "error", "-ss", str(second), "-i", str(video_path),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    # fmt: on
    proc = subprocess.run(cmd, capture_output=True, check=False)
    expected = width * height * 3
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()}")
    if len(proc.stdout) < expected:
        raise IndexError(f"frame {second} not present in {video_path}")
    return proc.stdout[:expected]


def decode_frame(video_path: Path, canvas: tuple[int, int], index: int, fps: int = 1) -> np.ndarray:
    """Decode a single frame by ordinal, for read-side hit resolution.

    At the default 1 fps the ordinal IS the timestamp. Seek lands on the
    nearest preceding keyframe and decodes forward, which is provably the
    same frame the sequential path returns (asserted in the test suite on
    both gop and all-intra media). At fps > 1 the seek target is half a
    frame period BEFORE the ordinal's pts, so the first frame at-or-after
    the target is exactly the requested ordinal regardless of pts rounding.
    """
    width, height = canvas
    ts = int(index) if fps <= 1 else (int(index) - 0.499) / fps
    raw = _read_one_frame(video_path, canvas, ts)
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def decode_frames_at(
    video_path: Path, canvas: tuple[int, int], ordinals: Sequence[int], fps: int = 1
) -> list[np.ndarray]:
    """Resolve k hit ordinals in ONE ffmpeg invocation.

    Seeks to the smallest ordinal and walks forward, capturing the requested
    frames as they pass: worst case decodes `max - min + 1` frames instead of
    `max` (the select scan) or k full keyframe walks (k separate seeks).
    Frames come back in the order the ordinals were given. fps > 1 shifts
    the seek target exactly like `decode_frame`.
    """
    if not ordinals:
        return []
    width, height = canvas
    order = sorted(int(o) for o in ordinals)
    first, last = order[0], order[-1]
    wanted = set(order)
    ts = first if fps <= 1 else (first - 0.499) / fps
    # -frames:v bounds the walk to the span: without it ffmpeg keeps
    # decoding past `last` and dies on the closed pipe with a nonzero exit
    # once the pipe buffer is full, which on any real-size stream (k hits
    # out of thousands of frames, measured 2026-09-13 on 38627 cards)
    # turned every batched resolve into a raised RuntimeError. the 12-frame
    # test stream hid it because its last hit was the last frame.
    # fmt: off
    cmd = [
        "ffmpeg", "-v", "error", "-ss", str(ts), "-i", str(video_path),
        "-frames:v", str(last - first + 1),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    # fmt: on
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = width * height * 3
    found: dict[int, np.ndarray] = {}
    abandoned = True
    try:
        for current in range(first, last + 1):
            raw = _read_exact(proc.stdout, frame_bytes)
            if len(raw) < frame_bytes:
                raise IndexError(f"frame {current} not present in {video_path}")
            if current in wanted:
                found[current] = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        abandoned = False
    finally:
        proc.stdout.close()
        if abandoned:
            proc.terminate()
            proc.wait()
            proc.stderr.close()
        else:
            code = proc.wait()
            stderr = proc.stderr.read().decode()
            proc.stderr.close()
            if code != 0:
                raise RuntimeError(f"ffmpeg decode failed: {stderr}")
    return [found[int(o)] for o in ordinals]


def decode_avif(path: Path) -> np.ndarray:
    """Decode one avif to an RGB array through avifdec (dav1d).

    The intermediate png is written uncompressed: avifdec's default png
    compression was 127 of the 145 ms one 488x680 card took to decode
    (measured 2026-09-13, 8x), and the bytes are the same either way.
    """
    from PIL import Image

    out_path = Path(path).with_suffix(".decoded.png")
    cmd = ["avifdec", "--png-compress", "0", str(path), str(out_path)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"avifdec failed on {path}: {proc.stderr.decode()[-400:]}")
    try:
        with Image.open(out_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    finally:
        out_path.unlink(missing_ok=True)


def frame_sha256(frame: np.ndarray) -> str:
    """Content hash of one decoded frame, shape included.

    The shape is hashed with the bytes so a geometry bug and a pixel bug
    are both caught.
    """
    digest = hashlib.sha256()
    digest.update(f"{frame.shape}|{frame.dtype}".encode())
    digest.update(np.ascontiguousarray(frame).tobytes())
    return "sha256:" + digest.hexdigest()


def verify_frame_hashes(video_path: Path, canvas: tuple[int, int], expected: Sequence[str]) -> None:
    """Re-decode and compare every frame hash.

    The frame-count guard catches a LOST frame; it cannot catch REORDERED
    frames, and a reordered corpus hands every citation the wrong image.
    Raises ValueError naming the first mismatch.
    """
    actual = [frame_sha256(frame) for batch in decode_frames(video_path, canvas) for frame in batch]
    if len(actual) != len(expected):
        raise ValueError(f"decoded {len(actual)} frames for {len(expected)} hashes")
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if got != want:
            raise ValueError(f"frame {i} hash mismatch: media and manifest disagree")


def decode_jxl(path: Path) -> np.ndarray:
    """Decode one .jxl (or a copied source image) to an RGB array via djxl.

    Copied fallbacks from the jxl-transcode policy keep their source suffix
    and are opened directly; only real .jxl files go through djxl.
    """
    from PIL import Image

    if path.suffix != ".jxl":
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    import tempfile

    # ppm, not png: same 8-bit rgb, no deflate on the way out. 30 ms to
    # 11 ms per 488x680 card (measured 2026-09-13).
    with tempfile.NamedTemporaryFile(suffix=".ppm") as tmp:
        proc = subprocess.run(["djxl", str(path), tmp.name], capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"djxl failed for {path}: {proc.stderr.decode()[:200]}")
        with Image.open(tmp.name) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
