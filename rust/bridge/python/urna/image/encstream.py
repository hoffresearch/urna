"""Encode side of image corpus media: the av1 stream encoder.

Backend selection and the control corpus live in `urna/image/orchestra.py`;
the per-image encoders (avif, jxl) in `urna/image/encstills.py`; the
gop probe in `urna/image/gopprober.py`. This module keeps the stream
encoder and the shared provenance record.

Provenance is recorded on every encode: the decoded pixels depend on the
exact toolchain (ffmpeg version, encoder, parameters), and another version
produces other pixels, other embeddings, and an index that silently stops
matching the media. `provenance_sha256` fingerprints that toolchain.

Every encode probes what was actually written (frame count, pix_fmt):
an encoder that falls back silently turns a flag into a lie.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from urna.image import mediabase


def _tool_version(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0].strip()


def provenance_sha256(payload: dict) -> str:
    """Fingerprint the toolchain record, canonically."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# bounded gop for inter coding: best size on the near-dup matrix (g=16 beat
# g=8, g=32, and single-keyframe) while capping random-access decode cost at
# 16 frames. one constant so the probe and the build always agree.
INTER_KEYINT = 16


def probe_tune_still(cache: dict = {}) -> int | None:  # noqa: B006
    """Resolve the SVT-AV1 'Still Picture' tune number for the local encoder.

    The number varies by SVT-AV1 version, so it is probed (a 16x16
    one-frame encode) instead of assumed: an unsupported value must become
    a loud warning and a recorded fallback, never a silently ignored flag.
    The probe carries keyint=1 because the IQ tune (3 on SVT 4.2) accepts
    all-intra only - probing without it rejected tune=3 and silently fell
    back to tune=4 (MS_SSIM); measured 2026-08-31 on the card corpus,
    tune=3 is +1.26 ssim2 for +1.4% bytes. Still-tune encodes are always
    all-intra (see encode_av1), so the probe matches what ships.
    Cached per process.
    """
    if "value" in cache:
        return cache["value"]
    import sys

    for candidate in (3, 4):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            # fmt: off
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:size=16x16:rate=1",
                "-frames:v", "1", "-c:v", "libsvtav1",
                "-svtav1-params", f"tune={candidate}:keyint=1", tmp.name,
            ]
            # fmt: on
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                cache["value"] = candidate
                return candidate
    print(
        "[forge] warning: local SVT-AV1 has no Still Picture tune; using default tune",
        file=sys.stderr,
    )
    cache["value"] = None
    return None


def encode_av1(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    canvas: tuple[int, int],
    crf: int = 35,
    preset: int = 8,
    fps: int = 1,
    lp: int = 2,
    keyint: int | None = None,
    pix_fmt: str = "yuv420p",
    tune: str = "default",
) -> dict:
    """Encode an ordered image list to one AV1 mp4 through a rawvideo pipe.

    `keyint=1` makes every frame a keyframe. Measured across three axes in
    fase 0 (ph2 dermoscopy, wsi tiles in scan order, pdf pages), all-intra
    beat the default gop on size at the same crf on every one: unrelated
    frames give motion estimation nothing to find, and every frame being a
    keyframe makes random access O(1). Still a lever, not the default,
    because the policy decision belongs to the builder's probe.

    The requested `pix_fmt` is probed after the encode and a mismatch
    raises: encoders fall back silently, and a 444 flag that writes 420
    bytes is a published lie.

    Raises if the encoded frame count disagrees with the image count: a
    corpus whose `#frame=N` pointers are off by one is worse than no corpus.
    """
    from PIL import Image

    width, height = canvas
    source_bytes = sum(p.stat().st_size for p in image_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svt_params = f"lp={lp}" + (f":keyint={keyint}" if keyint else "")
    if keyint != 1:
        # inter gop: cards/images are not scene cuts. SVT's scene-change
        # detector re-inserts the keyframes inter exists to avoid; measured
        # 2026-08-31 (near-dup reprints, crf35/s6): scd=0 + keyint=16 is
        # -29% vs all-intra where default scd erased the inter gain.
        svt_params += ":scd=0"
    # still tune ships only with an all-intra gop: SVT's IQ tune supports
    # all-intra (and low-delay, measured much worse) only - with keyint!=1
    # the flag would hard-error the encode. tune_resolved=None in the record
    # says the requested tune did not ship for this stream.
    tune_resolved: int | None = None
    if tune == "still" and keyint == 1:
        tune_resolved = probe_tune_still()
        if tune_resolved is not None:
            svt_params += f":tune={tune_resolved}"

    # fmt: off
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libsvtav1", "-crf", str(crf), "-preset", str(preset),
        "-svtav1-params", svt_params,
        "-pix_fmt", pix_fmt, "-frames:v", str(len(image_paths)),
        str(output_path),
    ]
    # fmt: on
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for path in image_paths:
            with Image.open(path) as img:
                frame = mediabase.letterbox(img, canvas)
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        failure = "ffmpeg encode failed" if proc.wait() != 0 else None
    except BrokenPipeError:
        proc.wait()
        failure = "ffmpeg closed the stream early"
    stderr = proc.stderr.read().decode()
    proc.stderr.close()
    if failure:
        raise RuntimeError(f"{failure}: {stderr}")

    actual_fmt = mediabase.probe_pix_fmt(output_path)
    if actual_fmt != pix_fmt:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"encoder wrote {actual_fmt} for a requested {pix_fmt}: it fell back "
            "silently. use the avif backend for 444, or drop the flag"
        )

    frames = mediabase.probe_frame_count(output_path)
    if frames != len(image_paths):
        raise RuntimeError(
            f"encoded {frames} frames for {len(image_paths)} images; "
            "frame pointers would be misaligned"
        )

    toolchain = {
        "ffmpeg": _tool_version(["ffmpeg", "-version"]),
        "encoder": "libsvtav1",
        "params": {
            "crf": crf,
            "preset": preset,
            "fps": fps,
            "lp": lp,
            "keyint": keyint,
            "pix_fmt": actual_fmt,
            "tune": tune,
            "tune_resolved": tune_resolved,
        },
    }
    output_bytes = output_path.stat().st_size
    return {
        "backend": "av1",
        "codec": "libsvtav1",
        "crf": crf,
        "preset": preset,
        "fps": fps,
        "tune": tune,
        "keyint": keyint,
        "pix_fmt": actual_fmt,
        "canvas": [width, height],
        "frame_count": frames,
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "compression_ratio": round(source_bytes / output_bytes, 2) if output_bytes else 0.0,
        "media_sha256": mediabase.sha256_file(output_path),
        "toolchain": toolchain,
        "provenance_sha256": provenance_sha256(toolchain),
    }
