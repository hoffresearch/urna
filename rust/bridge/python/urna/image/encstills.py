"""Per-image (still) encoders of the image corpus media: avif and jxl.

Carved out of `urna/image/encstream.py` (which keeps the av1 stream encoder
and the shared provenance record). Both encoders here write one file per
source image and report the same manifest shape as the stream backend:
`source_bytes` is the byte size of the ORIGINAL source files for every
backend, so `compression_ratio` is comparable across av1, avif and jxl.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from urna.image.encstream import _tool_version, provenance_sha256

AVIF_JOBS = 8  # see encode_avif: >= 2 is one byte-identical class, 1 is another


def encode_avif(
    image_paths: Sequence[Path],
    out_dir: Path,
    *,
    quality: int = 35,
    yuv: str = "420",
    speed: int = 8,
    source_bytes: int | None = None,
) -> dict:
    """Encode one avif per image into `out_dir`, with the same uri contract.

    `image_paths` are what avifenc reads (the backend hands it letterboxed
    pngs from a tempdir). `source_bytes`, when given, is the byte size of
    the ORIGINAL files and is what the record's `source_bytes` and
    `compression_ratio` describe; the encoder-input sum is kept apart as
    `letterboxed_input_bytes`. Without it the two are the same number.

    The avif backend's value is not compression (measured in fase 0: the
    av1 stream wins at every matched size on ph2); it is per-image O(1)
    semantics without an ordinal or an alignment guard, and real yuv444,
    which the measured melanoma breakdown (CP-0.6) asks for on medical
    corpora. The requested `yuv` is verified with `avifdec --info` on the
    first file.

    `-j` is pinned to AVIF_JOBS. libaom writes different bytes with one
    worker than with two or more (measured 2026-09-13 on 256 cards, libavif
    1.4.2 / aom 3.15.0: every file differed between -j 1 and -j 2, and -j 2,
    3, 4, 8, 16 and all were byte-identical), so avifenc's default of "all"
    made the output a function of the machine's core count. A pinned count
    keeps the file_hash a property of the recipe and the toolchain.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    input_bytes = 0
    total = 0
    for path in image_paths:
        input_bytes += path.stat().st_size
        out = out_dir / f"{Path(path).stem}.avif"
        # fmt: off
        cmd = [
            "avifenc", "-j", str(AVIF_JOBS), "-q", str(quality), "--speed", str(speed),
            "--yuv", yuv, str(path), str(out),
        ]
        # fmt: on
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"avifenc failed on {path}: {proc.stderr.decode()[-400:]}")
        total += out.stat().st_size

    probe = subprocess.run(
        ["avifdec", "--info", str(out_dir / f"{Path(image_paths[0]).stem}.avif")],
        capture_output=True,
        text=True,
    )
    actual_yuv = "444" if "YUV444" in probe.stdout else "420"
    if actual_yuv != yuv:
        raise RuntimeError(f"avifenc wrote yuv{actual_yuv} for a requested yuv{yuv}")

    toolchain = {
        "ffmpeg": _tool_version(["avifenc", "--version"]),
        "encoder": "avifenc/aom",
        "params": {"quality": quality, "speed": speed, "yuv": actual_yuv, "jobs": AVIF_JOBS},
    }
    if source_bytes is None:
        source_bytes = input_bytes
    return {
        "backend": "avif",
        "codec": "avifenc",
        "quality": quality,
        "speed": speed,
        "yuv": actual_yuv,
        "frame_count": len(image_paths),
        "source_bytes": source_bytes,
        "letterboxed_input_bytes": input_bytes,
        "output_bytes": total,
        "compression_ratio": round(source_bytes / total, 2) if total else 0.0,
        "toolchain": toolchain,
        "provenance_sha256": provenance_sha256(toolchain),
    }


_JPEG_SUFFIXES = {".jpg", ".jpeg"}


def encode_jxl_dir(
    image_paths: Sequence[Path],
    out_dir: Path,
    *,
    transcode: bool,
    on_unsupported_jpeg: str = "copy-source",
    verify_roundtrip: bool = True,
) -> dict:
    """One .jxl per source image: `transcode` = bit-exact reversible JPEG
    repack (--lossless_jpeg=1), else lossless of the source pixels (-d 0).

    Preservation contract: transcode preserves the original JPEG BYTES
    (verified by reconstructing with djxl and comparing sha256 when
    `verify_roundtrip`); lossless preserves decoded pixels. Timestamps and
    filenames live in the manifest only. A JPEG the encoder refuses follows
    `on_unsupported_jpeg` (error | copy-source | lossless-jxl) and the
    per-file decision is recorded - a silent fallback would claim a
    reversibility the corpus does not have.
    """
    import shutil

    for tool in ("cjxl",) + (("djxl",) if (transcode and verify_roundtrip) else ()):
        if shutil.which(tool) is None:
            raise RuntimeError(f"jxl backend needs '{tool}' on PATH: brew install jpeg-xl")
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_cjxl(src: Path, dst: Path, args: list[str]) -> bool:
        return (
            subprocess.run(["cjxl", str(src), str(dst), *args], capture_output=True).returncode == 0
        )

    def one(i: int, path: Path) -> tuple[str, dict]:
        name = f"{i:06d}.jxl"
        action, verified = "lossless", None
        if transcode:
            supported = path.suffix.lower() in _JPEG_SUFFIXES and run_cjxl(
                path, out_dir / name, ["--lossless_jpeg=1"]
            )
            if supported:
                action = "transcode"
                if verify_roundtrip:
                    verified = _verify_jpeg_roundtrip(path, out_dir / name)
                    if not verified:
                        (out_dir / name).unlink(missing_ok=True)
                        supported = False
            if not supported:
                if on_unsupported_jpeg == "error":
                    raise RuntimeError(
                        f"jxl-transcode: {path} is not a reversibly-transcodable jpeg"
                    )
                if on_unsupported_jpeg == "copy-source":
                    name = f"{i:06d}{path.suffix.lower()}"
                    (out_dir / name).write_bytes(path.read_bytes())
                    action, verified = "copied", True
                else:  # lossless-jxl
                    if not run_cjxl(path, out_dir / name, ["-d", "0"]):
                        raise RuntimeError(f"cjxl -d 0 failed for {path}")
                    action, verified = "lossless", None
        else:
            if not run_cjxl(path, out_dir / name, ["-d", "0"]):
                raise RuntimeError(f"cjxl -d 0 failed for {path}")
        return name, {"file": name, "action": action, "verified": verified}

    # cjxl is one process per image and each file's outcome depends only on
    # its own source, so the fan-out changes wall time, never bytes.
    import os
    from concurrent.futures import ThreadPoolExecutor

    source_bytes = sum(p.stat().st_size for p in image_paths)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        results = list(pool.map(one, range(len(image_paths)), image_paths))
    files = [name for name, _ in results]
    decisions = [d for _, d in results]

    output_bytes = sum((out_dir / f).stat().st_size for f in files)
    toolchain = {"cjxl": _tool_version(["cjxl", "--version"]), "transcode": transcode}
    return {
        "backend": "jxl-transcode" if transcode else "jxl",
        "canvas": None,
        "frame_count": len(files),
        "files": files,
        "decisions": decisions,
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "compression_ratio": round(source_bytes / output_bytes, 2) if output_bytes else 0.0,
        "toolchain": toolchain,
        "provenance_sha256": provenance_sha256(toolchain),
    }


def _verify_jpeg_roundtrip(original: Path, jxl_path: Path) -> bool:
    """Reconstruct the JPEG from the .jxl and compare bytes exactly."""
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        proc = subprocess.run(["djxl", str(jxl_path), tmp.name], capture_output=True)
        if proc.returncode != 0:
            return False
        rebuilt = hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest()
    return rebuilt == hashlib.sha256(original.read_bytes()).hexdigest()
