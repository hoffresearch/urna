"""crf="auto": pick the largest crf that clears a DUAL quality gate (RFC-2).

Two floors, never traded against each other:
- perceptual: SSIMULACRA2 p10 per stratum >= visual_floor_p10 and global min
  >= visual_floor_min. A global average would let one whole stratum degrade.
- vector: cosine(embed(source), embed(decoded)) p10 >= drift_floor_p10 with
  the gate model. An image can look fine to humans and still move in
  embedding space, which is what retrieval actually serves. A negative
  floor disables the leg (recorded as drift_pass = true).
- optional third leg (quality_utility): text-to-image hit@1 of the sample
  must stay within utility_tol of the lossless source and above
  utility_floor_hit1. Enabled by a non-negative floor; it is what a
  retrieval-only profile gates on once drift is disabled.

Strata come from cheap deterministic heuristics; they are themselves policy,
so BUCKET_HEURISTICS_VERSION is recorded in the report and participates in
the embedding recipe hash of gated builds. Full retrieval recall lives in
the sweep (RFC-5), outside this ladder loop, where it costs O(1) per variant.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from forge import image_media
from forge.image_decode import decode_frames
from forge.image_encode import encode_av1
from forge.quality_utility import UtilityGate

BUCKET_HEURISTICS_VERSION = 1


def _gray_thumb(path: Path, size: int = 128) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("L")
        img.thumbnail((size, size))
        return np.asarray(img, dtype=np.float32)


def bucket_of(path: Path, buckets: list[str]) -> tuple[str, ...]:
    """Deterministic stratum key (version BUCKET_HEURISTICS_VERSION)."""
    from PIL import Image

    parts: list[str] = []
    if set(buckets) & {"resolution", "alpha"}:
        with Image.open(path) as img:
            size_max, mode, info = max(img.size), img.mode, img.info
    thumb = _gray_thumb(path) if set(buckets) & {"entropy", "has_text"} else None
    for name in buckets:
        if name == "resolution":
            parts.append("small" if size_max < 512 else "medium" if size_max < 1024 else "large")
        elif name == "entropy":
            gy, gx = np.gradient(thumb)
            var = float(np.var(np.hypot(gx, gy)))
            parts.append("low" if var < 100 else "mid" if var < 1000 else "high")
        elif name == "has_text":
            gy, gx = np.gradient(thumb)
            frac = float(np.mean(np.hypot(gx, gy) > 40.0))
            parts.append("text" if frac > 0.10 else "plain")
        elif name == "alpha":
            parts.append("alpha" if ("A" in mode or "transparency" in info) else "opaque")
        elif name == "source_format":
            parts.append(path.suffix.lower().lstrip("."))
        else:
            raise ValueError(f"media.quality.buckets: unknown bucket '{name}'")
    return tuple(parts)


def stratified_sample(paths: list[Path], buckets: list[str], per_bucket: int) -> list[int]:
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(paths):
        groups.setdefault(bucket_of(p, buckets), []).append(i)
    picked: list[int] = []
    for key in sorted(groups):
        members = groups[key]
        step = max(1, len(members) // per_bucket)
        picked.extend(members[::step][:per_bucket])
    return sorted(picked)


def _ssimulacra2(orig_png: Path, dist_png: Path) -> float:
    out = subprocess.run(
        ["ssimulacra2", str(orig_png), str(dist_png)], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise RuntimeError(f"ssimulacra2 failed: {out.stderr[:200]}")
    return float(out.stdout.strip().split()[-1])


def choose_crf(
    paths: list[Path],
    canvas: tuple[int, int],
    media_spec,
    gate_adapter,
    labels: list[str] | None = None,
):
    """Return (chosen_crf, report). See module docstring for the gate.
    `labels` (one per path) render the utility queries; required only when
    utility_floor_hit1 >= 0."""
    from PIL import Image

    if shutil.which("ssimulacra2") is None:
        raise RuntimeError('media.crf="auto" needs ssimulacra2 on PATH: brew install jpeg-xl')
    q = media_spec.quality
    use_utility = q.utility_floor_hit1 >= 0
    if use_utility and (labels is None or len(labels) != len(paths)):
        from forge.build_spec import SpecError

        raise SpecError("media.quality.utility_floor_hit1: needs one label per item")
    idx = stratified_sample(paths, q.buckets, q.sample_per_bucket)
    sample = [paths[i] for i in idx]
    keys = [bucket_of(p, q.buckets) for p in sample]

    with tempfile.TemporaryDirectory(prefix="urna-crf-auto-") as tmp:
        tmp = Path(tmp)
        src_arrays: list[np.ndarray] = []
        src_pngs: list[Path] = []
        for i, p in enumerate(sample):
            with Image.open(p) as img:
                arr = np.asarray(image_media.letterbox(img, canvas), dtype=np.uint8)
            src_arrays.append(arr)
            png = tmp / f"src-{i:04d}.png"
            Image.fromarray(arr).save(png)
            src_pngs.append(png)
        src_emb = gate_adapter.embed_arrays(src_arrays)
        utility = None
        if use_utility:
            utility = UtilityGate(q, gate_adapter, [labels[i] for i in idx], src_emb)

        ladder_report: dict[str, dict] = {}
        passing: list[int] = []
        for crf in sorted(q.crf_ladder):
            mp4 = tmp / f"crf{crf}.mp4"
            encode_av1(
                sample,
                mp4,
                canvas=canvas,
                crf=crf,
                preset=media_spec.speed,
                keyint=1,
                pix_fmt=media_spec.pix_fmt,
                tune=media_spec.tune,
            )
            decoded = [f for batch in decode_frames(mp4, canvas) for f in batch]
            scores, dist_pngs = [], []
            for i, frame in enumerate(decoded):
                png = tmp / f"crf{crf}-{i:04d}.png"
                Image.fromarray(frame).save(png)
                scores.append(_ssimulacra2(src_pngs[i], png))
                dist_pngs.append(png)
            dec_emb = gate_adapter.embed_arrays(decoded)
            drift = np.sum(src_emb * dec_emb, axis=1)

            by_bucket: dict[str, float] = {}
            ok_buckets = True
            for key in sorted(set(keys)):
                vals = [s for s, k in zip(scores, keys, strict=True) if k == key]
                p10 = float(np.percentile(vals, 10))
                by_bucket["/".join(key)] = round(p10, 2)
                ok_buckets &= p10 >= q.visual_floor_p10
            ssim_min = float(np.min(scores))
            drift_p10 = float(np.percentile(drift, 10))
            visual_ok = ok_buckets and ssim_min >= q.visual_floor_min
            drift_ok = q.drift_floor_p10 < 0 or drift_p10 >= q.drift_floor_p10
            rung = {
                "ssim_p10_by_bucket": by_bucket,
                "ssim_min": round(ssim_min, 2),
                "drift_p10": round(drift_p10, 5),
                "visual_pass": visual_ok,
                "drift_pass": drift_ok,
            }
            ok = visual_ok and drift_ok
            if utility is not None:
                hit1, utility_ok = utility.check(dec_emb)
                rung["hit1"] = round(hit1, 4)
                rung["utility_pass"] = utility_ok
                ok = ok and utility_ok
            rung["pass"] = ok
            ladder_report[str(crf)] = rung
            if ok:
                passing.append(crf)
            for png in dist_pngs:
                png.unlink()

    warning = None
    if passing:
        chosen = max(passing)
    else:
        chosen = min(q.crf_ladder)
        floors = (
            f"visual p10>={q.visual_floor_p10}, min>={q.visual_floor_min}, "
            f"drift p10>={q.drift_floor_p10}"
        )
        if utility is not None:
            floors += f", utility hit@1>={utility.threshold:.4f}"
        warning = f"no ladder crf met the floors ({floors}); using smallest crf {chosen}"
        print(f"[forge] warning: {warning}")
    report = {
        "bucket_heuristics_version": BUCKET_HEURISTICS_VERSION,
        "buckets": q.buckets,
        "n_sampled": len(sample),
        "sample_indices": idx,
        "ladder": ladder_report,
        "chosen_crf": chosen,
        "warning": warning,
        "gate_model_hash": gate_adapter.model_hash,
    }
    if utility is not None:
        report["utility"] = utility.report()
    return chosen, report
