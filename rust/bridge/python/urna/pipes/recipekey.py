"""The identity side of the embed-cache triad: the embedding recipe (what
usage the vectors were computed under) and the model_hash probe (which
model computed them), both spec-independent so cache entries can be
shared across specs and output dirs.

The decoder fingerprint enters the recipe ONLY for models that embed the
decoded media (`image = "space"`): a text-only model's vectors do not
change with the crf, so its recipe must not either, otherwise every media
variant recomputes the same text table.
"""

from __future__ import annotations

import json
from pathlib import Path

from urna.model import presetmap
from urna.pipes.vectcache import atomic_write_json, canonical_hash
from urna.specs.specparse import CorpusSpec, ModelSpec

ADAPTER_VERSION = 1


def model_dir_fingerprint(preset, model_path: str | None) -> str | None:
    """Cheap identity of the resolved model dir: sorted (relpath, size)
    pairs, hashed. Catches a swapped snapshot without reading weights;
    None for presets with no on-disk dir (vendored potion, fake)."""
    d = presetmap.resolve_model_dir(preset, model_path)
    if d is None or not Path(d).is_dir():
        return None
    listing = sorted(
        (str(p.relative_to(d)), p.stat().st_size) for p in Path(d).rglob("*") if p.is_file()
    )
    return canonical_hash({"dir": listing})


def recipe(spec: CorpusSpec, media: dict | None, ms: ModelSpec, preset) -> dict:
    mode = spec.image_input_mode()
    out = {
        "adapter_version": ADAPTER_VERSION,
        "preset": ms.preset,
        "image_input_mode": mode,
        "text_corpus_mode": ms.text_corpus_mode or preset.text_corpus_mode,
        "text_query_mode": ms.text_query_mode or preset.text_query_mode,
        "image_mode": ms.image_mode or preset.image_mode,
        "image_prompt": ms.image_prompt or preset.image_prompt,
        "normalize": ms.normalize,
        "preprocess_version": ms.preprocess_version or preset.preprocess_version,
        "image_max_side": ms.image_max_side or preset.image_max_side,
        "image_doc_format": getattr(preset, "image_doc_format", "dict"),
        "encode_kwargs": ms.encode_kwargs,
        "model_dtype": ms.dtype,
        "device_class": ms.device or "auto",
    }
    if mode == "decoded_media" and media is not None and ms.image == "space":
        out["decoder"] = {
            "backend": media.get("backend"),
            "canvas": media.get("canvas"),
            "crf": media.get("crf"),
            "pix_fmt": media.get("pix_fmt"),
            "provenance_sha256": media.get("provenance_sha256"),
        }
    return out


def probe_knobs(ms: ModelSpec) -> dict:
    """The spec knobs that enter an st model_hash (stbackend.fingerprint_for:
    normalize + dtype policy, the latter resolved from the device) plus the
    model path. Two specs that differ on any of these have different
    model_hashes for the same preset, so they must not share a probe."""
    return {
        "normalize": ms.normalize,
        "model_dtype": ms.dtype,
        "device_class": ms.device or "auto",
        "model_path": ms.model_path or None,
    }


def probe_path(cache_dir: Path, ms: ModelSpec) -> Path:
    knobs = canonical_hash(probe_knobs(ms)).removeprefix("sha256:")[:16]
    return cache_dir / "models" / f"model_hash.{ms.preset}.{knobs}.json"


def write_probe(cache_dir: Path, preset, ms: ModelSpec, model_hash: str) -> None:
    dir_fp = model_dir_fingerprint(preset, ms.model_path or None)
    payload = {"model_hash": model_hash, "dir_fingerprint": dir_fp, "knobs": probe_knobs(ms)}
    atomic_write_json(probe_path(cache_dir, ms), payload)


def probe_model_hash(cache_dir: Path, preset, ms: ModelSpec, get_adapter) -> str:
    """The triad needs model_hash, which needs a loaded model. Cache a probe
    of it under `<cache_dir>/models/`, keyed by preset + the knobs that enter
    the fingerprint, and guarded by a cheap fingerprint of the resolved model
    dir, so swapping the snapshot on disk invalidates the probe instead of
    silently emitting the old model's vectors. The probe is spec-independent,
    so it lives in the shared cache root. A loaded model is the ground truth:
    the embed stage overwrites a probe that disagrees with it (write_probe)."""
    dir_fp = model_dir_fingerprint(preset, ms.model_path or None)
    probe_file = probe_path(cache_dir, ms)
    if probe_file.is_file():
        probed = json.loads(probe_file.read_text())
        if probed.get("dir_fingerprint") == dir_fp:
            return probed["model_hash"]
    model_hash = get_adapter().model_hash
    write_probe(cache_dir, preset, ms, model_hash)
    return model_hash
