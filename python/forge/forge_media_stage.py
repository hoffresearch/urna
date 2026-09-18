"""The media stage of the declarative build: encode the unique frames once
(crf=auto gate, similarity/cluster ordering, sharding) and record completion
under `<out>/.forge-state/media.json` keyed by canonical_hash(input_hash +
media spec), so `--resume` can verify the files and skip the encode.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from forge import image_media, model_registry
from forge.build_spec import CorpusSpec
from forge.forge_cache import atomic_write_json, canonical_hash

if TYPE_CHECKING:
    from forge.forge_pipeline import _Ctx


def _media_params(ctx: _Ctx) -> str:
    return canonical_hash({"input": ctx.input_hash, "media": asdict(ctx.spec.media)})


def media_stage(ctx: _Ctx, *, resume: bool) -> None:
    spec = ctx.spec
    if spec.media is None or not any(r.image_path for r in ctx.unique):
        return
    missing = [r.key for r in ctx.rows if r.image_path is None]
    if missing:
        from forge.forge_pipeline import ForgeError  # lazy: pipeline imports this module

        raise ForgeError(
            f"media enabled but {len(missing)} rows have no image (first: {missing[0]})"
        )

    state_file = ctx.state_dir / "media.json"
    params = _media_params(ctx)
    media_dir = image_media.media_dir_for(ctx.out_dir / f"{spec.name}.urna")
    if resume and state_file.is_file():
        st = json.loads(state_file.read_text())
        if st.get("params") == params and _media_files_ok(media_dir, st["media"], st["frame_uris"]):
            ctx.media, ctx.frame_uris = st["media"], st["frame_uris"]
            ctx.timings["media"] = 0.0
            return
        if resume and st.get("params") != params:
            pass  # spec/input changed: re-encode
    t0 = time.time()
    ctx.media, ctx.frame_uris = _encode_media(ctx, media_dir)
    ctx.timings["media"] = round(time.time() - t0, 3)
    atomic_write_json(
        state_file,
        {"params": params, "media": ctx.media, "frame_uris": ctx.frame_uris, "done": True},
    )


def _media_files_ok(media_dir: Path, media: dict, frame_uris: list[str]) -> bool:
    segments = media.get("segments")
    if not segments:
        # per-image backends (jxl/avif/control): every frame's file must
        # still exist non-empty; a bare "the dir has entries" check would
        # let resume package deleted or truncated media.
        for uri in frame_uris:
            p = media_dir / uri.removeprefix("media://").split("#frame=")[0]
            if not p.is_file() or p.stat().st_size == 0:
                return False
        return True
    for seg in segments:
        p = media_dir / seg["uri"]
        if not p.is_file():
            return False
        if seg.get("media_sha256") and image_media.sha256_file(p) != seg["media_sha256"]:
            return False
    return True


def _gate_adapter(ctx: _Ctx, preset_name: str):
    ms = next((m for m in ctx.spec.models if m.preset == preset_name), None)
    if ms is None:  # validate() mirrors this; keep the crash typed regardless
        from forge.forge_pipeline import ForgeError

        raise ForgeError(f"media gate/cluster model '{preset_name}' is not a spec model")
    return model_registry.create_embedder(
        preset_name,
        model_path=ms.model_path or None,
        device=ms.device or None,
        batch_size=ms.batch_size,
        allow_remote_code=frozenset(ctx.spec.output.allow_remote_code),
        allow_heavy=True,
    )


def _encode_media(ctx: _Ctx, media_dir: Path) -> tuple[dict, list[str]]:
    from forge import image_backends

    m = ctx.spec.media
    paths = [r.image_path for r in ctx.unique]
    canvas = image_media.canvas_size(paths, m.width)

    crf = m.crf
    quality_report = None
    if crf == "auto":
        from forge import quality_gate

        gate = _gate_adapter(ctx, m.quality.gate_model or first_image_preset(ctx.spec))
        labels = [r.label or r.text for r in ctx.unique]  # utility queries, one per frame
        crf, quality_report = quality_gate.choose_crf(paths, canvas, m, gate, labels=labels)

    order = None
    if m.order in ("similarity", "cluster"):
        from forge import image_order

        gate = _gate_adapter(ctx, m.cluster.space or first_image_preset(ctx.spec))
        vecs = gate.embed_paths(paths)
        order = (
            image_order.similarity_order(vecs)
            if m.order == "similarity"
            else image_order.cluster_order(vecs, m.cluster.threshold)
        )

    built = image_backends.build_media(
        paths,
        ctx.out_dir / f"{ctx.spec.name}.urna",
        ctx.spec.name,
        backend=m.backend,
        canvas=canvas,
        crf=int(crf),
        speed=m.speed,
        all_intra=(m.gop == "intra"),
        pix_fmt=m.pix_fmt,
        avif_quality=int(crf) if isinstance(crf, int) else 35,
        control=(m.backend == "control"),
        gop_policy=m.gop if m.gop != "intra" else "intra",
        order=order,
        shard_size=m.shard_size,
        tune=m.tune,
        fps=m.fps,
        jxl_transcode=m.jxl_transcode,
    )
    media = built["media"]
    if order is not None and media.get("order_permutation"):
        media["order"] = m.order  # the spec's method, not the backend's generic label
    media["dedup"] = {"n_items": len(ctx.rows), "n_unique_frames": len(ctx.unique)}
    if quality_report is not None:
        media["crf_auto"] = quality_report
    return media, built["uris"]


def first_image_preset(spec: CorpusSpec) -> str:
    return next(m.preset for m in spec.models if m.image == "space")
