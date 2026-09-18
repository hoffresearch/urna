"""Emit side of the declarative build: blob tables, space payloads,
the per-output `urna.build` calls (atomic commit), manifest v1 and
build.lock.json. Split from forge_pipeline (stages) along the
orchestrate/emit seam to honor the file-size contract.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from forge import image_media, model_registry
from forge.build_spec import default_model, emitted_spaces
from forge.forge_cache import atomic_write_json
from forge.forge_manifest import build_lock, check_lock, redact_path, write_manifest


def _blob_tables(ctx) -> tuple[list[dict] | None, list[dict] | None, list[str] | None]:
    if ctx.media is None:
        return None, None, None
    media_dir = image_media.media_dir_for(ctx.out_dir / f"{ctx.spec.name}.urna")
    embed = ctx.spec.output.embed_media
    refs: list[dict] = []
    paths: list[str] = []
    index_of: dict[str, int] = {}

    def add_ref(rel: str, sha: str | None) -> int:
        if rel not in index_of:
            p = media_dir / rel
            index_of[rel] = len(refs)
            refs.append(
                {
                    "content_hash": (sha or image_media.sha256_file(p)).removeprefix("sha256:"),
                    "original_uri": f"media://{rel}",
                    "byte_len": p.stat().st_size,
                    "inlined": embed,
                }
            )
            paths.append(str(p))
        return index_of[rel]

    spans = []
    if "#frame=" in ctx.frame_uris[0]:  # stream backend: blob per segment, span = frame index
        for seg in ctx.media["segments"]:
            add_ref(seg["uri"], seg.get("media_sha256"))
        for frame_idx in ctx.frame_of_row:
            seg_name, frag = ctx.frame_uris[frame_idx].removeprefix("media://").split("#frame=")
            spans.append(
                {
                    "blob_ref_index": index_of[seg_name],
                    "byte_start": int(frag),
                    "byte_end": int(frag) + 1,
                }
            )
    else:  # per-image backend: blob per file, span = whole file
        for frame_idx in ctx.frame_of_row:
            rel = ctx.frame_uris[frame_idx].removeprefix("media://")
            i = add_ref(rel, None)
            spans.append({"blob_ref_index": i, "byte_start": 0, "byte_end": refs[i]["byte_len"]})
    return refs, spans, (paths if embed else None)


def _spaces_payload(ctx, only_preset: str | None = None) -> list[dict]:
    spaces = []
    for ms, modality, dim, name in emitted_spaces(ctx.spec):
        if only_preset and ms.preset != only_preset:
            continue
        arrays = ctx.vectors[ms.preset]
        if modality == "image":  # noqa: SIM108, the branch is clearer than a ternary here
            vecs = arrays["image_unique"][ctx.frame_of_row]
        else:
            vecs = arrays["text"]
        if dim is not None:
            vecs = model_registry.slice_renorm(vecs, dim)
        spaces.append(
            {
                "name": name,
                "model_hash": ctx.model_meta[ms.preset]["model_hash"],
                "dtype": ms.space_dtype,
                "vectors": vecs.tolist(),
            }
        )
    return spaces


def _emit(ctx) -> dict:
    import urna

    spec = ctx.spec
    dm = default_model(spec)
    d_preset = model_registry.get_preset(dm.preset)
    text_vecs = ctx.vectors[dm.preset]["text"]
    blob_refs, chunk_spans, blob_paths = _blob_tables(ctx)
    chunks = [
        {
            "canonical_text": r.canonical_text,
            "source_uri": r.source_uri,
            "byte_start": r.ordinal,
            "byte_end": r.ordinal + 1,
            "embedding": text_vecs[i].tolist(),
        }
        for i, r in enumerate(ctx.rows)
    ]

    def emit_one(filename: str, spaces: list[dict]) -> dict:
        tmp = ctx.tmp_dir / filename
        final = ctx.out_dir / filename
        t0 = time.time()
        urna.build(
            str(tmp),
            d_preset.embedding_model,
            int(text_vecs.shape[1]),
            spec.chunker_version,
            ctx.model_meta[dm.preset]["model_hash"],
            chunks,
            title=spec.title or spec.name,
            version=spec.version,
            reproducible=spec.reproducible,
            preset=spec.build.preset,
            dtype=spec.build.dtype or None,
            mrl_dim=spec.build.mrl_dim or None,
            with_graph=spec.build.with_graph,
            graph_top_m=spec.build.graph_top_m,
            blob_refs=blob_refs,
            blob_data_paths=blob_paths,
            chunk_blob_spans=chunk_spans,
            spaces=spaces or None,
            provenance={"dataset": spec.name, "corpus_input_hash": ctx.input_hash},
        )
        os.replace(tmp, final)
        db = urna.open(str(final))
        db.validate()
        return {
            "file": str(final),
            "bytes": final.stat().st_size,
            "file_hash": db.file_hash,
            "build_s": round(time.time() - t0, 3),
        }

    outputs = {}
    if spec.output.mode in ("single", "both"):
        outputs[f"{spec.name}.urna"] = emit_one(f"{spec.name}.urna", _spaces_payload(ctx))
    if spec.output.mode in ("per-model", "both"):
        for ms in spec.models:
            if ms.preset == dm.preset and ms.image == "none" and spec.output.mode == "per-model":
                continue  # the default model alone would duplicate the single file's core
            outputs[f"{spec.name}-{ms.preset}.urna"] = emit_one(
                f"{spec.name}-{ms.preset}.urna", _spaces_payload(ctx, only_preset=ms.preset)
            )
    return {
        "outputs": outputs,
        "n_items": len(ctx.rows),
        "n_unique_frames": len(ctx.unique),
        "corpus_input_hash": ctx.input_hash,
        "timings": ctx.timings,
    }


def _finalize(ctx, result: dict, *, strict_env: bool, rebuild_only: bool) -> None:
    spec = ctx.spec
    spec_dir = Path(spec.spec_path).parent if spec.spec_path else ctx.out_dir
    mode = spec.output.provenance
    model_hashes = {p: m["model_hash"] for p, m in ctx.model_meta.items()}
    lock = build_lock(spec, model_hashes, device=os.environ.get("URNA_ST_DEVICE", "auto"))
    lock_path = ctx.out_dir / f"{spec.name}.build.lock.json"
    if rebuild_only and lock_path.is_file():
        diffs = check_lock(json.loads(lock_path.read_text()), lock)
        if diffs:
            msg = "build.lock divergence (L3 not claimable): " + "; ".join(diffs[:8])
            if strict_env:
                from forge.forge_pipeline import ForgeError

                raise ForgeError(msg)
            print(f"[forge] warning: {msg}")
    if ctx.models_filtered and lock_path.is_file():
        # a --models subset run must not poison the full build's lock: a
        # later full --rebuild-only would diff its models against the
        # subset and fail even though every cached vector is identical.
        print("[forge] --models subset: keeping the existing build.lock untouched")
    else:
        if ctx.models_filtered:
            lock["models_subset"] = sorted(m.preset for m in spec.models)
        atomic_write_json(lock_path, lock)

    manifest = {
        "name": spec.name,
        "n_items": result["n_items"],
        "n_unique_frames": result["n_unique_frames"],
        "corpus_input_hash": ctx.input_hash,
        "text_quality": "identity-only" if not spec.source.text.template else "template",
        "models": ctx.model_meta,
        "spaces": [
            {
                "name": name,
                "preset": ms.preset,
                "modality": modality,
                "dim": dim,
                "space_dtype": ms.space_dtype,
            }
            for ms, modality, dim, name in emitted_spaces(spec)
        ],
        "media": ({**ctx.media, "embedded": spec.output.embed_media} if ctx.media else None),
        "outputs": {
            k: {**v, "file": redact_path(v["file"], mode, spec_dir)}
            for k, v in result["outputs"].items()
        },
        "timings": ctx.timings,
        "sql": spec.source.query if mode == "full" else "",
        "items": [
            {
                "ordinal": r.ordinal,
                "key": r.key,
                "label": r.label,
                "media_uri": ctx.frame_uris[ctx.frame_of_row[i]] if ctx.frame_uris else None,
                "image_path": redact_path(str(r.image_path), mode, spec_dir)
                if r.image_path
                else None,
            }
            for i, r in enumerate(ctx.rows)
        ],
        # row -> unique frame; write_manifest keeps it only for a compact
        # manifest whose dedup collapsed rows (full items carry media_uri).
        "frame_of_row": list(ctx.frame_of_row),
    }
    write_manifest(ctx.out_dir / f"{spec.name}.manifest.json", manifest, mode, spec_dir)
    result["manifest"] = str(ctx.out_dir / f"{spec.name}.manifest.json")
    result["build_lock"] = str(lock_path)
