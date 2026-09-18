"""Declarative build orchestration (RFC-1): rows → dedup → media → embed → emit.

Transactional (RFC-0 N7): each expensive stage records completion + a params
hash under `<out>/.forge-state/`; outputs are written to `<out>/.tmp/` and
committed by atomic rename; `resume=True` skips stages whose params match and
whose artifacts verify. Embedding caches are their own state: content-
addressed by the triad under a shared root (forge_cache.cache_root), so they
live outside `<out>` and are reused across specs and output dirs. Rows are
cheap and always recomputed; their corpus_input_hash is what the other
stages key on.

Sharing (N4): per-model outputs reuse the one media encode, blob table and
cached vectors; space 0 of every file is the one text="default" model (N14).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from forge import forge_recipe, image_media, model_registry
from forge.build_spec import CorpusSpec, SpecError, validate
from forge.corpus_sources import Row, corpus_input_hash, load_rows
from forge.forge_cache import EmbedCache, cache_root, canonical_hash
from forge.forge_media_stage import media_stage


class ForgeError(RuntimeError):
    pass


@dataclass
class _Ctx:
    spec: CorpusSpec
    out_dir: Path
    cache_dir: Path  # shared, content-addressed: embed/<preset>/<triad>.npz + models/
    rows: list[Row] = field(default_factory=list)
    unique: list[Row] = field(default_factory=list)  # dedup: first occurrence per image hash
    frame_of_row: list[int] = field(default_factory=list)
    input_hash: str = ""
    media: dict | None = None
    frame_uris: list[str] = field(default_factory=list)  # per UNIQUE frame, item order
    vectors: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # preset -> arrays
    model_meta: dict[str, dict] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    models_filtered: bool = False  # --models subset: never overwrite a full build.lock

    @property
    def state_dir(self) -> Path:
        return self.out_dir / ".forge-state"

    @property
    def tmp_dir(self) -> Path:
        return self.out_dir / ".tmp"


def build(
    spec: CorpusSpec,
    *,
    sample: int | None = None,
    seed: int = 42,
    models_filter: list[str] | None = None,
    resume: bool = False,
    rebuild_only: bool = False,
    strict_env: bool = False,
    allow_heavy: bool = False,
) -> dict:
    validate(spec, allow_heavy=allow_heavy)
    if models_filter:
        spec.models = [m for m in spec.models if m.preset in models_filter]
        validate(spec, allow_heavy=allow_heavy)
    ctx = _Ctx(
        spec=spec,
        out_dir=Path(spec.output.dir),
        cache_dir=cache_root(spec.output.cache_dir),
        models_filtered=bool(models_filter),
    )
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    ctx.state_dir.mkdir(exist_ok=True)
    ctx.tmp_dir.mkdir(exist_ok=True)
    _ensure_cache_root(ctx.cache_dir)

    t0 = time.time()
    ctx.rows = load_rows(spec, sample=sample, seed=seed)
    if not ctx.rows:
        raise ForgeError("source produced no rows")
    ctx.input_hash = corpus_input_hash(ctx.rows)
    _dedup(ctx)
    ctx.timings["rows"] = round(time.time() - t0, 3)

    media_stage(ctx, resume=resume or rebuild_only)
    _embed_stage(ctx, rebuild_only=rebuild_only)
    from forge import forge_emit

    result = forge_emit._emit(ctx)
    forge_emit._finalize(ctx, result, strict_env=strict_env, rebuild_only=rebuild_only)
    return result


def _ensure_cache_root(path: Path) -> None:
    """First touch of a user-controlled path: name the setting on failure
    instead of leaking an OSError traceback out of the cli."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SpecError(
            f"output.cache_dir: cache root {path} is not a usable directory ({e.strerror}); "
            "set via [output] cache_dir, --cache-dir or URNA_CACHE_DIR"
        ) from e
    if not path.is_dir():
        raise SpecError(
            f"output.cache_dir: cache root {path} is not a directory; "
            "set via [output] cache_dir, --cache-dir or URNA_CACHE_DIR"
        )


def _dedup(ctx: _Ctx) -> None:
    dedup_on = ctx.spec.media.dedup if ctx.spec.media else True
    seen: dict[str, int] = {}
    for row in ctx.rows:
        key = row.image_sha256 if (row.image_sha256 and dedup_on) else f"row:{row.ordinal}"
        if key not in seen:
            seen[key] = len(ctx.unique)
            ctx.unique.append(row)
        ctx.frame_of_row.append(seen[key])


def _embed_stage(ctx: _Ctx, *, rebuild_only: bool) -> None:
    spec = ctx.spec
    for ms in spec.models:
        preset = model_registry.get_preset(ms.preset)
        adapter = None
        recipe = forge_recipe.recipe(spec, ctx.media, ms, preset)
        recipe_hash = canonical_hash(recipe)

        def get_adapter(ms=ms, recipe=recipe):
            nonlocal adapter
            if adapter is None:
                adapter = model_registry.create_embedder(
                    ms.preset,
                    model_path=ms.model_path or None,
                    device=ms.device or None,
                    batch_size=ms.batch_size,
                    allow_remote_code=frozenset(spec.output.allow_remote_code),
                    allow_heavy=True,
                    usage=recipe,
                )
            return adapter

        model_hash = forge_recipe.probe_model_hash(ctx.cache_dir, preset, ms, get_adapter)
        want_text = ms.text in ("default", "space")
        want_image = ms.image == "space"
        triad = {
            "model_hash": model_hash,
            "embedding_recipe_hash": recipe_hash,
            "corpus_input_hash": ctx.input_hash,
            # not part of the RFC-0 triad proper, but part of the cache key
            # (it enters the content-addressed file name): WHICH arrays this
            # spec needs, and whether dedup shaped the image rows. a spec
            # edit that changes any of these must miss.
            "arrays": {
                "text": want_text,
                "image": want_image,
                "dedup": bool(spec.media.dedup) if spec.media else None,
            },
        }
        cache = EmbedCache(ctx.cache_dir, ms.preset, triad)

        arrays = cache.load()
        required = {k for k, want in (("text", want_text), ("image_unique", want_image)) if want}
        if arrays is not None and not required <= set(arrays):
            arrays = None  # pre-fix cache entry that lacks an array emit needs
        if arrays is None:
            if rebuild_only:
                raise ForgeError(
                    f"--rebuild-only: cache for '{ms.preset}' is missing or stale "
                    "(triad mismatch); run a full build"
                )
            t0 = time.time()
            arrays = {}
            if ms.text in ("default", "space"):
                arrays["text"] = get_adapter().embed_texts([r.canonical_text for r in ctx.rows])
            if ms.image == "space":
                arrays["image_unique"] = _embed_images(ctx, get_adapter())
            elapsed = max(time.time() - t0, 1e-9)
            n = sum(a.shape[0] for a in arrays.values())
            ctx.timings[f"embed.{ms.preset}"] = round(elapsed, 3)
            ctx.model_meta.setdefault(ms.preset, {})["items_per_s"] = round(n / elapsed, 2)
            # the loaded model is the ground truth: a probe that disagrees is
            # stale (planted by a spec with other knobs, or an unfingerprinted
            # model swap). rewrite it and key the entry by the real hash.
            if adapter.model_hash != model_hash:
                model_hash = adapter.model_hash
                forge_recipe.write_probe(ctx.cache_dir, preset, ms, model_hash)
                triad["model_hash"] = model_hash
                cache = EmbedCache(ctx.cache_dir, ms.preset, triad)
            cache.store(arrays)
        else:
            ctx.timings[f"embed.{ms.preset}"] = 0.0
        if adapter is not None and hasattr(adapter, "close"):
            adapter.close()  # st workers: return the model's memory before the next model
        ctx.vectors[ms.preset] = arrays
        ctx.model_meta.setdefault(ms.preset, {}).update(
            {"model_hash": model_hash, "embedding_recipe_hash": recipe_hash, "recipe": recipe}
        )


def _embed_images(ctx: _Ctx, adapter) -> np.ndarray:
    mode = ctx.spec.image_input_mode()
    paths = [r.image_path for r in ctx.unique]
    if mode == "source" or ctx.media is None:
        return adapter.embed_paths(paths)
    from forge import image_backends
    from forge.image_corpus import _embed_compressed

    media_dir = image_media.media_dir_for(ctx.out_dir / f"{ctx.spec.name}.urna")
    frames_fn = image_backends.decoded_frames_fn(media_dir, ctx.media, ctx.frame_uris)
    vecs, _hashes = _embed_compressed(adapter, frames_fn, len(ctx.unique))
    perm = ctx.media.get("order_permutation")
    if perm:  # stream order -> item order (same inverse as image_corpus.build_corpus)
        vecs = vecs[np.argsort(np.asarray(perm))]
    return vecs
