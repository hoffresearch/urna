"""Validation rules over a parsed CorpusSpec (RFC-1): every violation is a
SpecError naming the offending key, and the emitted-space enumeration both
sides (pipeline and validator) share. Split from build_spec (the dataclass
contract + parser) along the schema/rules seam.
"""

from __future__ import annotations

from forge import model_registry
from forge.build_spec import MAX_SPACES, CorpusSpec, ModelSpec, SpecError


def emitted_spaces(spec: CorpusSpec) -> list[tuple[ModelSpec, str, int | None, str]]:
    """All named spaces the spec emits: (model, modality, dim|None=native, name)."""
    out = []
    for m in spec.models:
        dims: list[int | None] = list(m.dims) or [None]
        if m.image == "space":
            for d in dims:
                out.append((m, "image", d, m.preset if d is None else f"{m.preset}@{d}"))
        if m.text == "space":
            for d in dims:
                name = f"{m.preset}-text" if d is None else f"{m.preset}-text@{d}"
                out.append((m, "text", d, name))
    return out


def default_model(spec: CorpusSpec) -> ModelSpec:
    return next(m for m in spec.models if m.text == "default")


def validate(spec: CorpusSpec, *, allow_heavy: bool = False) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise SpecError(msg)

    need(bool(spec.name), "corpus.name: required")
    need(bool(spec.chunker_version), "corpus.chunker_version: required")
    kinds = {"sqlite", "image_dir", "pdf_dir", "csv", "jsonl"}
    need(spec.source.kind in kinds, f"source.kind: must be one of {sorted(kinds)}")
    if spec.source.kind == "sqlite":
        need(bool(spec.source.db) and bool(spec.source.query), "source.db/source.query: required")
        need(bool(spec.source.order_by), "source.order_by: required for total ordering (RFC-0 N1)")
    if spec.source.kind in ("csv", "jsonl"):
        need(bool(spec.source.path), "source.path: required")
        need(bool(spec.source.order_by), "source.order_by: required for total ordering (RFC-0 N1)")
    if spec.source.kind in ("image_dir", "pdf_dir"):
        need(bool(spec.source.input_dir), "source.input_dir: required")

    need(
        spec.image_input.mode in ("", "source", "decoded_media"),
        "embedding.image_input.mode: source | decoded_media",
    )
    if spec.image_input.mode == "decoded_media":
        need(spec.media is not None, "embedding.image_input.mode=decoded_media requires [media]")

    need(len(spec.models) > 0, "models: at least one [[models]] required")
    defaults = [m for m in spec.models if m.text == "default"]
    need(
        len(defaults) == 1,
        f'models: exactly one text="default" required (RFC-0 N14), found {len(defaults)}',
    )
    seen = set()
    image_presets = []
    for m in spec.models:
        need(m.preset not in seen, f"models: duplicate preset '{m.preset}'")
        seen.add(m.preset)
        preset = model_registry.get_preset(m.preset)  # RegistryError lists valid names
        need(m.text in ("default", "space", "none"), f"models.{m.preset}.text: default|space|none")
        need(m.image in ("space", "none"), f"models.{m.preset}.image: space|none")
        need(
            m.text != "none" or m.image != "none",
            f"models.{m.preset}: text=none and image=none emits nothing",
        )
        if m.text in ("default", "space"):
            need("text" in preset.modalities, f"models.{m.preset}: preset has no text tower")
        if m.image == "space":
            need("image" in preset.modalities, f"models.{m.preset}: preset has no image tower")
            has_images = spec.source.kind in ("image_dir", "pdf_dir") or bool(
                spec.source.image.path_template
            )
            need(has_images, f"models.{m.preset}.image=space: source declares no images")
            image_presets.append(m.preset)
        if m.dims:
            need(
                preset.mrl.supported,
                f"models.{m.preset}.dims: preset is not MRL-trained; dims not allowed",
            )
            for d in m.dims:
                need(
                    d in preset.mrl.dims,
                    f"models.{m.preset}.dims: {d} not in the validated ladder "
                    f"{sorted(preset.mrl.dims)} (method {preset.mrl.method})",
                )
        need(
            m.space_dtype in ("float32", "float16", "int8", "int4"),
            f"models.{m.preset}.space_dtype: float32|float16|int8|int4",
        )
        if m.space_dtype == "int4":
            for d in m.dims or [preset.default_dim]:
                need(d == 0 or d % 64 == 0, f"models.{m.preset}: int4 requires dim%64==0, got {d}")
        if not preset.executable:
            need(
                allow_heavy,
                f"models.{m.preset}: flagged too heavy for this machine; pass --allow-heavy",
            )
        if preset.trust_remote_code:
            need(
                m.preset in spec.output.allow_remote_code,
                f"models.{m.preset}: executes model-repo code; opt in with "
                f'output.allow_remote_code = ["{m.preset}"] (RFC-0 N11)',
            )

    spaces = emitted_spaces(spec)
    need(len(spaces) <= MAX_SPACES, f"models: {len(spaces)} named spaces > max {MAX_SPACES}")

    if spec.media is not None:
        m = spec.media
        from forge.build_spec import MEDIA_PROFILES

        need(
            m.profile in ("", *MEDIA_PROFILES),
            f"media.profile: '' | {' | '.join(sorted(MEDIA_PROFILES))}",
        )
        need(
            m.backend in ("av1", "avif", "jxl", "jxl-transcode", "control"),
            "media.backend: av1|avif|jxl|jxl-transcode|control",
        )
        need(m.gop in ("auto", "intra", "inter"), "media.gop: auto|intra|inter")
        need(m.order in ("none", "similarity", "cluster"), "media.order: none|similarity|cluster")
        need(m.tune in ("default", "still"), "media.tune: default|still")
        need(isinstance(m.crf, int) or m.crf == "auto", 'media.crf: int | "auto"')
        need(
            m.jxl_transcode.on_unsupported_jpeg in ("error", "copy-source", "lossless-jxl"),
            "media.jxl_transcode.on_unsupported_jpeg: error|copy-source|lossless-jxl",
        )
        # validate each knob against ITS OWN pipeline resolution: crf=auto
        # gates on quality.gate_model, ordering clusters on cluster.space,
        # both falling back to the first image model, never on each other.
        if m.crf == "auto":
            # the gate encodes the ladder with the av1 stream; an av1 crf
            # is not an avifenc -q, and the jxl backends ignore crf.
            need(
                m.backend == "av1",
                'media.crf: "auto" gates the av1 ladder only (backend = "av1", '
                'or profile = "stills-av1"); the avif and jxl backends take a fixed crf',
            )
            gate = m.quality.gate_model or (image_presets[0] if image_presets else "")
            need(bool(gate), "media.quality.gate_model: crf=auto needs an image model")
            need(
                gate in image_presets,
                f"media.quality.gate_model '{gate}' must be a spec model with image=space",
            )
            _validate_utility(m.quality, gate, need)
        if m.order in ("similarity", "cluster"):
            cspace = m.cluster.space or (image_presets[0] if image_presets else "")
            need(bool(cspace), f"media.cluster.space: order={m.order} needs an image model")
            need(
                cspace in image_presets,
                f"media.cluster.space '{cspace}' must be a spec model with image=space",
            )

    need(spec.build.graph_space == "default", 'build.graph_space: only "default" is supported')
    need(spec.output.mode in ("single", "per-model", "both"), "output.mode: single|per-model|both")
    need(
        spec.output.provenance in ("minimal", "standard", "full"),
        "output.provenance: minimal|standard|full",
    )
    need(
        not (spec.output.embed_media and spec.media is None),
        "output.embed_media requires a [media] section (there is nothing to inline)",
    )


def _validate_utility(q, gate: str, need) -> None:
    """The task-utility floor: ranges, the query template, and a gate model
    that can embed text (hit@1 is text-to-image)."""
    if q.utility_floor_hit1 < 0:
        return  # disabled; the other utility keys are inert
    need(
        0.0 <= q.utility_floor_hit1 <= 1.0,
        "media.quality.utility_floor_hit1: hit@1 floor must be within 0..1 (negative = off)",
    )
    need(
        q.utility_queries >= 0,
        "media.quality.utility_queries: must be >= 0 (0 = every sampled item)",
    )
    need(
        "{label}" in q.utility_query_template,
        'media.quality.utility_query_template: must contain "{label}"',
    )
    need(0.0 <= q.utility_tol <= 1.0, "media.quality.utility_tol: must be within 0..1")
    preset = model_registry.get_preset(gate)
    need(
        "text" in preset.modalities,
        f"media.quality.utility_floor_hit1: gate model '{gate}' has no text tower",
    )
