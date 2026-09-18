"""Declarative build spec: parse + validate the TOML/JSON contract (RFC-1).

Every user-facing choice lives here as a typed field with its default; every
violation raises SpecError naming the offending key. Unknown keys are errors,
not silence: a typo'd knob that silently does nothing is a lie in a config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from forge.media_profiles import MEDIA_PROFILES, merge_profile


class SpecError(ValueError):
    """Spec violation; message names the offending key."""


def _section(cls, data: dict, where: str):
    """Build a dataclass from a dict, refusing unknown keys."""
    allowed = {f.name for f in fields(cls)}
    for key in data:
        if key not in allowed:
            raise SpecError(f"{where}: unknown key '{key}' (valid: {', '.join(sorted(allowed))})")
    return cls(**data)


@dataclass
class SourceJoin:
    query: str
    on: str


@dataclass
class SourceText:
    template: str = ""


@dataclass
class SourceImage:
    path_template: str = ""
    label_template: str = ""


@dataclass
class SourceSpec:
    kind: str = ""
    db: str = ""
    query: str = ""
    order_by: list[str] = field(default_factory=list)
    derive: dict[str, str] = field(default_factory=dict)
    joins: list[SourceJoin] = field(default_factory=list)
    text: SourceText = field(default_factory=SourceText)
    image: SourceImage = field(default_factory=SourceImage)
    input_dir: str = ""  # image_dir | pdf_dir
    labels: str = ""  # image_dir | pdf_dir label file
    path: str = ""  # csv | jsonl


@dataclass
class ImageInputSpec:
    mode: str = ""  # "" = decoded_media when [media] present, else source


@dataclass
class QualitySpec:
    strategy: str = "stratified"
    buckets: list[str] = field(
        default_factory=lambda: ["resolution", "entropy", "has_text", "alpha", "source_format"]
    )
    sample_per_bucket: int = 12
    # floors a real corpus can reach. measured on the mtg cards at 488x680
    # yuv420 (2048 sample, av1 still speed 6): crf30 gives ssim2 p10 65.3,
    # min 58.6, drift p10 0.967; crf35 gives p50 62.7, p10 55.7, min 45.3,
    # drift p10 0.965. the earlier floors (p10 85, min 72, drift 0.98) were
    # set for large photos and no rung of the ladder reached them, so the
    # gate always fell back to the smallest crf with a warning. with these
    # defaults crf30 passes and crf35 fails on p10: a default that picks a
    # rung. a spec overrides each floor under [media.quality].
    visual_floor_p10: float = 60.0
    visual_floor_min: float = 45.0
    drift_floor_p10: float = 0.95  # negative = the drift leg is disabled
    gate_model: str = ""
    crf_ladder: list[int] = field(default_factory=lambda: [25, 30, 35, 40, 45, 50])
    # task-utility floor (RFC-2b): text-to-image hit@1 of one query per
    # sampled item against the decoded frames of the sample, measured by
    # the gate model's text tower. negative = disabled. a rung passes when
    # hit1_decoded >= max(utility_floor_hit1, hit1_source - utility_tol).
    utility_floor_hit1: float = -1.0  # absolute floor in [0, 1]
    utility_queries: int = 0  # queries drawn from the sample; 0 = every sampled item
    utility_query_template: str = "{label}"  # rendered per item; must keep {label}
    utility_tol: float = 0.0  # allowed hit@1 loss against the lossless source


@dataclass
class ClusterSpec:
    space: str = ""
    threshold: float = 0.92


@dataclass
class JxlTranscodeSpec:
    on_unsupported_jpeg: str = "copy-source"  # error | copy-source | lossless-jxl
    verify_roundtrip: bool = True
    keep_metadata: bool = False


@dataclass
class MediaSpec:
    profile: str = ""  # "" | a MEDIA_PROFILES name (media_profiles.py)
    backend: str = "av1"  # av1 | avif | jxl | jxl-transcode | control
    width: int = 1024
    crf: int | str = 35  # int | "auto"; the avif backend maps it to avifenc -q
    # still | default. svt-av1's still-picture tune, probed against the local
    # encoder (image_encode.probe_tune_still). measured 2026-09-12 on 2048
    # cards at crf 35: ssimulacra2 p50 62.7 against 51.8 for the default
    # tune, for +10% bytes. every av1 profile already set it; the bare
    # default follows. asdict(media) enters the media state key, so an av1
    # build that relied on the implicit default re-encodes once on --resume.
    tune: str = "still"
    speed: int = 8
    fps: int = 1
    pix_fmt: str = "yuv420p"
    shard_size: int = 2048
    gop: str = "auto"  # auto | intra | inter
    order: str = "none"  # none | similarity | cluster
    dedup: bool = True
    quality: QualitySpec = field(default_factory=QualitySpec)
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    jxl_transcode: JxlTranscodeSpec = field(default_factory=JxlTranscodeSpec)


@dataclass
class ModelSpec:
    preset: str = ""
    text: str = "none"  # default | space | none
    image: str = "none"  # space | none
    dims: list[int] = field(default_factory=list)
    model_path: str = ""
    device: str = ""
    batch_size: int = 32
    dtype: str = ""
    space_dtype: str = "int8"
    text_corpus_mode: str = ""  # "" = preset default
    text_query_mode: str = ""
    image_mode: str = ""
    image_prompt: str = ""
    normalize: bool = True
    preprocess_version: str = ""
    image_max_side: int = 0  # 0 = preset default
    encode_kwargs: dict = field(default_factory=dict)


@dataclass
class EngineBuildSpec:
    preset: str = "hybrid"
    dtype: str = ""
    with_graph: bool = True
    graph_top_m: int = 8
    graph_space: str = "default"
    mrl_dim: int = 0


@dataclass
class OutputSpec:
    mode: str = "single"  # single | per-model | both
    dir: str = "out"
    provenance: str = "standard"  # minimal | standard | full
    allow_remote_code: list[str] = field(default_factory=list)
    # inline the media bytes into the .urna (0x17): one self-contained file,
    # no sidecar needed at read time. the media dir remains as build cache.
    embed_media: bool = False
    # root of the shared, content-addressed embed cache (embed/<preset>/<triad>.npz
    # and models/ probes). "" = URNA_CACHE_DIR, else ${XDG_CACHE_HOME:-~/.cache}/urna.
    # `.forge-state/` and `.tmp/` stay in `dir` (transactional, same filesystem).
    cache_dir: str = ""


@dataclass
class CorpusSpec:
    name: str = ""
    title: str = ""
    version: str = "0.1.0"
    chunker_version: str = ""
    reproducible: bool = True
    source: SourceSpec = field(default_factory=SourceSpec)
    image_input: ImageInputSpec = field(default_factory=ImageInputSpec)
    media: MediaSpec | None = None
    models: list[ModelSpec] = field(default_factory=list)
    build: EngineBuildSpec = field(default_factory=EngineBuildSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    spec_path: str = ""

    def image_input_mode(self) -> str:
        if self.image_input.mode:
            return self.image_input.mode
        return "decoded_media" if self.media is not None else "source"


MAX_SPACES = 15  # SPACE_BAND_LEN - 1


def load_spec(path: str | Path) -> CorpusSpec:
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".toml":
        import tomllib

        data = tomllib.loads(raw.decode())
    elif path.suffix == ".json":
        data = json.loads(raw)
    elif path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise SpecError("yaml specs need pyyaml: pip install pyyaml (or use .toml)") from e
        data = yaml.safe_load(raw)
    else:
        raise SpecError(f"unsupported spec extension '{path.suffix}' (use .toml or .json)")
    # ${VAR} (strict) then ~/ over every string: specs stay machine-portable.
    return _parse(expand_paths(data), str(path))


_KNOWN_TABLES = frozenset({"corpus", "source", "media", "models", "embedding", "build", "output"})


def _parse(data: dict, spec_path: str) -> CorpusSpec:
    unknown = sorted(set(data) - _KNOWN_TABLES)
    if unknown:
        raise SpecError(
            f"unknown top-level table(s) {unknown}: valid tables are {sorted(_KNOWN_TABLES)}"
        )
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list) or not all(isinstance(m, dict) for m in raw_models):
        raise SpecError("models must be an array of tables; write [[models]], not [models]")
    corpus = dict(data.get("corpus", {}))
    src = dict(data.get("source", {}))
    joins = [_section(SourceJoin, dict(j), "source.joins") for j in src.pop("joins", [])]
    text = _section(SourceText, dict(src.pop("text", {})), "source.text")
    image = _section(SourceImage, dict(src.pop("image", {})), "source.image")
    order_by = src.pop("order_by", [])
    if isinstance(order_by, str):
        order_by = [order_by]
    source = _section(SourceSpec, {**src, "order_by": order_by}, "source")
    source.joins, source.text, source.image = joins, text, image

    media = None
    if "media" in data:
        m = dict(data["media"])
        profile = m.pop("profile", "")
        if profile:
            if profile not in MEDIA_PROFILES:
                raise SpecError(
                    f"media.profile: unknown '{profile}' (valid: {sorted(MEDIA_PROFILES)})"
                )
            m = {**merge_profile(MEDIA_PROFILES[profile], m), "profile": profile}
        quality = _section(QualitySpec, dict(m.pop("quality", {})), "media.quality")
        cluster = _section(ClusterSpec, dict(m.pop("cluster", {})), "media.cluster")
        jxl = _section(JxlTranscodeSpec, dict(m.pop("jxl_transcode", {})), "media.jxl_transcode")
        media = _section(MediaSpec, m, "media")
        media.quality, media.cluster, media.jxl_transcode = quality, cluster, jxl

    models = [_section(ModelSpec, dict(m), "models") for m in raw_models]
    image_input = _section(
        ImageInputSpec,
        dict(data.get("embedding", {}).get("image_input", {})),
        "embedding.image_input",
    )
    build = _section(EngineBuildSpec, dict(data.get("build", {})), "build")
    output = _section(OutputSpec, dict(data.get("output", {})), "output")
    spec = _section(CorpusSpec, corpus, "corpus")
    spec.source, spec.image_input, spec.media = source, image_input, media
    spec.models, spec.build, spec.output, spec.spec_path = models, build, output, spec_path
    return spec


# re-exported rules: callers import the whole contract from build_spec.
# spec_rules imports SpecError from here, so it loads last; spec_paths
# resolves SpecError lazily and stays importable on its own.
from forge.spec_paths import expand_paths  # noqa: E402
from forge.spec_rules import default_model, emitted_spaces, validate  # noqa: E402

__all__ = [
    "CorpusSpec",
    "SourceSpec",
    "MediaSpec",
    "ModelSpec",
    "SpecError",
    "MAX_SPACES",
    "MEDIA_PROFILES",
    "load_spec",
    "default_model",
    "emitted_spaces",
    "validate",
]
