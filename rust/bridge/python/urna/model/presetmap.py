"""Named embedding-model presets: models are data, not per-project code.

A preset declares what a model IS (ids, dims, validated MRL ladder, modalities),
what it NEEDS (imports with the exact pip fix line), and how it is USED by
default (the asymmetric query/document contract). The build spec may override
the usage fields; everything that changes an embedding output belongs to the
embedding_recipe_hash (RFC-0 N2), while `model_hash` identifies the model
alone.

MRL: `mrl.dims` is the ladder the model card validates. Slicing at any other
dim is mathematically possible and semantically unsupported, so it is refused
- a generic "slice anywhere" would be false advertising (RFC-3).

remote code: a preset with `trust_remote_code` loads ONLY when the caller
passes its name in `allow_remote_code` AND every code file matches the pinned
`remote_code_hashes` allowlist; the opt-in is the consent, the pin the identity.

hub snapshot: an open_clip preset with `hf_repo` and `revision` loads from
`snapshots/<revision>` of the hf cache, every file in `snapshot_files` present,
never through `refs/main`; a missing file is a typed SnapshotMissing naming it
and the fetch line (URNA_ALLOW_DOWNLOAD=1 fetches the pinned files once).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# offline by default, the guarantee the forge package gave every registry
# import before the move: no hub round-trip unless the caller opts in. set
# here because every model load goes through the registry, and importing
# urna.embed no longer loads potiontab (which sets the same flags).
# tokenizers parallelism stays off, as potiontab sets it, whatever the opt-in.
if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from urna.model.installed import CODE_PINS, INSTALLS, InstallSpec  # noqa: E402, F401

if TYPE_CHECKING:
    import numpy as np


def refuse_connections() -> None:
    """Make every network connection from this process fail, unless the
    caller opted in with URNA_ALLOW_DOWNLOAD=1. the hub's offline flags only
    hold the hub client; reviewed model code can still open a socket of its
    own (jina v5 omni fetches any http(s) string it is given as media), and
    a query must never leave the machine. local sockets stay usable."""
    if os.environ.get("URNA_ALLOW_DOWNLOAD") == "1":
        return
    import socket

    net = {socket.AF_INET, socket.AF_INET6}

    def refuse(*_args, **_kwargs):
        raise OSError("urna: the embedder runs offline; URNA_ALLOW_DOWNLOAD=1 allows downloads")

    def connect(self, address, _orig=socket.socket.connect):
        if self.family in net:
            refuse()
        return _orig(self, address)

    def connect_ex(self, address, _orig=socket.socket.connect_ex):
        if self.family in net:
            refuse()
        return _orig(self, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = refuse
    socket.getaddrinfo = refuse


class RegistryError(ValueError):
    """Typed registry failure: unknown preset, missing dep, refused load."""


class CapabilityError(RuntimeError):
    """The preset does not support the requested modality."""


class SnapshotMissing(FileNotFoundError):
    """A file of a preset's pinned hub snapshot is not on disk."""

    def __init__(self, message: str, repo: str):
        super().__init__(message)
        self.repo = repo


@dataclass(frozen=True)
class MrlSpec:
    supported: bool = False
    dims: tuple[int, ...] = ()
    method: str = "prefix_slice_l2"


@dataclass(frozen=True)
class ModelPreset:
    name: str
    kind: str  # "potion" | "open_clip" | "st_multimodal" | "st_text" | "fake"
    embedding_model: str  # manifest name; unique across presets (reverse lookup)
    model_id: str = ""
    pretrained: str | None = None
    default_dim: int = 0  # 0 = probe at load
    mrl: MrlSpec = field(default_factory=MrlSpec)
    modalities: frozenset[str] = frozenset({"text"})
    requires: tuple[tuple[str, str], ...] = ()  # (import_name, pip fix line)
    trust_remote_code: bool = False
    remote_code_hashes: tuple[tuple[str, str], ...] = ()  # (filename, sha256)
    local_dir: str | None = None
    executable: bool = True  # False: registered but refused without allow_heavy
    # asymmetric usage contract defaults (spec may override; all in recipe_hash)
    text_corpus_mode: str = "plain"  # "plain" | "document" | "query"
    text_query_mode: str = "plain"
    image_mode: str = "image_document"
    image_prompt: str = "Represent this image."
    normalize: bool = True
    preprocess_version: str = "processor-native-v1"
    image_max_side: int = 0  # 0 = model-native; recipe-hashed preprocessing
    # how an image document is handed to ST encode(): "dict" = {"image", "text": prompt}
    # (wemm's contract); "bare" = the PIL image alone (jina: the dict form collapses
    # its image embeddings onto the shared prompt text - measured, sims 0.99 across
    # different images vs 0.45 bare). recipe-hashed.
    image_doc_format: str = "dict"
    encode_kwargs: tuple[tuple[str, object], ...] = ()  # ST encode() extras (recipe-hashed)
    # the prompt a query is encoded with when the model names it otherwise
    # than `query` (encode_query only finds that name); "" = the model's own.
    query_prompt_name: str = ""
    # the dtype the model loads in on mps when float16 fails it; "" = float16.
    # part of the model_hash (the dtype policy), like every dtype.
    mps_dtype: str = ""
    install: InstallSpec | None = None  # validated install; None = not offered by setup
    license: str = ""  # the license the hub card declares, as setup shows it
    # the hub snapshot an open_clip preset loads from: repo, pinned commit, the
    # checkpoint file and every file the load and the tokenizer read.
    hf_repo: str = ""
    revision: str = ""
    weights_file: str = ""
    snapshot_files: tuple[str, ...] = ()


_ST_REQUIRES = (
    ("torch", "pip install torch"),
    ("sentence_transformers", 'pip install "sentence-transformers>=5.7,<6"'),
    ("transformers", 'pip install "transformers==5.2.0"'),
)
_WEMM_REQUIRES = _ST_REQUIRES + (("qwen_vl_utils", 'pip install "qwen-vl-utils==0.0.14"'),)
# jina v5 loads its four task adapters (LoRA) through peft.
_JINA_REQUIRES = _ST_REQUIRES + (("peft", "pip install peft"),)

_OPEN_CLIP_REQUIRES = (
    ("torch", "pip install torch"),
    ("open_clip", "pip install open_clip_torch"),
    ("PIL", "pip install pillow"),
)


def _st_text(name: str, repo: str, dim: int, license: str, **kw) -> ModelPreset:
    """A text model with its own query and document prompts, run by the
    sentence-transformers backend (`stbackend`), which encodes a query and a
    document through their own routes."""
    kw.setdefault("requires", _ST_REQUIRES)
    return ModelPreset(
        name=name,
        kind="st_multimodal",
        embedding_model=repo,
        model_id=repo,
        default_dim=dim,
        text_corpus_mode="document",
        text_query_mode="query",
        license=license,
        **kw,
    )


PRESETS: dict[str, ModelPreset] = {
    p.name: p
    for p in (
        ModelPreset(
            name="potion",
            kind="potion",
            embedding_model="minishlab/potion-base-8M/v1",
            model_id="minishlab/potion-base-8M",
            default_dim=256,
            requires=(("numpy", "pip install numpy"), ("tokenizers", "pip install tokenizers")),
            license="MIT",
        ),
        ModelPreset(
            name="minilm-multilingual",
            kind="st_text",
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            default_dim=384,
            requires=(("sentence_transformers", 'pip install "sentence-transformers"'),),
            license="Apache-2.0",
            # the revision and files the pt-br corpora were built from (the
            # files sentence-transformers itself fetches; the repo also holds
            # onnx, openvino and a pytorch_model.bin that are never fetched).
            install=INSTALLS["minilm-multilingual"],
        ),
        _st_text(
            "harrier-270m",
            "microsoft/harrier-oss-v1-270m",
            640,
            "MIT",
            query_prompt_name="web_search_query",
            # gemma3's activations overflow float16: every vector is NaN.
            mps_dtype="bfloat16",
            install=INSTALLS["harrier-270m"],
        ),
        _st_text(
            "jina-v5-text-nano",
            "jinaai/jina-embeddings-v5-text-nano",
            768,
            "CC-BY-NC-4.0",
            mrl=MrlSpec(True, (32, 64, 128, 256, 512, 768)),
            requires=_JINA_REQUIRES,
            trust_remote_code=True,
            remote_code_hashes=CODE_PINS["jina-v5-text-nano"],
            encode_kwargs=(("task", "retrieval"),),
            install=INSTALLS["jina-v5-text-nano"],
        ),
        _st_text(
            "qwen3-embedding-0.6b",
            "Qwen/Qwen3-Embedding-0.6B",
            1024,
            "Apache-2.0",
            install=INSTALLS["qwen3-embedding-0.6b"],
        ),
        _st_text(
            "octen-embedding-0.6b",
            "Octen/Octen-Embedding-0.6B",
            1024,
            "Apache-2.0",
            install=INSTALLS["octen-embedding-0.6b"],
        ),
        ModelPreset(
            name="clip-vit-b32",
            kind="open_clip",
            embedding_model="open_clip/ViT-B-32/openai",
            model_id="ViT-B-32",
            pretrained="openai",
            default_dim=512,
            modalities=frozenset({"text", "image"}),
            requires=_OPEN_CLIP_REQUIRES,
            license="Apache-2.0",
            # the hub copy of the openai weights open_clip resolves the tag to
            hf_repo="timm/vit_base_patch32_clip_224.openai",
            revision="a6f597a30f7b82c51704746581f9a4e41421e878",
            weights_file="open_clip_model.safetensors",
            snapshot_files=(
                "open_clip_model.safetensors",
                "open_clip_config.json",
            ),
            install=INSTALLS["clip-vit-b32"],
        ),
        ModelPreset(
            name="siglip2",
            kind="open_clip",
            embedding_model="open_clip/ViT-B-16-SigLIP2/webli",
            model_id="ViT-B-16-SigLIP2",
            pretrained="webli",
            default_dim=768,
            modalities=frozenset({"text", "image"}),
            # its text tower tokenizes through transformers' AutoTokenizer
            requires=_OPEN_CLIP_REQUIRES + (("transformers", 'pip install "transformers==5.2.0"'),),
            # the revision whose weights fingerprint to the model_hash of the
            # mtg stills-5models build (sha256:a9946db1...), checked 2026-10-02
            hf_repo="timm/ViT-B-16-SigLIP2",
            revision="eee10eff6dd8cabae2d7f379d4e8cfcd352030aa",
            weights_file="open_clip_model.safetensors",
            snapshot_files=(
                "open_clip_model.safetensors",
                "open_clip_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ),
            license="Apache-2.0",
            install=INSTALLS["siglip2"],
        ),
        ModelPreset(
            name="jina-v5-omni-nano",
            kind="st_multimodal",
            embedding_model="jinaai/jina-embeddings-v5-omni-nano",
            model_id="jinaai/jina-embeddings-v5-omni-nano",
            mrl=MrlSpec(True, (32, 64, 128, 256, 512, 768)),
            modalities=frozenset({"text", "image", "video"}),
            requires=_JINA_REQUIRES,
            trust_remote_code=True,
            remote_code_hashes=CODE_PINS["jina-v5-omni-nano"],
            text_corpus_mode="document",
            text_query_mode="query",
            image_doc_format="bare",
            encode_kwargs=(("task", "retrieval"),),
            license="CC-BY-NC-4.0",
            install=INSTALLS["jina-v5-omni-nano"],
        ),
        ModelPreset(
            name="jina-v5-omni-small",
            kind="st_multimodal",
            embedding_model="jinaai/jina-embeddings-v5-omni-small",
            model_id="jinaai/jina-embeddings-v5-omni-small",
            mrl=MrlSpec(True, (32, 64, 128, 256, 512, 768, 1024)),
            modalities=frozenset({"text", "image", "video"}),
            requires=_JINA_REQUIRES,
            trust_remote_code=True,
            remote_code_hashes=CODE_PINS["jina-v5-omni-small"],
            text_corpus_mode="document",
            text_query_mode="query",
            image_doc_format="bare",
            encode_kwargs=(("task", "retrieval"),),
            license="CC-BY-NC-4.0",
            install=INSTALLS["jina-v5-omni-small"],
        ),
        # the multilingual MiniLM the pt-br corpora were built with, before
        # the registry existed. kind st_text loads it through rust/bridge/python/urna/
        # searchtxt.py's own functions, so a corpus built then keeps its
        # model_hash. no transformers pin: the st pin above is wemm's.
        ModelPreset(
            name="wemm-2b",
            kind="st_multimodal",
            embedding_model="tencent/WeMM-Embedding-2B",
            model_id="tencent/WeMM-Embedding-2B",
            default_dim=2048,
            mrl=MrlSpec(True, (128, 256, 512, 1024, 2048)),
            modalities=frozenset({"text", "image", "video"}),
            requires=_WEMM_REQUIRES,
            trust_remote_code=True,
            remote_code_hashes=CODE_PINS["wemm-2b"],
            text_corpus_mode="document",
            text_query_mode="query",
            image_max_side=768,
            license="Apache-2.0",
            install=INSTALLS["wemm-2b"],
        ),
        ModelPreset(
            name="fake-test",
            kind="fake",
            embedding_model="urna-forge-fake-test/v1",
            default_dim=8,
            mrl=MrlSpec(True, (8, 4)),
            modalities=frozenset({"text", "image"}),
        ),
    )
}


def get_preset(name: str) -> ModelPreset:
    if name == "fake-test" and os.environ.get("URNA_ENABLE_FAKE_PRESET") != "1":
        raise RegistryError("preset 'fake-test' requires URNA_ENABLE_FAKE_PRESET=1 (test-only)")
    preset = PRESETS.get(name)
    if preset is None:
        valid = ", ".join(sorted(PRESETS))
        raise RegistryError(f"unknown model preset '{name}'. valid presets: {valid}")
    return preset


def preset_for_embedding_model(embedding_model: str) -> ModelPreset | None:
    """Reverse lookup for the query embedder: manifest name -> preset."""
    for preset in PRESETS.values():
        if preset.embedding_model == embedding_model:
            return preset
    return None


def check_deps(preset: ModelPreset) -> None:
    """Raise RegistryError naming the exact pip line for the first missing dep."""
    import importlib.util

    for module, fix in preset.requires:
        if importlib.util.find_spec(module) is None:
            raise RegistryError(
                f"preset '{preset.name}' needs the '{module}' package. install with: {fix}"
            )


def resolve_model_dir(preset: ModelPreset, model_path: str | os.PathLike | None = None):
    """explicit > URNA_MODEL_DIR_<NAME> env > preset.local_dir > hf cache > None."""
    if model_path:
        return Path(model_path)
    env_key = "URNA_MODEL_DIR_" + preset.name.upper().replace("-", "_")
    if os.environ.get(env_key):
        return Path(os.environ[env_key])
    if preset.local_dir:
        local = Path(preset.local_dir).expanduser()
        if local.is_dir():
            return local
    if preset.kind == "st_multimodal":
        from urna.model.modelhash import hf_cache_snapshot

        try:
            return hf_cache_snapshot(preset.model_id)
        except (FileNotFoundError, ValueError, RuntimeError):
            return None
    return None


def pinned_snapshot(preset: ModelPreset, model_path: str | os.PathLike | None = None) -> Path:
    """The directory holding the preset's pinned hub snapshot, every file present.

    explicit model_path > URNA_MODEL_DIR_<NAME> > `snapshots/<revision>` in the
    hf cache. refs/main is never read: it names whatever was fetched last, not
    the revision the preset verified. A missing file raises SnapshotMissing,
    unless URNA_ALLOW_DOWNLOAD=1, which fetches the pinned files first.
    """
    env_key = "URNA_MODEL_DIR_" + preset.name.upper().replace("-", "_")
    explicit = model_path or os.environ.get(env_key)
    if explicit:
        snap = Path(explicit)
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache = hf_home / "hub" / f"models--{preset.hf_repo.replace('/', '--')}"
        snap = cache / "snapshots" / preset.revision
    missing = [f for f in preset.snapshot_files if not (snap / f).is_file()]
    if missing and not explicit and os.environ.get("URNA_ALLOW_DOWNLOAD") == "1":
        snap = _fetch_snapshot(preset)
        missing = [f for f in preset.snapshot_files if not (snap / f).is_file()]
    if missing:
        files = " ".join(preset.snapshot_files)
        fetch = f"hf download {preset.hf_repo} {files} --revision {preset.revision}"
        where = str(snap) if explicit else f"the hf cache ({preset.hf_repo}@{preset.revision[:12]})"
        raise SnapshotMissing(
            f"preset '{preset.name}': {', '.join(missing)} missing from {where}. fetch the pinned "
            f"files with: {fetch}, or run once with URNA_ALLOW_DOWNLOAD=1",
            preset.hf_repo,
        )
    return snap


def _fetch_snapshot(preset: ModelPreset) -> Path:
    """Download the preset's pinned files: the explicit URNA_ALLOW_DOWNLOAD=1 opt-in."""
    # importing potiontab defaults the hub offline; the opt-in wins,
    # as in stbackend. huggingface_hub reads the flag when it is first imported.
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(k, None)
    from huggingface_hub import hf_hub_download

    path = None
    for f in preset.snapshot_files:
        path = hf_hub_download(preset.hf_repo, f, revision=preset.revision)
    return Path(path).parent


def verify_remote_code(preset: ModelPreset, model_dir: Path) -> None:
    """Pin gate (N11): every allowlisted code file must match its sha256."""
    for filename, expected in preset.remote_code_hashes:
        path = model_dir / filename
        if not path.is_file():
            raise RegistryError(f"preset '{preset.name}': pinned code file missing: {filename}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RegistryError(
                f"preset '{preset.name}': {filename} sha256 {actual} does not match the "
                f"pinned allowlist ({expected}). refusing to execute unreviewed remote code."
            )


def slice_renorm(vecs: np.ndarray, dim: int) -> np.ndarray:
    """First-`dim` prefix slice + re-L2 (same semantics as build_fn's mrl_dim).
    numpy loads here, not at import: the catalog generator reads the
    registry under a bare python3 (the release's stage step)."""
    import numpy as np

    v = np.asarray(vecs, dtype=np.float32)
    if not 0 < dim <= v.shape[1]:
        raise ValueError(f"slice dim {dim} out of range for source dim {v.shape[1]}")
    sliced = np.ascontiguousarray(v[:, :dim])
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    return (sliced / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def create_embedder(
    name: str,
    *,
    model_path: str | os.PathLike | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    allow_remote_code: frozenset[str] | set[str] = frozenset(),
    allow_heavy: bool = False,
    usage: dict | None = None,
):
    """Return a unified adapter (embed_texts/embed_paths/embed_arrays, dim,
    model_hash, fingerprint()) for the preset, enforcing the RFC-0 gates."""
    preset = get_preset(name)
    if not preset.executable and not allow_heavy:
        raise RegistryError(
            f"preset '{name}' is registered but flagged too heavy for this machine; "
            "pass --allow-heavy to load it anyway"
        )
    if preset.trust_remote_code and name not in allow_remote_code:
        raise RegistryError(
            f"preset '{name}' executes model-repo code (trust_remote_code). opt in "
            f'explicitly with allow_remote_code = ["{name}"] in the spec'
        )
    check_deps(preset)
    from urna.model import embedders as _adapters

    if preset.kind == "fake":
        return _adapters._FakeAdapter(preset)
    if preset.kind == "potion":
        from urna.embed import potiontab

        return _adapters._PotionAdapter(preset, potiontab.PotionEmbedder())
    if preset.kind == "open_clip":
        from urna.embed import visionemb

        snapshot = pinned_snapshot(preset, model_path) if preset.revision else None
        inner = visionemb.ImageEmbedder(
            model_id=preset.model_id,
            pretrained=preset.pretrained,
            device=device,
            batch_size=batch_size or 32,
            snapshot=snapshot,
            weights=preset.weights_file or None,
        )
        return _adapters._OpenClipAdapter(preset, inner)
    if preset.kind == "st_text":
        return _adapters._STTextAdapter(preset, model_path=model_path)
    if preset.kind == "st_multimodal":
        model_dir = resolve_model_dir(preset, model_path)
        if model_dir is not None and preset.remote_code_hashes:
            verify_remote_code(preset, model_dir)
        return _adapters._SubprocessSTAdapter(
            preset,
            model_dir=model_dir,
            model_path=model_path,
            device=device,
            batch_size=batch_size or 8,
            usage=usage or {},
        )
    raise RegistryError(f"preset '{name}': unknown kind '{preset.kind}'")
