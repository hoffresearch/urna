"""Reproducible model fingerprint for sentence-transformers / HF models.

The corpus' `model_hash` must uniquely identify the model that produced
the embeddings — otherwise `urna search-text` could feed a query
embedded by a *different* model and return cosine-valid garbage.

A naive `sha256(model_dir)` is unstable: it pulls in cache files,
locks, and snapshot directories. Instead this module hashes only the
files that actually affect inference:

  - config.json, config_sentence_transformers.json
  - modules.json, sentence_bert_config.json
  - tokenizer.json, tokenizer_config.json, special_tokens_map.json
  - 1_Pooling/config.json
  - model.safetensors  (or pytorch_model.bin)

The resulting structured fingerprint includes the file hash, tokenizer
hash, pooling config hash, embedding dim, and `normalize_embeddings`
flag. The compact `model_hash` (the single string stored in the
manifest) is `sha256:` + sha256 of the fingerprint serialized as
canonical JSON.

Two corpora built on different machines from the same upstream model
snapshot get the same `model_hash`. Snapshots that differ in tokenizer
or weights produce different hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Force HF/sentence-transformers OFFLINE by default before any (lazy) hub
# access. Opt into a first-time model download with URNA_ALLOW_DOWNLOAD=1.
if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_k, "1")

# Files whose contents affect inference output. Other files in the
# snapshot dir (LICENSE, README.md, .git*) are ignored.
RELEVANT_FILES: tuple[str, ...] = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "1_Pooling/config.json",
    "model.safetensors",
    "pytorch_model.bin",
)

PLACEHOLDER_HASH = "sha256:" + "0" * 64


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json_safe(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass(frozen=True)
class ModelFingerprint:
    """Structured fingerprint of a sentence-transformers / HF model snapshot.

    Stable across machines: only depends on the bytes of the relevant
    files plus a few config-derived facts. Serializable via `to_dict`.
    """

    model_id: str
    files_hash: str  # sha256 over the relevant files (sorted)
    tokenizer_hash: str | None
    pooling_config_hash: str | None
    embedding_dim: int
    normalize_embeddings: bool

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "files_hash": self.files_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "pooling_config_hash": self.pooling_config_hash,
            "embedding_dim": self.embedding_dim,
            "normalize_embeddings": self.normalize_embeddings,
        }


def compute_model_fingerprint(
    model_dir: str | Path,
    *,
    model_id: str | None = None,
) -> ModelFingerprint:
    """Compute a reproducible fingerprint of a model snapshot directory.

    `model_dir` should be the local path of an unpacked HF / sentence-
    transformers model — the same directory `SentenceTransformer` would
    load. Missing files are tolerated (some models don't ship every
    relevant file); `RELEVANT_FILES` order is the canonical order.

    If `model_id` is provided (e.g. the HF identifier the caller used),
    it goes into the fingerprint verbatim. Otherwise we read it from
    `config.json`'s `_name_or_path` (which can be local-path-flavored
    and unstable across machines).
    """
    md = Path(model_dir).resolve()
    if not md.is_dir():
        raise FileNotFoundError(f"model directory not found: {md}")

    files_h = hashlib.sha256()
    for rel in RELEVANT_FILES:
        p = md / rel
        sha = _sha256_file(p)
        if sha is None:
            continue
        files_h.update(rel.encode("utf-8"))
        files_h.update(b"\0")
        files_h.update(sha.encode("utf-8"))
        files_h.update(b"\0")
    files_hash = "sha256:" + files_h.hexdigest()

    cfg = _read_json_safe(md / "config.json")
    st_cfg = _read_json_safe(md / "config_sentence_transformers.json")
    pooling_cfg = _read_json_safe(md / "1_Pooling/config.json")

    resolved_id = model_id or cfg.get("_name_or_path") or md.name
    embedding_dim = int(
        st_cfg.get("hidden_size")
        or cfg.get("hidden_size")
        or cfg.get("dim")
        or pooling_cfg.get("word_embedding_dimension")
        or 0
    )
    normalize = bool(
        st_cfg.get("normalize_embeddings", pooling_cfg.get("pooling_mode_mean_tokens", True))
    )

    tokenizer_sha = _sha256_file(md / "tokenizer.json")
    tokenizer_hash = f"sha256:{tokenizer_sha}" if tokenizer_sha else None

    pooling_sha = _sha256_file(md / "1_Pooling/config.json")
    pooling_hash = f"sha256:{pooling_sha}" if pooling_sha else None

    return ModelFingerprint(
        model_id=str(resolved_id),
        files_hash=files_hash,
        tokenizer_hash=tokenizer_hash,
        pooling_config_hash=pooling_hash,
        embedding_dim=embedding_dim,
        normalize_embeddings=normalize,
    )


def fingerprint_to_model_hash(fp: ModelFingerprint) -> str:
    """Compact `sha256:<hex>` hash of the canonical JSON of `fp`.

    Stored in the manifest as `model_hash`. Reproducible across
    machines because the JCS-style serialization (sort_keys, no
    whitespace) is stable.
    """
    canonical = json.dumps(fp.to_dict(), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_placeholder(model_hash: str) -> bool:
    """True if `model_hash` is the legacy zero-placeholder."""
    return model_hash == PLACEHOLDER_HASH


def hf_cache_snapshot(model_id: str) -> Path | None:
    """Resolve the HF Hub cache snapshot dir for `model_id`, picking the
    revision that `refs/main` points to — the one HF actually loads — rather
    than an arbitrary alphabetical snapshot.

    Returns `None` if the model is not cached, or if the loaded revision
    cannot be determined unambiguously (multiple snapshots and no usable
    `refs/main`). Callers must then fail closed / require an explicit
    `--model-path`, never guess: hashing the wrong snapshot would stamp a
    corpus with a `model_hash` that does not match the model that produced
    its vectors, defeating the honesty gate.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    snap_dir = cache_dir / "snapshots"
    if not snap_dir.is_dir():
        return None
    ref = cache_dir / "refs" / "main"
    if ref.is_file():
        commit = ref.read_text().strip()
        cand = snap_dir / commit
        if cand.is_dir():
            return cand.resolve()
    snaps = [d for d in snap_dir.iterdir() if d.is_dir()]
    if len(snaps) == 1:
        return snaps[0].resolve()
    # zero snapshots, or ambiguous (multiple with no usable ref): do not guess.
    return None


def resolve_model_dir(model_name_or_path: str) -> Path:
    """Resolve a model name (HF id) or local path to its snapshot dir.

    Strategy, in order:
      1. If `model_name_or_path` is already a directory, use it.
      2. Look up the HF Hub cache layout
         (`$HF_HOME/hub/models--<org>--<name>/snapshots/<rev>`). This
         covers sentence-transformers v3+ which only carries the HF id
         in its config, not the local path.
      3. Try to import sentence_transformers and inspect the loaded
         model's `_modules` for an on-disk `name_or_path` (older
         sentence-transformers compatibility).

    Raises `FileNotFoundError` with an actionable message if nothing
    works — the caller should pass `--model-path` explicitly.
    """
    p = Path(model_name_or_path).expanduser()
    if p.is_dir():
        return p.resolve()

    # hF Hub cache: standard layout, resolving the revision refs/main points
    # to (not an arbitrary alphabetical snapshot). Finds the model regardless
    # of whether sentence-transformers is importable.
    snap = hf_cache_snapshot(model_name_or_path)
    if snap is not None:
        return snap

    # older sentence-transformers (v2.x) put the snapshot path in the
    # autoModel config. Importing is heavy; only do it as a last resort.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise FileNotFoundError(
            f"could not resolve model {model_name_or_path!r} to a local directory; "
            f"pass --model-path explicitly. (sentence_transformers not importable: {e})"
        ) from e
    model = SentenceTransformer(model_name_or_path)
    for module in model._modules.values():
        auto_model = getattr(module, "auto_model", None)
        if auto_model is None:
            continue
        cfg = getattr(auto_model, "config", None)
        nop = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        if nop and Path(nop).is_dir():
            return Path(nop).resolve()
    raise FileNotFoundError(
        f"could not resolve model {model_name_or_path!r} to a local directory; "
        f"pass --model-path explicitly."
    )
