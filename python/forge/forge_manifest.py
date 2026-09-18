"""Manifest v1 (RFC-0 N9), build.lock.json (N1) and provenance redaction (N10).

The manifest is a versioned contract, not an ad-hoc log: required fields are
asserted at write time, serialization is canonical (sort_keys), and readers
must check `manifest_schema_version` first. The build lock pins everything an
L3 (byte-identical) claim depends on; `--rebuild-only` compares against it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from forge.forge_cache import atomic_write_bytes

MANIFEST_SCHEMA_VERSION = 1
_REQUIRED = (
    "manifest_schema_version",
    "name",
    "n_items",
    "n_unique_frames",
    "models",
    "spaces",
    "media",
    "provenance_mode",
    "timings",
)


def redact_path(value: str, mode: str, spec_dir: Path) -> str:
    if mode == "full":
        return value
    home = str(Path.home())
    out = value.replace(home, "~") if value.startswith(home) else value
    if mode == "minimal":
        return Path(out).name
    try:
        return str(Path(out).relative_to(spec_dir))
    except ValueError:
        return out


# what the compact items[] of a minimal manifest leaves out: the two
# redacted source fields and the media uri (derived, see item_frame).
COMPACT_DROPPED = ("image_path", "label", "media_uri")


def write_manifest(path: Path, payload: dict, mode: str, spec_dir: Path) -> None:
    payload = dict(payload, manifest_schema_version=MANIFEST_SCHEMA_VERSION, provenance_mode=mode)
    frame_of_row = payload.pop("frame_of_row", None)
    payload["items_compact"] = mode == "minimal"
    if mode == "minimal":
        # 38627 items with image_path, key, label, media_uri and ordinal
        # were 12 MB of a 13 to 16 MB file. key + ordinal is the identity
        # a reader needs; the media frame is derived (item_frame), the
        # source path and the label are what minimal redacts anyway.
        payload.pop("sql", None)
        payload["items"] = [
            {"key": it["key"], "ordinal": it["ordinal"]} for it in payload.get("items", [])
        ]
        if frame_of_row and any(f != i for i, f in enumerate(frame_of_row)):
            # dedup collapsed rows: the row -> unique frame map is not the
            # identity and nothing else in the manifest records it.
            payload["frame_of_row"] = list(frame_of_row)
    missing = [k for k in _REQUIRED if k not in payload]
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    atomic_write_bytes(path, json.dumps(payload, sort_keys=True, indent=1).encode())


def manifest_items(manifest: dict, need: tuple[str, ...] = ()) -> list[dict]:
    """items[] of a forge manifest. `need` names the per-item fields the
    caller reads; a compact manifest (provenance = "minimal") carries key
    and ordinal only, and asking for a dropped field is a clear error naming
    the provenance mode that records it, never a KeyError deep in a loop."""
    items = manifest.get("items") or []
    if manifest.get("items_compact"):
        dropped = [f for f in need if f in COMPACT_DROPPED]
        if dropped:
            raise ValueError(
                f'manifest items are compact (provenance = "minimal"): '
                f"{', '.join(dropped)} not recorded per item; rebuild with "
                f'output.provenance = "standard" (or "full") to read it'
            )
    return items


_FRAME_RE = re.compile(r"^media://([^#]*)#frame=(\d+)$")


def frame_resolver(manifest: dict):
    """item -> global stream frame (the index media_resolver-style readers
    map to a segment and a shard-local frame). the dedup map (frame_of_row,
    identity when absent) and the similarity/cluster permutation of the
    media block are enough for a compact manifest; a full item whose
    media_uri names a segment keeps that as the authority (older manifests
    with a collapsed dedup map carry no frame_of_row)."""
    media = manifest.get("media") or {}
    starts = {s.get("uri"): s.get("start_frame", 0) for s in media.get("segments") or []}
    perm = media.get("order_permutation")
    stream_pos = {item: pos for pos, item in enumerate(perm)} if perm else None
    frame_of_row = manifest.get("frame_of_row")

    def resolve(item: dict) -> int:
        m = _FRAME_RE.match(item.get("media_uri") or "")
        if m:
            return starts.get(m.group(1), 0) + int(m.group(2))
        ordinal = int(item["ordinal"])
        frame = frame_of_row[ordinal] if frame_of_row else ordinal
        return stream_pos[frame] if stream_pos else frame

    return resolve


def _tool_fingerprint(name: str) -> dict | None:
    exe = shutil.which(name)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
        text = out.stdout or out.stderr
        version = text.splitlines()[0].strip() if text else ""
    except (subprocess.SubprocessError, OSError):
        version = ""
    return {
        "path": exe,
        "version": version,
        "sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
    }


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    versions = {"python": sys.version.split()[0]}
    import contextlib

    for pkg in (
        "numpy",
        "torch",
        "transformers",
        "sentence-transformers",
        "open-clip-torch",
        "tokenizers",
        "pillow",
    ):
        with contextlib.suppress(md.PackageNotFoundError):
            versions[pkg] = md.version(pkg)
    return versions


def build_lock(spec, model_hashes: dict[str, str], device: str) -> dict:
    return {
        "lock_schema_version": 1,
        "platform": {
            "os": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "packages": _package_versions(),
        "tools": {
            name: _tool_fingerprint(name)
            for name in ("ffmpeg", "ffprobe", "cjxl", "djxl", "ssimulacra2")
        },
        "models": model_hashes,
        "device": device,
        "resolved_spec": _spec_dict(spec),
    }


def _spec_dict(spec) -> dict:
    raw = asdict(spec)
    raw["media"] = asdict(spec.media) if spec.media is not None else None
    # where the embed cache lives is not identity: the same entries are read
    # whether the root came from the spec, --cache-dir or URNA_CACHE_DIR, and
    # a machine path does not belong in the L3 record.
    raw["output"].pop("cache_dir", None)
    return raw


def check_lock(previous: dict, current: dict) -> list[str]:
    """Return human-readable divergences between two build locks (L3 gate)."""
    diffs: list[str] = []

    def walk(prefix: str, a, b) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(f"{prefix}.{k}" if prefix else k, a.get(k), b.get(k))
        elif a != b:
            diffs.append(f"{prefix}: {_short(a)} -> {_short(b)}")

    walk("", previous, current)
    # where the spec, the outputs and the inputs live is not a byte-affecting
    # parameter (row content is guarded per item by input_hash), so a rebuild
    # under another data root (${VAR} re-exported, another home) stays L3.
    return [d for d in diffs if not _LOCATION_KEYS.match(d)]


_LOCATION_KEYS = re.compile(
    r"resolved_spec\.(spec_path|output\.(dir|cache_dir)|source\.(path|db|image\.path_template))\b"
)


def _short(v) -> str:
    s = json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= 60 else s[:57] + "..."
