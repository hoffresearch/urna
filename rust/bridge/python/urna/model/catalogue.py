"""The model catalog `urna setup` and the explorer offer, generated from the
registry: every preset whose install was validated end to end (a pinned hub
revision, the exact files fetched at it and the `model_hash` they
fingerprint to) and whose packages do not contradict another offered
model's. every other preset is listed under `excluded` with the reason, so
the installer can say why a model is not offered instead of hiding it.

the catalog ships in the payload as `urna/model/catalogue.json`; no remote index.
`rust/bridge/python/urna/model/catalogue.json` is this module's output, checked in:
`tool/tasks/embedpack.py` and `tool/tests/test_embedpack.py` refuse a stale copy.

usage:  python rust/bridge/python/urna/model/catalogue.py            print the catalog
        python rust/bridge/python/urna/model/catalogue.py --write    rewrite catalogue.json
        python rust/bridge/python/urna/model/catalogue.py --check    exit 1 when stale
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# run as a file (rust/bridge/python/ or the payload python/), the folder that
# holds urna/ goes first; imported as a module, the package is already there.
if not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from urna.model.presetmap import PRESETS, ModelPreset  # noqa: E402

SCHEMA = 1
CATALOG = Path(__file__).resolve().with_name("catalogue.json")


def exclusion(preset: ModelPreset) -> str | None:
    """Why a preset is not offered, or None when it is installable."""
    if preset.kind == "potion":
        return "ships inside the payload; nothing to install"
    if not preset.executable:
        return "flagged too heavy for a default machine"
    if preset.trust_remote_code and not preset.remote_code_hashes:
        return "runs model-repo code that has not been reviewed and pinned"
    if preset.install is not None:
        return None
    if preset.kind == "open_clip" and preset.revision:
        return (
            "pinned hub snapshot, fetched on first use with URNA_ALLOW_DOWNLOAD=1; the "
            "installer cannot verify an open_clip model_hash yet (it hashes the loaded tensors)"
        )
    if preset.kind == "open_clip":
        return "open_clip fetches its own weights; no pinned hub revision yet"
    if preset.local_dir:
        return "validated against a local snapshot only; no pinned hub revision yet"
    return "install not validated yet: no pinned revision and file list"


def _exact_pin(spec: str) -> tuple[str, str] | None:
    """`transformers==5.2.0` -> (`transformers`, `5.2.0`); other specs None."""
    if "==" not in spec:
        return None
    name, version = spec.split("==", 1)
    return name.strip().lower().replace("_", "-"), version.strip()


def _clash(packages: tuple[str, ...], pins: dict[str, tuple[str, str]]) -> str | None:
    for spec in packages:
        pin = _exact_pin(spec)
        if pin is None:
            continue
        name, version = pin
        held = pins.get(name)
        if held is not None and held[0] != version:
            return (
                f"pins {name}=={version}, which conflicts with {held[1]}'s "
                f"{name}=={held[0]} in the same env"
            )
    return None


def _entry(preset: ModelPreset) -> dict:
    spec = preset.install
    assert spec is not None
    return {
        "name": preset.name,
        "embedding_model": preset.embedding_model,
        "kind": preset.kind,
        "repo": preset.model_id,
        "revision": spec.revision,
        "files": [{"path": p, "size": n} for p, n in spec.files],
        "bytes": sum(n for _, n in spec.files),
        "model_hash": spec.model_hash,
        "packages": list(spec.packages),
        "imports": [module for module, _ in preset.requires],
        "remote_code": preset.trust_remote_code,
        "remote_code_hashes": [{"path": p, "sha256": h} for p, h in preset.remote_code_hashes],
        "dim": preset.default_dim,
        "modalities": sorted(preset.modalities),
    }


def build(presets: dict[str, ModelPreset] = PRESETS) -> dict:
    """The catalog for `presets`, in registry order; the test-only fake
    preset is left out of both lists."""
    models: list[dict] = []
    excluded: list[dict] = []
    pins: dict[str, tuple[str, str]] = {}
    for preset in presets.values():
        if preset.kind == "fake":
            continue
        reason = exclusion(preset)
        if reason is None and preset.install is not None:
            reason = _clash(preset.install.packages, pins)
        if reason is not None:
            excluded.append(
                {"name": preset.name, "embedding_model": preset.embedding_model, "reason": reason}
            )
            continue
        assert preset.install is not None
        for spec in preset.install.packages:
            pin = _exact_pin(spec)
            if pin is not None:
                pins.setdefault(pin[0], (pin[1], preset.name))
        models.append(_entry(preset))
    return {"schema": SCHEMA, "models": models, "excluded": excluded}


def render(catalog: dict) -> str:
    return json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    text = render(build())
    if args == ["--write"]:
        CATALOG.write_text(text)
        print(f"wrote {CATALOG}")
        return 0
    if args == ["--check"]:
        if not CATALOG.is_file() or CATALOG.read_text() != text:
            print(
                f"{CATALOG} is stale: run python {Path(__file__).resolve()} --write",
                file=sys.stderr,
            )
            return 1
        return 0
    if args:
        print(__doc__, file=sys.stderr)
        return 2
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
