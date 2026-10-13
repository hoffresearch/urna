# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""The model catalog `urna setup` and the explorer offer, generated from the
registry: every preset whose install was validated end to end (a pinned hub
revision, the exact files fetched at it and the `model_hash` they
fingerprint to). every other preset is listed under `excluded` with the
reason, so the installer can say why a model is not offered instead of
hiding it.

the offered models share one env, so their packages are resolved together
into one lock (`modellock.txt`, exact versions for every platform, torch's
cpu build); setup installs a model's packages constrained by it. a
conflict between two models fails the lock, not a user's install.

the catalog ships in the payload as `urna/model/catalogue.json`; no remote index.
`rust/bridge/python/urna/model/catalogue.json` is this module's output, checked in:
`tool/tasks/embedpack.py` and `tool/tests/test_embedpack.py` refuse a stale copy.

beside the models it carries what setup takes from upstreams: the uv release
it downloads when a machine has none (a url and sha256 per platform), the
python range the lock's torch has wheels for, the size of that torch per
platform, the lock and the sample corpus it offers (a dataset commit, path,
size and sha256). `tool/tasks/upstreams.py` resolves them over the network
into `upstreams.json` and `modellock.txt`; this module only copies the one
and checks the other (its inputs are this registry's, its bytes the ones
upstreams.json names), so `--check` stays a bare-python, offline check.

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
UPSTREAMS = Path(__file__).resolve().with_name("upstreams.json")
LOCK = Path(__file__).resolve().with_name("modellock.txt")
# what the env holds before any model: the potion embedder's packages.
BASE_PACKAGES = ("numpy", "tokenizers")


# the bytes a parameter takes once loaded, by the device it runs on: the
# sentence-transformers backend loads in float16 on mps and float32 on the
# cpu (`stbackend._default_dtype`); open_clip and the plain
# sentence-transformers path load float32 everywhere.
DTYPE_BYTES = {"st_multimodal": {"cpu": 4, "mps": 2}}
FLOAT32 = {"cpu": 4, "mps": 4}


class Stale(Exception):
    """The lock does not belong to this registry: run upstreams.py --write."""


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
    return "install not validated yet: no pinned revision and file list"


def _entry(preset: ModelPreset) -> dict:
    spec = preset.install
    assert spec is not None
    return {
        "name": preset.name,
        "embedding_model": preset.embedding_model,
        "kind": preset.kind,
        # where its files are on the hub: an open_clip preset names its
        # architecture as model_id and its pinned snapshot as hf_repo.
        "repo": preset.hf_repo or preset.model_id,
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
        "license": preset.license,
    }


def load_upstreams(path: Path | None = None) -> dict:
    """`upstreams.json`: `{"uv": {...}, "python": {...}, "sample": {...}}`, resolved by
    `tool/tasks/upstreams.py`."""
    return json.loads((path or UPSTREAMS).read_text())


def offered(presets: dict[str, ModelPreset] = PRESETS) -> list[ModelPreset]:
    """The presets setup offers, in registry order."""
    return [p for p in presets.values() if p.kind != "fake" and exclusion(p) is None]


def lock_inputs(presets: dict[str, ModelPreset] = PRESETS) -> list[str]:
    """What the lock resolves: the env's base packages and every offered
    model's, each spec once, sorted."""
    specs = set(BASE_PACKAGES)
    for preset in offered(presets):
        assert preset.install is not None
        specs.update(preset.install.packages)
    return sorted(specs)


def check_lock(lock: dict, presets: dict[str, ModelPreset], text: str | None) -> None:
    """Raise Stale unless `lock` (upstreams.json's entry) was resolved from
    this registry's inputs and `text` is the file it names."""
    import hashlib

    if lock.get("inputs") != lock_inputs(presets):
        raise Stale("the lock was resolved for other packages than the registry's")
    if text is None or hashlib.sha256(text.encode()).hexdigest() != lock.get("sha256"):
        raise Stale(f"{LOCK.name} is not the file upstreams.json names")


def with_params(entry: dict, params: dict) -> dict:
    """`entry` with its parameter count (upstreams.json's, read at the same
    revision) and the bytes a parameter takes per device; raises Stale when
    upstreams.json counted another revision or none."""
    held = params.get(entry["name"])
    if held is None or held.get("revision") != entry["revision"]:
        raise Stale(f"{entry['name']}'s parameters were not counted at its pinned revision")
    return {
        **entry,
        "params": held["params"],
        "dtype_bytes": DTYPE_BYTES.get(entry["kind"], FLOAT32),
    }


def build(
    presets: dict[str, ModelPreset] = PRESETS,
    upstreams: dict | None = None,
    lock_text: str | None = None,
) -> dict:
    """The catalog for `presets`, in registry order, with the upstream
    values of `upstreams` (default: `upstreams.json`); the test-only fake
    preset is left out of both lists. raises Stale when the lock (default:
    `modellock.txt`) does not belong to `presets`."""
    models: list[dict] = []
    excluded: list[dict] = []
    for preset in presets.values():
        if preset.kind == "fake":
            continue
        reason = exclusion(preset)
        if reason is not None:
            excluded.append(
                {"name": preset.name, "embedding_model": preset.embedding_model, "reason": reason}
            )
            continue
        models.append(_entry(preset))
    up = load_upstreams() if upstreams is None else upstreams
    models = [with_params(m, up.get("params", {})) for m in models]
    if lock_text is None and LOCK.is_file():
        lock_text = LOCK.read_text()
    check_lock(up["lock"], presets, lock_text)
    return {
        "schema": SCHEMA,
        "uv": up["uv"],
        "python": up["python"],
        "torch": up["torch"],
        "lock": {"file": LOCK.name, "sha256": up["lock"]["sha256"]},
        "sample": up.get("sample"),
        "models": models,
        "excluded": excluded,
    }


def render(catalog: dict) -> str:
    return json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        text = render(build())
    except Stale as e:
        print(f"{e}: run python tool/tasks/upstreams.py --write", file=sys.stderr)
        return 1
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
