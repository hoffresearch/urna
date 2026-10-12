"""stage the offline embedder payload for release archives and installers.

the `urna` binary embeds queries OFFLINE by shelling out to a query
embedder script: `urna/embed/potionqry.py` with its vendored table for
potion corpora, `urna/embed/presetqry.py` (the model registry) for
corpora whose default model is a registry model (wemm, clip, jina). a
released binary has no repo around it, so the release archives and the
one-liner installer carry this payload and lay it down where the cli looks
(`<exe>/../share/urna/python/` or `$XDG_DATA_HOME/urna/python/`; see
rust/clitui/src/cmd/embed_gate.rs `installed_script_in`). the payload keeps
the package layout of rust/bridge/python/urna/, so each script puts the
`python/` dir that holds `urna/` on sys.path and imports `urna.<sub>.<mod>`.

the registry path ships its scripts only, not its model dependencies: the
setup venv has numpy and tokenizers, and `presetqry.py` names the
exact `pip install` line for what a registry model still needs (torch,
sentence-transformers, open_clip), exit 4.

usage:  python tool/tasks/embedpack.py <dest> [--tar <out.tar.gz>]
writes: <dest>/urna/VERSION       (the workspace version: setup replaces a
                                   payload of another release)
        <dest>/urna/python/urna/  (the modules below, PKG-relative, and
                                   model/potionb8m/...)

with --tar, also packs the staged `urna/` tree as a single gzipped tarball
(the release artifact the one-liner installer downloads and extracts into
the data dir). git-lfs pointer files are rejected; run `git lfs pull`
first. `urna doctor` validates exactly this layout post-install.
"""

from __future__ import annotations

import re
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "rust" / "bridge" / "python" / "urna"

MODULES = [
    # the package root: the extension loads on first use, so the payload
    # imports it without _urna.
    "__init__.py",
    "embed/__init__.py",
    "embed/lexifloor.py",
    "embed/potiontab.py",
    "embed/potionqry.py",
    # the registry query path (`ask`/`retrieve` on a wemm, clip or jina
    # corpus): the embedder, the registry, its adapters and the two model
    # backends the adapters import lazily.
    "embed/presetqry.py",
    "embed/stbackend.py",
    "embed/stprocess.py",
    "embed/visionemb.py",
    # the sentence-transformers query embedder: `search-text`'s default and
    # the fallback of `presetqry.py` for a model no preset names.
    "embed/searchtxt.py",
    "model/__init__.py",
    "model/presetmap.py",
    # the validated installs and reviewed code pins the registry attaches.
    "model/installed.py",
    "model/embedders.py",
    "model/modelhash.py",
    # the models setup and the explorer offer to install (generated from the
    # registry by catalogue.py; a stale copy fails the stage).
    "model/catalogue.json",
    # the lock every model's packages install from (`-c`), beside it.
    "model/modellock.txt",
    # the fetch half of the model install setup and the explorer share.
    "model/installer.py",
]


def workspace_version(manifest: Path | None = None) -> str:
    """The version every crate and the wheel ship under; at release time it is
    the tag's, so the stamp names the release the payload came from.

    dist's global release job runs this with the runner's python3, which is
    3.10 on ubuntu-22.04 and has no tomllib (3.11+), so without it the
    `version` line of [workspace.package] is read directly."""
    text = (manifest or ROOT / "Cargo.toml").read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:
        section = re.search(r"^\[workspace\.package\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        version = section and re.search(r'^version\s*=\s*"([^"]+)"\s*$', section.group(1), re.M)
        if not version:
            msg = "embedpack: no version in [workspace.package] of Cargo.toml"
            raise SystemExit(msg) from None
        return version.group(1)
    return tomllib.loads(text)["workspace"]["package"]["version"]


def catalog_drift() -> str | None:
    """An error when model/catalogue.json is not what the registry generates
    now: the payload must offer exactly the validated presets."""
    sys.path.insert(0, str(PKG.parent))
    from urna.model import catalogue

    path = PKG / "model" / "catalogue.json"
    try:
        want = catalogue.render(catalogue.build())
    except catalogue.Stale as e:
        return f"{e}: run python tool/tasks/upstreams.py --write"
    have = path.read_text() if path.is_file() else ""
    if have != want:
        rel = path.relative_to(ROOT)
        return f"{rel} is stale: run python {rel.with_suffix('.py')} --write"
    return None


def fail(msg: str) -> None:
    print(f"embedpack: error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    args = sys.argv[1:]
    tar_out: Path | None = None
    if "--tar" in args:
        i = args.index("--tar")
        tar_out = Path(args[i + 1]).resolve()
        del args[i : i + 2]
    if len(args) != 1:
        fail("usage: embedpack.py <dest> [--tar <out.tar.gz>]")
    sources = [PKG / n for n in MODULES]
    for src in sources:
        if not src.is_file():
            fail(f"missing source: {src}")
    stale = catalog_drift()
    if stale:
        fail(stale)
    home = Path(args[0]).resolve() / "urna"
    dest = home / "python" / "urna"
    if dest.parent.exists():
        shutil.rmtree(dest.parent)
    for rel in MODULES:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PKG / rel, dest / rel)
    home.joinpath("VERSION").write_text(workspace_version() + "\n")
    model_src = PKG / "model" / "potionb8m"
    if not model_src.is_dir():
        fail(f"missing model dir: {model_src}")
    shutil.copytree(model_src, dest / "model" / "potionb8m")
    for f in sorted(dest.rglob("*")):
        if f.is_file():
            with f.open("rb") as fh:
                if fh.read(64).startswith(b"version https://git-lfs"):
                    fail(f"{f} is a git-lfs pointer; run `git lfs pull` first")
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"staged embedder payload at {dest} ({total / 1e6:.1f} MB)")
    if tar_out is not None:
        tar_out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_out, "w:gz") as tar:
            tar.add(home, arcname="urna")
        # the one-liner installer verifies this against the downloaded file,
        # in the same `<hex> *<name>` format sha256sum emits.
        import hashlib

        digest = hashlib.sha256(tar_out.read_bytes()).hexdigest()
        sha_path = tar_out.with_name(tar_out.name + ".sha256")
        sha_path.write_text(f"{digest} *{tar_out.name}\n")
        print(f"wrote {tar_out} ({tar_out.stat().st_size / 1e6:.1f} MB)")
        print(f"wrote {sha_path}")


if __name__ == "__main__":
    main()
