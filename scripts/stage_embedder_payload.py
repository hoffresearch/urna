"""stage the offline embedder payload for release archives and installers.

the `urna` binary embeds queries OFFLINE by shelling out to the potion
embedder script (`forge/embed_query_potion.py`) with its vendored table.
a released binary has no repo around it, so the release archives and the
one-liner installer carry this payload and lay it down where the cli looks
(`<exe>/../share/urna/forge/` or `$XDG_DATA_HOME/urna/forge/`; see
crates/urna-cli/src/cmd/util.rs `default_potion_embedder_path`).

usage:  python scripts/stage_embedder_payload.py <dest> [--tar <out.tar.gz>]
writes: <dest>/urna/forge/__init__.py
        <dest>/urna/forge/embed_default.py
        <dest>/urna/forge/embed_potion.py
        <dest>/urna/forge/embed_query_potion.py
        <dest>/urna/forge/models/potion-base-8M/...

with --tar, also packs the staged `urna/` tree as a single gzipped tarball
(the release artifact the one-liner installer downloads and extracts into
the data dir). git-lfs pointer files are rejected; run `git lfs pull`
first. `urna doctor` validates exactly this layout post-install.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT / "python" / "forge"

MODULES = [
    "__init__.py",
    "embed_default.py",
    "embed_potion.py",
    "embed_query_potion.py",
]


def fail(msg: str) -> None:
    print(f"stage_embedder_payload: error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    args = sys.argv[1:]
    tar_out: Path | None = None
    if "--tar" in args:
        i = args.index("--tar")
        tar_out = Path(args[i + 1]).resolve()
        del args[i : i + 2]
    if len(args) != 1:
        fail("usage: stage_embedder_payload.py <dest> [--tar <out.tar.gz>]")
    dest = Path(args[0]).resolve() / "urna" / "forge"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in MODULES:
        src = FORGE / name
        if not src.is_file():
            fail(f"missing source: {src}")
        shutil.copyfile(src, dest / name)
    model_src = FORGE / "models" / "potion-base-8M"
    if not model_src.is_dir():
        fail(f"missing model dir: {model_src}")
    shutil.copytree(model_src, dest / "models" / "potion-base-8M")
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
            tar.add(dest.parent, arcname="urna")
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
