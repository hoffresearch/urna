"""Build the Debian source package a Launchpad PPA compiles, from one tag.

    python tool/tasks/ppasource.py --tag v0.5.5 --series noble --out DIR

Run from a checkout at the tag, on a machine with git, cargo and
dpkg-buildpackage (Ubuntu). Writes into DIR the two original tarballs and
the unsigned source package:

- `urna_<version>.orig.tar.xz`: `git archive` of the tag;
- `urna_<version>.orig-vendor.tar.xz`: every crate of Cargo.lock
  (`cargo vendor --locked`), because Launchpad builds with no network;
- `urna_<version>-1~<series>1.dsc`, `.debian.tar.xz` and
  `_source.changes`, from `dpkg-buildpackage -S`.

Both tarballs carry the tag commit's time and no owner, so each series
uploads the same bytes.

`debian/` is `pkgs/linux/apt/debian`, filled with `string.Template` (`$$`
is a literal `$`, a missing name is an error): the version, the series and
the date in `changelog`, the summary and description from
`[workspace.metadata.pkgs]` in Cargo.toml in `control`. The tag must be
the workspace version, so a source package never names a release it is
not. Signing (`debsign`) and the upload (`dput`) are the caller's.
"""

from __future__ import annotations

import argparse
import email.utils
import json
import re
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = re.compile(r"^v(\d+\.\d+\.\d+)$")
SERIES = re.compile(r"^[a-z]+$")
TEMPLATED = ("changelog", "control")


class SourceError(Exception):
    pass


def texts(root: Path) -> dict[str, str]:
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    package = cargo["workspace"]["package"]
    pkgs = cargo["workspace"]["metadata"]["pkgs"]
    return {
        "version": package["version"],
        "repository": package["repository"],
        "summary": " ".join(pkgs["summary"].split()),
        "description": control_text(pkgs["description"]),
    }


def control_text(text: str) -> str:
    """A long description in control syntax: each line indented by one
    space, at most 79 columns, a paragraph break as ` .`."""
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
    lines: list[str] = []
    for i, paragraph in enumerate(paragraphs):
        if i:
            lines.append(" .")
        lines += textwrap.wrap(paragraph, 78, initial_indent=" ", subsequent_indent=" ")
    return "\n".join(lines)


def clear_file_sums(vendor: Path) -> None:
    """Empty each vendored crate's per-file checksums, as Debian's Rust
    packages do: `dpkg-source -x` leaves out every `*.orig` file, so cargo
    would miss `Cargo.toml.orig`. The crate's own checksum, which cargo
    holds against Cargo.lock, stays."""
    for sums in vendor.glob("*/.cargo-checksum.json"):
        data = json.loads(sums.read_text())
        sums.write_text(json.dumps({"files": {}, "package": data.get("package")}))


def owned_by_root(mtime: int):
    """A tar filter that keeps names, modes and content and drops what
    differs between two runs: owner and mtime. Both tarballs come out byte
    for byte the same for one tag, so each series of the matrix can upload
    them: Launchpad refuses a second orig tarball of the same name with
    other bytes."""

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = mtime
        return info

    return normalize


def render(debian: Path, values: dict[str, str]) -> None:
    for name in TEMPLATED:
        path = debian / name
        try:
            path.write_text(string.Template(path.read_text()).substitute(values))
        except (KeyError, ValueError) as err:
            raise SourceError(f"debian/{name}: unknown or malformed placeholder ({err})") from err


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def build(root: Path, tag: str, series: str, out: Path) -> Path:
    found = TAG.match(tag)
    if not found:
        raise SourceError(f"{tag!r} is not vX.Y.Z")
    if not SERIES.match(series):
        raise SourceError(f"{series!r} is not an Ubuntu series name")
    values = texts(root)
    version = found.group(1)
    if values["version"] != version:
        raise SourceError(f"the checkout is version {values['version']}, not {version}")
    debversion = f"{version}-1~{series}1"
    out.mkdir(parents=True, exist_ok=True)
    orig = out / f"urna_{version}.orig.tar.xz"
    vendor_tar = out / f"urna_{version}.orig-vendor.tar.xz"
    tree = out / f"urna-{version}"
    if tree.exists():
        shutil.rmtree(tree)
    commit_time = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    normalize = owned_by_root(int(commit_time))
    run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--prefix=urna-{version}/",
            "-o",
            str(out / "orig.tar"),
            "HEAD",
        ],
        root,
    )
    with tarfile.open(out / "orig.tar") as src, tarfile.open(orig, "w:xz") as dst:
        for member in src.getmembers():
            dst.addfile(normalize(member), src.extractfile(member) if member.isfile() else None)
    (out / "orig.tar").unlink()
    with tempfile.TemporaryDirectory() as tmp:
        vendor = Path(tmp) / "vendor"
        run(["cargo", "vendor", "--locked", "--versioned-dirs", str(vendor)], root)
        clear_file_sums(vendor)
        with tarfile.open(vendor_tar, "w:xz") as dst:
            dst.add(vendor, arcname="vendor", filter=normalize)
    with tarfile.open(orig) as src:
        src.extractall(out, filter="tar")
    with tarfile.open(vendor_tar) as src:
        src.extractall(tree, filter="tar")
    shutil.copytree(root / "pkgs/linux/apt/debian", tree / "debian")
    render(
        tree / "debian",
        {
            **values,
            "tag": tag,
            "series": series,
            "debversion": debversion,
            "date": email.utils.format_datetime(datetime.now(UTC)),
        },
    )
    run(["dpkg-buildpackage", "-S", "-sa", "-d", "-us", "-uc"], tree)
    return out / f"urna_{debversion}_source.changes"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        changes = build(args.root.resolve(), args.tag, args.series, args.out.resolve())
    except (SourceError, subprocess.CalledProcessError) as err:
        print(f"ppasource: {err}", file=sys.stderr)
        return 1
    print(f"ppasource: {changes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
