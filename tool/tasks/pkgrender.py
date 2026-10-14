"""Render each package channel's manifest from one release.

    python tool/tasks/pkgrender.py render --channel scoop \
        --manifest dist-manifest.json --sums sha256.sum --out DIR

Channels: scoop, wingt (winget), conda, choco, dnf (the COPR spec) and nfpm
(one config per Linux architecture, `nfpm-amd64.yaml` and `nfpm-arm64.yaml`).
Each file lands under DIR at its path below `pkgs/`.

The release is the only source of what changes per version: the version and
tag from cargo-dist's `dist-manifest.json`, the URL of every archive from its
hosting block, and each archive's SHA-256 from the release's `sha256.sum`.
The texts every channel shows (summary, description, tags) come from
`[workspace.metadata.pkgs]` in Cargo.toml, the license and the links from
`[workspace.package]`. Templates are the files in `pkgs/`, filled with
`string.Template`: `$$` is a literal `$`, and a missing name is an error, so
a template can never ship with a placeholder left in it.

The texts are escaped for the file they land in (JSON and YAML strings, XML
text, one line of spec text) and a PowerShell script gets none of them;
URLs, hashes, the version and the links are checked against their shape
first, so nothing read here can break out of a field. Standard library
only; reads local files, never the network.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import string
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# short name -> cargo-dist target triple, for every archive a channel uses.
TARGETS = {
    "x86_64_linux": "x86_64-unknown-linux-musl",
    "aarch64_linux": "aarch64-unknown-linux-musl",
    "x86_64_macos": "x86_64-apple-darwin",
    "aarch64_macos": "aarch64-apple-darwin",
    "x86_64_windows": "x86_64-pc-windows-msvc",
}
# channel -> its template files under pkgs/; `per_target` renders once per
# listed target, with that target's values also under the plain names
# (`url`, `sha256`, `dir`, ...) and `arch` in nfpm's naming.
CHANNELS = {
    "scoop": {"files": ["scoop/urna.json"]},
    "wingt": {
        "files": [
            "wingt/HoffResearch.Urna.yaml",
            "wingt/HoffResearch.Urna.installer.yaml",
            "wingt/HoffResearch.Urna.locale.en-US.yaml",
        ]
    },
    "conda": {"files": ["conda/recipe.yaml"]},
    "choco": {"files": ["choco/urna.nuspec", "choco/tools/chocolateyInstall.ps1"]},
    "dnf": {"files": ["linux/dnf/urna.spec"]},
    "nfpm": {"files": ["linux/nfpm.yaml"], "per_target": ["x86_64_linux", "aarch64_linux"]},
}
NFPM_ARCH = {"x86_64_linux": "amd64", "aarch64_linux": "arm64"}

VERSION = re.compile(r"^\d+\.\d+\.\d+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/releases/download/v\d+\.\d+\.\d+/[\w.-]+$")


class RenderError(Exception):
    pass


def read_sums(path: Path) -> dict[str, str]:
    """`sha256.sum` lines (`<hex> *<name>` or `<hex>  <name>`) as name -> hex."""
    sums = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition(" ")
        name = name.lstrip(" *")
        if not SHA256.match(digest) or not name:
            raise RenderError(f"{path.name}: not a checksum line: {line!r}")
        sums[name] = digest
    return sums


def release(manifest: Path, sums: Path, expect: str | None = None) -> dict[str, str]:
    """The per-version values: version, tag and each target's archive. With
    `expect`, the manifest must be that tag's, so a job's own tag and the
    version it renders cannot part."""
    data = json.loads(manifest.read_text())
    tag = data.get("announcement_tag", "")
    version = tag.removeprefix("v")
    if not VERSION.match(version):
        raise RenderError(f"{manifest.name}: announcement_tag {tag!r} is not vX.Y.Z")
    if expect is not None and tag != expect:
        raise RenderError(f"{manifest.name} is the release of {tag}, not {expect}")
    rel = next((r for r in data.get("releases", []) if r.get("app_name") == "urna"), None)
    if rel is None or rel.get("app_version") != version:
        raise RenderError(f"{manifest.name}: no urna release at {version}")
    gh = rel.get("hosting", {}).get("github", {})
    base = gh.get("artifact_base_url", "") + gh.get("artifact_download_path", "")
    digests = read_sums(sums)
    values = {"version": version, "tag": tag}
    for short, triple in TARGETS.items():
        art = next(
            (
                (name, a)
                for name, a in data.get("artifacts", {}).items()
                if a.get("kind") == "executable-zip" and a.get("target_triples") == [triple]
            ),
            None,
        )
        if art is None:
            raise RenderError(f"{manifest.name}: no archive for {triple}")
        name, a = art
        url = f"{base}/{name}"
        if not URL.match(url):
            raise RenderError(f"{manifest.name}: {name} has no release URL: {url}")
        if name not in digests:
            raise RenderError(f"{sums.name}: no SHA-256 for {name}")
        exe = next((x["path"] for x in a.get("assets", []) if x.get("kind") == "executable"), None)
        if exe is None:
            raise RenderError(f"{manifest.name}: {name} lists no executable")
        # cargo-dist puts a tarball's files under a directory named after the
        # archive; a zip keeps them at its root.
        folder = "" if name.endswith(".zip") else name.split(".tar.")[0]
        values |= {
            f"url_{short}": url,
            f"sha256_{short}": digests[name],
            f"archive_{short}": name,
            f"dir_{short}": folder,
            f"exe_{short}": exe,
        }
    return values


def metadata(root: Path) -> dict:
    """The texts and links every channel shows, from Cargo.toml."""
    cargo = tomllib.loads((root / "Cargo.toml").read_text())
    ws = cargo.get("workspace", {})
    pkg = ws.get("package", {})
    meta = ws.get("metadata", {}).get("pkgs", {})
    missing = [k for k in ("summary", "description", "tags") if not meta.get(k)]
    if missing:
        raise RenderError(f"Cargo.toml [workspace.metadata.pkgs] lacks {', '.join(missing)}")
    if len(meta["summary"]) > 80:
        raise RenderError("Cargo.toml [workspace.metadata.pkgs] summary is over 80 characters")
    links = {k: pkg.get(k, "") for k in ("license", "homepage", "repository")}
    if not re.fullmatch(r"[A-Za-z0-9.+-]+", links["license"]):
        raise RenderError(f"Cargo.toml license {links['license']!r} is not one SPDX id")
    for key in ("homepage", "repository"):
        if not re.fullmatch(r"https://[\w./-]+", links[key]):
            raise RenderError(f"Cargo.toml {key} {links[key]!r} is not a plain https URL")
    summary = " ".join(meta["summary"].split())
    description = " ".join(meta["description"].split())
    return {
        "texts": {
            "summary": summary,
            "description": description,
            "about": f"{summary}\n{description}",
        },
        "tags": [str(t) for t in meta["tags"]],
        "links": links,
    }


def escaped(meta: dict, suffix: str) -> dict[str, str]:
    """The texts written for one file kind, tags in the list form it uses.

    JSON and YAML get JSON strings (a YAML double-quoted scalar is JSON), so
    the template leaves them unquoted; XML gets escaped text; a spec gets the
    one-line text as it is; a PowerShell script gets only the links."""
    texts, tags = meta["texts"], meta["tags"]
    if suffix == ".ps1":
        return dict(meta["links"])
    if suffix in (".json", ".yaml"):
        out = {k: json.dumps(v) for k, v in texts.items()} | {"tags_list": json.dumps(tags)}
    elif suffix == ".nuspec":
        out = {k: html.escape(v) for k, v in texts.items()} | {
            "tags_list": html.escape(" ".join(tags))
        }
    else:
        out = dict(texts) | {"tags_list": " ".join(tags)}
    return out | meta["links"]


def render(channel: str, rel: dict[str, str], meta: dict, root: Path, out: Path) -> list[Path]:
    """Every file of one channel, all rendered before any is written."""
    prefix = f"{meta['links']['repository']}/releases/download/{rel['tag']}/"
    for key, url in rel.items():
        if key.startswith("url_") and not url.startswith(prefix):
            raise RenderError(f"{url} is not under this release ({prefix})")
    spec = CHANNELS[channel]
    bodies = []
    for target in spec.get("per_target", [None]):
        values = dict(rel)
        if target:
            values |= {
                k.removesuffix(f"_{target}"): v for k, v in rel.items() if k.endswith(f"_{target}")
            }
            values["arch"] = NFPM_ARCH[target]
        for rel_path in spec["files"]:
            src = root / "pkgs" / rel_path
            text = src.read_text()
            if not text.strip():
                raise RenderError(f"pkgs/{rel_path} is empty")
            try:
                body = string.Template(text).substitute(values | escaped(meta, src.suffix))
            except (KeyError, ValueError) as err:
                raise RenderError(
                    f"pkgs/{rel_path}: unknown or malformed placeholder {err}"
                ) from None
            dest = out / rel_path
            if target:
                dest = dest.with_name(f"{dest.stem}-{values['arch']}{dest.suffix}")
            bodies.append((dest, body))
    for dest, body in bodies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
    return [dest for dest, _ in bodies]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="fill one channel's templates")
    r.add_argument("--channel", required=True, choices=sorted(CHANNELS))
    r.add_argument("--manifest", required=True, type=Path)
    r.add_argument("--sums", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--tag", help="refuse a manifest of any other tag")
    r.add_argument("--root", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        files = render(
            args.channel,
            release(args.manifest, args.sums, args.tag),
            metadata(args.root),
            args.root,
            args.out,
        )
    except (RenderError, OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as err:
        print(f"pkgrender: {err}", file=sys.stderr)
        return 1
    for f in files:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
