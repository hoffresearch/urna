"""The values urna's model catalogue takes from its upstreams instead of
having them written by hand, resolved once here and committed as
`rust/bridge/python/urna/model/upstreams.json`:

- uv: the latest release on github and the sha256 it publishes next to
  each archive, for every platform urna ships. `urna setup` downloads that
  archive when a machine has no uv and refuses any other bytes.
- python: the range torch publishes wheels for, read off the wheel names
  of torch's latest release on pypi (`cp310` .. `cp314` gives
  `>=3.10,<3.15`), so setup never builds its env on a python the models
  cannot install torch on.

they are read when the catalogue is generated, never when a user runs
setup: the catalogue ships in the payload, `catalogue.py --check` (bare
python, no network) refuses one that does not match this file, and a user
only gets a uv or a range urna's ci has run. a value resolved on the user's
machine would reach every user untested; one kept by hand ages.

usage:  python tool/tasks/upstreams.py           print what the upstreams say now
        python tool/tasks/upstreams.py --write   rewrite upstreams.json and catalogue.json
        python tool/tasks/upstreams.py --check   exit 1 when an upstream moved

GH_TOKEN or GITHUB_TOKEN, when set, lifts github's anonymous rate limit.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rust" / "bridge" / "python"))

from urna.model import catalogue  # noqa: E402

UPSTREAMS = catalogue.UPSTREAMS
UA = "urna-upstreams"
UV_REPO = "astral-sh/uv"

# urna's release targets, keyed the way the binary names its own platform
# (`std::env::consts::{ARCH, OS}`), to the uv archive that runs there. the
# musl builds are static, so one archive serves every linux distro.
UV_ARCHIVES = {
    "aarch64-macos": "uv-aarch64-apple-darwin.tar.gz",
    "x86_64-macos": "uv-x86_64-apple-darwin.tar.gz",
    "aarch64-linux": "uv-aarch64-unknown-linux-musl.tar.gz",
    "x86_64-linux": "uv-x86_64-unknown-linux-musl.tar.gz",
    "x86_64-windows": "uv-x86_64-pc-windows-msvc.zip",
}


def _get(url: str, accept: str = "application/json") -> bytes:
    headers = {"User-Agent": UA, "Accept": accept}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return r.read()


def parse_sha256(text: str) -> str:
    """`<64 hex>  <file>` (or the bare hash) to the hash, lowercased."""
    word = text.split()[0] if text.split() else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", word):
        raise ValueError(f"not a sha256 line: {text[:80]!r}")
    return word.lower()


def resolve_uv() -> dict:
    release = json.loads(_get(f"https://api.github.com/repos/{UV_REPO}/releases/latest"))
    version = release["tag_name"].lstrip("v")
    names = {a["name"] for a in release["assets"]}
    archives = {}
    for platform, name in UV_ARCHIVES.items():
        if name not in names or f"{name}.sha256" not in names:
            raise SystemExit(f"uv {version} publishes no {name} with its .sha256")
        url = f"https://github.com/{UV_REPO}/releases/download/{version}/{name}"
        sha = parse_sha256(_get(f"{url}.sha256", accept="*/*").decode())
        archives[platform] = {"url": url, "sha256": sha}
    return {"version": version, "archives": archives}


def python_range(wheels: list[str]) -> str:
    """The `>=3.A,<3.B` range a set of wheel names covers on every platform
    they ship for: the cpython minors each platform has a wheel for,
    intersected, which must be contiguous. free-threaded tags (`cp314t`)
    count as their minor."""
    by_platform: dict[str, set[int]] = {}
    for name in wheels:
        if not name.endswith(".whl"):
            continue
        parts = name[: -len(".whl")].split("-")
        tag, platform = parts[-3], parts[-1]
        m = re.fullmatch(r"cp3(\d+)t?", tag)
        if m:
            by_platform.setdefault(platform, set()).add(int(m.group(1)))
    if not by_platform:
        raise ValueError("no cpython wheels")
    minors = sorted(set.intersection(*by_platform.values()))
    if not minors or minors != list(range(minors[0], minors[-1] + 1)):
        raise ValueError(f"the cpython minors every platform has are not contiguous: {minors}")
    return f">=3.{minors[0]},<3.{minors[-1] + 1}"


def resolve_python() -> dict:
    # torch's latest release until the models' lock (#534) names the torch
    # they resolve to; the only constraint the offered models put on it is
    # sentence-transformers' floor, which every recent torch meets.
    info = json.loads(_get("https://pypi.org/pypi/torch/json"))
    version = info["info"]["version"]
    wheels = [u["filename"] for u in info["urls"]]
    return {"requires": python_range(wheels), "torch": version}


def resolve() -> dict:
    return {"uv": resolve_uv(), "python": resolve_python()}


def render(upstreams: dict) -> str:
    return json.dumps(upstreams, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args not in ([], ["--write"], ["--check"]):
        print(__doc__, file=sys.stderr)
        return 2
    text = render(resolve())
    if args == ["--write"]:
        UPSTREAMS.write_text(text)
        print(f"wrote {UPSTREAMS}")
        return catalogue.main(["--write"])
    if args == ["--check"]:
        held = UPSTREAMS.read_text() if UPSTREAMS.is_file() else ""
        if held != text:
            print(
                f"an upstream moved since {UPSTREAMS.name} was written:\n{text}"
                f"run python tool/tasks/upstreams.py --write and open a pull request",
                file=sys.stderr,
            )
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
