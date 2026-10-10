"""The edits release.yml needs that cargo-dist has no setting for.

``.github/workflows/release.yml`` is dist's template. Each edit below is
applied after ``dist generate``, and ``allow-dirty = ["ci"]`` in Cargo.toml
keeps dist's plan from refusing the edited file. That setting also stops
``dist generate`` from writing the file, so ``generate`` here runs dist with
the line taken out, then applies the edits. Every edit must match dist's
output exactly once, so a dist upgrade that moves a line stops here until
this file learns it.

Edits:

- the npm publish job's ``setup-node`` sets ``package-manager-cache: false``
  (zizmor ``cache-poisoning``): a job that publishes restores no cache, and
  setup-node v5+ turns caching on by default.

Subcommands:

- ``generate [--check]``: regenerate release.yml with the edits, or fail
  when the checked-in file differs from that (gatecheck.yml);
  ``$DIST`` names the dist binary (default ``dist``), whose version must be
  Cargo.toml's ``cargo-dist-version``.

    python tool/tasks/distpatch.py generate --check
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARGO = ROOT / "Cargo.toml"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
DIRTY = 'allow-dirty = ["ci"]\n'

# (what dist writes, what release.yml carries)
EDITS = [
    (
        "      - uses: actions/setup-node@{sha}\n"
        "        with:\n"
        "          node-version: '20.x'\n"
        "          registry-url: 'https://registry.npmjs.org'\n",
        "      - uses: actions/setup-node@{sha}\n"
        "        with:\n"
        "          node-version: '20.x'\n"
        "          registry-url: 'https://registry.npmjs.org'\n"
        "          package-manager-cache: false\n",
    ),
]


class Refused(Exception):
    pass


def _pins(cargo: str) -> dict[str, str]:
    import tomllib

    dist = tomllib.loads(cargo)["workspace"]["metadata"]["dist"]
    return {
        "sha": dist.get("github-action-commits", {}).get("actions/setup-node", ""),
        "version": dist["cargo-dist-version"],
    }


def patch(text: str, sha: str) -> str:
    """dist's release.yml with every edit applied; each must match once."""
    for before, after in EDITS:
        before, after = before.format(sha=sha), after.format(sha=sha)
        if after in text:
            raise Refused("release.yml: an edit is already applied; patch dist's output")
        found = text.count(before)
        if found != 1:
            raise Refused(f"release.yml: dist's text for an edit found {found} times, not once")
        text = text.replace(before, after)
    return text


def _dist_generate(cargo: str, dist: str) -> None:
    if cargo.count(DIRTY) != 1:
        raise Refused(f"Cargo.toml: expected one line {DIRTY.strip()}")
    CARGO.write_text(cargo.replace(DIRTY, ""), encoding="utf-8")
    try:
        subprocess.run([dist, "generate", "--mode", "ci"], cwd=ROOT, check=True)
    finally:
        CARGO.write_text(cargo, encoding="utf-8")


def generate(check: bool) -> int:
    dist = os.environ.get("DIST", "dist")
    cargo = CARGO.read_text(encoding="utf-8")
    pins = _pins(cargo)
    found = subprocess.run([dist, "--version"], capture_output=True, text=True, check=True)
    version = re.search(r"\d+\.\d+\.\d+\S*", found.stdout)
    if not version or version.group(0) != pins["version"]:
        raise Refused(f"{dist} is {found.stdout.strip()}, Cargo.toml pins {pins['version']}")
    checked_in = RELEASE.read_text(encoding="utf-8")
    _dist_generate(cargo, dist)
    text = patch(RELEASE.read_text(encoding="utf-8"), pins["sha"])
    if check:
        RELEASE.write_text(checked_in, encoding="utf-8")
        if text != checked_in:
            print(
                "distpatch: release.yml is not dist's output with the edits; "
                "run generate, never edit it by hand",
                file=sys.stderr,
            )
            return 1
        print("distpatch: release.yml is dist's output with the edits")
        return 0
    RELEASE.write_text(text, encoding="utf-8")
    print(f"distpatch: wrote {RELEASE.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        return generate(args.check)
    except Refused as err:
        print(f"distpatch: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
