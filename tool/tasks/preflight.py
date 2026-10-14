# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""Check that a release's version and origin agree before anything is built.

Tree mode (every pull request, ``gatecheck.yml``): the workspace version in
``Cargo.toml`` is the one version everywhere a release names it:

- the ``urna-format`` and ``urna-engine`` pins in ``[workspace.dependencies]``
  (``cargo publish`` keeps the version and strips the path);
- every crate under ``rust/`` inherits it (``version.workspace = true``),
  except one that is its own workspace (``rust/ingest``, excluded at the root);
- ``Cargo.lock`` carries it for urna, urna-format, urna-engine, urna-bridge;
- ``CITATION.cff``: ``version``, the versioned ``repository-artifact`` and
  release URLs, and a ``date-released`` equal to the changelog's date;
- ``docs/CHANGELOG`` has a ``## [X.Y.Z] - YYYY-MM-DD`` section, and nothing
  under ``## [Unreleased]``: an unreleased change is a file in ``docs/news/``,
  named ``<issue>.<type>.md`` (``+<name>`` for no issue) with a type of
  ``[tool.towncrier]`` in ``pyproject.toml``, which
  ``tool/tasks/releasepr.sh`` turns into the release's section, so no two
  pull requests edit the same line of the changelog;
- the package texts: the ``description`` of ``rust/clitui/Cargo.toml``
  (crates.io, npm, Homebrew) and of ``pkgs/wheel/pyproject.toml`` (PyPI)
  are one sentence, each free to close it with a clause of its own after
  the last comma (", and the urna CLI", ", read from Python"), and the
  ``summary`` of ``[workspace.metadata.pkgs]`` fits in 80 characters.

Tag mode (``--tag vX.Y.Z``, the release's ``tagverify.yml`` after the
signature check): the same checks on the files of the tag's own commit,
read through git and never from the working tree, plus the tag names that
commit's version, the commit is on the protected ``main`` and the release
date is not after the day the tag was made. Both days are UTC days: the
release date comes from cargo-release (``tool/tasks/releasepr.sh``), which
dates in UTC, so a tag made at 22:30 in Brasilia (01:30 UTC the next day) the
evening a release was prepared fits it, and a release dated after the tag's
UTC day is still refused.

Needs Python 3.11+ (tomllib); the jobs that run it are on ubuntu-24.04.

    python tool/tasks/preflight.py
    python tool/tasks/preflight.py --tag v0.5.3 --main-ref origin/main
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINNED = ("urna-format", "urna-engine")
LOCKED = ("urna", "urna-format", "urna-engine", "urna-bridge")
CRATES_URL = "https://crates.io/crates/urna/{v}"
RELEASE_URL = "https://github.com/hoffresearch/urna/releases/tag/v{v}"
RELEASE_DESCRIPTION = "Release of Urna version {v}."
CRATE = "rust/clitui/Cargo.toml"
WHEEL = "pkgs/wheel/pyproject.toml"
SUMMARY_MAX = 80


def _toml(text: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        raise SystemExit("preflight: needs Python 3.11+ (tomllib)") from None

    return tomllib.loads(text)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


class Tree:
    """The release files of the checkout, or of one commit (tag mode).

    Tag mode reads every file from the tag's commit through git, never from
    the working tree: a checkout that differs from the tag (a later fix, a
    local edit) must not vouch for what the tag releases.
    """

    def __init__(self, root: Path, commit: str | None = None):
        self.root, self.commit = root, commit

    def read(self, rel: str) -> str:
        if self.commit is None:
            return (self.root / rel).read_text(encoding="utf-8")
        shown = _git(self.root, "show", f"{self.commit}:{rel}")
        if shown.returncode != 0:
            raise FileNotFoundError(f"{rel} is not in {self.commit[:12]}")
        return shown.stdout

    def news(self) -> list[str]:
        if self.commit is None:
            return sorted(p.name for p in (self.root / "docs" / "news").glob("*") if p.is_file())
        listed = _git(self.root, "ls-tree", "--name-only", self.commit, "--", "docs/news/")
        return sorted(rel.removeprefix("docs/news/") for rel in listed.stdout.splitlines())

    def crate_manifests(self) -> list[str]:
        if self.commit is None:
            paths = (p.relative_to(self.root) for p in self.root.glob("rust/*/Cargo.toml"))
            return sorted(p.as_posix() for p in paths)
        listed = _git(self.root, "ls-tree", "-r", "--name-only", self.commit, "--", "rust")
        return sorted(
            rel
            for rel in listed.stdout.splitlines()
            if re.fullmatch(r"rust/[^/]+/Cargo\.toml", rel)
        )


def _cff_field(text: str, key: str) -> str | None:
    # top-level and list-item scalars, quoted or bare; CITATION.cff keeps
    # one value per line, so a line match is exact enough without a yaml dep.
    match = re.search(rf'^\s*(?:-\s+)?{re.escape(key)}:\s*"?([^"\n]*?)"?\s*$', text, re.M)
    return match.group(1) if match else None


def _iso_date(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value or "")
    except ValueError:
        return None


def _version(tree: Tree) -> str:
    return _toml(tree.read("Cargo.toml"))["workspace"]["package"]["version"]


def workspace_version(root: Path) -> str:
    return _version(Tree(root))


def _release_date(tree: Tree, version: str) -> dt.date | None:
    """The date of the changelog section for `version`, or None."""
    heading = rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$"
    match = re.search(heading, tree.read("docs/CHANGELOG"), re.M)
    return _iso_date(match.group(1)) if match else None


def _check_manifests(tree: Tree, v: str) -> list[str]:
    errors = []
    deps = _toml(tree.read("Cargo.toml"))["workspace"].get("dependencies", {})
    for name in PINNED:
        pin = deps.get(name, {}).get("version") if isinstance(deps.get(name), dict) else None
        if pin != v:
            errors.append(f"Cargo.toml: [workspace.dependencies] {name} pins {pin!r}, not {v!r}")
    for rel in tree.crate_manifests():
        manifest = _toml(tree.read(rel))
        if "workspace" in manifest:
            continue  # its own workspace (rust/ingest), excluded by the root manifest
        version = manifest["package"].get("version")
        if version != {"workspace": True} and version != v:
            errors.append(f"{rel}: version {version!r} neither inherits nor equals {v!r}")
    locked = {p["name"]: p["version"] for p in _toml(tree.read("Cargo.lock")).get("package", [])}
    for name in LOCKED:
        if locked.get(name) != v:
            errors.append(f"Cargo.lock: {name} is {locked.get(name)!r}, not {v!r}")
    return errors


def _check_citation(tree: Tree, v: str, released: dt.date | None) -> list[str]:
    text = tree.read("CITATION.cff")
    want = {
        "version": v,
        "repository-artifact": CRATES_URL.format(v=v),
        "value": RELEASE_URL.format(v=v),
        "description": RELEASE_DESCRIPTION.format(v=v),
    }
    errors = [
        f"CITATION.cff: {key} is {_cff_field(text, key)!r}, not {value!r}"
        for key, value in want.items()
        if _cff_field(text, key) != value
    ]
    cited = _iso_date(_cff_field(text, "date-released"))
    if cited is None:
        errors.append("CITATION.cff: date-released is missing or not YYYY-MM-DD")
    elif released is not None and cited != released:
        msg = f"date-released {cited} differs from the changelog's {released}"
        errors.append(f"CITATION.cff: {msg}")
    return errors


def _check_news(tree: Tree) -> list[str]:
    log = tree.read("docs/CHANGELOG")
    unreleased = re.search(r"^## \[Unreleased\]$(.*?)(?=^## \[)", log, re.M | re.S)
    errors = []
    if unreleased is None or re.search(r"^(- |### )", unreleased.group(1), re.M):
        errors.append("docs/CHANGELOG: [Unreleased] holds entries; a change goes in docs/news/")
    config = _toml(tree.read("pyproject.toml"))["tool"]["towncrier"]
    kinds = "|".join(re.escape(k["directory"]) for k in config["type"])
    shape = rf"(\d+|\+[a-z0-9-]+)\.({kinds})(\.\d+)?\.md"
    for name in tree.news():
        if not re.fullmatch(shape, name):
            errors.append(f"docs/news/{name}: not <issue>.<type>.md with a type of pyproject.toml")
    return errors


def _check_texts(tree: Tree) -> list[str]:
    crate = _toml(tree.read(CRATE))["package"].get("description", "")
    wheel = _toml(tree.read(WHEEL))["project"].get("description", "")
    errors = []
    # the sentence before each channel's own closing clause
    if crate.rsplit(", ", 1)[0] != wheel.rsplit(", ", 1)[0]:
        errors.append(f"{WHEEL}: description {wheel!r} does not open like {CRATE}'s {crate!r}")
    workspace = _toml(tree.read("Cargo.toml"))["workspace"]
    summary = workspace.get("metadata", {}).get("pkgs", {}).get("summary", "")
    if not 0 < len(summary) <= SUMMARY_MAX:
        msg = f"summary is {len(summary)} characters, not 1 to {SUMMARY_MAX}"
        errors.append(f"Cargo.toml: [workspace.metadata.pkgs] {msg}")
    return errors


def _check(tree: Tree) -> list[str]:
    v = _version(tree)
    released = _release_date(tree, v)
    errors = _check_manifests(tree, v)
    if released is None:
        errors.append(f"docs/CHANGELOG: no '## [{v}] - YYYY-MM-DD' section")
    errors += _check_news(tree) + _check_texts(tree)
    return errors + _check_citation(tree, v, released)


def check_tree(root: Path = ROOT) -> list[str]:
    """Every place the checkout names its version agrees; returns the errors."""
    try:
        return _check(Tree(root))
    except FileNotFoundError as err:
        return [f"missing release file: {err}"]


def check_tag(tag: str, root: Path = ROOT, main_ref: str = "origin/main") -> list[str]:
    """The checks on the tag's own commit, plus its name, place on main and date."""
    commit = _git(root, "rev-parse", "--verify", "-q", f"refs/tags/{tag}^{{commit}}").stdout.strip()
    if not commit:
        return [f"tag {tag} does not exist"]
    tree = Tree(root, commit)
    try:
        errors = [f"{tag}: {e}" for e in _check(tree)]
        v = _version(tree)
        released = _release_date(tree, v)
    except FileNotFoundError as err:
        return [f"{tag}: missing release file: {err}"]
    if tag != f"v{v}":
        errors.append(f"tag {tag} does not name the version v{v} of its own commit")
    if _git(root, "merge-base", "--is-ancestor", commit, main_ref).returncode != 0:
        errors.append(f"tag {tag} points at {commit[:12]}, which is not on {main_ref}")
    fmt = "--format=%(taggerdate:unix)"
    stamp = _git(root, "for-each-ref", fmt, f"refs/tags/{tag}").stdout.strip()
    tagged = dt.datetime.fromtimestamp(int(stamp), dt.UTC).date() if stamp.isdigit() else None
    if tagged is None:
        errors.append(f"tag {tag} has no tagger date (a lightweight tag)")
    elif released is not None and released > tagged:
        errors.append(f"release date {released} is after the tag date {tagged}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tag", help="also check this release tag (tag mode)")
    parser.add_argument("--main-ref", default="origin/main", help="the protected branch ref")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    errors = check_tag(args.tag, args.root, args.main_ref) if args.tag else check_tree(args.root)
    for error in errors:
        print(f"preflight: {error}", file=sys.stderr)
    if errors:
        return 1
    scope = f"tag {args.tag} (its own commit)" if args.tag else "tree"
    print(f"preflight: {scope} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
