# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""Prove tool/tasks/releasepr.sh prepares a release and nothing else.

every case runs the real script, with the pinned cargo-release, in a scratch
repository built from this checkout's tracked files (git-lfs files left out,
no release step reads them) and signs with a throwaway ssh key. the script
runs with --no-push, so nothing leaves the machine:

- happy path: the next patch version lands in the workspace version, both
  pins, the four crates of Cargo.lock, a dated changelog section below the
  towncrier marker that holds every news/ fragment, and the five versioned
  CITATION.cff fields; every other CITATION.cff line is unchanged; the
  preflight passes; one signed commit on release-X.Y.Z in its own worktree
  touches exactly those four files and removes every fragment; the base
  branch, the checkout and the tags do not move;
- error path: the current version, a malformed version, and a cargo-release
  or a towncrier other than the pinned one are refused before any branch or
  worktree exists;
- edge case: a news/ with no fragment is refused.

skips when the pinned cargo-release or towncrier is not installed (the
script names the install command).

Run: python tool/tests/test_releasepr.py
"""

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "tool/tasks/releasepr.sh"
PINNED = re.search(r"^CARGO_RELEASE_VERSION=(\S+)$", (REPO / SCRIPT).read_text(), re.M).group(1)
TOWNCRIER = re.search(r"^TOWNCRIER_VERSION=(\S+)$", (REPO / SCRIPT).read_text(), re.M).group(1)
MARKER = "<!-- towncrier: each release lands below this line -->"
FILES = ["CITATION.cff", "Cargo.lock", "Cargo.toml", "docs/CHANGELOG"]
CITED = ("version:", "date-released:", "repository-artifact:", "    value:", "    description:")


def run(args, cwd, env=None, check=True):
    proc = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"{args} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


def git(cwd, *args, check=True):
    return run(["git", *args], cwd, check=check)


def scratch(tmp: Path) -> tuple[Path, Path]:
    """A repository holding this checkout's tracked files, plus a signing key."""
    repo = tmp / "repo"
    listed = run(["git", "ls-files", "-z"], REPO).stdout.split("\0")
    lfs = set(run(["git", "lfs", "ls-files", "-n"], REPO, check=False).stdout.split())
    for rel in filter(None, listed):
        src = REPO / rel
        if rel in lfs or not src.is_file() or src.is_symlink() and not src.exists():
            continue
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)
    key = tmp / "key"
    run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "t@t", "-f", str(key)], tmp)
    signers = tmp / "trustkeys"
    signers.write_text(f"t@t {(tmp / 'key.pub').read_text().strip()}\n")
    git(tmp, "init", "-q", "-b", "main", str(repo))
    for k, v in [
        ("user.name", "t"),
        ("user.email", "t@t"),
        ("gpg.format", "ssh"),
        ("user.signingkey", str(key)),
        ("gpg.ssh.allowedSignersFile", str(signers)),
        ("commit.gpgsign", "false"),
    ]:
        git(repo, "config", k, v)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo, signers


def workspace_version(repo: Path) -> str:
    text = (repo / "Cargo.toml").read_text()
    return re.search(r'^\[workspace\.package\]\n(?:.*\n)*?version = "([^"]+)"', text, re.M).group(1)


def prepare(repo: Path, version: str, worktree: Path, env=None):
    args = ["bash", SCRIPT, version, "--base", "main", "--worktree", str(worktree), "--no-push"]
    return run(args, repo, env=env, check=False)


def untouched(repo: Path, branch: str, worktree: Path):
    assert git(repo, "branch", "--list", branch).stdout.strip() == "", f"{branch} was created"
    assert not worktree.exists(), f"{worktree} was created"
    assert git(repo, "status", "--porcelain").stdout == "", "the checkout changed"


def case_happy(repo: Path, signers: Path, tmp: Path):
    old = workspace_version(repo)
    major, minor, patch = map(int, old.split("."))
    new = f"{major}.{minor}.{patch + 1}"
    base = git(repo, "rev-parse", "main").stdout.strip()
    fragments = sorted(p.name for p in (repo / "news").iterdir())
    entries = [(repo / "news" / name).read_text().strip() for name in fragments]
    assert entries, "the checkout has no news/ fragment to release"
    cff_before = (repo / "CITATION.cff").read_text().splitlines()
    wt = tmp / "wt-happy"
    proc = prepare(repo, new, wt)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    today = dt.datetime.now(dt.UTC).date().isoformat()

    assert workspace_version(wt) == new
    manifest = (wt / "Cargo.toml").read_text()
    for pin, path in (("urna-format", "rust/format"), ("urna-engine", "rust/engine")):
        assert f'{pin} = {{ path = "{path}", version = "{new}" }}' in manifest, pin
    lock = (wt / "Cargo.lock").read_text()
    for name in ("urna", "urna-format", "urna-engine", "urna-bridge"):
        assert f'name = "{name}"\nversion = "{new}"' in lock, f"Cargo.lock {name}"
    log = (wt / "docs/CHANGELOG").read_text()
    assert f"last-updated: {today}\n" in log
    head = f"{MARKER}\n\n## [{new}] - {today}\n"
    assert head in log.split("## [Unreleased]\n", 1)[1], "changelog"
    section = log.split(head, 1)[1].split(f"\n## [{old}]", 1)[0]
    for entry in entries:
        assert f"\n- {entry}\n" in section, entry
    assert not (wt / "news").exists() or not any((wt / "news").iterdir()), "fragments left"
    cff = (wt / "CITATION.cff").read_text()
    for line in (
        f'version: "{new}"',
        f'date-released: "{today}"',
        f"repository-artifact: https://crates.io/crates/urna/{new}",
        f"    value: https://github.com/hoffresearch/urna/releases/tag/v{new}",
        f"    description: Release of Urna version {new}.",
    ):
        assert line in cff.splitlines(), line
    rest = [ln for ln in cff.splitlines() if not ln.startswith(CITED)]
    assert rest == [ln for ln in cff_before if not ln.startswith(CITED)], "CITATION.cff moved"

    assert git(wt, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == f"release-{new}"
    assert git(wt, "rev-list", "--count", f"{base}..HEAD").stdout.strip() == "1"
    shown = git(wt, "log", "-1", "--format=%G?%n%s").stdout.split("\n")
    assert shown[:2] == ["G", f"Release {new}"], shown
    names = git(wt, "diff", "--name-only", f"{base}..HEAD").stdout.split()
    assert sorted(n for n in names if not n.startswith("news/")) == FILES, names
    assert sorted(n for n in names if n.startswith("news/")) == [f"news/{f}" for f in fragments]
    assert git(wt, "status", "--porcelain").stdout == "", "left uncommitted changes"
    run([sys.executable, "tool/tasks/preflight.py"], wt)
    assert git(repo, "rev-parse", "main").stdout.strip() == base, "main moved"
    assert git(repo, "tag", "--list").stdout == "", "a tag was created"
    assert git(repo, "status", "--porcelain").stdout == "", "the checkout changed"
    print(f"case 1 (prepare {old} -> {new}, signed, four files, fragments in, preflight ok): OK")
    return new


def case_refused(repo: Path, tmp: Path):
    old = workspace_version(repo)
    for version, why in [(old, "is not after"), ("0.6", "usage:")]:
        wt = tmp / f"wt-{version}"
        proc = prepare(repo, version, wt)
        assert proc.returncode != 0 and why in proc.stderr, proc.stderr
        untouched(repo, f"release-{version}", wt)
    print("case 2 (the current and a malformed version refused, nothing created): OK")


def case_wrong_tool(repo: Path, tmp: Path):
    shim = tmp / "shim"
    shim.mkdir()
    real = shutil.which("cargo")
    (shim / "cargo").write_text(
        f'#!/bin/sh\nif [ "$1 $2" = "release --version" ]; then echo "cargo-release 0.0.1"; '
        f'else exec "{real}" "$@"; fi\n'
    )
    (shim / "cargo").chmod(0o755)
    env = dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}")
    wt = tmp / "wt-tool"
    proc = prepare(repo, "99.0.0", wt, env=env)
    assert proc.returncode != 0 and f"needs cargo-release {PINNED}, found 0.0.1" in proc.stderr
    untouched(repo, "release-99.0.0", wt)
    (shim / "cargo").unlink()
    (shim / "towncrier").write_text('#!/bin/sh\necho "towncrier, version 0.0.1"\n')
    (shim / "towncrier").chmod(0o755)
    proc = prepare(repo, "99.0.0", wt, env=env)
    assert proc.returncode != 0 and f"needs towncrier {TOWNCRIER}, found 0.0.1" in proc.stderr
    untouched(repo, "release-99.0.0", wt)
    print("case 3 (another cargo-release or towncrier refused, the install named): OK")


def case_no_news(repo: Path, tmp: Path):
    git(repo, "rm", "-q", "-r", "news")
    git(repo, "commit", "-q", "-m", "nothing to release")
    wt = tmp / "wt-empty"
    proc = prepare(repo, "99.0.0", wt)
    assert proc.returncode != 0 and "news/ has nothing to release" in proc.stderr, proc.stderr
    untouched(repo, "release-99.0.0", wt)
    print("case 4 (a news/ with no fragment refused): OK")


def main():
    found = run(["cargo", "release", "--version"], REPO, check=False).stdout.split()
    if found[1:2] != [PINNED]:
        print(f"releasepr: skipped, cargo-release {PINNED} is not installed")
        return
    found = []
    if shutil.which("towncrier"):
        found = run(["towncrier", "--version"], REPO, check=False).stdout.split()
    if found[2:3] != [TOWNCRIER]:
        print(f"releasepr: skipped, towncrier {TOWNCRIER} is not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        repo, signers = scratch(tmp)
        case_happy(repo, signers, tmp)
        case_refused(repo, tmp)
        case_wrong_tool(repo, tmp)
        case_no_news(repo, tmp)
    print("releasepr: 4 cases passed")


if __name__ == "__main__":
    main()
