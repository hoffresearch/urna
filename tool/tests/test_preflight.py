# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""Prove the release preflight agrees with the tree and refuses every drift.

`tool/tasks/preflight.py` runs on every pull request (tree mode) and
on the release tag after its signature check (tag mode). this suite runs it
against the real checkout, then against temp copies of the files a release
names (the manifests, the lockfile, CITATION.cff, the changelog), one field
changed per case, and against throwaway git repos for the tag checks:

- happy path: the checkout passes; an annotated tag on main naming the
  version of its own commit passes, even when the checkout is broken;
- error path: a moved workspace pin, a crate that stops inheriting, a stale
  lockfile entry, each versioned CITATION field, a date the changelog does
  not carry, a missing changelog section; a tag naming another version, on
  a commit off main, dated before the release, or lightweight; a tag whose
  commit has a stale CITATION while main and the checkout carry the fix
  (tag mode reads the tag's commit, not the working tree); a release file
  missing from the tag's commit;
- edge case: a pin that drifts only in Cargo.lock (the manifests agree).

Run: python tool/tests/test_preflight.py
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tool" / "tasks" / "preflight.py"
spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

FILES = ["Cargo.toml", "Cargo.lock", "CITATION.cff", "docs/CHANGELOG", "pyproject.toml"]
VERSION = preflight.workspace_version(REPO)
MARKER = "<!-- towncrier: each release lands below this line -->"
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
GIT += ["-c", "tag.gpgsign=false"]


def copy_tree(dst: Path) -> Path:
    news = [str(p.relative_to(REPO)) for p in REPO.glob("news/*")]
    for rel in FILES + news + [str(p.relative_to(REPO)) for p in REPO.glob("rust/*/Cargo.toml")]:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst / rel)
    return dst


def edit(root: Path, rel: str, old: str, new: str, count: int = 1) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{rel} lost {old!r}; update this test"
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def git_repo(root: Path, tag_date: str | None = None) -> None:
    def run(*args, env=None):
        subprocess.run(GIT + list(args), cwd=root, check=True, capture_output=True, env=env)

    run("init", "-q", "-b", "main")
    run("add", "-A")
    run("commit", "-q", "-m", "release")
    run("update-ref", "refs/remotes/origin/main", "HEAD")
    env = dict(os.environ, GIT_COMMITTER_DATE=tag_date) if tag_date else None
    run("tag", "-a", f"v{VERSION}", "-m", f"v{VERSION}", env=env)


def bump(v: str) -> str:
    major, minor, patch = v.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def test_checkout_passes() -> None:
    assert preflight.check_tree(REPO) == [], preflight.check_tree(REPO)
    assert preflight.main([]) == 0


def test_each_drift_is_named() -> None:
    other = bump(VERSION)
    pin = f'"rust/engine", version = "{VERSION}"'
    cases = [
        ("Cargo.toml", pin, pin.replace(VERSION, other), "urna-engine pins"),
        (
            "rust/clitui/Cargo.toml",
            "version.workspace = true",
            f'version = "{other}"',
            "inherits",
        ),
        ("CITATION.cff", f'version: "{VERSION}"', f'version: "{other}"', "CITATION.cff: version"),
        ("CITATION.cff", f"crates/urna/{VERSION}", f"crates/urna/{other}", "repository-artifact"),
        ("CITATION.cff", f"tag/v{VERSION}", f"tag/v{other}", "CITATION.cff: value"),
        ("CITATION.cff", f"version {VERSION}.", f"version {other}.", "CITATION.cff: description"),
        ("docs/CHANGELOG", f"## [{VERSION}] - ", f"## [{VERSION}] ", "no '## ["),
        ("docs/CHANGELOG", MARKER, MARKER + "\n### Added\n\n- an entry\n", "[Unreleased] holds"),
    ]
    for rel, old, new, expected in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_tree(Path(tmp))
            edit(root, rel, old, new)
            errors = preflight.check_tree(root)
            assert any(expected in e for e in errors), (rel, expected, errors)
            assert preflight.main(["--root", str(root)]) == 1
    # the citation date has to be the changelog's, whatever day it is today.
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        cff = (root / "CITATION.cff").read_text(encoding="utf-8")
        cff = re.sub(r'date-released: "[^"]+"', 'date-released: "2001-01-01"', cff)
        (root / "CITATION.cff").write_text(cff, encoding="utf-8")
        errors = preflight.check_tree(root)
        assert any("differs from the changelog" in e for e in errors), errors
    # a fragment towncrier would not read is named, not lost.
    for name in ("544.feature.md", "notes.md", "544.added.txt"):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_tree(Path(tmp))
            (root / "news").mkdir(exist_ok=True)
            (root / "news" / name).write_text("an entry\n", encoding="utf-8")
            errors = preflight.check_tree(root)
            assert any(f"news/{name}" in e for e in errors), (name, errors)


def test_a_drift_only_in_the_lockfile() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        lock = f'name = "urna-bridge"\nversion = "{VERSION}"'
        edit(root, "Cargo.lock", lock, lock.replace(VERSION, bump(VERSION)))
        errors = preflight.check_tree(root)
        assert errors == [f"Cargo.lock: urna-bridge is {bump(VERSION)!r}, not {VERSION!r}"], errors


def test_a_tag_on_main_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        git_repo(root)
        assert preflight.check_tag(f"v{VERSION}", root) == []


def test_tags_that_do_not_fit_are_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        git_repo(root)
        assert preflight.check_tag(f"v{bump(VERSION)}", root) == [
            f"tag v{bump(VERSION)} does not exist"
        ]
        run = ["tag", "-a", f"v{bump(VERSION)}", "-m", "x", "HEAD"]
        subprocess.run(GIT + run, cwd=root, check=True, capture_output=True)
        errors = preflight.check_tag(f"v{bump(VERSION)}", root)
        assert any(
            f"does not name the version v{VERSION} of its own commit" in e for e in errors
        ), errors
        # a commit main never received: the tag moves there, main stays put.
        subprocess.run(GIT + ["commit", "-q", "--allow-empty", "-m", "side"], cwd=root, check=True)
        subprocess.run(
            GIT + ["tag", "-f", "-a", f"v{VERSION}", "-m", "x"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        errors = preflight.check_tag(f"v{VERSION}", root)
        assert any("which is not on origin/main" in e for e in errors), errors
        subprocess.run(
            GIT + ["tag", "-f", f"v{VERSION}", "origin/main"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        errors = preflight.check_tag(f"v{VERSION}", root)
        assert errors == [f"tag v{VERSION} has no tagger date (a lightweight tag)"], errors
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        git_repo(root, tag_date="2001-01-01T12:00:00+00:00")
        errors = preflight.check_tag(f"v{VERSION}", root)
        assert any("is after the tag date 2001-01-01" in e for e in errors), errors


def test_the_tag_day_is_the_utc_day() -> None:
    # releasepr.sh dates the release in UTC. prepared and tagged the same
    # evening in Brasilia (22:30 at -03:00 is 01:30 UTC the next day), the
    # release carries the next day's date; read in the tagger's timezone the
    # tag was dated the day before and was refused.
    def dated(root: Path, day: str) -> None:
        log = (root / "docs/CHANGELOG").read_text(encoding="utf-8")
        log = re.sub(
            rf"^## \[{re.escape(VERSION)}\] - \S+$",
            f"## [{VERSION}] - {day}",
            log,
            count=1,
            flags=re.M,
        )
        (root / "docs/CHANGELOG").write_text(log, encoding="utf-8")
        cff = (root / "CITATION.cff").read_text(encoding="utf-8")
        cff = re.sub(r'date-released: "[^"]+"', f'date-released: "{day}"', cff)
        (root / "CITATION.cff").write_text(cff, encoding="utf-8")

    evening = "2026-10-04T22:30:00-03:00"
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        dated(root, "2026-10-05")
        git_repo(root, tag_date=evening)
        assert preflight.check_tag(f"v{VERSION}", root) == []
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        dated(root, "2026-10-06")
        git_repo(root, tag_date=evening)
        errors = preflight.check_tag(f"v{VERSION}", root)
        assert errors == ["release date 2026-10-06 is after the tag date 2026-10-05"], errors


def test_the_tag_commit_is_what_is_checked() -> None:
    # the tag points at a commit with a stale CITATION; main and the checkout
    # carry the fix afterwards. the files of the tag's commit decide.
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        edit(root, "CITATION.cff", f'version: "{VERSION}"', 'version: "0.0.1"')
        git_repo(root)
        edit(root, "CITATION.cff", 'version: "0.0.1"', f'version: "{VERSION}"')
        subprocess.run(GIT + ["commit", "-q", "-am", "fix"], cwd=root, check=True)
        subprocess.run(
            GIT + ["update-ref", "refs/remotes/origin/main", "HEAD"], cwd=root, check=True
        )
        assert preflight.check_tree(root) == []
        errors = preflight.check_tag(f"v{VERSION}", root)
        assert errors == [f"v{VERSION}: CITATION.cff: version is '0.0.1', not {VERSION!r}"], errors
        assert preflight.main(["--root", str(root), "--tag", f"v{VERSION}"]) == 1
    # the other way round: a broken checkout does not fail a sound tag.
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        git_repo(root)
        edit(
            root,
            "Cargo.lock",
            f'name = "urna"\nversion = "{VERSION}"',
            'name = "urna"\nversion = "0.0.1"',
        )
        assert preflight.check_tree(root) != []
        assert preflight.check_tag(f"v{VERSION}", root) == []
    # a release file the tag's commit does not carry is named, not a traceback.
    with tempfile.TemporaryDirectory() as tmp:
        root = copy_tree(Path(tmp))
        (root / "CITATION.cff").unlink()
        git_repo(root)
        errors = preflight.check_tag(f"v{VERSION}", root)
        assert errors[0].startswith(f"v{VERSION}: missing release file: CITATION.cff is not in"), (
            errors
        )


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"preflight: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
