"""Prove the workspaces outside the root keep its lint policy.

rust/ingest (the root excludes it) and fuzz/ (cargo-fuzz needs nightly)
are cargo workspaces of their own, so they inherit nothing: their `[lints]`
tables are a copy of the root's `[workspace.lints]`, and rust/ingest's
`rust-version` a copy of the root's (fuzz/ runs on nightly and pins none).
a copy drifts silently; this suite makes the drift a failure:

- happy path: every root lint is in each copy at the same level and
  priority, and rust/ingest's rust-version agrees with the root's;
- error path: a lint missing from a copy, or at another level, is named
  with the copy it is missing from;
- edge case: a copy may add a lint the root cannot have (rust/ingest's
  `unsafe_code = "forbid"`: the engine's SIMD keeps the root from it).

Run: python tool/tests/test_lintmatch.py
"""

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "Cargo.toml"
INGEST = REPO / "rust" / "ingest" / "Cargo.toml"
COPIES = {"rust/ingest": INGEST, "fuzz": REPO / "fuzz" / "Cargo.toml"}


def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def drift(root: dict, copy: dict, where: str = "rust/ingest") -> list[str]:
    """what the copy lacks or sets differently, one line per lint."""
    found = []
    for tool, lints in root.items():
        theirs = copy.get(tool, {})
        for name, level in lints.items():
            if name not in theirs:
                found.append(f"{tool}::{name} missing from {where}")
            elif theirs[name] != level:
                found.append(f"{tool}::{name} is {theirs[name]!r}, the root has {level!r}")
    return found


def test_every_copy_has_the_root_lints():
    root = load(ROOT)["workspace"]["lints"]
    for where, path in COPIES.items():
        found = drift(root, load(path).get("lints", {}), where)
        assert found == [], found


def test_ingest_keeps_the_root_rust_version():
    root = load(ROOT)["workspace"]["package"]["rust-version"]
    ingest = load(INGEST)["package"]["rust-version"]
    assert ingest == root, (ingest, root)


def test_a_drift_is_named():
    root = {"clippy": {"todo": "deny", "expect_used": "deny"}}
    copy = {"clippy": {"todo": "warn"}}
    assert drift(root, copy, "fuzz") == [
        "clippy::todo is 'warn', the root has 'deny'",
        "clippy::expect_used missing from fuzz",
    ], drift(root, copy, "fuzz")


def test_ingest_may_be_stricter():
    idioms = {"level": "deny", "priority": -1}
    root = {"rust": {"rust_2018_idioms": idioms}}
    copy = {"rust": {"rust_2018_idioms": dict(idioms), "unsafe_code": "forbid"}}
    assert drift(root, copy) == [], drift(root, copy)


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"lintmatch: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
