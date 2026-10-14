"""Prove the name check holds the tree to the length table and the lexicon.

`tool/tasks/namecheck.py` runs in fullcheck.sh and gatecheck.yml. this suite
runs it against the real checkout, then against throwaway git repos built
from a small tree, one drift per case:

- happy path: the checkout passes; a minimal tree with the right lengths,
  the exceptions and a lexicon passes;
- error path: a root folder, a module, a test and a workflow of the wrong
  length; a project doc in lowercase; an abbreviation listed twice; a
  lexicon entry citing a name that is not in the tree or does not contain it;
- edge case: hyphens, underscores, extensions, the leading dot and the
  `test_` prefix do not count; a name fixed by a tool is exempt only at the
  path its tool puts it (`SKILL.md` in a skill, `release.yml` in workflows),
  never by name elsewhere; untracked files are out of scope.

Run: python tool/tests/test_namecheck.py
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tool" / "tasks" / "namecheck.py"
spec = importlib.util.spec_from_file_location("namecheck", SCRIPT)
namecheck = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = namecheck  # dataclasses look the module up
spec.loader.exec_module(namecheck)

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
LEXICON = """# Terms

| Abbreviation | Meaning | Names |
| --- | --- | --- |
| `qry` | query | `potionqry.py` |
| `cmp` | compare | `vectorcmp.py` |
"""
BASE = {
    "rust/bridge/Cargo.toml": "",
    "rust/bridge/python/urna/__init__.py": "",
    "rust/bridge/python/urna/embed/__init__.py": "",
    "rust/bridge/python/urna/embed/potionqry.py": "",
    "tool/bench/vectorcmp.py": "",
    "tool/tests/test_potiontab.py": "",
    "docs/TERMS.md": LEXICON,
    "docs/CONTRIBUTING.md": "",
    ".github/workflows/gatecheck.yml": "",
    ".github/workflows/release.yml": "",
    "pkgs/choco/urna.nuspec": "",
    "demo/x.json": "",
}


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="urna-namecheck-"))
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    subprocess.run(GIT + ["init", "-q"], cwd=root, check=True)
    subprocess.run(GIT + ["add", "-A"], cwd=root, check=True)
    return root


def problems(extra: dict[str, str] | None = None, drop: tuple[str, ...] = ()) -> list[str]:
    files = {k: v for k, v in BASE.items() if k not in drop}
    files.update(extra or {})
    root = tree(files)
    paths = namecheck.tracked(root)
    out = namecheck.check_lengths(paths)
    return out + namecheck.check_lexicon(paths, (root / "docs/TERMS.md").read_text())


def test_checkout_passes():
    assert namecheck.main(REPO) == 0


def test_minimal_tree_passes():
    assert problems() == [], problems()


def test_measure_counts_letters_and_digits_only():
    assert namecheck.measure("av1stream.py", "files") == 9
    assert namecheck.measure("potionb8m", "dirs") == 9
    assert namecheck.measure(".github", "hidden") == 6
    assert namecheck.measure("test_namecheck.py", "files", "test_") == 9
    assert namecheck.measure("potion-base-8M", "dirs") == 12
    assert namecheck.measure("spec_rule.py", "files") == 8


def test_wrong_lengths_are_named():
    found = problems(
        {
            "scripts/x.sh": "",
            "rust/bridge/python/urna/embed/query.py": "",
            "tool/tests/test_spec.py": "",
            ".github/workflows/ci.yml": "",
        }
    )
    for path in (
        "scripts:",
        "embed/query.py:",
        "tool/tests/test_spec.py:",
        ".github/workflows/ci.yml:",
    ):
        assert any(path in p for p in found), (path, found)
    assert len(found) == 4, found


def test_project_docs_are_uppercase():
    found = problems({"docs/usage.md": ""})
    assert found == ["docs/usage.md: files of docs/ are uppercase"], found


def test_exceptions_hold_only_where_they_belong():
    fixed = {
        "rust/bridge/build.rs": "",
        "pkgs/linux/aur/PKGBUILD": "",
        "rust/bridge/python/urna/embed/__init__.py": "",
        ".github/pull_request_template.md": "",
        ".github/zizmor.yml": "",
        "docs/CHANGELOG": "",
    }
    assert problems(fixed) == [], problems(fixed)
    # the same names anywhere else are held to the rule: an exception is a
    # place, not a name.
    found = problems(
        {
            "tool/tasks/recipe.yaml": "",
            "tool/bench/__init__.py": "",
            "tool/tasks/SKILL.md": "",
            ".github/release.yml": "",
        }
    )
    assert sorted(found) == [
        ".github/release.yml: 7 characters, files of .github/ have 9",
        "tool/bench/__init__.py: 4 characters, files of tool/bench/ have 9",
        "tool/tasks/SKILL.md: 5 characters, files of tool/tasks/ have 9",
        "tool/tasks/SKILL.md: files of tool/tasks/ are lowercase",
        "tool/tasks/recipe.yaml: 6 characters, files of tool/tasks/ have 9",
    ], found


def test_untracked_files_are_out_of_scope():
    root = tree(BASE)
    (root / "target").mkdir()
    (root / "target" / "debug.rs").write_text("")
    assert namecheck.check_lengths(namecheck.tracked(root)) == []


def test_lexicon_rejects_a_second_meaning():
    twice = LEXICON + "| `cmp` | compile | `vectorcmp.py` |\n"
    found = problems({"docs/TERMS.md": twice})
    assert found == ["docs/TERMS.md: `cmp` listed twice (compare; compile)"], found


def test_lexicon_cites_real_names_that_carry_it():
    ghost = LEXICON + "| `emb` | embedding | `visionemb.py` |\n"
    assert problems({"docs/TERMS.md": ghost}) == [
        "docs/TERMS.md: `emb` cites `visionemb.py`, which is not in the tree"
    ]
    wrong = LEXICON + "| `img` | image | `vectorcmp.py` |\n"
    assert problems({"docs/TERMS.md": wrong}) == [
        "docs/TERMS.md: `img` cites `vectorcmp.py`, which does not contain it"
    ]


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"namecheck: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
