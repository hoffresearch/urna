"""Prove the fullcheck step cache skips a step only when nothing it reads moved.

`tool/tasks/stepcache.py` keys each fullcheck step by a hash of its inputs.
this suite runs it against throwaway git repos:

- happy path: the same tree gives the same key; a step marked with a key
  is a hit for that key and a miss for any other;
- error path: editing a tracked file (committed or not), adding an
  untracked one, deleting a tracked one, changing a --file or an --extra
  each change the key; a deleted or rewritten --output turns a hit into a
  miss; a corrupt stamp is a miss; an output kept beside the stamp (named
  like it) is never overwritten by it;
- edge case: an ignored file and a file outside the input paths leave the
  key alone; no input paths hash no files (git would list the whole tree).

Run: python tool/tests/test_stepcache.py
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tool" / "tasks" / "stepcache.py"
spec = importlib.util.spec_from_file_location("stepcache", SCRIPT)
stepcache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stepcache)

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix="urna-stepcache-"))
    files = {
        "rust/format/src/lib.rs": "pub fn a() {}\n",
        "rust/engine/src/lib.rs": "pub fn b() {}\n",
        "docs/USAGE.md": "# usage\n",
        ".gitignore": "/target\n*.so\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    subprocess.run(GIT + ["init", "-q"], cwd=root, check=True)
    subprocess.run(GIT + ["add", "-A"], cwd=root, check=True)
    subprocess.run(GIT + ["commit", "-qm", "t"], cwd=root, check=True)
    return root


def key(root: Path, paths=("rust",), files=(), extras=("rustc 1",)) -> str:
    return stepcache.key(list(paths), list(files), list(extras), root)


def test_same_tree_same_key():
    root = tree()
    assert key(root) == key(root)


def test_input_changes_move_the_key():
    root = tree()
    before = key(root)
    lib = root / "rust/format/src/lib.rs"
    lib.write_text("pub fn a() { 1; }\n")  # uncommitted edit
    edited = key(root)
    assert edited != before
    subprocess.run(GIT + ["commit", "-qam", "edit"], cwd=root, check=True)
    assert key(root) == edited, "committing an edit changes no content"
    (root / "rust/engine/src/new.rs").write_text("")  # untracked, not ignored
    added = key(root)
    assert added != edited
    (root / "rust/engine/src/lib.rs").unlink()  # tracked, deleted
    assert key(root) != added


def test_ignored_and_outside_files_leave_the_key():
    root = tree()
    before = key(root)
    (root / "rust/format/_ext.so").write_text("binary")
    (root / "target").mkdir()
    (root / "target/out").write_text("x")
    (root / "docs/USAGE.md").write_text("# changed\n")
    assert key(root) == before
    assert key(root, paths=(".",)) != key(tree(), paths=(".",))


def test_files_and_extras_are_in_the_key():
    root = tree()
    corpus = root.parent / f"{root.name}-corpus.urna"
    corpus.write_bytes(b"one")
    with_file = key(root, files=(corpus,))
    corpus.write_bytes(b"two")
    assert key(root, files=(corpus,)) != with_file
    assert key(root, extras=("rustc 2",)) != key(root)
    corpus.unlink()
    assert key(root, files=(corpus,)) != with_file, "a missing file is a change"


def test_no_paths_hash_no_files():
    root = tree()
    assert stepcache.tree_files([], root) == []
    assert stepcache.tree_files(["rust/format"], root) == ["rust/format/src/lib.rs"]


def test_hit_needs_the_key_and_the_outputs():
    root = tree()
    stamps = root / "target/fullcheck"
    out = root / "target/presetrun.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{}")
    assert not stepcache.hit("presetrun", "k1", [str(out)], stamps)
    stepcache.mark("presetrun", "k1", [str(out)], stamps)
    assert stepcache.hit("presetrun", "k1", [str(out)], stamps)
    assert not stepcache.hit("presetrun", "k2", [str(out)], stamps)
    out.write_text('{"other": 1}')
    assert not stepcache.hit("presetrun", "k1", [str(out)], stamps), "rewritten output"
    stepcache.mark("presetrun", "k1", [str(out)], stamps)
    out.unlink()
    assert not stepcache.hit("presetrun", "k1", [str(out)], stamps), "deleted output"
    stepcache.stamp_path("presetrun", stamps).write_text("{not json")
    assert not stepcache.hit("presetrun", "k1", [], stamps), "corrupt stamp"


def test_an_output_beside_the_stamp_survives_the_mark():
    root = tree()
    stamps = root / "target/fullcheck"
    stamps.mkdir(parents=True)
    out = stamps / "presetrun.json"
    out.write_text('{"presets": []}')
    stepcache.mark("presetrun", "k1", [str(out)], stamps)
    assert out.read_text() == '{"presets": []}'
    assert stepcache.hit("presetrun", "k1", [str(out)], stamps)


def test_cli_round_trip():
    root = tree()
    stamps = root / "target/fullcheck"
    run = [sys.executable, str(SCRIPT)]
    k = subprocess.run(
        run + ["key", "--extra", "x", "rust"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert len(k) == 64, k
    # the cli keys the checkout it lives in, so only hit/mark are exercised
    # against `stamps`, through the module (their storage is the same code).
    stepcache.mark("names", k, [], stamps)
    assert stepcache.hit("names", k, [], stamps)


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"stepcache: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
