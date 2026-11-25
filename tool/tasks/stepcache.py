"""remember which fullcheck steps passed, keyed by a hash of their inputs.

a step that passed with the same inputs passes again, so `fullcheck.sh`
skips it: the principle of bazel and turborepo, by content hash, never by
modification time. the key of a step is the sha-256 of:

- every file under its input paths that git sees (tracked, plus untracked
  ones not ignored), each by path and content, so an edit not yet
  committed counts and a generated file under target/ does not. an input
  path under which git sees no file is an error, never an empty hash: a
  renamed or mistyped path would give a key that never moves;
- any file outside the tree it names with --file (the bench corpus);
- the --extra strings: the toolchain versions and the step's own text.

a stamp records the key and the sha-256 of each --output, so a step whose
product was deleted or rebuilt by something else runs again. a step whose
run skipped a test for a missing need (tool/tests/testneeds.py: the lines
the skip log gained past --skip-from) is not recorded, and an older stamp
of it is dropped: a pass that left something unchecked is not a pass to
remember.

    key  [--file PATH]... [--extra TEXT]... [PATH...]   print the key
    hit  NAME KEY [--output PATH]...   exit 0 if NAME passed with KEY
    mark NAME KEY [--output PATH]... [--skip-log PATH --skip-from BYTES]
         record that NAME passed with KEY, unless the skip log grew

stamps live in target/fullcheck/ (ignored; `cargo clean` drops them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAMPS = ROOT / "target" / "fullcheck"


def file_digest(path: Path) -> str:
    """sha-256 of the content of `path`, or a marker when it does not exist
    (a tracked file deleted in the working tree is an input change too)."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while block := f.read(1 << 20):
                h.update(block)
    except FileNotFoundError:
        return "missing"
    return h.hexdigest()


class MissingInput(ValueError):
    """an input path under which git sees no file."""


def tree_files(paths: list[str], root: Path = ROOT) -> list[str]:
    """the files under `paths` that git sees: tracked and untracked, minus
    ignored ones, sorted. no paths means no files (git would list all); a
    path under which git sees none raises MissingInput."""
    files: set[str] = set()
    for path in paths:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", path],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        found = {p for p in out.decode().split("\0") if p}
        if not found:
            raise MissingInput(path)
        files |= found
    return sorted(files)


def key(paths: list[str], files: list[str], extras: list[str], root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for rel in tree_files(paths, root):
        h.update(f"tree\0{rel}\0{file_digest(root / rel)}\0".encode())
    for f in files:
        p = Path(f).resolve()
        h.update(f"file\0{p}\0{file_digest(p)}\0".encode())
    for e in extras:
        h.update(f"extra\0{len(e)}\0{e}\0".encode())
    return h.hexdigest()


def stamp_path(name: str, stamps: Path) -> Path:
    # its own suffix, so a step's output kept beside it (presetrun's
    # metrics, presetrun.json) is never the stamp.
    return stamps / f"{name}.stamp"


def outputs_of(outputs: list[str]) -> dict[str, str]:
    return {str(Path(o).resolve()): file_digest(Path(o)) for o in outputs}


def hit(name: str, k: str, outputs: list[str], stamps: Path = STAMPS) -> bool:
    try:
        stamp = json.loads(stamp_path(name, stamps).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    now = outputs_of(outputs)
    if "missing" in now.values():
        return False
    return stamp.get("key") == k and stamp.get("outputs") == now


def skipped(log: Path | None, since: int = 0) -> list[str]:
    """the lines the skip log gained past byte `since`: the tests the step
    skipped, as `need\ttest\treason`."""
    if log is None:
        return []
    try:
        data = log.read_bytes()[since:]
    except FileNotFoundError:
        return []
    return [line for line in data.decode(errors="replace").splitlines() if line]


def mark(
    name: str, k: str, outputs: list[str], stamps: Path = STAMPS, skips: list[str] | None = None
) -> bool:
    """record that `name` passed with `k`; with `skips`, drop its stamp
    instead and return False."""
    if skips:
        stamp_path(name, stamps).unlink(missing_ok=True)
        return False
    stamps.mkdir(parents=True, exist_ok=True)
    tmp = stamp_path(name, stamps).with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": k, "outputs": outputs_of(outputs)}, indent=2) + "\n")
    tmp.replace(stamp_path(name, stamps))
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("key")
    k.add_argument("--file", action="append", default=[])
    k.add_argument("--extra", action="append", default=[])
    k.add_argument("paths", nargs="*")
    for cmd in ("hit", "mark"):
        s = sub.add_parser(cmd)
        s.add_argument("name")
        s.add_argument("key")
        s.add_argument("--output", action="append", default=[])
        if cmd == "mark":
            s.add_argument("--skip-log", type=Path)
            s.add_argument("--skip-from", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "key":
        try:
            print(key(args.paths, args.file, args.extra))
        except MissingInput as e:
            print(f"stepcache: no file under the input path {e}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "hit":
        return 0 if hit(args.name, args.key, args.output) else 1
    skips = skipped(args.skip_log, args.skip_from)
    if not mark(args.name, args.key, args.output, skips=skips):
        needs = sorted({line.split("\t", 1)[0] for line in skips})
        print(
            f"  NOT CACHED: {args.name} skipped tests for {', '.join(needs)}; it runs again",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
