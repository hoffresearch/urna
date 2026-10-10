"""check the naming convention of docs/adr/structure/0003 against the tree.

one rule: siblings of the same kind have the same length. LEVELS below says,
for each folder, which children are checked and how long they must be, so
no level is guessed and none is skipped. a length counts letters and digits
only: no hyphen, no underscore, no extension, no leading dot of a hidden
folder and no `test_` prefix (`av1stream.py` and `potionb8m/` are 9).

the check reads the tracked files (`git ls-files`), so generated and ignored
folders (target/, .venv/, pkgs/stage/) are out of its scope. a name fixed by
a tool, a distribution channel or a project convention is an exception and
is listed in EXCEPTIONS; nothing else is.

it also checks the abbreviation lexicon in docs/TERMS.md: one abbreviation,
one meaning (no abbreviation listed twice), and every name an entry cites
exists in the tree and contains the abbreviation.

run:  python tool/tasks/namecheck.py        (exit 1 and one line per problem)
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
TERMS = "docs/TERMS.md"


@dataclass(frozen=True)
class Level:
    parent: str  # folder, relative to the root ("" is the root); * matches one folder
    kind: str  # "dirs", "hidden" (dirs with a leading dot) or "files"
    length: int
    case: str = "lower"  # "lower" or "upper"
    prefix: str = ""  # a prefix that does not count (test_)


LEVELS = [
    Level("", "dirs", 4),
    Level("", "hidden", 6),
    Level("rust", "dirs", 6),
    Level("rust/bridge/python/urna", "dirs", 5),
    Level("rust/bridge/python/urna/*", "files", 9),
    Level("rust/bridge/python/urna/model", "dirs", 9),
    Level("tool", "dirs", 5),
    Level("tool/bench", "files", 9),
    Level("tool/tasks", "files", 9),
    Level("tool/tests", "files", 9, prefix="test_"),
    Level("demo", "dirs", 7),
    Level("demo/corpora", "dirs", 5),
    Level("docs", "dirs", 3),
    Level("docs", "files", 5, case="upper"),
    Level("docs/adr", "dirs", 9),
    Level("pkgs", "dirs", 5),
    Level("pkgs/linux", "dirs", 3),
    Level(".github", "files", 9),
    Level(".github/workflows", "files", 9),
]

# names outside the rule, each held to the place where it is fixed: a name
# is exempt only at the path its tool, channel or convention puts it, never
# by its name alone. levels the table does not check need no entry here.
EXCEPTIONS = [
    # tools
    (re.compile(r"^rust/bridge/python/urna/[a-z]+/__init__\.py$"), "python package marker"),
    (re.compile(r"^docs/(CHANGELOG|CODE_OF_CONDUCT\.md|CONTRIBUTING\.md|SECURITY\.md)$"), "GitHub"),
    (re.compile(r"^\.github/pull_request_template\.md$"), "GitHub"),
    (re.compile(r"^\.github/workflows/release\.yml$"), "cargo-dist"),
    (re.compile(r"^\.github/zizmor\.yml$"), "zizmor's config discovery"),
    (re.compile(r"^\.zed$"), "the Zed editor's project settings"),
]


def tracked(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\0") if p]


def measure(name: str, kind: str, prefix: str = "") -> int:
    """letters and digits of the name, without extension, leading dot or prefix."""
    stem = name[1:] if name.startswith(".") else name
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    if kind == "files":
        stem = stem.split(".", 1)[0]
    return sum(c.isalnum() for c in stem)


def children(paths: list[str]) -> dict[str, tuple[set[str], set[str]]]:
    """folder -> (child folders, child files), for every folder of the tree."""
    tree: dict[str, tuple[set[str], set[str]]] = {}
    for p in paths:
        parts = PurePosixPath(p).parts
        for i in range(len(parts)):
            parent = "/".join(parts[:i])
            dirs, files = tree.setdefault(parent, (set(), set()))
            (files if i == len(parts) - 1 else dirs).add(parts[i])
    return tree


def matches(pattern: str, folder: str) -> bool:
    a, b = pattern.split("/") if pattern else [], folder.split("/") if folder else []
    return len(a) == len(b) and all(x in ("*", y) for x, y in zip(a, b, strict=True))


def is_exception(path: str) -> bool:
    return any(pattern.match(path) for pattern, _ in EXCEPTIONS)


def check_lengths(paths: list[str]) -> list[str]:
    tree = children(paths)
    problems = []
    for level in LEVELS:
        for folder, (dirs, files) in sorted(tree.items()):
            if not matches(level.parent, folder):
                continue
            if level.kind == "files":
                names = files
            elif level.kind == "hidden":
                names = {d for d in dirs if d.startswith(".")}
            else:
                names = {d for d in dirs if not d.startswith(".")}
            for name in sorted(names):
                path = f"{folder}/{name}" if folder else name
                if is_exception(path):
                    continue
                size = measure(name, level.kind, level.prefix)
                if size != level.length:
                    where = f"{level.parent or 'the root'}/"
                    problems.append(
                        f"{path}: {size} characters, {level.kind} of {where} have {level.length}"
                    )
                stem = name.split(".", 1)[0] if level.kind == "files" else name
                cased = stem.upper() if level.case == "upper" else stem.lower()
                if stem != cased:
                    where = f"{folder or 'the root'}/"
                    problems.append(f"{path}: {level.kind} of {where} are {level.case}case")
    return problems


ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$")


def lexicon(text: str) -> list[tuple[str, str, list[str]]]:
    """the rows of the lexicon table: (abbreviation, meaning, cited names)."""
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            names = re.findall(r"`([^`]+)`", m.group(3))
            rows.append((m.group(1), m.group(2), names))
    return rows


def check_lexicon(paths: list[str], text: str) -> list[str]:
    rows = lexicon(text)
    if not rows:
        return [f"{TERMS}: no lexicon table found"]
    problems = []
    seen: dict[str, str] = {}
    stems, where = set(), set()
    for p in paths:
        parts = PurePosixPath(p).parts
        for i, part in enumerate(parts):
            stems.add(part.lstrip(".").split(".", 1)[0].removeprefix("test_"))
            where.add("/".join(parts[: i + 1]))
    for abbr, meaning, names in rows:
        if abbr in seen:
            problems.append(f"{TERMS}: `{abbr}` listed twice ({seen[abbr]}; {meaning})")
        seen[abbr] = meaning
        if not names:
            problems.append(f"{TERMS}: `{abbr}` cites no name that uses it")
        for name in names:
            # a name with a folder in front (`docs/adr/`) must be that path.
            path = name.rstrip("/")
            last = PurePosixPath(path).name
            stem = last.lstrip(".").split(".", 1)[0].removeprefix("test_")
            if stem not in stems or ("/" in path and path not in where):
                problems.append(f"{TERMS}: `{abbr}` cites `{name}`, which is not in the tree")
            elif abbr.lower() not in stem.lower():
                problems.append(f"{TERMS}: `{abbr}` cites `{name}`, which does not contain it")
    return problems


def main(root: Path = ROOT) -> int:
    paths = tracked(root)
    terms = root / TERMS
    problems = check_lengths(paths)
    if terms.is_file():
        problems += check_lexicon(paths, terms.read_text(encoding="utf-8"))
    else:
        problems.append(f"{TERMS}: missing")
    for line in problems:
        print(f"namecheck: {line}", file=sys.stderr)
    if problems:
        return 1
    print(f"namecheck: {len(paths)} tracked files follow the length table and the lexicon")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
