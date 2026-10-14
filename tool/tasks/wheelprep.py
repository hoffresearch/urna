"""stage the maturin wheel project under pkgs/stage/.

the repo keeps the `urna` package at rust/bridge/python/urna/ with the
extension copied in as _urna.so; the published wheel ships only the public
surface of that package. this script copies it into pkgs/stage/ so
`maturin build` there produces the official `urna` wheel without touching
the dev flow.

contents staged (PKG = rust/bridge/python/urna):
  stage/pyproject.toml              <- pkgs/wheel/pyproject.toml (verbatim;
                                       its paths resolve from stage/)
  stage/README.md                   <- README.md
  stage/LICENSE                     <- LICENSE (license-files in the
                                       pyproject; the wheel ships the text)
  stage/urna/__init__.py            <- PKG/__init__.py
  stage/urna/entry/clidriver.py     <- PKG/entry/clidriver.py (the console)
  stage/urna/embed/potiontab.py     <- PKG/embed/potiontab.py
  stage/urna/embed/lexifloor.py     <- PKG/embed/lexifloor.py
  stage/urna/model/potionb8m/       <- PKG/model/potionb8m/
  (and the __init__.py of entry/, embed/ and model/)

the potion table (~30 MB) is bundled on purpose: the installed package must
embed offline by construction, so no lazy fetch path exists. git-lfs pointer
files are rejected; run `git lfs pull` before staging.

run:  python tool/tasks/wheelprep.py
then: cd pkgs/stage && maturin build --release
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "rust" / "bridge" / "python" / "urna"
STAGING = ROOT / "pkgs" / "stage"

# the offline potion embedder is self-contained (stdlib + numpy + tokenizers)
# and resolves its table relative to __file__ (../model/potionb8m), so the
# package keeps the repo's layout. embed/__init__.py re-exports it with the
# lexical floor.
MODULES = [
    "__init__.py",
    "entry/__init__.py",
    "entry/clidriver.py",
    "embed/__init__.py",
    "embed/lexifloor.py",
    "embed/potiontab.py",
    "model/__init__.py",
]
COPIES = [
    (ROOT / "pkgs" / "wheel" / "pyproject.toml", STAGING / "pyproject.toml"),
    (ROOT / "README.md", STAGING / "README.md"),
    (ROOT / "LICENSE", STAGING / "LICENSE"),
] + [(PKG / m, STAGING / "urna" / m) for m in MODULES]

MODEL_SRC = PKG / "model" / "potionb8m"
MODEL_DST = STAGING / "urna" / "model" / "potionb8m"


def fail(msg: str) -> None:
    print(f"wheelprep: error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_not_lfs_pointer(path: Path) -> None:
    with path.open("rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        fail(f"{path} is a git-lfs pointer; run `git lfs pull` first")


def main() -> None:
    if not MODEL_SRC.is_dir():
        fail(f"missing model dir: {MODEL_SRC}")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    for src, dst in COPIES:
        if not src.is_file():
            fail(f"missing source: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.copytree(MODEL_SRC, MODEL_DST)
    for f in sorted(MODEL_DST.rglob("*")):
        if f.is_file():
            check_not_lfs_pointer(f)
    total = sum(f.stat().st_size for f in STAGING.rglob("*") if f.is_file())
    print(f"staged wheel project at {STAGING} ({total / 1e6:.1f} MB)")
    print("next: cd pkgs/stage && maturin build --release")


if __name__ == "__main__":
    main()
