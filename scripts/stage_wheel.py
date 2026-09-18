"""stage the maturin wheel project under packaging/staging/.

the repo keeps a dev layout (python/urna.py + python/_urna.so side by side);
the published wheel needs a package layout (urna/__init__.py + urna._urna).
this script copies the public surface into packaging/staging/ so
`maturin build` there produces the official `urna` wheel without touching
the dev flow.

contents staged:
  staging/pyproject.toml               <- packaging/pyproject.toml (verbatim;
                                          its paths resolve from staging/)
  staging/README.md                    <- README.md
  staging/urna/__init__.py             <- python/urna.py
  staging/urna/models/potion-base-8M/  <- python/forge/models/potion-base-8M/

the potion table (~30 MB) is bundled on purpose: the installed package must
embed offline by construction, so no lazy fetch path exists. git-lfs pointer
files are rejected; run `git lfs pull` before staging.

run:  python scripts/stage_wheel.py
then: cd packaging/staging && maturin build --release
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "packaging" / "staging"

COPIES = [
    (ROOT / "packaging" / "pyproject.toml", STAGING / "pyproject.toml"),
    (ROOT / "README.md", STAGING / "README.md"),
    (ROOT / "python" / "urna.py", STAGING / "urna" / "__init__.py"),
    (ROOT / "python" / "urna_cli.py", STAGING / "urna" / "_cli.py"),
    # the offline potion embedder is self-contained (stdlib + numpy +
    # tokenizers) and resolves its model dir relative to __file__, so it
    # drops into the package unchanged next to the bundled table.
    (ROOT / "python" / "forge" / "embed_potion.py", STAGING / "urna" / "embed_potion.py"),
]

MODEL_SRC = ROOT / "python" / "forge" / "models" / "potion-base-8M"
MODEL_DST = STAGING / "urna" / "models" / "potion-base-8M"


def fail(msg: str) -> None:
    print(f"stage_wheel: error: {msg}", file=sys.stderr)
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
    print("next: cd packaging/staging && maturin build --release")


if __name__ == "__main__":
    main()
