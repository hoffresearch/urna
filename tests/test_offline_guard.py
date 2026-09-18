"""Guard test: the sentence-transformers entrypoints force HuggingFace
OFFLINE by default, and honor URNA_ALLOW_DOWNLOAD=1 as the explicit opt-in
(audit findings S5 / P1).

Importing `embed_query` must set HF_HUB_OFFLINE=1 before any hub access, so a
hostile/misconfigured corpus model name can never trigger a download mid-run
(e.g. while the box is handling PHI). Runs in a subprocess with a clean env so
the module-load-time guard is observed in isolation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PYDIR = str(Path(__file__).resolve().parent.parent / "python")
SNIPPET = (
    "import os, sys; sys.path.insert(0, os.environ['PYDIR']); "
    "import embed_query; print(os.environ.get('HF_HUB_OFFLINE'))"
)


def _run(extra_env: dict) -> str:
    env = dict(os.environ)
    env["PYDIR"] = PYDIR
    # clean slate: the guard uses setdefault, so a pre-set value would mask it.
    _forced = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    for k in (*_forced, "URNA_ALLOW_DOWNLOAD"):
        env.pop(k, None)
    env.update(extra_env)
    proc = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_offline_forced_by_default() -> None:
    assert _run({}) == "1", "HF_HUB_OFFLINE must be forced to 1 by default"
    print("offline forced by default: ok")


def test_opt_in_download_disables_force() -> None:
    assert _run({"URNA_ALLOW_DOWNLOAD": "1"}) == "None", (
        "URNA_ALLOW_DOWNLOAD=1 must NOT force offline (explicit opt-in)"
    )
    print("URNA_ALLOW_DOWNLOAD opt-in respected: ok")


if __name__ == "__main__":
    test_offline_forced_by_default()
    test_opt_in_download_disables_force()
    print("offline guard tests OK")
