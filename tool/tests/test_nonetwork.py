"""Guard test: the sentence-transformers entrypoints force HuggingFace
OFFLINE by default, and honor URNA_ALLOW_DOWNLOAD=1 as the explicit opt-in
(audit findings S5 / P1).

Importing `searchtxt`, the model registry `presetmap` every other model load
goes through, and the image path (`visionemb`, which loads open_clip outside
the registry, and the two entry points that import it) must set
HF_HUB_OFFLINE=1 before any hub access, so a hostile/misconfigured corpus
model name can never trigger a download mid-run
(e.g. while the box is handling PHI). Runs in a subprocess with a clean env so
the module-load-time guard is observed in isolation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PYDIR = str(Path(__file__).resolve().parents[2] / "rust" / "bridge" / "python")
SNIPPET = (
    "import os, sys; sys.path.insert(0, os.environ['PYDIR']); "
    "import {module}; print(os.environ.get('HF_HUB_OFFLINE'))"
)


def _run(extra_env: dict, module: str = "urna.embed.searchtxt") -> str:
    env = dict(os.environ)
    env["PYDIR"] = PYDIR
    # clean slate: the guard uses setdefault, so a pre-set value would mask it.
    _forced = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    for k in (*_forced, "URNA_ALLOW_DOWNLOAD"):
        env.pop(k, None)
    env.update(extra_env)
    code = SNIPPET.format(module=module)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
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


def test_registry_forces_offline_by_default() -> None:
    assert _run({}, "urna.model.presetmap") == "1", (
        "importing urna.model.presetmap must force HF_HUB_OFFLINE=1 by default"
    )
    print("registry: offline forced by default: ok")


def test_registry_honors_the_opt_in() -> None:
    assert _run({"URNA_ALLOW_DOWNLOAD": "1"}, "urna.model.presetmap") == "None", (
        "URNA_ALLOW_DOWNLOAD=1 must NOT force offline through the registry either"
    )
    print("registry: URNA_ALLOW_DOWNLOAD opt-in respected: ok")


IMAGE_PATH = ("urna.embed.visionemb", "urna.entry.imgcorpus", "urna.entry.imgsearch")


def test_image_path_forces_offline_by_default() -> None:
    for module in IMAGE_PATH:
        assert _run({}, module) == "1", f"importing {module} must force HF_HUB_OFFLINE=1"
    print("image path: offline forced by default: ok")


def test_image_path_honors_the_opt_in() -> None:
    for module in IMAGE_PATH:
        assert _run({"URNA_ALLOW_DOWNLOAD": "1"}, module) == "None", (
            f"URNA_ALLOW_DOWNLOAD=1 must NOT force offline through {module}"
        )
    print("image path: URNA_ALLOW_DOWNLOAD opt-in respected: ok")


GUARD = (
    "import os, sys, socket, urllib.request; sys.path.insert(0, os.environ['PYDIR'])\n"
    "from urna.model import presetmap\n"
    "presetmap.refuse_connections()\n"
    "a, b = socket.socketpair(); a.send(b'x'); assert b.recv(1) == b'x'\n"
    "try:\n"
    "    urllib.request.urlopen('http://127.0.0.1:9/', timeout=2)\n"
    "except OSError as e:\n"
    "    print(e)\n"
)


def test_the_embedder_refuses_connections() -> None:
    """Reviewed model code can open a socket of its own (jina v5 omni fetches
    any http(s) query as media): the query embedders refuse every network
    connection unless the caller opted in; local sockets keep working."""
    env = {k: v for k, v in os.environ.items() if k != "URNA_ALLOW_DOWNLOAD"}
    env["PYDIR"] = PYDIR
    run = subprocess.run([sys.executable, "-c", GUARD], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert "runs offline" in run.stdout, run.stdout
    env["URNA_ALLOW_DOWNLOAD"] = "1"
    run = subprocess.run([sys.executable, "-c", GUARD], capture_output=True, text=True, env=env)
    assert run.returncode == 0 and "runs offline" not in run.stdout, run.stdout
    print("embedder: connections refused by default, the opt-in lifts it: ok")


if __name__ == "__main__":
    test_offline_forced_by_default()
    test_opt_in_download_disables_force()
    test_registry_forces_offline_by_default()
    test_registry_honors_the_opt_in()
    test_image_path_forces_offline_by_default()
    test_image_path_honors_the_opt_in()
    test_the_embedder_refuses_connections()
    print("offline guard tests OK")
