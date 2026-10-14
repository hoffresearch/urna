"""Prove the siglip2 query path offline against its pinned hub snapshot.

Each case runs rust/bridge/python/urna/embed/presetqry.py in a child process with the
hub offline and HF_HOME pointed at a temporary cache that holds only
`snapshots/<revision>` (links to the real files): no refs/main and no
.no_exist markers, so nothing but the pinned files can be read.

  1. every file present: exit 0, and the model_hash is the one the files
     fingerprinted to when the mtg stills-5models corpus was built, so
     loading from the snapshot keeps the identity of a load by tag.
  2. tokenizer.json missing: exit 3, the file named, the urna-fetch line,
     no traceback and no hub call.

Needs torch, open_clip, transformers and the pinned snapshot in the local hf
cache (hf download timm/ViT-B-16-SigLIP2 <files> --revision <pin>); skips
without them.

Run: .venv/bin/python tool/tests/test_clipsnaps.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = Path(__file__).resolve().parents[2] / "rust" / "bridge" / "python"
sys.path.insert(0, str(PYTHON))

from urna.model import presetmap as mr  # noqa: E402

SCRIPT = PYTHON / "urna" / "embed" / "presetqry.py"
PRESET = mr.PRESETS["siglip2"]
# model_hash of the siglip2 space in the mtg-urna-benchmark release
# v0.3/stills-5models (built 2026-09-12 to 14 with the load by tag)
BUILT_HASH = "sha256:a9946db1d33627e773569c415ab202cd759befcdc047ac0d8768937298775f3c"


def _real_snapshot() -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snap = (
        hf_home
        / "hub"
        / f"models--{PRESET.hf_repo.replace('/', '--')}"
        / "snapshots"
        / PRESET.revision
    )
    return snap if all((snap / f).is_file() for f in PRESET.snapshot_files) else None


def _pinned_only_cache(tmp: str, real: Path, skip: str | None = None) -> None:
    cache = Path(tmp) / "hub" / f"models--{PRESET.hf_repo.replace('/', '--')}"
    snap = cache / "snapshots" / PRESET.revision
    snap.mkdir(parents=True)
    for name in PRESET.snapshot_files:
        if name != skip:
            (snap / name).symlink_to((real / name).resolve())


def _query(hf_home: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "URNA_ALLOW_DOWNLOAD"}
    env.update(HF_HOME=hf_home, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--preset",
            "siglip2",
            PRESET.embedding_model,
            "a red dragon",
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def main() -> None:
    missing = [m for m, _ in PRESET.requires if importlib.util.find_spec(m) is None]
    real = _real_snapshot()
    if missing or real is None:
        why = f"missing {', '.join(missing)}" if missing else "pinned snapshot not cached"
        print(f"test_clipsnaps: SKIP ({why})")
        return

    with tempfile.TemporaryDirectory() as tmp:
        _pinned_only_cache(tmp, real)
        p = _query(tmp)
        assert p.returncode == 0, p.stderr[-2000:]
        out = json.loads(p.stdout)
        assert out["model_hash"] == BUILT_HASH, out["model_hash"]
        assert out["embedding_dim"] == 768 and len(out["vector"]) == 768
        norm = sum(x * x for x in out["vector"])
        assert abs(norm - 1.0) < 1e-4, norm
    print("case 1 (pinned snapshot only, offline: exit 0, the build's model_hash): OK")

    with tempfile.TemporaryDirectory() as tmp:
        _pinned_only_cache(tmp, real, skip="tokenizer.json")
        p = _query(tmp)
        assert p.returncode == 3, (p.returncode, p.stderr[-2000:])
        assert "tokenizer.json missing" in p.stderr, p.stderr
        assert f"urna-fetch: {PRESET.hf_repo}" in p.stderr, p.stderr
        assert "Traceback" not in p.stderr, p.stderr
    print("case 2 (tokenizer.json missing: exit 3, file named, urna-fetch): OK")


if __name__ == "__main__":
    main()
