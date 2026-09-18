"""End-to-end test for the search-text model_hash gate.

Builds three corpora with different model_hashes and shells out to a
fake embedder that reports a fixed model_hash. Asserts:

  1. Match: corpus.model_hash == embedder.model_hash → search succeeds.
  2. Mismatch: corpus.model_hash != embedder.model_hash → CLI bails with
     a typed error containing "model_hash mismatch".
  3. Placeholder: corpus.model_hash == sha256:0...0 → CLI bails with
     "legacy placeholder" hint, even when the embedder reports the
     same placeholder. Caller must opt-in via --skip-model-hash-check.

Doesn't require real sentence-transformers — the fake embedder is a
~20-line Python script that produces a deterministic unit vector.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))
import urna  # noqa: E402

# Path to the release CLI binary.
CLI = REPO / "target" / "release" / "urna"
if not CLI.exists():
    raise SystemExit("build the CLI first: cargo build --release -p urna-cli")

# A fake embedder script that ignores the model and returns a fixed
# vector + a fingerprint chosen by an env var. Lets us simulate any
# model_hash the test wants without needing real ML.
FAKE_EMBEDDER_SRC = """\
import json, os, sys
mh = os.environ.get("FAKE_MODEL_HASH", "sha256:" + "ff" * 32)
dim = int(os.environ.get("FAKE_DIM", "4"))
model_name = sys.argv[-2] if len(sys.argv) >= 2 else ""
# Always produce a normalized unit vector.
vec = [1.0] + [0.0] * (dim - 1)
print(json.dumps({
    "model_hash": mh,
    "fingerprint": {"fake": True},
    "embedding_model": model_name,
    "embedding_dim": dim,
    "vector": vec,
}))
"""


def build_corpus(out_path: Path, model_hash: str, dim: int = 4) -> None:
    """Build a 2-chunk corpus with the requested model_hash."""
    chunks = [
        dict(
            canonical_text="primeiro chunk de teste",
            source_uri="test://t/0",
            byte_start=0,
            byte_end=23,
            embedding=[1.0] + [0.0] * (dim - 1),
        ),
        dict(
            canonical_text="segundo chunk de teste",
            source_uri="test://t/1",
            byte_start=23,
            byte_end=45,
            embedding=[0.0, 1.0] + [0.0] * (dim - 2),
        ),
    ]
    if out_path.exists():
        out_path.unlink()
    urna.build(
        output_path=str(out_path),
        embedding_model="fake/test-model",
        embedding_dim=dim,
        chunker_version="test/v1",
        model_hash=model_hash,
        chunks=chunks,
        reproducible=True,
    )


def run_search_text(
    corpus: Path,
    embedder: Path,
    fake_model_hash: str,
    *,
    skip_check: bool = False,
) -> tuple[int, str, str]:
    """Run `urna search-text` with the given fake embedder. Returns
    (exit_code, stdout, stderr)."""
    cmd = [
        str(CLI),
        "search-text",
        str(corpus),
        "vacina",
        "-k",
        "1",
        "--embedder",
        str(embedder),
    ]
    if skip_check:
        cmd.append("--skip-model-hash-check")
    env = dict(os.environ)
    env["FAKE_MODEL_HASH"] = fake_model_hash
    env["FAKE_DIM"] = "4"
    proc = subprocess.run(cmd, capture_output=True, env=env, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> None:
    real_hash = "sha256:" + "ab" * 32
    other_hash = "sha256:" + "cd" * 32
    placeholder = "sha256:" + "00" * 32

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        embedder = td / "fake_embedder.py"
        embedder.write_text(FAKE_EMBEDDER_SRC)

        # Case 1: match — succeeds.
        c_match = td / "match.urna"
        build_corpus(c_match, real_hash)
        rc, stdout, stderr = run_search_text(c_match, embedder, real_hash)
        assert rc == 0, f"match case should succeed, got rc={rc}\nstderr={stderr}"
        assert "chunk_id=sha256:" in stdout, f"expected hits in output, got:\n{stdout}"
        print("case 1 (match): OK")

        # Case 2: mismatch — fails with typed error.
        c_mismatch = td / "mismatch.urna"
        build_corpus(c_mismatch, real_hash)
        rc, stdout, stderr = run_search_text(c_mismatch, embedder, other_hash)
        assert rc != 0, f"mismatch case should fail, got rc=0\nstdout={stdout}"
        assert "model_hash mismatch" in stderr, (
            f"expected 'model_hash mismatch' in stderr, got:\n{stderr}"
        )
        print("case 2 (mismatch): OK")

        # Case 3: placeholder — fails even when embedder reports same placeholder.
        c_placeholder = td / "placeholder.urna"
        build_corpus(c_placeholder, placeholder)
        rc, stdout, stderr = run_search_text(c_placeholder, embedder, placeholder)
        assert rc != 0, "placeholder case should fail, got rc=0"
        assert "legacy placeholder" in stderr, (
            f"expected 'legacy placeholder' in stderr, got:\n{stderr}"
        )
        print("case 3 (placeholder): OK")

        # Case 4: placeholder + --skip-model-hash-check → succeeds.
        rc, stdout, stderr = run_search_text(c_placeholder, embedder, placeholder, skip_check=True)
        assert rc == 0, f"placeholder + skip should succeed, got rc={rc}\nstderr={stderr}"
        print("case 4 (placeholder + skip): OK")

        # Case 5: dim mismatch — embedder reports different dim.
        # Build a corpus with dim=8 (different from the embedder's 4).
        c_dim = td / "dim_mismatch.urna"
        build_corpus(c_dim, real_hash, dim=8)
        rc, stdout, stderr = run_search_text(c_dim, embedder, real_hash)
        assert rc != 0, "dim mismatch should fail, got rc=0"
        assert "dim mismatch" in stderr, f"expected 'dim mismatch' in stderr, got:\n{stderr}"
        print("case 5 (dim mismatch): OK")

    print("all model_hash gate tests passed")


if __name__ == "__main__":
    main()
