"""Prove the query-embedder ROUTING (RFC-4): a manifest whose
embedding_model is a registry model routes ask/retrieve through
embed_query_model.py (observable: the fake preset's env gate error surfaces
when the env var is missing, and the query succeeds when it is set); a
corpus built with mrl_dim gets --mrl-dim so the truncated-dim gate passes.
The three-layer gate itself is covered by test_search_text_model_hash.py.

Run: .venv/bin/python tests/test_query_embedder_routing.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
CLI = REPO / "target" / "release" / "urna"
if not CLI.exists():
    raise SystemExit("build the CLI first: cargo build --release --workspace")

os.environ["URNA_ENABLE_FAKE_PRESET"] = "1"

import urna
from forge import model_registry as mr


def _build(out: str, mrl_dim: int | None) -> None:
    emb = mr.create_embedder("fake-test")
    texts = [f"fake chunk number {i}" for i in range(4)]
    vecs = emb.embed_texts(texts)
    chunks = [
        {
            "canonical_text": t,
            "source_uri": "test://f",
            "byte_start": i,
            "byte_end": i + 1,
            "embedding": vecs[i].tolist(),
        }
        for i, t in enumerate(texts)
    ]
    urna.build(
        out,
        emb.embedding_model,
        8,
        "t/1",
        emb.model_hash,
        chunks,
        preset="exact",
        reproducible=True,
        mrl_dim=mrl_dim,
    )


def _ask(corpus: str, env_fake: bool) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["URNA_PYTHON"] = str(REPO / ".venv" / "bin" / "python")
    if not env_fake:
        env.pop("URNA_ENABLE_FAKE_PRESET", None)
    p = subprocess.run(
        [str(CLI), "ask", corpus, "fake chunk number 2", "-k", "1"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )
    return p.returncode, p.stdout, p.stderr


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = str(Path(tmp) / "r.urna")
        _build(corpus, mrl_dim=None)

        rc, out, err = _ask(corpus, env_fake=True)
        assert rc == 0, err
        assert "fake chunk number" in out and "urna://" in out
        print("case 1 (registry model routed + gate passes): OK")

        rc, _, err = _ask(corpus, env_fake=False)
        assert rc != 0 and "URNA_ENABLE_FAKE_PRESET" in err, (
            "the failure must come from embed_query_model.py's registry path, "
            f"proving the routing; got: {err}"
        )
        print("case 2 (routing observable via registry error): OK")

        mrl = str(Path(tmp) / "mrl.urna")
        _build(mrl, mrl_dim=4)
        info = json.loads(
            subprocess.run(
                [str(CLI), "inspect", "--json", mrl], capture_output=True, text=True, cwd=REPO
            ).stdout
        )
        assert info["manifest"]["full_dim"] == 8 and info["manifest"]["embedding_dim"] == 4
        rc, out, err = _ask(mrl, env_fake=True)
        assert rc == 0, f"--mrl-dim must be passed for truncated corpora: {err}"
        assert "urna://" in out
        print("case 3 (mrl corpus gets --mrl-dim, dim gate passes): OK")

        # sanity: sliced query really is the engine's truncation (scores align)
        emb = mr.create_embedder("fake-test")
        q = mr.slice_renorm(emb.embed_texts(["fake chunk number 2"]), 4)[0]
        db = urna.open(mrl)
        hits = db.search([float(x) for x in q], k=1)
        assert hits[0].score > 0.999
        print("case 4 (slice_renorm query == stored truncation): OK")

    print("all query embedder routing tests passed")


if __name__ == "__main__":
    main()
