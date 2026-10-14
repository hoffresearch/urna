"""Prove the query-embedder ROUTING (RFC-4): a manifest whose
embedding_model is a registry model routes ask/retrieve through
presetqry.py (observable: the fake preset's env gate error surfaces
when the env var is missing, and the query succeeds when it is set); a
corpus built with mrl_dim gets --mrl-dim so the truncated-dim gate passes;
and the sentence-transformers path (searchtxt.py, behind search-text)
slices a query the same way the registry does. A manifest model no preset
names falls through to searchtxt.py, so ask works on a corpus built with
any sentence-transformers model (cases 6 to 9).
The three-layer gate itself is covered by test_hashguard.py.

Run: .venv/bin/python tool/tests/test_askrouter.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEARCHTXT = REPO / "rust" / "bridge" / "python" / "urna" / "embed" / "searchtxt.py"
sys.path.insert(0, str(REPO / "rust" / "bridge" / "python"))
CLI = REPO / "target" / "release" / "urna"
if not CLI.exists():
    raise SystemExit("build the CLI first: cargo build --release --workspace")

os.environ["URNA_ENABLE_FAKE_PRESET"] = "1"

import urna
from urna.model import presetmap as mr


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
            "the failure must come from presetqry.py's registry path, "
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

        # the search-text embedder slices the same way (one geometry for both
        # query paths), and refuses an out-of-range dim.
        from urna.embed import searchtxt

        full = [float(x) for x in emb.embed_texts(["fake chunk number 2"])[0]]
        st_q = searchtxt.slice_renorm(full, 4)
        assert len(st_q) == 4
        assert all(abs(a - float(b)) < 1e-6 for a, b in zip(st_q, q, strict=True))
        assert db.search(st_q, k=1)[0].score > 0.999
        proc = subprocess.run(
            [sys.executable, str(SEARCHTXT), "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0 and "--mrl-dim" in proc.stdout, proc.stdout
        print("case 5 (searchtxt.slice_renorm == registry slice_renorm): OK")

        _st_fallback_cases(tmp)

    print("all query embedder routing tests passed")


ST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_SCRIPT = REPO / "rust" / "bridge" / "python" / "urna" / "embed" / "presetqry.py"


def _has_st(py: str) -> bool:
    probe = "import importlib.util as u; print(u.find_spec('sentence_transformers') is not None)"
    p = subprocess.run([py, "-c", probe], capture_output=True, text=True)
    return p.stdout.strip() == "True"


def _model_embed(py: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([py, str(MODEL_SCRIPT), *args], capture_output=True, text=True, cwd=REPO)


def _st_python() -> str | None:
    """The interpreter for the sentence-transformers cases (URNA_ST_PYTHON,
    else this one), or None when it cannot import sentence_transformers."""
    st_py = os.environ.get("URNA_ST_PYTHON", sys.executable)
    return st_py if _has_st(st_py) else None


def _pinned_sha256() -> str:
    """the sha-256 tool/tasks/benchdata.py pins for the benchmark corpus."""
    spec = importlib.util.spec_from_file_location(
        "benchdata", REPO / "tool" / "tasks" / "benchdata.py"
    )
    benchdata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchdata)
    return benchdata.SHA256


def _offline_env() -> dict:
    return {k: v for k, v in os.environ.items() if k != "URNA_ALLOW_DOWNLOAD"}


def _st_fallback_cases(tmp: str) -> None:
    """The MiniLM corpus and any other sentence-transformers model go through
    rust/bridge/python/urna/embed/searchtxt.py, so they are askable with the model_hash they were
    built with. The st cases skip on exactly two named conditions: no
    interpreter with sentence-transformers, or the embedder reporting the
    model outside the local cache (its `urna-fetch:` line). Any other failure
    of the probe fails the suite."""
    if not _has_st(sys.executable):
        for model in ("acme/unknown-model", ST_MODEL):
            p = _model_embed(sys.executable, model, "q")
            assert p.returncode == 4, (model, p.returncode, p.stderr)
            assert "urna-needs: sentence-transformers" in p.stderr, p.stderr
        print("case 6 (no sentence-transformers: exit 4 naming the package): OK")
    st_py = _st_python()
    if st_py is None:
        print("cases 7-10 skipped: no interpreter with sentence-transformers (URNA_ST_PYTHON)")
        return
    probe = subprocess.run(
        [st_py, str(MODEL_SCRIPT), ST_MODEL, "probe"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=_offline_env(),
    )
    if probe.returncode == 3 and f"urna-fetch: {ST_MODEL}" in probe.stderr:
        print(f"cases 7-10 skipped: {ST_MODEL} is not in the local cache")
        return
    assert probe.returncode == 0, f"the MiniLM probe failed for a real reason: {probe.stderr}"

    texts = ["o consumidor tem direito à troca", "receita de bolo de cenoura", "pix sem tarifa"]
    payloads = [json.loads(_model_embed(st_py, ST_MODEL, t).stdout) for t in texts]
    corpus = str(Path(tmp) / "st.urna")
    urna.build(
        corpus,
        ST_MODEL,
        payloads[0]["embedding_dim"],
        "t/1",
        payloads[0]["model_hash"],
        [
            {
                "canonical_text": t,
                "source_uri": "test://st",
                "byte_start": 0,
                "byte_end": len(t),
                "embedding": pl["vector"],
            }
            for t, pl in zip(texts, payloads, strict=True)
        ],
        preset="exact",
        reproducible=True,
    )
    env = dict(os.environ, URNA_PYTHON=st_py)
    p = subprocess.run(
        [str(CLI), "ask", corpus, "direito do consumidor", "-k", "1"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )
    assert p.returncode == 0, p.stderr
    assert "o consumidor tem direito" in p.stdout, p.stdout
    print("case 7 (MiniLM corpus: ask routed, gate passes, right hit): OK")

    p = subprocess.run(
        [st_py, str(MODEL_SCRIPT), "acme/not-a-cached-model", "q"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=_offline_env(),
    )
    assert p.returncode == 3, (p.returncode, p.stderr)
    assert "urna-fetch: acme/not-a-cached-model" in p.stderr, p.stderr
    assert "not in the local cache" in p.stderr and "Traceback" not in p.stderr, p.stderr
    print("case 8 (a model outside the cache: exit 3, urna-fetch, no hub call): OK")

    p = _model_embed(st_py, "--mrl-dim", "128", ST_MODEL, "q")
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["embedding_dim"] == 128 and len(out["vector"]) == 128
    assert out["model_hash"] == payloads[0]["model_hash"]
    print("case 9 (--mrl-dim passes through, model_hash unchanged): OK")

    _minilm_preset_case(st_py, payloads[0]["model_hash"])


def _minilm_preset_case(st_py: str, query_hash: str) -> None:
    """The registry preset keeps the MiniLM's embedding path and fingerprint:
    the build-side adapter, the registry query route (by name and by
    --preset) and the search-text embedder agree on model_hash and vectors,
    and so does the benchmark corpus built before the preset existed."""
    preset = mr.preset_for_embedding_model(ST_MODEL)
    assert preset is not None and preset.name == "minilm-multilingual", preset
    script = (
        "import json, sys; sys.path.insert(0, 'rust/bridge/python');"
        "from urna.model import presetmap as mr;"
        "e = mr.create_embedder('minilm-multilingual');"
        "v = e.embed_texts(['pix sem tarifa'])[0];"
        "print(json.dumps({'h': e.model_hash, 'd': e.dim, 'v': [float(x) for x in v]}))"
    )
    a = subprocess.run(
        [st_py, "-c", script], capture_output=True, text=True, cwd=REPO, env=_offline_env()
    )
    assert a.returncode == 0, a.stderr
    adapter = json.loads(a.stdout)
    routes = {
        "registry": _model_embed(st_py, ST_MODEL, "pix sem tarifa"),
        "--preset": _model_embed(
            st_py, "--preset", "minilm-multilingual", ST_MODEL, "pix sem tarifa"
        ),
        "search-text": subprocess.run(
            [st_py, str(SEARCHTXT), ST_MODEL, "pix sem tarifa"],
            capture_output=True,
            text=True,
            cwd=REPO,
            env=_offline_env(),
        ),
    }
    for name, r in routes.items():
        assert r.returncode == 0, (name, r.stderr)
        out = json.loads(r.stdout)
        assert out["model_hash"] == adapter["h"] == query_hash, name
        assert out["embedding_dim"] == adapter["d"] == 384, name
        cos = sum(x * y for x, y in zip(out["vector"], adapter["v"], strict=True))
        assert cos > 0.99999, (name, cos)
    # presetrun builds the variants of the pinned corpus under its hash.
    built = REPO / "target" / "bench" / _pinned_sha256()[:16] / "corpus_hybrid.urna"
    if built.exists():
        assert urna.open(str(built)).inspect()["manifest"]["model_hash"] == adapter["h"]
        print("case 10 (minilm-multilingual preset == searchtxt.py == the benchmark corpus): OK")
    else:
        print("case 10 (minilm-multilingual preset == searchtxt.py): OK; corpus_hybrid not built")


if __name__ == "__main__":
    main()
