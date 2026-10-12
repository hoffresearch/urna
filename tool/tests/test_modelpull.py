"""Prove the model fetch setup and the explorer share
(`rust/bridge/python/urna/model/installer.py`) downloads only what was confirmed, only
the pinned files, and only keeps a model whose fingerprint is the one the
catalog promised.

the download cases fetch a real 18 MB sentence-transformers test model
(`sentence-transformers-testing/stsb-bert-tiny-safetensors`) at a pinned
revision into a scratch HF_HOME, through a catalog shaped like the shipped
one; they skip, by name, only when huggingface.co does not answer.

- happy path: fetch reports progress up to the total, lays down exactly the
  catalog's files (never the repo's pytorch_model.bin), points refs/main at
  the pin and returns the pinned model_hash; a second fetch is offline;
- error path: no download confirmation is exit 3 with `urna-fetch:` and an
  untouched cache; a remote-code model without its own consent is exit 5
  with `urna-consent:` even with downloads allowed; a corpus hash the entry
  does not have is exit 6 before any download; a model the catalog leaves
  out is exit 7 with the reason; a catalog hash the files do not produce,
  or a reviewed code file that changed, is exit 6;
- edge: a cache whose refs/main names another revision is left alone and
  the fetch fails naming both; the two progress bars xet drives for one
  file (reconstructed and transferred bytes) print monotonic lines capped
  at the file's size.

Run: .venv/bin/python tool/tests/test_modelpull.py
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "rust" / "bridge" / "python" / "urna" / "model" / "installer.py"

TINY = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
TINY_REV = "f3cb857cba53019a20df283396bcca179cf051a4"
TINY_HASH = "sha256:94da16921acd348f121867f8ba8c23db98d7c612fbc02dd5f410c327af2de4fe"
TINY_FILES = [
    ("1_Pooling/config.json", 270),
    ("README.md", 4033),
    ("config.json", 622),
    ("config_sentence_transformers.json", 123),
    ("model.safetensors", 17547912),
    ("modules.json", 229),
    ("sentence_bert_config.json", 53),
    ("special_tokens_map.json", 125),
    ("tokenizer.json", 711649),
    ("tokenizer_config.json", 1270),
    ("vocab.txt", 231508),
]


def _entry(name: str = "tiny", **over) -> dict:
    e = {
        "name": name,
        "embedding_model": TINY,
        "kind": "st_text",
        "repo": TINY,
        "revision": TINY_REV,
        "files": [{"path": p, "size": n} for p, n in TINY_FILES],
        "bytes": sum(n for _, n in TINY_FILES),
        "model_hash": TINY_HASH,
        "packages": ["sentence-transformers>=3"],
        "imports": ["sentence_transformers"],
        "remote_code": False,
        "remote_code_hashes": [],
        "dim": 128,
        "modalities": ["text"],
    }
    e.update(over)
    return e


def _catalog(d: Path, *models: dict, tag: str = "catalog") -> Path:
    path = d / f"{tag}.json"
    excluded = [{"name": "big", "embedding_model": "acme/big", "reason": "too heavy"}]
    path.write_text(json.dumps({"schema": 1, "models": list(models), "excluded": excluded}))
    return path


def _run(hf: Path, catalog: Path, *args: str, download: bool = False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "URNA_"))}
    env["HF_HOME"] = str(hf)
    if download:
        env["URNA_ALLOW_DOWNLOAD"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--catalog", str(catalog)],
        capture_output=True,
        text=True,
        env=env,
        cwd=hf.parent,
    )


def _repo(hf: Path) -> Path:
    return hf / "hub" / f"models--{TINY.replace('/', '--')}"


def test_refusals_before_any_download(d: Path) -> None:
    hf = d / "hf"
    cat = _catalog(d, _entry(), _entry("coded", remote_code=True))
    plan = _run(hf, cat, "plan", "tiny")
    assert plan.returncode == 0, plan.stderr
    p = json.loads(plan.stdout)
    assert not p["cached"] and p["missing_bytes"] == p["bytes"] == sum(n for _, n in TINY_FILES)
    assert p["revision"] == TINY_REV and p["model_hash"] == TINY_HASH
    no = _run(hf, cat, "fetch", "tiny")
    assert no.returncode == 3 and f"urna-fetch: {TINY}" in no.stderr, (no.returncode, no.stderr)
    coded = _run(hf, cat, "fetch", "coded", download=True)
    assert coded.returncode == 5, (coded.returncode, coded.stderr)
    assert "urna-consent: remote-code coded" in coded.stderr, coded.stderr
    other = _run(hf, cat, "fetch", "tiny", "--expect-hash", "sha256:" + "1" * 64, download=True)
    assert other.returncode == 6 and "corpus was built with" in other.stderr, other.stderr
    left = _run(hf, cat, "plan", "acme/big")
    assert left.returncode == 7 and "too heavy" in left.stderr, left.stderr
    assert not _repo(hf).exists(), "a refused fetch must not touch the cache"
    print(
        "error path (no confirmation 3, no remote-code consent 5, other corpus 6, excluded 7): OK"
    )


def test_a_device_dtype_hash_is_left_to_the_query(d: Path) -> None:
    """A sentence-transformers backend model stamps its corpora with a hash
    that names the dtype of the device it ran on: the plan cannot compare it
    with the catalog's file fingerprint, the query setup runs next does."""
    hf = d / "hf2"
    cat = _catalog(d, _entry("multi", kind="st_multimodal"), tag="st")
    plan = _run(hf, cat, "plan", "multi", "--expect-hash", "sha256:" + "2" * 64)
    assert plan.returncode == 0, plan.stderr
    print("edge (an st_multimodal corpus hash does not refuse the plan): OK")


def test_two_bars_for_one_file_never_go_back() -> None:
    # xet drives two bars per file with the reporter class (the reconstructed
    # bytes, then the fewer transferred bytes, each from zero): the lines
    # stay monotonic and never pass the file's size. no network.
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("edge (two bars for one file) skipped: huggingface_hub is not installed")
        return
    import contextlib
    import importlib.util
    import io

    spec = importlib.util.spec_from_file_location("installer", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base, size, total = 5048, 17547912, 18497794
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        report = mod._reporter(base, size, total, "model.safetensors")
        reconstruct, transfer = report(total=size), report(total=size)
        for _ in range(16):
            reconstruct.update(1 << 20)
        transfer.update(16535263)  # the compressed bytes, behind what was shown
        # an overshoot (a fresh reporter, one big update) is capped at the size.
        mod._reporter(base, size, total, "model.safetensors")(total=size).update(2 * size)
    done = [int(line.split()[1]) for line in out.getvalue().splitlines()]
    assert done[:16] == sorted(done[:16]) and len(done) == 17, done
    assert base + 16535263 not in done, "the transfer bar printed a lower count"
    assert done[-1] == base + size, done
    print("edge (two progress bars for one file: monotonic, capped at the size): OK")


def _hub_up() -> bool:
    url = f"https://huggingface.co/api/models/{TINY}/revision/{TINY_REV}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.status == 200
    except OSError:
        return False


def test_fetch_lays_down_the_pinned_files(d: Path) -> None:
    hf = d / "hf"
    cat = _catalog(d, _entry())
    got = _run(hf, cat, "fetch", "tiny", download=True)
    assert got.returncode == 0, got.stderr
    progress = [
        line.split()[1:3] for line in got.stdout.splitlines() if line.startswith("urna-progress:")
    ]
    done = [int(a) for a, _ in progress]
    assert done == sorted(done) and done[-1] == int(progress[-1][1]), progress
    # the 17.5 MB weights report as they arrive, not only when whole (the
    # hub serves them over xet, which writes no partial file to watch).
    weights = {
        a
        for line in got.stdout.splitlines()
        if line.endswith("model.safetensors")
        for a in line.split()[1:2]
    }
    assert len(weights) >= 5, weights
    out = json.loads(got.stdout.splitlines()[-1])
    assert out["model_hash"] == TINY_HASH
    snap = _repo(hf) / "snapshots" / TINY_REV
    laid = sorted(str(p.relative_to(snap)) for p in snap.rglob("*") if not p.is_dir())
    assert laid == sorted(p for p, _ in TINY_FILES), laid
    assert (_repo(hf) / "refs" / "main").read_text() == TINY_REV
    again = _run(hf, cat, "fetch", "tiny")
    assert again.returncode == 0 and "urna-progress" not in again.stdout, again.stderr
    print("happy path (pinned files only, progress to the total, refs/main, offline rerun): OK")

    bad = _catalog(d, _entry(model_hash="sha256:" + "2" * 64), tag="bad")
    r = _run(hf, bad, "fetch", "tiny")
    assert r.returncode == 6 and "not the catalog's" in r.stderr, r.stderr
    pinned = _entry(remote_code=True, remote_code_hashes=[{"path": "config.json", "sha256": "0"}])
    r = _run(hf, _catalog(d, pinned, tag="pinned"), "fetch", "tiny", "--allow-remote-code")
    assert r.returncode == 6 and "config.json" in r.stderr, r.stderr
    print("error path (a hash the files do not produce 6, a changed reviewed file 6): OK")

    ref = _repo(hf) / "refs" / "main"
    ref.write_text("0" * 40)
    r = _run(hf, cat, "fetch", "tiny")
    assert r.returncode == 6 and "000000000000" in r.stderr, (r.returncode, r.stderr)
    assert TINY_REV[:12] in r.stderr, r.stderr
    assert ref.read_text() == "0" * 40, "another revision's refs/main is not ours to move"
    print("edge (refs/main of another revision left alone, both named): OK")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="urna-install-") as tmp:
        test_refusals_before_any_download(Path(tmp))
        test_a_device_dtype_hash_is_left_to_the_query(Path(tmp))
    test_two_bars_for_one_file_never_go_back()
    if not _hub_up():
        print("download cases skipped: huggingface.co is unreachable")
    else:
        with tempfile.TemporaryDirectory(prefix="urna-install-") as tmp:
            test_fetch_lays_down_the_pinned_files(Path(tmp))
    print("all model install tests passed")


if __name__ == "__main__":
    main()
