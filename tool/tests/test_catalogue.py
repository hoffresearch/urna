"""Prove the model catalog setup and the explorer offer is the registry's.

`urna/model/catalogue.py` generates `urna/model/catalogue.json` from
the registry and the payload ships it; the installer reads nothing else.

- happy path: the checked-in catalog is exactly what the registry
  generates; the MiniLM the pt-br corpora use is offered with its pinned
  revision, files and model_hash; generation needs no numpy (the release
  stages the payload under a bare python3);
- error path: every preset not offered is listed with a reason (only potion,
  which ships in the payload; a preset without a validated install says
  so), a stale catalog fails the check, and a lock resolved for other
  packages than the registry's, or bytes other than the ones upstreams.json
  names, fails the build;
- edge: the test-only fake preset is in neither list.

Run: .venv/bin/python tool/tests/test_catalogue.py
"""

import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rust" / "bridge" / "python"))

from urna.model import catalogue as mc
from urna.model import presetmap as mr

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_HASH = "sha256:97a500de1ad2dc9ffb41fbe3dec27d4fd26ddaa13143aa50f905d3d1f6eff0af"


def test_checked_in_catalog_is_the_registrys() -> None:
    have = json.loads(mc.CATALOG.read_text())
    assert have == mc.build(), "rust/bridge/python/urna/model/catalogue.json is stale"
    assert mc.main(["--check"]) == 0
    # the release's stage step runs under a python3 without numpy.
    script = str(mc.CATALOG.with_name("catalogue.py"))
    blocked = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, runpy; sys.modules['numpy'] = None; sys.argv = ['c', '--check'];"
            f"runpy.run_path({script!r}, run_name='__main__')",
        ],
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 0, blocked.stderr
    print("happy path (catalogue.json == registry output, generated without numpy): OK")


def test_minilm_is_offered_pinned() -> None:
    cat = mc.build()
    by = {m["name"]: m for m in cat["models"]}
    m = by["minilm-multilingual"]
    assert m["embedding_model"] == m["repo"] == MINILM
    assert len(m["revision"]) == 40 and m["model_hash"] == MINILM_HASH
    paths = [f["path"] for f in m["files"]]
    assert "model.safetensors" in paths and "pytorch_model.bin" not in paths
    assert m["bytes"] == sum(f["size"] for f in m["files"]) > 400_000_000
    assert m["packages"] == ["sentence-transformers>=3"] and not m["remote_code"]
    print("happy path (minilm-multilingual offered at a pinned revision and hash): OK")


def test_every_other_preset_says_why() -> None:
    cat = mc.build()
    offered = {m["name"] for m in cat["models"]}
    excluded = {e["name"]: e["reason"] for e in cat["excluded"]}
    real = {n for n, p in mr.PRESETS.items() if p.kind != "fake"}
    assert offered | set(excluded) == real and not offered & set(excluded)
    assert "fake-test" not in offered | set(excluded)
    # every preset but potion carries a validated install (#534).
    assert set(excluded) == {"potion"} and "payload" in excluded["potion"]
    bare = dataclasses.replace(mr.PRESETS["qwen3-embedding-0.6b"], install=None)
    assert "not validated" in mc.exclusion(bare)
    coded = dataclasses.replace(mr.PRESETS["wemm-2b"], remote_code_hashes=())
    assert "reviewed and pinned" in mc.exclusion(coded)
    print("error path (only potion is not offered, with its reason; fake hidden): OK")


def test_a_stale_catalog_fails_the_check() -> None:
    with tempfile.TemporaryDirectory(prefix="urna-catalog-") as tmp:
        stale = Path(tmp) / "catalogue.json"
        stale.write_text(mc.render({"schema": 1, "models": [], "excluded": []}))
        saved = mc.CATALOG
        mc.CATALOG = stale
        try:
            assert mc.main(["--check"]) == 1
            assert mc.main(["--write"]) == 0 and mc.main(["--check"]) == 0
        finally:
            mc.CATALOG = saved
    print("error path (a stale catalog fails --check, --write fixes it): OK")


def test_a_lock_for_other_packages_is_stale() -> None:
    up = mc.load_upstreams()
    text = mc.LOCK.read_text()
    base = mr.PRESETS["minilm-multilingual"]
    other = dataclasses.replace(
        base, install=dataclasses.replace(base.install, packages=("transformers==4.57.0",))
    )
    for presets, lock_text, needle in (
        ({**mr.PRESETS, "minilm-multilingual": other}, text, "other packages"),
        (mr.PRESETS, text + "# edited\n", "not the file"),
    ):
        try:
            mc.build(presets, up, lock_text)
        except mc.Stale as e:
            assert needle in str(e), e
        else:
            raise AssertionError(f"a stale lock was accepted ({needle})")
    assert mc.lock_inputs()[:2] == ["numpy", "open_clip_torch"]
    print("error path (a lock for other packages or other bytes is stale): OK")


def main() -> None:
    test_checked_in_catalog_is_the_registrys()
    test_minilm_is_offered_pinned()
    test_every_other_preset_says_why()
    test_a_stale_catalog_fails_the_check()
    test_a_lock_for_other_packages_is_stale()
    print("all model catalog tests passed")


if __name__ == "__main__":
    main()
