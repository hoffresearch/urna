"""Prove the staged embedder payload answers queries with no repo around it.

`tool/tasks/embedpack.py` is what the release archives and
`urna setup` lay down; an installed binary resolves both query embedders
inside that tree (`urna/python/urna/embed/potionqry.py` for potion corpora,
`urna/python/urna/embed/presetqry.py` for registry models,
`urna/python/urna/embed/searchtxt.py` for search-text and for
sentence-transformers models outside the registry). this suite stages
the payload into a temp dir, then runs the staged scripts from a cwd
outside the checkout, with only the staged tree on their path:

- happy path: the potion route embeds and reports the potion model_hash,
  byte-for-byte the value the repo copy reports;
- the registry route: a potion manifest name resolves through the
  registry and embeds; a remote-code preset without the opt-in is refused
  with the URNA_ALLOW_REMOTE_CODE hint (exit 4) before any dependency is
  looked at; an unknown manifest model is exit 4 naming the known models
  when sentence-transformers is absent, and is handed to searchtxt.py
  when it is present;
- error path: a payload staged from a tree missing a module fails the
  staging itself, so a release never ships half the route; so does a
  `catalogue.json` that is not what the registry generates.

Run: .venv/bin/python tool/tests/test_embedpack.py
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "tool" / "tasks" / "embedpack.py"
POTION = "minishlab/potion-base-8M/v1"

EXPECTED_FILES = {
    "urna/VERSION",
    "urna/python/urna/__init__.py",
    "urna/python/urna/embed/__init__.py",
    "urna/python/urna/embed/searchtxt.py",
    "urna/python/urna/model/__init__.py",
    "urna/python/urna/model/modelhash.py",
    "urna/python/urna/model/catalogue.json",
    "urna/python/urna/embed/lexifloor.py",
    "urna/python/urna/embed/potiontab.py",
    "urna/python/urna/embed/potionqry.py",
    "urna/python/urna/embed/presetqry.py",
    "urna/python/urna/model/presetmap.py",
    "urna/python/urna/model/embedders.py",
    "urna/python/urna/embed/stbackend.py",
    "urna/python/urna/embed/stprocess.py",
    "urna/python/urna/embed/visionemb.py",
    "urna/python/urna/model/installer.py",
}


def _run(
    script: Path, *args: str, cwd: Path, env: dict | None = None
) -> subprocess.CompletedProcess:
    # a cwd outside the checkout, and no PYTHONPATH: the staged tree has to
    # find its own modules, exactly like an installed binary's child.
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update(env or {})
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
        check=False,
    )


def test_stage_and_query(base: Path) -> None:
    dest = base / "payload"
    r = subprocess.run(
        [sys.executable, str(STAGE), str(dest)], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
    staged = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    missing = EXPECTED_FILES - staged
    assert not missing, f"payload is missing {sorted(missing)}"
    # what `urna setup` requires (cmd/payload.rs REQUIRED) is exactly what the stage
    # script ships: a shipped file it does not require could go missing
    # unnoticed, a required file it does not ship would fail every install.
    src = (REPO / "rust/clitui/src/cmd/payload.rs").read_text()
    block = src[src.index("pub const REQUIRED") : src.index("];", src.index("pub const REQUIRED"))]
    required = {"urna/" + r for r in re.findall(r'"([^"]+)"', block)}
    assert len(required) >= 8, required
    assert required == staged, (
        f"required but not shipped: {sorted(required - staged)}; "
        f"shipped but not required: {sorted(staged - required)}"
    )
    assert (dest / "urna" / "python" / "urna" / "model" / "potionb8m").is_dir()
    # the stamp `urna setup` compares against the release it wants.
    import tomllib

    with (REPO / "Cargo.toml").open("rb") as f:
        version = tomllib.load(f)["workspace"]["package"]["version"]
    assert (dest / "urna" / "VERSION").read_text() == version + "\n"

    outside = base / "elsewhere"
    outside.mkdir()
    pkg = dest / "urna" / "python" / "urna"

    # the potion route, and the registry route on the same manifest name,
    # both embed offline and agree on the model_hash (the gate compares it
    # against the manifest).
    potion = _run(pkg / "embed" / "potionqry.py", POTION, "hello world", cwd=outside)
    assert potion.returncode == 0, potion.stderr
    registry = _run(pkg / "embed" / "presetqry.py", POTION, "hello world", cwd=outside)
    assert registry.returncode == 0, registry.stderr
    a, b = json.loads(potion.stdout), json.loads(registry.stdout)
    assert a["model_hash"].startswith("sha256:") and a["model_hash"] == b["model_hash"]
    assert a["embedding_dim"] == b["embedding_dim"] == 256
    assert len(a["vector"]) == 256

    # a registry model needing remote code is refused without the explicit
    # opt-in, before dependencies come into it: the same answer on a box
    # with torch and on one without.
    wemm = _run(pkg / "embed" / "presetqry.py", "tencent/WeMM-Embedding-2B", "hello", cwd=outside)
    assert wemm.returncode == 4, (wemm.returncode, wemm.stderr)
    assert "URNA_ALLOW_REMOTE_CODE" in wemm.stderr, wemm.stderr

    # with the opt-in, the answer names what is missing (exit 4 with the pip
    # line) or goes on to the model assets (exit 3) when the deps exist; it
    # is never "embedder script not found", which is the failure this
    # payload retires.
    wemm_ok = _run(
        pkg / "embed" / "presetqry.py",
        "tencent/WeMM-Embedding-2B",
        "hello",
        cwd=outside,
        env={"URNA_ALLOW_REMOTE_CODE": "wemm-2b", "HF_HUB_OFFLINE": "1"},
    )
    assert wemm_ok.returncode in (3, 4), (wemm_ok.returncode, wemm_ok.stderr)
    if wemm_ok.returncode == 4:
        assert "install with:" in wemm_ok.stderr, wemm_ok.stderr

    # a model no preset names: without sentence-transformers, exit 4 naming
    # the known models and the pip line; with it, the staged searchtxt.py
    # takes over and fails offline on a model that is not cached.
    unknown = _run(pkg / "embed" / "presetqry.py", "acme/not-a-model", "hello", cwd=outside)
    if importlib.util.find_spec("sentence_transformers") is None:
        assert unknown.returncode == 4, (unknown.returncode, unknown.stderr)
        assert "no registry preset embeds" in unknown.stderr, unknown.stderr
        assert "sentence-transformers" in unknown.stderr, unknown.stderr
    else:
        assert unknown.returncode != 0, unknown.stdout
        assert "no registry preset embeds" not in unknown.stderr, unknown.stderr

    # the search-text embedder imports modelhash from the same staged
    # package.
    st = _run(pkg / "embed" / "searchtxt.py", "--help", cwd=outside)
    assert st.returncode == 0 and "--mrl-dim" in st.stdout, st.stderr
    print("stage + potion + registry + sentence-transformers routes: OK")


def test_stage_refuses_incomplete_tree(base: Path) -> None:
    # a copy of the checkout's urna package with one route module removed:
    # the staging script must fail, not ship the rest.
    import shutil

    fake_root = base / "fake-repo"
    fake_stage = fake_root / "tool" / "tasks" / STAGE.name
    fake_pkg = fake_root / "rust" / "bridge" / "python" / "urna"
    fake_stage.parent.mkdir(parents=True)
    shutil.copyfile(STAGE, fake_stage)
    shutil.copytree(
        REPO / "rust" / "bridge" / "python" / "urna",
        fake_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "potionb8m", "_urna*"),
    )
    (fake_pkg / "model" / "presetmap.py").unlink()
    r = subprocess.run(
        [sys.executable, str(fake_stage), str(base / "half")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "missing source" in r.stderr and "presetmap.py" in r.stderr, r.stderr
    print("incomplete tree refused: OK")

    # the registry back, the catalog edited by hand: the payload must offer
    # exactly what the registry generates, so staging refuses.
    shutil.copyfile(
        REPO / "rust" / "bridge" / "python" / "urna" / "model" / "presetmap.py",
        fake_pkg / "model" / "presetmap.py",
    )
    (fake_pkg / "model" / "catalogue.json").write_text('{"models": []}\n')
    r = subprocess.run(
        [sys.executable, str(fake_stage), str(base / "stale")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1 and "catalogue.json is stale" in r.stderr, (r.returncode, r.stderr)
    assert not (base / "stale").exists()
    print("stale catalog refused: OK")


def test_version_stamp_reads_without_tomllib(base: Path) -> None:
    """dist's global release job stages the payload with ubuntu-22.04's
    python3 (3.10, no tomllib); the stamp must come out the same there. The
    0.5.2 tag failed in that job on `import tomllib`."""
    spec = importlib.util.spec_from_file_location("embedpack", STAGE)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    with_tomllib = stage.workspace_version()
    saved = sys.modules.get("tomllib")
    sys.modules["tomllib"] = None  # makes `import tomllib` raise ModuleNotFoundError
    try:
        assert stage.workspace_version() == with_tomllib, "the fallback reads another version"
        # edge: other `version =` lines before and after the section
        manifest = base / "Cargo.toml"
        manifest.write_text(
            '[workspace.dependencies]\nserde = { version = "1" }\nversion = "9.9.9"\n\n'
            '[workspace.package]\nedition = "2024"\nversion = "1.2.3"\n\n'
            '[profile.dist]\nversion = "0.0.0"\n',
            encoding="utf-8",
        )
        assert stage.workspace_version(manifest) == "1.2.3"
        # error: no version in the section
        manifest.write_text('[workspace.package]\nedition = "2024"\n', encoding="utf-8")
        try:
            stage.workspace_version(manifest)
        except SystemExit as e:
            assert "no version in [workspace.package]" in str(e), e
        else:
            raise AssertionError("a manifest without a version was accepted")
    finally:
        if saved is None:
            sys.modules.pop("tomllib", None)
        else:
            sys.modules["tomllib"] = saved
    print(f"ok: version stamp {with_tomllib} with and without tomllib")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="urna-payload-") as tmp:
        base = Path(tmp)
        test_stage_and_query(base)
        test_stage_refuses_incomplete_tree(base)
        test_version_stamp_reads_without_tomllib(base)
    print("all embedder payload tests passed")


if __name__ == "__main__":
    main()
