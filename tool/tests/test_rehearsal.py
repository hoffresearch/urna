"""Prove the release rehearsal is the release's build, publishes nothing, and judges right.

`tool/tasks/rehearsal.py` generates `.github/workflows/rehearsal.yml`
from the `release.yml` dist writes, decides whether a change needs it, checks
the artifacts a release would upload and gives the required check's verdict:

- happy path: the checked-in rehearsal is what release.yml produces; its build
  jobs carry the release's steps, runners and matrix; a payload module, a model
  file, a staging script, a wheel source, a crate and the readme each need the
  rehearsal; a full artifact set passes; a real binary validates the golden
  fixture; impact with every build green passes, no impact passes dispensed;
- error path: an unknown job, a missing job, a changed condition, a changed
  plan command, a new top-level key and a kept job reading a secret are all
  refused; a docs or test change does not need it; a missing artifact, a
  checksum mismatch, a wrong payload VERSION, three wheels and a binary that
  rejects its input are named; a needed rehearsal with a failed, cancelled or
  skipped build fails the check, so does a failed impact job;
- edge case: an unknown base (a new branch's first push) runs the rehearsal;
  the generated workflow has no write permission, no secret but the run's own
  token, no attestation step and none of the jobs that host or publish.

Run: python tool/tests/test_rehearsal.py
"""

import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rehearsal", REPO / "tool" / "tasks" / "rehearsal.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

RELEASE = yaml.safe_load(rh.RELEASE.read_text(encoding="utf-8"))
VERSION = rh._load("embedpack").workspace_version()


def refused(release: dict, contains: str) -> None:
    try:
        rh.transform(release)
    except rh.Refused as err:
        assert contains in str(err), (contains, str(err))
        return
    raise AssertionError(f"transform accepted a release.yml with {contains!r}")


def test_the_rehearsal_is_the_release_build() -> None:
    assert rh.generate(check=True) == 0
    ours = yaml.safe_load(rh.REHEARSAL.read_text(encoding="utf-8"))
    local = [
        s
        for s in RELEASE["jobs"]["build-local-artifacts"]["steps"]
        if not str(s.get("uses", "")).startswith("actions/attest@")
    ]
    assert ours["jobs"]["build-local-artifacts"]["steps"] == local
    for key in ("runs-on", "strategy", "needs", "env"):
        assert ours["jobs"]["build-local-artifacts"].get(key) == RELEASE["jobs"][
            "build-local-artifacts"
        ].get(key), key
    assert ours["jobs"]["build-global-artifacts"] == RELEASE["jobs"]["build-global-artifacts"]
    assert ours["jobs"]["custom-wheelmake"]["uses"] == "./.github/workflows/wheelmake.yml"
    print("happy (the build jobs are the release's: steps, runners, matrix, wheels): OK")


def test_the_rehearsal_publishes_nothing() -> None:
    ours = yaml.safe_load(rh.REHEARSAL.read_text(encoding="utf-8"))
    text = rh.REHEARSAL.read_text(encoding="utf-8")
    assert not set(ours["jobs"]) & set(rh.DROP), set(ours["jobs"]) & set(rh.DROP)
    assert ours["permissions"] == {"contents": "read"}
    for name, job in ours["jobs"].items():
        assert job.get("permissions", {"contents": "read"}) == {"contents": "read"}, name
        assert "secrets" not in job, name
        assert not any(
            str(s.get("uses", "")).startswith("actions/attest@") for s in job.get("steps", [])
        ), name
    assert "secrets." not in text.replace("secrets.GITHUB_TOKEN", "")
    # the wheels workflow asks for nothing itself, so the read-only caller holds,
    # and it attests only on the release's tag push.
    wheels = yaml.safe_load((REPO / ".github/workflows/wheelmake.yml").read_text(encoding="utf-8"))
    assert "permissions" not in wheels and all(
        "permissions" not in j for j in wheels["jobs"].values()
    )
    attest = [
        s
        for s in wheels["jobs"]["wheels"]["steps"]
        if str(s.get("uses", "")).startswith("actions/attest@")
    ]
    assert attest[0]["if"] == "github.event_name == 'push' && github.ref_type == 'tag'"
    print("edge (no write permission, no secret, no attestation, no host or publish job): OK")


def test_unknown_shapes_are_refused() -> None:
    cases = []
    extra = copy.deepcopy(RELEASE)
    extra["jobs"]["publish-docker"] = {"runs-on": "ubuntu-24.04"}
    cases.append((extra, "unknown ['publish-docker']"))
    gone = copy.deepcopy(RELEASE)
    del gone["jobs"]["announce"]
    cases.append((gone, "missing ['announce']"))
    cond = copy.deepcopy(RELEASE)
    cond["jobs"]["build-local-artifacts"]["if"] = "${{ always() }}"
    cases.append((cond, "the build-local-artifacts condition"))
    plan = copy.deepcopy(RELEASE)
    step = next(s for s in plan["jobs"]["plan"]["steps"] if s.get("id") == "plan")
    step["run"] = "dist host --steps=create\n"
    cases.append((plan, "the plan command"))
    top = copy.deepcopy(RELEASE)
    top["concurrency"] = {"group": "x"}
    cases.append((top, "top-level keys"))
    secret = copy.deepcopy(RELEASE)
    secret["jobs"]["build-global-artifacts"]["env"]["NPM"] = "${{ secrets.NPM_TOKEN }}"
    cases.append((secret, "reads secrets ['secrets.NPM_TOKEN"))
    bracket = copy.deepcopy(RELEASE)
    bracket["jobs"]["build-global-artifacts"]["env"]["NPM"] = "${{ secrets['NPM_TOKEN'] }}"
    cases.append((bracket, "reads secrets [\"secrets['NPM_TOKEN']"))
    suffix = copy.deepcopy(RELEASE)
    suffix["jobs"]["build-global-artifacts"]["env"]["X"] = "${{ secrets.GITHUB_TOKEN_EXTRA }}"
    cases.append((suffix, "reads secrets ['secrets.GITHUB_TOKEN_EXTRA"))
    whole = copy.deepcopy(RELEASE)
    whole["jobs"]["plan"]["env"]["ALL"] = "${{ toJSON(secrets) }}"
    cases.append((whole, "reads secrets ['secrets)"))
    for release, message in cases:
        refused(release, message)
    print(
        "error (an unknown or missing job, a changed condition or command, a secret: refused): OK"
    )


def test_what_needs_the_rehearsal() -> None:
    needs = [
        "rust/bridge/python/urna/embed/presetqry.py",
        "rust/bridge/python/urna/model/catalogue.json",
        "rust/bridge/python/urna/model/potionb8m/model.safetensors",
        "rust/bridge/python/urna/embed/searchtxt.py",
        "tool/tasks/embedpack.py",
        "tool/tasks/wheelprep.py",
        "rust/bridge/python/urna/entry/clidriver.py",
        "pkgs/wheel/pyproject.toml",
        "rust/clitui/src/main.rs",
        "Cargo.lock",
        "README.md",
        "LICENSE",
        ".github/workflows/wheelmake.yml",
    ]
    for path in needs:
        assert rh.touches_release([path]) == [path], path
    for path in [
        "docs/USAGE.md",
        "tool/tests/test_pythonapi.py",
        "rust/ingest/src/lib.rs",
        ".github/workflows/setuptest.yml",
    ]:
        assert rh.touches_release([path]) == [], path
    assert rh.changed("0" * 40, "HEAD") is None and rh.changed("", "HEAD") is None
    # a module of the urna package the release does not ship stays out.
    assert rh.touches_release(["rust/bridge/python/urna/pipes/buildflow.py"]) == []
    # real history, when the checkout has it: #278 was docs only; the #479
    # move put every payload module at its new path.
    known = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", "62ddc83c^{commit}"], capture_output=True
    )
    if known.returncode == 0:
        assert rh.touches_release(rh.changed("7a417e1e", "577ab85b")) == []
        assert "rust/bridge/python/urna/model/installer.py" in rh.touches_release(
            rh.changed("62ddc83c^", "62ddc83c")
        )
    print("happy/error (payload, models, staging, wheels, crates need it; docs, tests do not): OK")


def _artifacts(directory: Path, plan: dict, version: str = VERSION, wheels: int = 4) -> None:
    for name in plan["artifacts"]:
        if name.endswith(".sha256") or name == "sha256.sum":
            continue
        path = directory / name
        if name == "urna-embedder-payload.tar.gz":
            with tarfile.open(path, "w:gz") as tar:
                data = f"{version}\n".encode()
                info = tarfile.TarInfo("urna/VERSION")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        else:
            path.write_bytes(f"artifact {name}".encode())
        if f"{name}.sha256" in plan["artifacts"]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            (directory / f"{name}.sha256").write_text(f"{digest} *{name}\n", encoding="utf-8")
    for tag in [
        "manylinux_2_34_x86_64",
        "manylinux_2_34_aarch64",
        "macosx_10_12_universal2",
        "win_amd64",
    ][:wheels]:
        wheel = directory / f"urna-{VERSION}-cp312-abi3-{tag}.whl"
        wheel.write_bytes(f"wheel {tag}".encode())
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        (directory / f"{wheel.name}.sha256").write_text(
            f"{digest} *{wheel.name}\n", encoding="utf-8"
        )
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()} *{p.name}"
        for p in sorted(directory.iterdir())
        if p.name in plan["artifacts"] and not p.name.endswith(".sha256")
    ]
    # dist writes a blank last line; so does this copy.
    (directory / "sha256.sum").write_text("\n".join(lines) + "\n\n", encoding="utf-8")


def _plan() -> dict:
    # the kinds dist's plan gives (dist plan --output-format=json).
    kinds = {
        "sha256.sum": "unified-checksum",
        "urna.rb": "installer",
        "urna-npm-package.tar.gz": "installer",
        "urna.cdx.xml": "sbom",
        "urna-embedder-payload.tar.gz": "extra-artifact",
        "urna-embedder-payload.tar.gz.sha256": "extra-artifact",
    }
    for target in ["aarch64-apple-darwin", "x86_64-unknown-linux-musl"]:
        kinds[f"urna-{target}.tar.xz"] = "executable-zip"
        kinds[f"urna-{target}.tar.xz.sha256"] = "checksum"
    return {"artifacts": {n: {"kind": k} for n, k in kinds.items()}}


def test_the_artifacts_a_release_uploads() -> None:
    plan = _plan()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _artifacts(d, plan)
        assert rh.check_assets(d, plan, VERSION) == []
        (d / "urna.rb").unlink()
        (d / "urna-aarch64-apple-darwin.tar.xz").write_bytes(b"rebuilt")
        errors = rh.check_assets(d, plan, VERSION)
        assert "missing urna.rb" in errors, errors
        mismatch = "urna-aarch64-apple-darwin.tar.xz does not match its .sha256"
        assert mismatch in errors, errors
        assert any(e.startswith("sha256.sum: ") for e in errors), errors
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _artifacts(d, plan, version="0.0.1", wheels=3)
        errors = rh.check_assets(d, plan, VERSION)
        assert f"payload VERSION is '0.0.1', not {VERSION!r}" in errors, errors
        assert f"3 urna {VERSION} wheels, expected 4" in errors, errors
    # an empty index cannot vouch for an altered npm package (it has no own .sha256).
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _artifacts(d, plan)
        (d / "sha256.sum").write_text("", encoding="utf-8")
        (d / "urna-npm-package.tar.gz").write_bytes(b"altered")
        errors = rh.check_assets(d, plan, VERSION)
        assert "sha256.sum lacks urna-npm-package.tar.gz" in errors, errors
        assert "sha256.sum lacks urna-x86_64-unknown-linux-musl.tar.xz" in errors, errors
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _artifacts(d, plan)
        index = d / "sha256.sum"
        kept = [ln for ln in index.read_text().splitlines() if "urna-npm-package" not in ln]
        index.write_text("\n".join(kept) + "\n", encoding="utf-8")
        lacks = "sha256.sum lacks urna-npm-package.tar.gz"
        assert rh.check_assets(d, plan, VERSION) == [lacks]
    print("happy/error (artifacts, wheels, checksums, sha256.sum coverage, payload VERSION): OK")


def test_a_real_binary_against_the_golden_fixture() -> None:
    binary = REPO / "target" / "release" / "urna"
    if not binary.is_file():
        print("binary case skipped: no target/release/urna (cargo build --release)")
        return
    golden = REPO / "rust/format/tests/fixtures/golden_v1_minimal.urna"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        with tarfile.open(d / "urna-host.tar.xz", "w:xz") as tar:
            tar.add(binary, arcname="urna-host/urna")
        # beside it, its checksum, as in a release (a glob once picked this).
        digest = hashlib.sha256((d / "urna-host.tar.xz").read_bytes()).hexdigest()
        (d / "urna-host.tar.xz.sha256").write_text(
            f"{digest} *urna-host.tar.xz\n", encoding="utf-8"
        )
        assert rh.check_binary(d, "host", golden, VERSION) == []
        broken = d / "broken.urna"
        data = bytearray(golden.read_bytes())
        data[100] ^= 0xFF
        broken.write_bytes(bytes(data))
        errors = rh.check_binary(d, "host", broken, VERSION)
        assert errors and "validate broken.urna exited" in errors[0], errors
        assert rh.check_binary(d, "other-target", golden, VERSION) == [
            "no urna-other-target archive"
        ]
        # the exact version: a binary saying 0.5.3 is not 0.5 (nor 0.5.30 not 0.5.3).
        prefix = VERSION.rsplit(".", 1)[0]
        errors = rh.check_binary(d, "host", golden, prefix)
        assert errors and f"not 'urna {prefix}'" in errors[0], errors
        # --version has to succeed: an executable that exits 1 is refused.
        fails = d / "fails.sh"
        fails.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fails.chmod(0o755)
        with tarfile.open(d / "urna-fails.tar.xz", "w:xz") as tar:
            tar.add(fails, arcname="urna-fails/urna")
        errors = rh.check_binary(d, "fails", golden, VERSION)
        assert errors and "(exit 1)" in errors[0], errors
    print("happy/error (a real binary validates the golden fixture, rejects a flipped byte): OK")


def test_the_required_check() -> None:
    green = {"result": "success"}
    needed = {"impact": {"result": "success", "outputs": {"run": "true"}}}
    needed.update({j: green for j in (*rh.REQUIRED, "assets")})
    assert rh.verdict(needed)[0]
    dispensed = {"impact": {"result": "success", "outputs": {"run": "false"}}}
    dispensed.update({j: {"result": "skipped"} for j in (*rh.REQUIRED, "assets")})
    ok, why = rh.verdict(dispensed)
    assert ok and "dispensed" in why
    for job, result in [
        ("build-global-artifacts", "skipped"),
        ("custom-wheelmake", "cancelled"),
        ("assets", "failure"),
        ("plan", "skipped"),
    ]:
        bad = json.loads(json.dumps(needed))
        bad[job] = {"result": result}
        ok, why = rh.verdict(bad)
        assert not ok and f"{job} {result}" in why, why
    assert not rh.verdict({"impact": {"result": "failure"}})[0]
    # only an explicit run=false dispenses: a missing or odd output fails.
    for outputs in ({}, {"run": ""}, {"run": "yes"}, None):
        ok, why = rh.verdict({"impact": {"result": "success", "outputs": outputs}})
        assert not ok and "neither 'true' nor 'false'" in why, (outputs, why)
    print("happy/error (dispensed passes; a needed build failed, cancelled or skipped fails): OK")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"rehearsal: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
