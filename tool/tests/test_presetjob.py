"""Prove the preset job refuses what it cannot judge and settles the device once.

`tool/tasks/presetjob.py` installs every preset through `urna setup` on the
four runners and measures torch's memory in the env. no urna, no venv and no
network here: `subprocess.run` is stubbed, and the data dir it builds in is
a temporary one.

- happy path: an mps probe that runs its matmul keeps the device mps, the
  install that built the env counts and no setup is pinned to a device;
  a measurement within a quarter of the catalogue's passes;
- error path: `--only` with a name setup does not offer is a usage error
  that names it and the offered ones; a measured 0, nothing or text is the
  probe failing, never a value to write; a measurement off by more than a
  quarter asks for the write;
- edge: mps reported available whose kernel does not compile (github's mac
  runner) runs every preset on the cpu with `URNA_ST_DEVICE=cpu` in each
  setup's env, the first install repeated on the cpu, said once and in the
  summary; the windows probe compiles.

Run: python tool/tests/test_presetjob.py
"""

import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tool" / "tasks"))

import presetjob as pj  # noqa: E402

TWO = ["minilm-multilingual", "harrier-270m"]
FAULT = (
    "Failed to created pipeline state object, error: Error Domain=CompilerError Code=2 "
    '"error: unsupported deferred-static-alloca-size function body"'
)
HELD = 448593920  # upstreams.json's torch.rss for aarch64-macos


def _done(stdout: str, code: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout, stderr)


class Stub:
    """A fake `subprocess.run`: setup lays down the venv's python and answers
    the query; the env's python answers the probes the way `mps` says."""

    def __init__(self, data: Path, mps: str) -> None:
        self.data, self.mps, self.setups = data, mps, []

    def __call__(self, cmd, **kw):
        if cmd[1:3] == ["setup", "--yes"]:
            pj.venv_python(self.data).parent.mkdir(parents=True, exist_ok=True)
            pj.venv_python(self.data).write_text("")
            name, env = cmd[cmd.index("--model") + 1], kw["env"]
            self.setups.append((name, env))
            # "model-breaks": the probe passes, but a model on mps hits
            # metal's compiler; on the cpu it answers.
            if self.mps == "model-breaks" and name != TWO[0] and "URNA_ST_DEVICE" not in env:
                return _done(f"fail(15) models: {name}: query failed: RuntimeError: {FAULT}", 15)
            return _done(f"answered a query\nok models: {cmd[-1]}")
        code = cmd[2]
        if code == pj.MPS_AVAILABLE:
            return _done("False" if self.mps == "none" else "True")
        if code == pj.MPS_PROBE:
            if self.mps in ("works", "model-breaks"):
                return _done(pj.MPS_PROBE_SUM)
            return _done("", 1, "RuntimeError: Failed to created pipeline state object")
        if code in (pj.MEASURE, pj.MEASURE_WINDOWS):
            return _done(str(HELD))
        raise AssertionError(cmd)


def _job(mps: str, *only: str) -> tuple[int, str, Stub]:
    saved = pj.subprocess.run, pj.target, pj.total_ram, pj.tempfile.mkdtemp
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        stub = Stub(data, mps)
        pj.subprocess.run = stub
        pj.target = lambda: "aarch64-macos"
        pj.total_ram = lambda: 64 << 30
        pj.tempfile.mkdtemp = lambda prefix="": str(data)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = pj.main(["--urna", "urna", "--only", *only])
        finally:
            pj.subprocess.run, pj.target, pj.total_ram, pj.tempfile.mkdtemp = saved
    return code, out.getvalue(), stub


def test_an_unknown_only_name_is_a_usage_error() -> None:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            pj.main(["--urna", "urna", "--only", "nome-errado", "minilm-multilingual"])
        except SystemExit as e:
            code = e.code
        else:
            code = None
    assert code == 2, code
    assert "unknown preset nome-errado" in err.getvalue(), err.getvalue()
    assert "harrier-270m" in err.getvalue() and "wemm-2b" in err.getvalue()
    print("error path (--only with a name setup does not offer exits 2 naming it): OK")


def test_a_measured_zero_is_the_probe_failing() -> None:
    catalog = {"torch": {"rss": {"x86_64-windows": HELD}}}
    saved = pj.subprocess.run
    try:
        for bad in ("0\n", "", "abc\n", "-5\n"):
            pj.subprocess.run = lambda *a, bad=bad, **k: _done(bad)
            got = pj.check_rss(catalog, Path("d"), "x86_64-windows")
            assert got and "the probe is broken" in got, (bad, got)
            assert "write" not in got.split("broken")[0], got
        pj.subprocess.run = lambda *a, **k: _done(f"{int(HELD * 1.1)}\n")
        with contextlib.redirect_stdout(io.StringIO()):
            assert pj.check_rss(catalog, Path("d"), "x86_64-windows") is None
            pj.subprocess.run = lambda *a, **k: _done(f"{HELD * 2}\n")
            got = pj.check_rss(catalog, Path("d"), "x86_64-windows")
            assert got and f'"x86_64-windows": {HELD * 2}' in got, got
            pj.subprocess.run = lambda *a, **k: _done("", 1, "ImportError: torch")
            got = pj.check_rss(catalog, Path("d"), "x86_64-windows")
            assert got and "ImportError" in got, got
    finally:
        pj.subprocess.run = saved
    compile(pj.MEASURE_WINDOWS, "<measure-windows>", "exec")
    print("error path (0, nothing or text is the probe failing; off by half asks the write): OK")


def test_a_broken_mps_runs_every_preset_on_the_cpu() -> None:
    code, out, stub = _job("broken", *TWO)
    assert code == 0, out
    names = [n for n, _ in stub.setups]
    assert names == ["minilm-multilingual", *TWO], names
    assert "URNA_ST_DEVICE" not in stub.setups[0][1], "the env build is not pinned"
    assert all(env["URNA_ST_DEVICE"] == "cpu" for _, env in stub.setups[1:])
    line = "cannot compile a kernel on this machine: the presets run on the cpu"
    assert out.count(line) == 1, out
    assert "presets: 2 passed, 0 failed, 0 skipped on aarch64-macos, device cpu" in out, out
    print("edge (mps that compiles no kernel: cpu for every setup, said once): OK")


def test_a_working_mps_keeps_the_device_and_the_first_install() -> None:
    code, out, stub = _job("works", *TWO)
    assert code == 0, out
    assert [n for n, _ in stub.setups] == TWO, stub.setups
    assert all("URNA_ST_DEVICE" not in env for _, env in stub.setups)
    assert "device mps" in out and "presets: 2 passed" in out, out
    assert out.count("ok models: minilm-multilingual") == 1, out
    code, out, stub = _job("none", "harrier-270m")
    assert code == 0 and "torch finds no mps device" in out and "device cpu" in out, out
    assert stub.setups[-1][1]["URNA_ST_DEVICE"] == "cpu"
    print("happy path (a working mps stays, the env build counts; no device is the cpu): OK")


def test_a_model_metal_refuses_moves_the_rest_to_the_cpu() -> None:
    code, out, stub = _job("model-breaks", *TWO, "qwen3-embedding-0.6b")
    assert code == 0, out
    names = [n for n, _ in stub.setups]
    assert names == [*TWO, "harrier-270m", "qwen3-embedding-0.6b"], names
    assert "URNA_ST_DEVICE" not in stub.setups[1][1], "the first try is on mps"
    assert all(env["URNA_ST_DEVICE"] == "cpu" for _, env in stub.setups[2:])
    assert out.count("the presets run on the cpu from here") == 1, out
    assert "== harrier-270m (again, on the cpu)" in out, out
    assert "presets: 3 passed, 0 failed, 0 skipped on aarch64-macos, device cpu" in out, out
    assert pj.mps_broke(FAULT) and not pj.mps_broke("ImportError: torch")
    print("edge (a model metal refuses after a passing probe: cpu from there, run again): OK")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"presetjob: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
