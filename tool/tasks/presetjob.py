"""Install every model preset the catalogue offers through `urna setup`, one
at a time, and prove each answers a query: what `.github/workflows/presetjob.yml`
runs on a clean runner whenever the lock changes.

  python tool/tasks/presetjob.py --urna target/release/urna [--only NAME ...]

the presets share one data dir, so one env, as on a user's machine; the
hugging face cache is a new, empty one for each, so every weights file comes
from the hub at its pinned revision. each runs `urna setup --yes --no-sample
--model NAME` (with --allow-remote-code NAME when it runs repo code), which
installs its packages from the lock, fetches and verifies its files, then
loads it and embeds one query with downloads off; setup must exit 0 and say
the query was answered. a preset whose estimated minimum ram is more than
this machine has is skipped and listed, not failed.

last, torch's own memory (torch and sentence-transformers imported, no
model) is measured in the env and compared with the catalogue's value for
this platform: off by more than a quarter, or missing, fails with the value
to write into upstreams.json's `torch.rss`.

exit codes: 0 every preset passed (skips listed), 1 a preset or the memory
check failed, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "rust" / "bridge" / "python" / "urna" / "model" / "catalogue.json"
RSS_SLACK = 0.25
MEASURE = (
    "import sys, resource\n"
    "import torch, transformers, sentence_transformers\n"
    "r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
    "print(r if sys.platform == 'darwin' else r * 1024)\n"
)
MEASURE_WINDOWS = (
    "import ctypes, ctypes.wintypes as w\n"
    "import torch, transformers, sentence_transformers\n"
    "class C(ctypes.Structure):\n"
    "    _fields_ = [('cb', w.DWORD), ('faults', w.DWORD)] + [\n"
    "        (n, ctypes.c_size_t)\n"
    "        for n in ('peak', 'ws', 'qpp', 'qp', 'qnpp', 'qnp', 'pf', 'ppf')]\n"
    "c = C(); c.cb = ctypes.sizeof(C)\n"
    "ctypes.windll.psapi.GetProcessMemoryInfo(\n"
    "    ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)\n"
    "print(c.peak)\n"
)


def target() -> str:
    """This machine the way the binary names it (`Scan::target`)."""
    arch = {"arm64": "aarch64", "amd64": "x86_64"}.get(platform.machine().lower())
    arch = arch or platform.machine().lower()
    osname = {"darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)
    return f"{arch}-{osname}"


def device(tgt: str) -> str:
    return "mps" if tgt == "aarch64-macos" else "cpu"


def total_ram() -> int | None:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout) if out.returncode == 0 else None
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
        return None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong)
            for n in ("phys", "avail", "pagef", "availpf", "virt", "availv", "ext")
        ]

    s = Status()
    s.len = ctypes.sizeof(Status)
    return s.phys if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)) else None


def real_device(data: Path, tgt: str) -> str:
    """The device the env's torch will run on: mps only where torch says it
    is there (a mac runner is a vm, which may have none)."""
    py = venv_python(data)
    if device(tgt) != "mps" or not py.is_file():
        return device(tgt)
    code = "import torch; print(torch.backends.mps.is_available())"
    out = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    return "mps" if out.stdout.strip() == "True" else "cpu"


def min_ram(entry: dict, catalog: dict, tgt: str, dev: str) -> int | None:
    """The estimate the model screen shows (`Entry::min_ram`), on `dev`."""
    per = entry.get("dtype_bytes", {}).get(dev)
    if not per or not entry.get("params"):
        return None
    return entry["params"] * per + catalog.get("torch", {}).get("rss", {}).get(tgt, 0)


def venv_python(data: Path) -> Path:
    venv = data / "urna" / "venv"
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run_one(urna: Path, entry: dict, data: Path) -> tuple[bool, str]:
    hf = Path(tempfile.mkdtemp(prefix=f"presetjob-hf-{entry['name']}-"))
    env = {**os.environ, "URNA_DATA_DIR": str(data), "HF_HOME": str(hf)}
    env.pop("URNA_ALLOW_DOWNLOAD", None)
    cmd = [str(urna), "setup", "--yes", "--no-sample", "--model", entry["name"]]
    if entry["remote_code"]:
        cmd += ["--allow-remote-code", entry["name"]]
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    finally:
        shutil.rmtree(hf, ignore_errors=True)
    text = out.stdout + out.stderr
    ok = out.returncode == 0 and "answered a query" in text
    return ok, text


def check_rss(catalog: dict, data: Path, tgt: str) -> str | None:
    """An error when torch's measured memory is not the catalogue's."""
    code = MEASURE_WINDOWS if sys.platform == "win32" else MEASURE
    out = subprocess.run([str(venv_python(data)), "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        return f"measuring torch's memory failed: {out.stderr.strip()[-400:]}"
    got = int(out.stdout.strip())
    held = catalog.get("torch", {}).get("rss", {}).get(tgt)
    print(f"torch's own memory on {tgt}: {got} bytes ({got / (1 << 30):.2f} GB)")
    if held is None or abs(got - held) > RSS_SLACK * held:
        return (
            f"upstreams.json's torch.rss.{tgt} is {held}, measured {got}: write "
            f'"{tgt}": {got} into it and run python tool/tasks/upstreams.py --write'
        )
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urna", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args(argv)
    catalog = json.loads(CATALOG.read_text())
    tgt = target()
    ram = total_ram()
    entries = [e for e in catalog["models"] if not args.only or e["name"] in args.only]
    data = Path(tempfile.mkdtemp(prefix="presetjob-data-"))
    failed, skipped = [], []
    for e in entries:
        need = min_ram(e, catalog, tgt, real_device(data, tgt))
        if need and ram and need > ram:
            skipped.append(
                f"{e['name']}: needs ~{need / (1 << 30):.1f} GB, this machine has "
                f"{ram / (1 << 30):.1f} GB"
            )
            print(f"skip {skipped[-1]}", flush=True)
            continue
        print(f"== {e['name']}", flush=True)
        ok, text = run_one(args.urna, e, data)
        # a pass shows the line setup prints for the model; a failure, all.
        done = [ln for ln in text.splitlines() if "ok models:" in ln]
        print("\n".join(done) if ok else text, flush=True)
        if not ok:
            failed.append(e["name"])
    rss = check_rss(catalog, data, tgt) if venv_python(data).is_file() else None
    print(
        f"\npresets: {len(entries) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped on {tgt}"
    )
    for s in skipped:
        print(f"  skipped {s}")
    for f in failed:
        print(f"  FAILED {f}")
    if rss:
        print(f"  FAILED {rss}")
    return 1 if failed or rss else 0


if __name__ == "__main__":
    raise SystemExit(main())
