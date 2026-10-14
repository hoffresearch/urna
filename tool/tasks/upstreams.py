"""The values urna's model catalogue takes from its upstreams instead of
having them written by hand, resolved once here and committed as
`rust/bridge/python/urna/model/upstreams.json`:

- uv: the latest release on github and the sha256 it publishes next to
  each archive, for every platform urna ships. `urna setup` downloads that
  archive when a machine has no uv and refuses any other bytes.
- lock: every offered model's packages and the env's base ones, resolved
  together by `uv pip compile --universal --torch-backend cpu` into
  `modellock.txt`: exact versions for linux, macos and windows, torch's cpu
  build (no `nvidia-*`). a conflict between two models fails here.
- python: the range the lock's torch publishes wheels for, read off its
  wheel names on pypi (`cp310` .. `cp314` gives `>=3.10,<3.15`), so setup
  never builds its env on a python the models cannot install torch on.
- torch: the size of the lock's torch per platform, read off pypi (macos)
  and the `content-length` of the cpu wheels on download.pytorch.org
  (linux, windows); a platform torch ships no build for has no size. its
  own memory (`rss`, torch and sentence-transformers imported, no model) is
  measured on each platform by the preset job, which refuses a value that
  moved; this script keeps the held one.
- params: each offered model's parameter count, summed off the headers of
  the safetensors files its install fetches, read with two range requests
  each at the pinned revision; with the dtype the model loads in on a
  device, it is the model's share of the minimum ram setup shows.
- sample: the corpus setup offers a first-time user, one release of the
  fakenews-ptbr benchmark on hugging face, taken by name (`--sample v0.1`).
  the dataset's commit is resolved once and the file's size and the
  release's `SHA256SUMS` are read at that commit, so a push between two
  reads cannot mix releases. the release must be built with the potion the
  payload ships (its manifest's `model_hash`), since setup offers it with
  no other model. a commit that serves the same bytes as the one held is
  not taken, so a push elsewhere in the dataset changes nothing here.

they are read when the catalogue is generated, never when a user runs
setup: the catalogue ships in the payload, `catalogue.py --check` (bare
python, no network) refuses one that does not match this file, and a user
only gets a uv, a range or a sample urna's ci has run. a value resolved on
the user's machine would reach every user untested; one kept by hand ages.

usage:  python tool/tasks/upstreams.py                  print what the upstreams say now
        python tool/tasks/upstreams.py --write          rewrite upstreams.json, modellock.txt
                                                        and catalogue.json
        python tool/tasks/upstreams.py --check          exit 1 when an upstream moved
        python tool/tasks/upstreams.py --sample v0.2 --write
                                                        take another sample release

without `--sample` the release upstreams.json holds is kept; nothing lists
the dataset's releases or orders them. the potion check loads the payload's
potion table, so it runs with numpy and tokenizers (the repo's `.venv`);
the lock needs uv on PATH.

GH_TOKEN or GITHUB_TOKEN, when set, lifts github's anonymous rate limit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rust" / "bridge" / "python"))

from urna.model import catalogue  # noqa: E402

UPSTREAMS = catalogue.UPSTREAMS
LOCK = catalogue.LOCK
UA = "urna-upstreams"
UV_REPO = "astral-sh/uv"

# urna's release targets, keyed the way the binary names its own platform
# (`std::env::consts::{ARCH, OS}`), to the uv archive that runs there. the
# musl builds are static, so one archive serves every linux distro.
UV_ARCHIVES = {
    "aarch64-macos": "uv-aarch64-apple-darwin.tar.gz",
    "x86_64-macos": "uv-x86_64-apple-darwin.tar.gz",
    "aarch64-linux": "uv-aarch64-unknown-linux-musl.tar.gz",
    "x86_64-linux": "uv-x86_64-unknown-linux-musl.tar.gz",
    "x86_64-windows": "uv-x86_64-pc-windows-msvc.zip",
}

# urna's platforms to the tag of the torch wheel that runs there, and where
# that wheel is listed: pypi for macos (its torch is the cpu+mps build),
# pytorch's cpu index for linux and windows (pypi's pulls in cuda).
TORCH_WHEELS = {
    "aarch64-macos": ("macosx_", "_arm64", "pypi"),
    "x86_64-macos": ("macosx_", "_x86_64", "pypi"),
    "aarch64-linux": ("manylinux_", "_aarch64", "cpu"),
    "x86_64-linux": ("manylinux_", "_x86_64", "cpu"),
    "x86_64-windows": ("win_amd64", "", "cpu"),
}
# the index of torch's cpu builds: setup's pip fallback installs from it,
# uv finds it itself (--torch-backend cpu).
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CPU_INDEX = f"{TORCH_INDEX}/torch/"
LOCK_HEADER = "# resolved by tool/tasks/upstreams.py from catalogue.lock_inputs(); do not edit\n"

# the sample corpus: the potion build of the benchmark, the one that runs
# on the payload alone (no torch, no other model).
SAMPLE_REPO = "brennercruvinel/fakenews-ptbr-urna-benchmark"
SAMPLE_BUILD = "potion-tiny"
SAMPLE_FILE = "fakenews.urna"


def _get(url: str, accept: str = "application/json") -> bytes:
    headers = {"User-Agent": UA, "Accept": accept}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return r.read()


def parse_sha256(text: str) -> str:
    """`<64 hex>  <file>` (or the bare hash) to the hash, lowercased."""
    word = text.split()[0] if text.split() else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", word):
        raise ValueError(f"not a sha256 line: {text[:80]!r}")
    return word.lower()


def resolve_uv() -> dict:
    release = json.loads(_get(f"https://api.github.com/repos/{UV_REPO}/releases/latest"))
    version = release["tag_name"].lstrip("v")
    names = {a["name"] for a in release["assets"]}
    archives = {}
    for platform, name in UV_ARCHIVES.items():
        if name not in names or f"{name}.sha256" not in names:
            raise SystemExit(f"uv {version} publishes no {name} with its .sha256")
        url = f"https://github.com/{UV_REPO}/releases/download/{version}/{name}"
        sha = parse_sha256(_get(f"{url}.sha256", accept="*/*").decode())
        archives[platform] = {"url": url, "sha256": sha}
    return {"version": version, "archives": archives}


def python_range(wheels: list[str]) -> str:
    """The `>=3.A,<3.B` range a set of wheel names covers on every platform
    they ship for: the cpython minors each platform has a wheel for,
    intersected, which must be contiguous. free-threaded tags (`cp314t`)
    count as their minor."""
    by_platform: dict[str, set[int]] = {}
    for name in wheels:
        if not name.endswith(".whl"):
            continue
        parts = name[: -len(".whl")].split("-")
        tag, platform = parts[-3], parts[-1]
        m = re.fullmatch(r"cp3(\d+)t?", tag)
        if m:
            by_platform.setdefault(platform, set()).add(int(m.group(1)))
    if not by_platform:
        raise ValueError("no cpython wheels")
    minors = sorted(set.intersection(*by_platform.values()))
    if not minors or minors != list(range(minors[0], minors[-1] + 1)):
        raise ValueError(f"the cpython minors every platform has are not contiguous: {minors}")
    return f">=3.{minors[0]},<3.{minors[-1] + 1}"


def _head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def torch_wheels(version: str) -> list[str]:
    """The wheel names of torch `version` on pypi."""
    info = json.loads(_get(f"https://pypi.org/pypi/torch/{version}/json"))
    return [u["filename"] for u in info["urls"]]


def resolve_python(version: str) -> dict:
    """The range torch `version` (the lock's) has wheels for."""
    return {"requires": python_range(torch_wheels(version)), "torch": version}


def _minors(requires: str) -> range:
    m = re.fullmatch(r">=3\.(\d+),<3\.(\d+)", requires)
    if not m:
        raise ValueError(f"not a python range: {requires}")
    return range(int(m.group(1)), int(m.group(2)))


def pick_size(
    names_sizes: list[tuple[str, int]], version: str, platform: str, minors
) -> int | None:
    """The largest wheel of torch `version` for `platform` over the cpython
    minors of the range: one size per platform, whichever python setup's
    env lands on."""
    head, tail, _ = TORCH_WHEELS[platform]
    best = None
    for name, size in names_sizes:
        parts = name[: -len(".whl")].split("-")
        if len(parts) < 5 or parts[1].split("+")[0] != version:
            continue
        tag, plat = parts[-3], parts[-1]
        m = re.fullmatch(r"cp3(\d+)", tag)
        if m and int(m.group(1)) in minors and plat.startswith(head) and plat.endswith(tail):
            best = max(best or 0, size)
    return best


def resolve_torch(version: str, requires: str) -> dict:
    """The download size of torch `version` on each urna platform."""
    minors = _minors(requires)
    info = json.loads(_get(f"https://pypi.org/pypi/torch/{version}/json"))
    pypi = [(u["filename"], u["size"]) for u in info["urls"]]
    listing = _get(TORCH_CPU_INDEX, accept="text/html").decode()
    cpu = []
    for href in re.findall(r'href="([^"]+\.whl)[^"]*"', listing):
        name = urllib.request.unquote(href.rsplit("/", 1)[-1])
        if name.startswith(f"torch-{version}+cpu-"):
            cpu.append((name, href))
    sizes = {}
    for platform, (_, _, where) in TORCH_WHEELS.items():
        if where == "pypi":
            size = pick_size(pypi, version, platform, minors)
        else:
            # only the wheels that run here are asked for their size.
            fits = [(n, h) for n, h in cpu if pick_size([(n, 1)], version, platform, minors)]
            got = [(n, _head_size(urllib.parse.urljoin(TORCH_CPU_INDEX, h))) for n, h in fits]
            size = pick_size(got, version, platform, minors)
        if size is not None:
            sizes[platform] = size
    return {"version": version, "index": TORCH_INDEX, "sizes": sizes}


def _get_range(url: str, start: int, end: int) -> bytes:
    headers = {"User-Agent": UA, "Range": f"bytes={start}-{end}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read()


def header_params(header: dict) -> int:
    """The parameter count a safetensors header describes."""
    total = 0
    for name, t in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for d in t["shape"]:
            n *= d
        total += n
    return total


def safetensors_params(url: str) -> int:
    """The parameters of the safetensors file at `url`, from its header
    alone: an 8-byte little-endian length, then that much json."""
    size = int.from_bytes(_get_range(url, 0, 7), "little")
    return header_params(json.loads(_get_range(url, 8, 8 + size - 1)))


def resolve_params(presets=None) -> dict:
    """Each offered model's parameter count at its pinned revision."""
    out = {}
    for preset in catalogue.offered(*([presets] if presets else [])):
        spec = preset.install
        weights = [path for path, _ in spec.files if path.endswith(".safetensors")]
        base = f"https://huggingface.co/{preset.hf_repo or preset.model_id}/resolve/{spec.revision}"
        out[preset.name] = {
            "revision": spec.revision,
            "params": sum(safetensors_params(f"{base}/{w}") for w in weights),
        }
    return out


def compile_lock(inputs: list[str], python_min: str) -> str:
    """`inputs` resolved for every platform with torch's cpu build."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("the lock needs uv on PATH")
    cmd = [
        uv,
        "pip",
        "compile",
        "-",
        "--universal",
        "--torch-backend",
        "cpu",
        "--no-header",
        "--python-version",
        python_min,
        "--quiet",
    ]
    out = subprocess.run(cmd, input="\n".join(inputs) + "\n", capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"the offered models' packages do not resolve together:\n{out.stderr}")
    text = LOCK_HEADER + out.stdout
    cuda = [ln for ln in text.splitlines() if ln.startswith("nvidia-")]
    if cuda:
        raise SystemExit(f"the lock pulls in cuda packages: {', '.join(cuda)}")
    return text


def lock_torch(text: str) -> str:
    """The torch version the lock pins (the same on every platform)."""
    found = {m.group(1) for m in re.finditer(r"^torch==([0-9.]+)(?:\+cpu)?\b", text, re.M)}
    if len(found) != 1:
        raise SystemExit(f"the lock pins torch {sorted(found) or 'nowhere'}, not one version")
    return found.pop()


def resolve_lock(held_python: dict | None) -> tuple[str, dict, dict, dict]:
    """The lock, its entry, the python range and torch's sizes. the lock is
    compiled from the floor of the range; when its torch moves the floor,
    once more from the new one."""
    inputs = catalogue.lock_inputs()
    floor = (held_python or {}).get("requires", ">=3.10,<3.15").split(",")[0][2:]
    for _ in range(2):
        text = compile_lock(inputs, floor)
        python = resolve_python(lock_torch(text))
        new_floor = python["requires"].split(",")[0][2:]
        if new_floor == floor:
            break
        floor = new_floor
    torch = resolve_torch(python["torch"], python["requires"])
    entry = {"inputs": inputs, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    return text, entry, python, torch


def parse_sums(text: str) -> dict[str, str]:
    """A `SHA256SUMS` file (`<hex>  <name>` per line) to `{name: hex}`."""
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parse_sha256(line)
    return sums


def potion_model_hash() -> str:
    """The `model_hash` of the potion the payload ships, from its table."""
    from urna.embed.potiontab import PotionEmbedder

    return PotionEmbedder().model_hash()


def resolve_sample(release: str, held: dict | None = None) -> dict:
    if not re.fullmatch(r"v\d+(\.\d+)*", release):
        raise SystemExit(f"--sample takes a release name like v0.1, not {release!r}")
    api = f"https://huggingface.co/api/datasets/{SAMPLE_REPO}"
    commit = json.loads(_get(f"{api}/revision/main"))["sha"]
    folder = f"release/{release}/{SAMPLE_BUILD}"
    path = f"{folder}/{SAMPLE_FILE}"

    def at(name: str) -> bytes:
        url = f"https://huggingface.co/datasets/{SAMPLE_REPO}/resolve/{commit}/{folder}/{name}"
        return _get(url, accept="*/*")

    try:
        listing = json.loads(_get(f"{api}/tree/{commit}/{folder}"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        listing = []
    tree = {e["path"]: e for e in listing}
    if path not in tree:
        raise SystemExit(f"{SAMPLE_REPO}@{commit[:8]} has no {path}")
    sha = parse_sums(at("SHA256SUMS").decode()).get(SAMPLE_FILE)
    if sha is None:
        raise SystemExit(f"{folder}/SHA256SUMS at {commit[:8]} lists no {SAMPLE_FILE}")
    manifest = json.loads(at("manifest.json"))["inspect"]["manifest"]
    want = potion_model_hash()
    if manifest["model_hash"] != want:
        raise SystemExit(
            f"{release} is built with {manifest['embedding_model']} "
            f"({manifest['model_hash']}), not the payload's potion ({want})"
        )
    if held and held.get("release") == release and held.get("sha256") == sha:
        commit = held["commit"]
    return {
        "repo": SAMPLE_REPO,
        "release": release,
        "commit": commit,
        "path": path,
        "url": f"https://huggingface.co/datasets/{SAMPLE_REPO}/resolve/{commit}/{path}",
        "size": tree[path]["size"],
        "sha256": sha,
    }


def held_upstreams() -> dict:
    return json.loads(UPSTREAMS.read_text()) if UPSTREAMS.is_file() else {}


def resolve(sample: str | None = None) -> tuple[dict, str]:
    """upstreams.json's values and the lock's text."""
    held_all = held_upstreams()
    held = held_all.get("sample")
    release = sample or (held or {}).get("release")
    if release is None:
        raise SystemExit("upstreams.json holds no sample: name its release with --sample v0.1")
    lock_text, lock, python, torch = resolve_lock(held_all.get("python"))
    torch["rss"] = (held_all.get("torch") or {}).get("rss", {})
    return {
        "uv": resolve_uv(),
        "python": python,
        "torch": torch,
        "lock": lock,
        "params": resolve_params(),
        "sample": resolve_sample(release, held),
    }, lock_text


def render(upstreams: dict) -> str:
    return json.dumps(upstreams, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    sample = None
    if "--sample" in args:
        i = args.index("--sample")
        if i + 1 >= len(args):
            print(__doc__, file=sys.stderr)
            return 2
        sample = args.pop(i + 1)
        args.pop(i)
    if args not in ([], ["--write"], ["--check"]) or (sample and args == ["--check"]):
        print(__doc__, file=sys.stderr)
        return 2
    values, lock_text = resolve(sample)
    text = render(values)
    if args == ["--write"]:
        UPSTREAMS.write_text(text)
        LOCK.write_text(lock_text)
        print(f"wrote {UPSTREAMS} and {LOCK}")
        return catalogue.main(["--write"])
    if args == ["--check"]:
        held = UPSTREAMS.read_text() if UPSTREAMS.is_file() else ""
        held_lock = LOCK.read_text() if LOCK.is_file() else ""
        if held != text or held_lock != lock_text:
            print(
                f"an upstream moved since {UPSTREAMS.name} was written:\n{text}"
                f"run python tool/tasks/upstreams.py --write and open a pull request",
                file=sys.stderr,
            )
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
