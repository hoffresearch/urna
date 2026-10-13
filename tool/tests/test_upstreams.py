# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""Prove the upstream values the catalogue carries are read, not written.

`tool/tasks/upstreams.py` resolves the uv release setup downloads (a url
and a sha256 per platform), the python range torch has wheels for and the
sample corpus setup offers, and writes them to `upstreams.json`;
`catalogue.py` copies that file into the catalogue the payload ships. no
network here: the github, pypi and hugging face answers are stubbed.

- happy path: the range is the cpython minors every platform has a wheel
  for (torch 2.14.1's names give `>=3.10,<3.15`); uv's archives are taken
  per urna platform with the sha256 their release publishes; the checked-in
  catalogue carries upstreams.json verbatim; the sample is read at one
  commit (its size from the tree, its sha256 from the release's
  `SHA256SUMS`), and a newer commit serving the same bytes keeps the held one;
- error path: a platform with no archive, or no `.sha256`, fails; minors
  with a gap fail; a moved upstream fails `--check`; a sample built with
  another model than the payload's potion, a release missing from the tree
  or a name that is no release fails;
- edge: a free-threaded tag (`cp314t`) counts as its minor, a platform with
  fewer minors narrows the range, an sdist is ignored.

Run: python tool/tests/test_upstreams.py
"""

import json
import sys
import tempfile
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tool" / "tasks"))
sys.path.insert(0, str(REPO / "rust" / "bridge" / "python"))

import upstreams as up  # noqa: E402

from urna.model import catalogue as mc  # noqa: E402


def wheels(tags: list[str], platforms: list[str]) -> list[str]:
    return [f"torch-2.14.1-{t}-{t}-{p}.whl" for t in tags for p in platforms]


def test_the_range_is_what_every_platform_has() -> None:
    plats = ["macosx_14_0_arm64", "manylinux_2_28_x86_64", "win_amd64"]
    names = wheels(["cp310", "cp311", "cp312", "cp313", "cp314"], plats)
    assert up.python_range(names) == ">=3.10,<3.15"
    # a cp315 wheel moves the top with no edit.
    assert up.python_range(names + wheels(["cp315"], plats)) == ">=3.10,<3.16"
    # one platform without cp314 narrows it for everyone.
    short = [n for n in names if not ("cp314" in n and "win_amd64" in n)]
    assert up.python_range(short) == ">=3.10,<3.14"
    free = wheels(["cp310", "cp311", "cp312", "cp313"], plats) + wheels(["cp314t"], plats)
    assert up.python_range(free + ["torch-2.14.1.tar.gz"]) == ">=3.10,<3.15"
    print("happy path (the range is the minors every platform has; cp314t counts): OK")


def test_gaps_and_nothing_fail() -> None:
    plats = ["win_amd64"]
    for bad in (wheels(["cp310", "cp312"], plats), ["torch-2.14.1.tar.gz"]):
        try:
            up.python_range(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    print("error path (minors with a gap, or no cpython wheel, fail): OK")


def stub_github(missing: str | None = None):
    names = sorted(up.UV_ARCHIVES.values())
    assets = [{"name": n} for n in names] + [{"name": f"{n}.sha256"} for n in names if n != missing]

    def get(url: str, accept: str = "application/json") -> bytes:
        if url.endswith("/releases/latest"):
            return json.dumps({"tag_name": "0.13.0", "assets": assets}).encode()
        if url.endswith(".sha256"):
            name = url.rsplit("/", 1)[1][: -len(".sha256")]
            return f"{'ab' * 32}  {name}\n".encode()
        raise AssertionError(url)

    return get


def test_uv_archives_per_platform() -> None:
    saved = up._get
    try:
        up._get = stub_github()
        uv = up.resolve_uv()
        assert uv["version"] == "0.13.0"
        assert set(uv["archives"]) == set(up.UV_ARCHIVES)
        mac = uv["archives"]["aarch64-macos"]
        assert mac["url"] == (
            "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-aarch64-apple-darwin.tar.gz"
        )
        assert mac["sha256"] == "ab" * 32
        assert uv["archives"]["x86_64-linux"]["url"].endswith("x86_64-unknown-linux-musl.tar.gz")
        assert uv["archives"]["x86_64-windows"]["url"].endswith(".zip")
        up._get = stub_github(missing="uv-x86_64-pc-windows-msvc.zip")
        try:
            up.resolve_uv()
        except SystemExit as e:
            assert "uv-x86_64-pc-windows-msvc.zip" in str(e)
        else:
            raise AssertionError("an archive without its .sha256 was accepted")
    finally:
        up._get = saved
    for bad in ("", "nothex  uv.tar.gz", "ab" * 31):
        try:
            up.parse_sha256(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    print("happy path (uv: one archive per urna platform, the release's sha256): OK")


def test_the_catalogue_carries_upstreams_verbatim() -> None:
    held = mc.load_upstreams()
    cat = json.loads(mc.CATALOG.read_text())
    assert cat["uv"] == held["uv"] and cat["python"] == held["python"]
    assert set(held["uv"]["archives"]) == set(up.UV_ARCHIVES)
    other = {
        **held,
        "uv": {"version": "9.9.9", "archives": {}},
        "python": {"requires": ">=3.11,<3.12"},
    }
    built = mc.build(upstreams=other)
    assert built["torch"] == held["torch"] and built["lock"]["sha256"] == held["lock"]["sha256"]
    assert built["uv"]["version"] == "9.9.9" and built["python"]["requires"] == ">=3.11,<3.12"
    print("happy path (catalogue.json carries upstreams.json, nothing written by hand): OK")


def test_a_moved_upstream_fails_check() -> None:
    saved = up.resolve, up.UPSTREAMS, up.LOCK
    with tempfile.TemporaryDirectory(prefix="urna-upstreams-") as tmp:
        held = Path(tmp) / "upstreams.json"
        lock = Path(tmp) / "modellock.txt"
        now = {"uv": {"version": "0.13.0"}, "python": {"requires": ">=3.10,<3.15"}}
        held.write_text(up.render(now))
        lock.write_text("torch==2.14.1\n")
        try:
            up.UPSTREAMS, up.LOCK = held, lock
            up.resolve = lambda _sample=None: (now, "torch==2.14.1\n")
            assert up.main(["--check"]) == 0
            up.resolve = lambda _sample=None: (
                {**now, "uv": {"version": "0.14.0"}},
                "torch==2.14.1\n",
            )
            assert up.main(["--check"]) == 1
            # a package that moved: the lock differs, upstreams.json alone does not.
            up.resolve = lambda _sample=None: (now, "torch==2.15.0\n")
            assert up.main(["--check"]) == 1
            assert up.main(["--nope"]) == 2
        finally:
            up.resolve, up.UPSTREAMS, up.LOCK = saved
    print("error path (a newer uv or a moved package fails --check until rewritten): OK")


POTION = "sha256:" + "11" * 32
SUMS = f"{'cd' * 32}  fakenews.urna\n{'ef' * 32}  manifest.json\n"


def stub_hub(commit: str, model_hash: str = POTION, release: str = "v0.1"):
    """The hub's answers for one dataset commit; every read is recorded,
    so the test sees each one names that commit."""
    seen: list[str] = []
    folder = f"release/{release}/potion-tiny"

    def get(url: str, accept: str = "application/json") -> bytes:
        seen.append(url)
        if url.endswith("/revision/main"):
            return json.dumps({"sha": commit}).encode()
        if f"/tree/{commit}/{folder}" in url:
            entry = {"path": f"{folder}/fakenews.urna", "size": 19604805}
            return json.dumps([entry]).encode()
        if url.endswith(f"/resolve/{commit}/{folder}/SHA256SUMS"):
            return SUMS.encode()
        if url.endswith(f"/resolve/{commit}/{folder}/manifest.json"):
            manifest = {"model_hash": model_hash, "embedding_model": "minishlab/potion-base-8M/v1"}
            return json.dumps({"inspect": {"manifest": manifest}}).encode()
        if "/tree/" in url:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)  # type: ignore[arg-type]
        raise AssertionError(f"unexpected read: {url}")

    return get, seen


def test_the_sample_is_read_at_one_commit() -> None:
    saved = up._get, up.potion_model_hash
    try:
        up.potion_model_hash = lambda: POTION
        up._get, seen = stub_hub("a" * 40)
        s = up.resolve_sample("v0.1")
        assert s["commit"] == "a" * 40
        assert s["path"] == "release/v0.1/potion-tiny/fakenews.urna"
        assert s["url"].endswith(f"/resolve/{'a' * 40}/release/v0.1/potion-tiny/fakenews.urna")
        assert s["size"] == 19604805 and s["sha256"] == "cd" * 32
        reads = [u for u in seen if not u.endswith("/revision/main")]
        assert reads and all("a" * 40 in u for u in reads), seen
        # a push elsewhere in the dataset: same bytes, the held commit stays.
        up._get, _ = stub_hub("b" * 40)
        assert up.resolve_sample("v0.1", held=s)["commit"] == "a" * 40
        assert up.resolve_sample("v0.1", held={**s, "sha256": "00" * 32})["commit"] == "b" * 40
    finally:
        up._get, up.potion_model_hash = saved
    print("happy path (the sample: one commit, its size and SHA256SUMS; same bytes keep it): OK")


def test_a_sample_the_payload_cannot_ask_is_refused() -> None:
    saved = up._get, up.potion_model_hash
    try:
        up.potion_model_hash = lambda: POTION
        other = stub_hub("a" * 40, model_hash="sha256:" + "22" * 32)[0]
        cases = [
            ("v0.1", other, "not the payload's potion"),
            ("v0.9", stub_hub("a" * 40)[0], "has no release/v0.9"),
            ("latest", stub_hub("a" * 40)[0], "a release name"),
        ]
        for release, get, why in cases:
            up._get = get
            try:
                up.resolve_sample(release)
            except SystemExit as e:
                assert why in str(e), str(e)
                continue
            raise AssertionError(f"{release} was accepted")
    finally:
        up._get, up.potion_model_hash = saved
    print("error path (another model, a missing release or no release name is refused): OK")


def test_the_lock_torch_and_params_are_read_not_written() -> None:
    lock = (
        "# header\ntorch==2.14.1 ; sys_platform == 'darwin'\n"
        "torch==2.14.1+cpu ; sys_platform != 'darwin'\n    # via x\n"
    )
    assert up.lock_torch(lock) == "2.14.1"
    try:
        up.lock_torch(lock + "torch==2.15.0 ; python_version >= '3.14'\n")
    except SystemExit as e:
        assert "not one version" in str(e)
    else:
        raise AssertionError("two torch versions were taken")
    names = [
        ("torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl", 190),
        ("torch-2.14.1+cpu-cp313-cp313-manylinux_2_28_x86_64.whl", 196),
        ("torch-2.14.1+cpu-cp315-cp315-manylinux_2_28_x86_64.whl", 999),
        ("torch-2.14.1+cpu-cp313-cp313-manylinux_2_28_aarch64.whl", 159),
    ]
    # the largest wheel of the range's minors for the platform; none for a mac.
    assert up.pick_size(names, "2.14.1", "x86_64-linux", range(10, 15)) == 196
    assert up.pick_size(names, "2.14.1", "aarch64-linux", range(10, 15)) == 159
    assert up.pick_size(names, "2.14.1", "x86_64-macos", range(10, 15)) is None
    header = {"__metadata__": {"format": "pt"}, "w": {"shape": [1024, 596]}, "b": {"shape": [7]}}
    assert up.header_params(header) == 1024 * 596 + 7
    print("happy path (torch from the lock, a size per platform, params from headers): OK")


def test_a_lock_that_pulls_in_cuda_fails() -> None:
    saved = up.shutil.which, up.subprocess.run

    class Out:
        returncode, stderr = 0, ""
        stdout = "torch==2.14.1\nnvidia-cublas-cu12==12.8\n"

    try:
        up.shutil.which = lambda _name: "/usr/bin/uv"
        up.subprocess.run = lambda *_a, **_k: Out()
        try:
            up.compile_lock(["torch"], "3.10")
        except SystemExit as e:
            assert "cuda packages: nvidia-cublas-cu12==12.8" in str(e), str(e)
        else:
            raise AssertionError("a lock with nvidia-* was accepted")
        Out.returncode, Out.stderr = 1, "No solution found"
        try:
            up.compile_lock(["a==1", "a==2"], "3.10")
        except SystemExit as e:
            assert "do not resolve together" in str(e) and "No solution" in str(e)
        else:
            raise AssertionError("a conflict was accepted")
    finally:
        up.shutil.which, up.subprocess.run = saved
    print("error path (cuda in the lock, or two models that conflict, fail the lock): OK")


def main() -> None:
    test_the_range_is_what_every_platform_has()
    test_gaps_and_nothing_fail()
    test_uv_archives_per_platform()
    test_the_catalogue_carries_upstreams_verbatim()
    test_a_moved_upstream_fails_check()
    test_the_sample_is_read_at_one_commit()
    test_a_sample_the_payload_cannot_ask_is_refused()
    test_the_lock_torch_and_params_are_read_not_written()
    test_a_lock_that_pulls_in_cuda_fails()
    print("all upstream tests passed")


if __name__ == "__main__":
    main()
