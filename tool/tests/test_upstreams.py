"""Prove the upstream values the catalogue carries are read, not written.

`tool/tasks/upstreams.py` resolves the uv release setup downloads (a url
and a sha256 per platform) and the python range torch has wheels for, and
writes them to `upstreams.json`; `catalogue.py` copies that file into the
catalogue the payload ships. no network here: the github and pypi answers
are stubbed.

- happy path: the range is the cpython minors every platform has a wheel
  for (torch 2.14.1's names give `>=3.10,<3.15`); uv's archives are taken
  per urna platform with the sha256 their release publishes; the checked-in
  catalogue carries upstreams.json verbatim;
- error path: a platform with no archive, or no `.sha256`, fails; minors
  with a gap fail; a moved upstream fails `--check`;
- edge: a free-threaded tag (`cp314t`) counts as its minor, a platform with
  fewer minors narrows the range, an sdist is ignored.

Run: python tool/tests/test_upstreams.py
"""

import json
import sys
import tempfile
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
    other = {"uv": {"version": "9.9.9", "archives": {}}, "python": {"requires": ">=3.11,<3.12"}}
    built = mc.build(upstreams=other)
    assert built["uv"]["version"] == "9.9.9" and built["python"]["requires"] == ">=3.11,<3.12"
    print("happy path (catalogue.json carries upstreams.json, nothing written by hand): OK")


def test_a_moved_upstream_fails_check() -> None:
    saved = up.resolve, up.UPSTREAMS
    with tempfile.TemporaryDirectory(prefix="urna-upstreams-") as tmp:
        held = Path(tmp) / "upstreams.json"
        now = {"uv": {"version": "0.13.0"}, "python": {"requires": ">=3.10,<3.15"}}
        held.write_text(up.render(now))
        try:
            up.UPSTREAMS = held
            up.resolve = lambda _sample=None: now
            assert up.main(["--check"]) == 0
            up.resolve = lambda _sample=None: {**now, "uv": {"version": "0.14.0"}}
            assert up.main(["--check"]) == 1
            assert up.main(["--nope"]) == 2
        finally:
            up.resolve, up.UPSTREAMS = saved
    print("error path (a newer uv fails --check until upstreams.json is rewritten): OK")


def main() -> None:
    test_the_range_is_what_every_platform_has()
    test_gaps_and_nothing_fail()
    test_uv_archives_per_platform()
    test_the_catalogue_carries_upstreams_verbatim()
    test_a_moved_upstream_fails_check()
    print("all upstream tests passed")


if __name__ == "__main__":
    main()
