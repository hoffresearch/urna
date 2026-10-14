"""Prove the PyPI upload takes the right wheels from the right run, once.

`tool/tasks/pypiindex.py` is what pypiindex.yml runs before its upload. this
suite serves controlled answers from a local HTTP server, standing in for
the GitHub runs API and the index's JSON API, over real wheel-shaped files
with real sha256 sidecars:

- happy path: a release run on the tag's commit with a successful host is a
  pypi source; a successful rehearsal run on a main commit is a testpypi
  source; an index without the version gets all four wheels;
- error path: a run of another workflow, of another commit, without a
  successful host, a failed rehearsal, a rehearsal off main; a wheel that
  does not match its .sha256, a missing sidecar, a wrong count, a wheel of
  another version; a file the index has with another sha256, a file the
  index has that this release did not build;
- edge case: a partial upload (two wheels already there, same sha256) plans
  only the other two, and a complete one plans none; a clash on the last
  wheel copies nothing for upload.

Run: python tool/tests/test_pypiindex.py
"""

import hashlib
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pypiindex", REPO / "tool" / "tasks" / "pypiindex.py")
pypi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pypi)

VERSION = "9.8.7"
TAGS = ["manylinux_2_17_x86_64", "manylinux_2_17_aarch64", "macosx_10_12_universal2", "win_amd64"]
ROUTES: dict[str, object] = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server's name)
        body = ROUTES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


def serve() -> str:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}"


BASE = serve()


def setup_function(_=None) -> None:
    """point the API at this file's server before each case: pytest runs
    other files in the same process, and test_chanprobe points
    GITHUB_API_URL at its own."""
    os.environ["GITHUB_API_URL"] = BASE
    os.environ.pop("GH_TOKEN", None)


def wheels(directory: Path, version: str = VERSION, count: int = 4) -> dict[str, str]:
    """Write `count` wheel files with sidecars; returns name -> sha256."""
    digests = {}
    for tag in TAGS[:count]:
        name = f"urna-{version}-cp312-abi3-{tag}.whl"
        data = f"wheel {name}".encode()
        (directory / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        (directory / f"{name}.sha256").write_text(f"{digest} *{name}\n", encoding="utf-8")
        digests[name] = digest
    return digests


def expect_error(fn, *args, contains: str, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except pypi.ReleaseError as err:
        assert contains in str(err), (contains, str(err))
        return
    raise AssertionError(f"expected a ReleaseError naming {contains!r}")


def git_main(root: Path) -> tuple[str, str]:
    """A repo whose main has one commit, plus one commit off main."""
    git = [
        "git",
        "-C",
        str(root),
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(git[:3] + ["init", "-q", "-b", "main"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "a"], check=True)
    on_main = subprocess.run(
        git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(git[:3] + ["update-ref", "refs/remotes/origin/main", on_main], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "b"], check=True)
    off = subprocess.run(
        git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return on_main, off


def release_run(
    run_id: int, sha: str, path: str = ".github/workflows/release.yml", host: str = "success"
):
    ROUTES[f"/repos/o/r/actions/runs/{run_id}"] = {
        "path": path,
        "head_sha": sha,
        "event": "push",
        "status": "in_progress",
        "conclusion": None,
    }
    ROUTES[f"/repos/o/r/actions/runs/{run_id}/jobs?per_page=100"] = {
        "jobs": [{"name": "plan", "conclusion": "success"}, {"name": "host", "conclusion": host}]
    }


def test_the_sources_each_index_allows() -> None:
    release_run(1, "abc")
    pypi.check_source("pypi", "1", "o/r", "abc")
    expect_error(pypi.check_source, "pypi", "1", "o/r", "def", contains="not the tag push")
    release_run(2, "abc", path=".github/workflows/gatecheck.yml")
    expect_error(
        pypi.check_source, "pypi", "2", "o/r", "abc", contains="takes wheels from release.yml"
    )
    release_run(3, "abc", host="failure")
    expect_error(pypi.check_source, "pypi", "3", "o/r", "abc", contains="no successful host")
    expect_error(pypi.check_source, "pypi", "404", "o/r", "abc", contains="does not exist")
    # a release run is never a testpypi source, a rehearsal never a pypi one.
    expect_error(pypi.check_source, "testpypi", "1", "o/r", "abc", contains="rehearsal.yml")
    with tempfile.TemporaryDirectory() as tmp:
        on_main, off = git_main(Path(tmp))
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            rehearsal = ".github/workflows/rehearsal.yml"
            ROUTES["/repos/o/r/actions/runs/10"] = {
                "path": rehearsal,
                "head_sha": on_main,
                "conclusion": "success",
            }
            pypi.check_source("testpypi", "10", "o/r", on_main)
            expect_error(
                pypi.check_source,
                "pypi",
                "10",
                "o/r",
                on_main,
                contains="takes wheels from release.yml",
            )
            ROUTES["/repos/o/r/actions/runs/11"] = {
                "path": rehearsal,
                "head_sha": off,
                "conclusion": "success",
            }
            expect_error(
                pypi.check_source, "testpypi", "11", "o/r", on_main, contains="not on origin/main"
            )
            ROUTES["/repos/o/r/actions/runs/12"] = {
                "path": rehearsal,
                "head_sha": on_main,
                "conclusion": "failure",
            }
            expect_error(
                pypi.check_source, "testpypi", "12", "o/r", on_main, contains="did not succeed"
            )
        finally:
            os.chdir(cwd)


def test_a_new_version_uploads_every_wheel() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp, "dist"), Path(tmp, "upload")
        src.mkdir()
        names = wheels(src)
        upload = pypi.plan_upload("pypi", src, VERSION, out, json_base=f"{BASE}/fresh")
        assert sorted(upload) == sorted(names), upload
        assert sorted(p.name for p in out.iterdir()) == sorted(names)


def test_the_local_wheels_must_be_the_released_ones() -> None:
    cases = [
        (lambda d: (d / next(iter(wheels(d)))).write_bytes(b"other bytes"), "does not match"),
        (lambda d: (d / (next(iter(wheels(d))) + ".sha256")).unlink(), "has no"),
        (lambda d: wheels(d, count=3), "has 3 wheels, expected 4"),
        (lambda d: wheels(d, version="1.0.0"), f"is not a urna {VERSION} wheel"),
    ]
    for prepare, message in cases:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            prepare(src)
            expect_error(
                pypi.plan_upload,
                "pypi",
                src,
                VERSION,
                src / "up",
                json_base=f"{BASE}/fresh",
                contains=message,
            )


def test_what_the_index_already_has() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp, "dist")
        src.mkdir()
        names = wheels(src)
        first, second, *rest = sorted(names)
        url = f"/partial/urna/{VERSION}/json"
        ROUTES[url] = {
            "urls": [{"filename": n, "digests": {"sha256": names[n]}} for n in (first, second)]
        }
        upload = pypi.plan_upload(
            "pypi", src, VERSION, Path(tmp, "up1"), json_base=f"{BASE}/partial"
        )
        assert sorted(upload) == rest, upload
        ROUTES[f"/done/urna/{VERSION}/json"] = {
            "urls": [{"filename": n, "digests": {"sha256": d}} for n, d in names.items()]
        }
        assert (
            pypi.plan_upload("pypi", src, VERSION, Path(tmp, "up2"), json_base=f"{BASE}/done") == []
        )
        assert not Path(tmp, "up2").exists()
        ROUTES[f"/clash/urna/{VERSION}/json"] = {
            "urls": [{"filename": first, "digests": {"sha256": "0" * 64}}]
        }
        expect_error(
            pypi.plan_upload,
            "pypi",
            src,
            VERSION,
            Path(tmp, "up3"),
            json_base=f"{BASE}/clash",
            contains=f"already has {first} with another sha256",
        )
        # a clash on the last wheel, the others new: nothing is copied at all.
        last = sorted(names)[-1]
        ROUTES[f"/clash-last/urna/{VERSION}/json"] = {
            "urls": [{"filename": last, "digests": {"sha256": "0" * 64}}]
        }
        expect_error(
            pypi.plan_upload,
            "pypi",
            src,
            VERSION,
            Path(tmp, "up5"),
            json_base=f"{BASE}/clash-last",
            contains=f"already has {last} with another sha256",
        )
        assert not Path(tmp, "up5").exists(), "a refused plan copied wheels for upload"
        stray = f"urna-{VERSION}.tar.gz"
        ROUTES[f"/stray/urna/{VERSION}/json"] = {
            "urls": [{"filename": stray, "digests": {"sha256": "1" * 64}}]
        }
        expect_error(
            pypi.plan_upload,
            "pypi",
            src,
            VERSION,
            Path(tmp, "up4"),
            json_base=f"{BASE}/stray",
            contains="did not build",
        )


def test_the_cli_reports_the_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp, "dist")
        src.mkdir()
        wheels(src)
        output = Path(tmp, "github_output")
        os.environ["GITHUB_OUTPUT"] = str(output)
        try:
            args = [
                "plan",
                "--index",
                "testpypi",
                "--dir",
                str(src),
                "--version",
                VERSION,
                "--out",
                str(Path(tmp, "up")),
                "--json-base",
                f"{BASE}/fresh",
            ]
            assert pypi.main(args) == 0
            assert output.read_text(encoding="utf-8") == "upload=4\n"
            assert (
                pypi.main(
                    ["source", "--index", "pypi", "--run", "404", "--repo", "o/r", "--sha", "x"]
                )
                == 1
            )
        finally:
            os.environ.pop("GITHUB_OUTPUT")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        setup_function()
        test()
        print(f"ok  {test.__name__}")
    print(f"pypiindex: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
