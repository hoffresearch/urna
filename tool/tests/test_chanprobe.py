"""Prove the channel probes wait for the exact version and the report survives a failure.

`tool/tasks/chanprobe.py` is what setuptest waits with and what
runreport.yml summarizes a release with. This suite serves controlled
answers from a local HTTP server standing in for the GitHub API, npm, the
crates.io sparse index, PyPI and the Homebrew tap:

- happy path: a version that appears after two absent answers is waited for
  and served; a release whose run succeeded, whose tag points at its commit
  and which every channel serves reports "Released everywhere" and exits 0;
- error path: a version that never appears fails at the deadline naming the
  last state; a cancelled run, an absent npm version, a crates index that
  answers 500 and a formula at another version are each in the summary, and
  the report exits 1 with the summary already written; a tag that points at
  another commit is named;
- edge case: the exact version only: npm serving 0.5.30, an index whose only
  0.5.3 line is yanked, and a PyPI version without files are not served; an
  API that does not answer, answers non-JSON or lists no pypiindex.yml run is
  reported as unavailable or absent, never as success, and an answer the
  report cannot read still leaves a summary and exit 1.

Run: python tool/tests/test_chanprobe.py
"""

import http.server
import importlib.util
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("chanprobe", REPO / "tool" / "tasks" / "chanprobe.py")
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)

ROUTES: dict[str, object] = {}
HITS: dict[str, int] = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server's name)
        HITS[self.path] = HITS.get(self.path, 0) + 1
        answer = ROUTES.get(self.path)
        if callable(answer):
            answer = answer(HITS[self.path])
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        if answer is None:
            status, body = 404, None
        data = (
            (body if isinstance(body, str) else json.dumps(body)).encode()
            if body is not None
            else b""
        )
        self.send_response(status)
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
    """point the endpoints at this file's server before each case: pytest
    runs other files in the same process, and test_pypiindex points
    GITHUB_API_URL at its own."""
    for var, path in [
        ("NPM_REGISTRY", "/npm"),
        ("CRATES_INDEX", "/index"),
        ("PYPI_URL", "/pypi-host"),
        ("RAW_GITHUB", "/raw"),
        ("GITHUB_API_URL", "/api"),
        ("CRATES_API", "/crates-api"),
    ]:
        os.environ[var] = BASE + path
    os.environ.pop("GH_TOKEN", None)
    os.environ.pop("GITHUB_STEP_SUMMARY", None)


V = "0.5.3"
SHA = "a" * 40
INDEX = "/index/ur/na/urna"
FORMULA = "/raw/hoffresearch/homebrew-urna/main/Formula/urna.rb"


def index_line(vers: str, yanked: bool = False) -> str:
    return json.dumps({"name": "urna", "vers": vers, "yanked": yanked})


WHEELS = {
    f"urna-{V}-cp312-abi3-{tag}.whl": f"{i:064x}" for i, tag in enumerate(["a", "b", "c", "d"], 1)
}
ARCHIVES = {f"urna-{t}.tar.xz": f"{i:064x}" for i, t in enumerate(["linux", "darwin"], 10)}


def released(version: str = V) -> None:
    """Every channel serving `version`, a green run on SHA with its tag, the
    same wheels on PyPI and the release, an attestation for every artifact."""
    assets = [{"name": n, "digest": f"sha256:{d}"} for n, d in {**ARCHIVES, **WHEELS}.items()]
    ROUTES.update(
        {
            f"/npm/@urna%2fcli/{version}": {"version": version},
            INDEX: index_line("0.5.2") + "\n" + index_line(version) + "\n",
            f"/pypi-host/pypi/urna/{version}/json": {
                "info": {"version": version},
                "urls": [{"filename": n, "digests": {"sha256": d}} for n, d in WHEELS.items()],
            },
            FORMULA: f'class Urna < Formula\n  version "{version}"\nend\n',
            f"/api/repos/o/r/releases/tags/v{version}": {"assets": assets},
            "/api/repos/o/r/actions/runs/7": {"conclusion": "success"},
            "/api/repos/o/r/actions/runs/7/jobs?per_page=100": {
                "jobs": [
                    {"name": "plan", "conclusion": "success"},
                    {"name": "host", "conclusion": "success"},
                ]
            },
            f"/api/repos/o/r/git/ref/tags/v{version}": {"object": {"type": "tag", "sha": "t" * 40}},
            f"/api/repos/o/r/git/tags/{'t' * 40}": {"object": {"type": "commit", "sha": SHA}},
            f"/api/repos/o/r/actions/workflows/pypiindex.yml/runs?head_sha={SHA}": {
                "workflow_runs": [{"conclusion": "success"}]
            },
            **{
                f"/api/repos/o/r/attestations/sha256:{d}": {"attestations": [{}]}
                for d in {**ARCHIVES, **WHEELS}.values()
            },
        }
    )


def test_crates_io_outages_are_not_absence() -> None:
    url = f"/crates-api/crates/urna/{V}"
    ROUTES[url] = {"version": {"num": V}}
    assert ch.crate_status("urna", V, delay=0)[0] == ch.PUBLISHED
    ROUTES[url] = None
    assert ch.crate_status("urna", V, delay=0)[0] == ch.ABSENT
    # a 429 then a 200: retried, then published, never "absent".
    ROUTES[url] = lambda hit: {"version": {"num": V}} if hit >= 2 else (429, "")
    HITS.pop(url, None)
    assert ch.crate_status("urna", V, delay=0)[0] == ch.PUBLISHED and HITS[url] == 2
    ROUTES[url] = (503, "")
    code, why = ch.crate_status("urna", V, attempts=3, delay=0)
    assert code == 1 and "answered 503" in why, (code, why)
    os.environ["CRATES_API"] = "http://127.0.0.1:9/crates-api"  # nothing listens
    try:
        code, why = ch.crate_status("urna", V, attempts=2, delay=0)
        assert code == 1 and "answered nothing" in why, (code, why)
    finally:
        os.environ["CRATES_API"] = BASE + "/crates-api"
    ROUTES[url] = None
    assert ch.main(["crate", "--name", "urna", "--version", V, "--delay", "0"]) == ch.ABSENT
    print("happy/error (200 published, 404 absent; 429 retried; 5xx and no answer stop): OK")


def test_wheels_and_attestations() -> None:
    released()
    lines, problems = ch.wheels_and_attestations(V, "o/r")
    assert problems == [] and any("the same 4 files" in ln for ln in lines), (lines, problems)
    assert any("has one (6)" in ln for ln in lines), lines
    first = next(iter(WHEELS))
    pypi = f"/pypi-host/pypi/urna/{V}/json"
    ROUTES[pypi]["urls"][0]["digests"]["sha256"] = "f" * 64
    ROUTES[f"/api/repos/o/r/attestations/sha256:{ARCHIVES['urna-linux.tar.xz']}"] = None
    _, problems = ch.wheels_and_attestations(V, "o/r")
    assert f"PyPI and the GitHub release differ on ['{first}']" in problems, problems
    assert "no attestation for ['urna-linux.tar.xz']" in problems, problems
    ROUTES[f"/api/repos/o/r/releases/tags/v{V}"] = {
        "assets": [{"name": n, "digest": f"sha256:{d}"} for n, d in ARCHIVES.items()]
    }
    _, problems = ch.wheels_and_attestations(V, "o/r")
    assert "the GitHub release carries no wheels" in problems, problems
    print("happy/error (same wheels on PyPI and the release, an attestation each; or named): OK")


def test_a_propagating_version_is_waited_for() -> None:
    url = f"/npm/@urna%2fcli/{V}"
    ROUTES[url] = lambda hit: {"version": V} if hit >= 3 else None
    HITS.pop(url, None)
    ok, state = ch.wait("npm", V, timeout=10, interval=0.05)
    assert ok and state == "served" and HITS[url] == 3, (ok, state, HITS.get(url))
    ROUTES["/pypi-host/pypi/urna/9.9.9/json"] = None
    ok, why = ch.wait("pypi", "9.9.9", timeout=0.3, interval=0.05)
    assert not ok and "did not serve 9.9.9 within" in why and "absent" in why, why
    assert ch.main(["wait", "--channel", "npm", "--version", f"v{V}", "--timeout", "1"]) == 0
    print(
        "happy/error (a propagating version is waited for; a missing one fails at the deadline): OK"
    )


def test_only_the_exact_version_is_served() -> None:
    ROUTES[f"/npm/@urna%2fcli/{V}"] = {"version": f"{V}0"}
    assert ch.probe_npm(V).startswith("absent"), ch.probe_npm(V)
    ROUTES[INDEX] = index_line(f"{V}0") + "\n" + index_line(V, yanked=True) + "\n"
    assert ch.probe_crates(V).startswith("absent"), ch.probe_crates(V)
    ROUTES[f"/pypi-host/pypi/urna/{V}/json"] = {"info": {"version": V}, "urls": []}
    assert ch.probe_pypi(V).startswith("absent"), ch.probe_pypi(V)
    ROUTES[INDEX] = (500, "")
    assert ch.probe_crates(V) == "unavailable (HTTP 500)", ch.probe_crates(V)
    print("edge (0.5.30, a yanked line, no files: not served; a 500 is unavailable): OK")


def test_a_full_release_reports_released() -> None:
    released()
    ok, text = ch.report("7", SHA, f"v{V}", "o/r")
    assert ok, text
    for want in [
        "**success**",
        "Jobs: 2 of 2 succeeded",
        "pypiindex.yml: success",
        "| npm | served |",
        "| PyPI | served (4 files) |",
        "| GitHub release | served (6 assets) |",
        "the same 4 files",
        "has one (6)",
        "Released everywhere.",
    ]:
        assert want in text, (want, text)
    print("happy (a green run, its tag, every channel serving: released everywhere): OK")


def test_a_failed_release_is_still_reported() -> None:
    released()
    ROUTES["/api/repos/o/r/actions/runs/7"] = {"conclusion": "cancelled"}
    ROUTES["/api/repos/o/r/actions/runs/7/jobs?per_page=100"] = {
        "jobs": [
            {"name": "host", "conclusion": "success"},
            {"name": "custom-pypiready", "conclusion": "cancelled"},
            {"name": "announce", "conclusion": "skipped"},
        ]
    }
    ROUTES[f"/npm/@urna%2fcli/{V}"] = None
    ROUTES[INDEX] = (500, "")
    ROUTES[FORMULA] = 'class Urna < Formula\n  version "0.5.2"\nend\n'
    with tempfile.TemporaryDirectory() as tmp:
        summary = Path(tmp, "summary.md")
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
        try:
            assert (
                ch.main(["report", "--run", "7", "--sha", SHA, "--tag", f"v{V}", "--repo", "o/r"])
                == 1
            )
        finally:
            os.environ.pop("GITHUB_STEP_SUMMARY")
        text = summary.read_text(encoding="utf-8")
    for want in [
        "**cancelled**",
        "- custom-pypiready: cancelled",
        "- announce: skipped",
        "| npm | absent",
        "| crates.io | unavailable (HTTP 500) |",
        "the formula is at 0.5.2",
        "the release run ended cancelled",
        "Not released everywhere:",
    ]:
        assert want in text, (want, text)
    print(
        "error (cancelled run, skipped jobs, absent npm, a 500, an old formula: in the summary): OK"
    )


def test_a_tag_on_another_commit_is_named() -> None:
    released()
    ROUTES[f"/api/repos/o/r/git/tags/{'t' * 40}"] = {"object": {"type": "commit", "sha": "b" * 40}}
    ok, text = ch.report("7", SHA, f"v{V}", "o/r")
    assert not ok and f"tag v{V} points at {'b' * 40}, not the run's {SHA[:12]}" in text, text
    print("error (a tag that points at another commit than the run's): OK")


def test_missing_answers_are_never_success() -> None:
    released()
    ROUTES["/api/repos/o/r/actions/runs/7/jobs?per_page=100"] = (500, "")
    runs = f"/api/repos/o/r/actions/workflows/pypiindex.yml/runs?head_sha={SHA}"
    ROUTES[runs] = {"workflow_runs": []}
    ROUTES[f"/npm/@urna%2fcli/{V}"] = "<html>not json</html>"
    ROUTES[f"/api/repos/o/r/attestations/sha256:{ARCHIVES['urna-linux.tar.xz']}"] = (502, "")
    ok, text = ch.report("7", SHA, f"v{V}", "o/r")
    assert not ok, text
    for want in [
        "the run's jobs are unavailable (HTTP 500)",
        "- pypiindex.yml: no run for this commit",
        "| npm | absent",
        "the attestations of ['urna-linux.tar.xz'] are unavailable",
    ]:
        assert want in text, (want, text)
    ROUTES[f"/api/repos/o/r/releases/tags/v{V}"] = (503, "")
    ROUTES["/api/repos/o/r/actions/runs/7"] = (502, "")
    ok, text = ch.report("7", SHA, f"v{V}", "o/r")
    assert not ok and "**unavailable (HTTP 502)**" in text, text
    assert f"the GitHub release v{V} is unavailable (HTTP 503)" in text, text
    # an answer the report cannot read still ends in a written summary and exit 1.
    released()
    ROUTES[f"/api/repos/o/r/releases/tags/v{V}"] = {"assets": [{"digest": "sha256:00"}]}
    with tempfile.TemporaryDirectory() as tmp:
        summary = Path(tmp, "summary.md")
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
        try:
            args = ["report", "--run", "7", "--sha", SHA, "--tag", f"v{V}", "--repo", "o/r"]
            assert ch.main(args) == 1
        finally:
            os.environ.pop("GITHUB_STEP_SUMMARY")
        assert "the report could not finish (KeyError" in summary.read_text(encoding="utf-8")
    print("error (unavailable jobs, no pypi run, non-JSON, unreadable answers: never success): OK")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        ROUTES.clear()
        setup_function()
        test()
    print(f"chanprobe: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
