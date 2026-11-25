"""Prove a test that lacks a need skips only when the run does not require it.

`tool/tests/testneeds.py` is the contract every suite that can lack an
external resource goes through (rust/clitui/tests/common/needs.rs is the
same for the Rust tests):

- happy path: a need the run does not require prints a `skip:` line and
  appends `need\\ttest\\treason` to URNA_SKIP_LOG; the report lists it and
  exits 0;
- error path: URNA_REQUIRE_<NEED>=1 or URNA_REQUIRE_ALL=1 turns the skip
  into a failure naming the need and how to provide it, and logs nothing;
  the report of a required or unknown need exits 1; a need the table does
  not list is refused;
- edge case: no log, or an empty one, reports nothing skipped; every
  URNA_REQUIRE_* the tree names is a need of the table (or ALL).

Run: python tool/tests/test_testneeds.py
"""

import contextlib
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import testneeds  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_an_optional_need_logs_the_skip(tmp: Path) -> None:
    log = tmp / "skips.log"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        testneeds.skip("cards", "t1", "URNA_CARDS_CORPUS unset", env={"URNA_SKIP_LOG": str(log)})
        testneeds.skip("media", "t2", "no ffmpeg", env={"URNA_SKIP_LOG": str(log)})
    assert err.getvalue() == "skip: cards: t1: URNA_CARDS_CORPUS unset\nskip: media: t2: no ffmpeg\n"
    assert log.read_text() == "cards\tt1\tURNA_CARDS_CORPUS unset\nmedia\tt2\tno ffmpeg\n"
    with contextlib.redirect_stderr(io.StringIO()):
        testneeds.skip("cards", "t3", "no log named", env={})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert testneeds.report(log, env={}) == 0
    assert re.search(r"cards\s+1\s+URNA_CARDS_CORPUS naming", out.getvalue()), out.getvalue()
    print("happy (an optional need prints, logs and reports 0): OK")


def test_a_required_need_fails(tmp: Path) -> None:
    log = tmp / "required.log"
    for env in ({"URNA_REQUIRE_FORGE": "1"}, {"URNA_REQUIRE_ALL": "1"}):
        env = dict(env, URNA_SKIP_LOG=str(log))
        try:
            testneeds.skip("forge", "t", "no numpy", env=env)
        except testneeds.NeedMissing as e:
            msg = str(e)
        else:
            raise AssertionError(f"{env}: a required need skipped")
        assert "forge is required (URNA_REQUIRE_FORGE or URNA_REQUIRE_ALL)" in msg, msg
        assert "t: no numpy" in msg and "uv sync --group forge" in msg, msg
    assert not log.exists(), "a failure is not a skip: nothing logged"
    # another need's variable requires nothing here.
    with contextlib.redirect_stderr(io.StringIO()):
        testneeds.skip("forge", "t", "no numpy", env={"URNA_REQUIRE_NATIVE": "1"})
    try:
        testneeds.skip("forgee", "t", "typo", env={})
    except KeyError:
        pass
    else:
        raise AssertionError("a need the table does not list was accepted")
    print("error (URNA_REQUIRE_<NEED> and URNA_REQUIRE_ALL fail; an unknown need is refused): OK")


def test_the_report_fails_on_a_required_or_unknown_need(tmp: Path) -> None:
    log = tmp / "report.log"
    log.write_text("cards\tparity\tunset\nforge\tpotion\tno numpy\n")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        assert testneeds.report(log, env={}) == 0
        assert testneeds.report(log, env={"URNA_REQUIRE_FORGE": "1"}) == 1
        log.write_text("nope\tt\tr\n")
        assert testneeds.report(log, env={}) == 1
    assert "REQUIRED" in sink.getvalue() and "UNKNOWN" in sink.getvalue(), sink.getvalue()
    # the command line: what fullcheck.sh and gatecheck.yml run.
    script = Path(testneeds.__file__)
    log.write_text("forge\tpotion\tno numpy\n")
    env = {"PATH": "/usr/bin:/bin", "URNA_REQUIRE_FORGE": "1"}
    p = subprocess.run([sys.executable, script, "report", log], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "FAIL: forge skipped: potion" in p.stderr, (p.returncode, p.stderr)
    print("error (the report exits 1 on a required or an unknown need): OK")


def test_no_log_reports_nothing(tmp: Path) -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert testneeds.report(tmp / "absent.log", env={"URNA_REQUIRE_ALL": "1"}) == 0
        (tmp / "empty.log").write_text("")
        assert testneeds.report(tmp / "empty.log", env={"URNA_REQUIRE_ALL": "1"}) == 0
    assert "no test skipped" in out.getvalue()
    print("edge (no log or an empty one: nothing skipped): OK")


def test_every_required_variable_names_a_need() -> None:
    table = testneeds.needs()
    assert all(table.values()), "a need without how to provide it"
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    named = set()
    for rel in filter(None, files):
        path = REPO / rel
        if rel.startswith(("docs/CHANGELOG", "docs/news/")):
            continue  # history may name a variable that is gone
        if path.suffix in {".py", ".rs", ".sh", ".yml", ".md", ".toml"} and path.is_file():
            named |= set(re.findall(r"URNA_REQUIRE_([A-Z0-9_]+[A-Z0-9])", path.read_text(errors="replace")))
    stale = sorted(n for n in named - {"ALL"} if n.lower() not in table)
    assert not stale, f"URNA_REQUIRE_* naming no need of testneeds.tsv: {stale}"
    print(f"edge (the {len(named)} URNA_REQUIRE_* the tree names are needs of the table): OK")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="urna-testneeds-") as d:
        tmp = Path(d)
        test_an_optional_need_logs_the_skip(tmp)
        test_a_required_need_fails(tmp)
        test_the_report_fails_on_a_required_or_unknown_need(tmp)
        test_no_log_reports_nothing(tmp)
    test_every_required_variable_names_a_need()
    print("testneeds: 5 cases passed")


if __name__ == "__main__":
    main()
