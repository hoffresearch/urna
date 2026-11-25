"""the needs of a test: a named external resource it can lack.

a test that lacks one calls `skip(need, test, reason)`. when the run
requires the need (URNA_REQUIRE_<NEED>=1, or URNA_REQUIRE_ALL=1) that is a
failure naming the need and how to provide it; otherwise it prints
`skip: <need>: <test>: <reason>` and, when URNA_SKIP_LOG names a file,
appends `<need>\\t<test>\\t<reason>` to it, so the gate that ran the test can
say what did not run and refuse to remember the step as passed.

the needs and how to provide each live in testneeds.tsv beside this file;
rust/clitui/tests/common/needs.rs holds the Rust tests to the same table
and the same contract.

    python tool/tests/testneeds.py skip NEED TEST REASON
        `skip` for a shell gate (fullcheck.sh's own steps); exit 1 when
        the need is required
    python tool/tests/testneeds.py report LOG
        print the needs LOG records, one line each; exit 1 when one of
        them is required by the environment or is not in the table
"""

from __future__ import annotations

import os
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path

TABLE = Path(__file__).with_name("testneeds.tsv")


class NeedMissing(AssertionError):
    """a test lacks a need the run requires."""


def needs(table: Path = TABLE) -> dict[str, str]:
    """need -> how to provide it, from the table."""
    out = {}
    for line in table.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            need, provide = line.split("\t", 1)
            out[need] = provide
    return out


def env_name(need: str) -> str:
    return "URNA_REQUIRE_" + need.upper()


def required(need: str, env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("URNA_REQUIRE_ALL") == "1" or env.get(env_name(need)) == "1"


def skip(need: str, test: str, reason: str, env: Mapping[str, str] | None = None) -> None:
    """the test lacks `need`: raise when the run requires it, else log the skip."""
    env = os.environ if env is None else env
    table = needs()
    if need not in table:
        raise KeyError(f"{need}: not a need of {TABLE.name}")
    if required(need, env):
        raise NeedMissing(
            f"{need} is required ({env_name(need)} or URNA_REQUIRE_ALL): {test}: {reason}; "
            f"provide {table[need]}"
        )
    print(f"skip: {need}: {test}: {reason}", file=sys.stderr)
    log = env.get("URNA_SKIP_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{need}\t{test}\t{reason}\n")


def skip_case(case: unittest.TestCase, need: str, reason: str) -> None:
    """`skip` for a unittest case, which then reports itself skipped."""
    skip(need, case.id(), reason)
    case.skipTest(f"{need}: {reason}")


def report(log: Path, env: Mapping[str, str] | None = None) -> int:
    """one line per need LOG records; 1 when one is required or unknown."""
    lines = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
    if not lines:
        print("  no test skipped for a missing need")
        return 0
    table = needs()
    tests: dict[str, list[str]] = {}
    for line in lines:
        need, test, _reason = (line.split("\t") + ["", ""])[:3]
        tests.setdefault(need, []).append(test)
    width = max(len(n) for n in tests)
    bad = []
    print(f"  {'need':<{width}}  tests  how to provide it")
    for need, names in sorted(tests.items()):
        mark = ""
        if need not in table:
            mark, provide = " UNKNOWN", "not a need of testneeds.tsv"
            bad.append(need)
        else:
            provide = table[need]
            if required(need, env):
                mark = " REQUIRED"
                bad.append(need)
        print(f"  {need:<{width}}  {len(names):>5}  {provide}{mark}")
    for need in bad:
        print(f"  FAIL: {need} skipped: {', '.join(tests[need])}", file=sys.stderr)
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["skip"] and len(argv) == 4:
        try:
            skip(*argv[1:])
        except (NeedMissing, KeyError) as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 1
        return 0
    if argv[:1] == ["report"] and len(argv) == 2:
        return report(Path(argv[1]))
    print(__doc__.rsplit("\n\n", 1)[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
