"""distpatch.py's edits against dist's text.

`tool/tasks/distpatch.py` applies to release.yml the edits cargo-dist has no
setting for. Each edit matches dist's output exactly once: a dist upgrade
that moves the line, a file that carries it twice, or a file already edited
is refused instead of edited blindly. The checked-in release.yml carries
every edit. `generate --check` itself needs the dist binary and runs in
gatecheck.yml.

Run: python tool/tests/test_distpatch.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("distpatch", REPO / "tool" / "tasks" / "distpatch.py")
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)

SHA = dp._pins((REPO / "Cargo.toml").read_text(encoding="utf-8"))["sha"]
DIST = (
    "jobs:\n  publish-npm:\n    steps:\n"
    + dp.EDITS[0][0].format(sha=SHA)
    + "      - run: npm publish\n"
)


def refused(text: str) -> bool:
    try:
        dp.patch(text, SHA)
    except dp.Refused:
        return True
    return False


def test_each_edit_lands_once():
    out = dp.patch(DIST, SHA)
    assert out.count("package-manager-cache: false") == 1, out
    assert out.endswith("      - run: npm publish\n"), out


def test_moved_doubled_or_applied_text_is_refused():
    assert refused(DIST.replace("node-version: '20.x'", "node-version: '22.x'"))
    assert refused(DIST + DIST)
    assert refused(dp.patch(DIST, SHA))
    assert refused(DIST.replace(SHA, "0" * 40))


def test_release_yml_carries_every_edit():
    text = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for _, after in dp.EDITS:
        assert text.count(after.format(sha=SHA)) == 1, after


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"distpatch: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
