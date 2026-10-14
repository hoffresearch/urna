"""Prove docs/ATLAS.toml's summary says what urna is today, not its history.

the summary grew into a changelog: one dated paragraph per change, 17,789
characters. the history lives in docs/CHANGELOG; the atlas keeps the
current state, and the rule at the top of the file says so. this suite
holds the summary to that:

- happy path: the checked-in summary has at most three sentences, fits the
  size limit and carries no date;
- error path: a dated paragraph, a fourth sentence or an oversized summary
  is named, and so is history without a date: a tracker id (`p1#09`,
  `#262`), a release version (`v0.6`) or a change verb (`added`, `moved`);
- edge case: the rule stays at the top of the file; `e.g.`, `i.e.`, a path
  or a file name with dots do not end a sentence, and the format version
  `v1` is the current state, not history.

Run: python tool/tests/test_atlasdocs.py
"""

import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ATLAS = REPO / "docs" / "ATLAS.toml"

MAX_CHARS = 639  # the number of the 639-line guard, as a character cap
MAX_SENTENCES = 3
# what a changelog entry looks like, dated or not.
HISTORY = [
    (re.compile(r"\b20\d\d-\d\d-\d\d\b"), "a date"),
    (re.compile(r"(?<![\w/])(?:p\d+)?#\d+\b"), "a tracker id"),
    (re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b"), "a release version"),
    (
        re.compile(
            r"\b(?:added|landed|moved|renamed|removed|dropped|replaced|introduced"
            r"|previously|formerly|no longer)\b",
            re.IGNORECASE,
        ),
        "a change verb",
    ),
]
# abbreviations whose dot is not a sentence end.
ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|cf)\.", re.IGNORECASE)
# a sentence ends at . ! or ? followed by whitespace or the end; `.urna` and
# `urna.open()` do not end one.
END = re.compile(r"[.!?](?=\s|$)")


def problems(summary: str) -> list[str]:
    found = []
    if len(summary) > MAX_CHARS:
        found.append(f"{len(summary)} characters, over {MAX_CHARS}")
    sentences = len(END.findall(ABBREV.sub("", summary.strip())))
    if sentences > MAX_SENTENCES:
        found.append(f"{sentences} sentences, over {MAX_SENTENCES}")
    for pattern, what in HISTORY:
        for hit in pattern.findall(summary):
            found.append(f"history ({what}: {hit}) belongs in docs/CHANGELOG")
    return found


def test_summary_is_current_state():
    summary = tomllib.loads(ATLAS.read_text(encoding="utf-8"))["summary"]
    assert problems(summary) == [], problems(summary)


def test_history_is_named():
    dated = "urna is a stack. 2026-08-10: the media pillar landed."
    assert any("2026-08-10" in p for p in problems(dated)), problems(dated)
    for undated, hit in (
        ("urna is a stack; p1#09 holds the graph.", "p1#09"),
        ("urna is a stack; the CLI follows #262.", "#262"),
        ("urna is a stack; v0.6 carries the CLI.", "v0.6"),
        ("urna is a stack; the CLI added a verb.", "added"),
        ("urna is a stack; the bridge moved to rust/.", "moved"),
    ):
        assert any(hit in p for p in problems(undated)), (undated, problems(undated))
    four = "One. Two. Three. Four."
    assert problems(four) == ["4 sentences, over 3"], problems(four)
    long = "x" * (MAX_CHARS + 1)
    assert problems(long) == [f"{MAX_CHARS + 1} characters, over {MAX_CHARS}"]


def test_dots_inside_a_sentence_do_not_end_it():
    one = "A `.urna` file in rust/format, e.g. corpus.urna, read by urna.open(), i.e. the reader."
    assert problems(one) == [], problems(one)
    three = "One, e.g. this. Two, i.e. that. Three, frozen at v1."
    assert problems(three) == [], problems(three)


def test_rule_is_at_the_top():
    head = ATLAS.read_text(encoding="utf-8").split("\n\n", 1)[0]
    assert "docs/CHANGELOG" in head and "Current state only" in head, head


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"atlasdocs: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
