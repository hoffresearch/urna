"""Prove the PPA source package's texts and guards before dpkg-buildpackage runs.

`tool/tasks/ppasource.py` builds what ppaupload.yml signs and uploads. The
build itself needs git, cargo and dpkg-buildpackage; this suite covers what
decides its content, with no network and no Debian tools:

- happy path: the repository's own debian/ templates fill with the
  workspace texts, the long description is control syntax (one leading
  space, at most 79 columns, ` .` between paragraphs) and `$${shlibs:Depends}`
  stays a substitution variable for dpkg; a vendored crate keeps its own
  checksum and loses only the per-file ones;
- error path: a tag that is not vX.Y.Z, a series that is not a name and a
  tag that is not the workspace version are refused before anything is
  written; a template with an unknown placeholder names its file;
- edge case: a description with hard-wrapped lines and extra spaces comes
  out as the same words, one paragraph per blank line.

Run: python tool/tests/test_ppasource.py
"""

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ppasource", REPO / "tool" / "tasks" / "ppasource.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

VALUES = {
    "tag": "v1.2.3",
    "series": "noble",
    "debversion": "1.2.3-1~noble1",
    "date": "Wed, 14 Oct 2026 10:00:00 +0000",
}


def test_templates_fill_with_the_workspace_texts() -> None:
    values = ps.texts(REPO)
    with tempfile.TemporaryDirectory() as tmp:
        debian = Path(tmp, "debian")
        shutil.copytree(REPO / "pkgs/linux/apt/debian", debian)
        ps.render(debian, {**values, **VALUES})
        control = (debian / "control").read_text()
        changelog = (debian / "changelog").read_text()
    assert f"Description: {values['summary']}\n" in control, control
    assert "${shlibs:Depends}" in control and "$$" not in control, control
    assert changelog.startswith("urna (1.2.3-1~noble1) noble; urgency="), changelog
    assert VALUES["date"] in changelog, changelog
    for line in values["description"].split("\n"):
        assert line.startswith(" ") and len(line) <= 79, repr(line)
    print("happy (templates filled, control syntax, shlibs left for dpkg): OK")


def test_control_text_rewraps_paragraphs() -> None:
    text = ps.control_text("one  two\nthree\n\n\nfour   five\n")
    assert text == " one two three\n .\n four five", repr(text)
    long = ps.control_text(" ".join(["word"] * 40))
    assert all(len(line) <= 79 and line.startswith(" ") for line in long.split("\n")), long
    print("edge (hard wraps and spaces rejoined, one paragraph per blank line): OK")


def test_clear_file_sums_keeps_the_crate_checksum() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        crate = Path(tmp, "serde-1.0.0")
        crate.mkdir()
        sums = crate / ".cargo-checksum.json"
        sums.write_text(
            json.dumps({"files": {"Cargo.toml.orig": "ab", "src/lib.rs": "cd"}, "package": "ef"})
        )
        ps.clear_file_sums(Path(tmp))
        assert json.loads(sums.read_text()) == {"files": {}, "package": "ef"}
    print("happy (per-file sums cleared, the crate's own checksum kept): OK")


def test_guards_refuse_before_writing() -> None:
    version = ps.texts(REPO)["version"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp, "out")
        for tag, series, needle in [
            (f"v{version}-rc.1", "noble", "is not vX.Y.Z"),
            (f"v{version}", "Noble 24", "is not an Ubuntu series name"),
            ("v0.0.1", "noble", f"the checkout is version {version}, not 0.0.1"),
        ]:
            try:
                ps.build(REPO, tag, series, out)
            except ps.SourceError as err:
                assert needle in str(err), err
            else:
                raise AssertionError(f"{tag} {series} was accepted")
        assert not out.exists()
        debian = Path(tmp, "debian")
        debian.mkdir()
        (debian / "changelog").write_text("urna ($nosuch)\n")
        (debian / "control").write_text("Source: urna\n")
        try:
            ps.render(debian, VALUES)
        except ps.SourceError as err:
            assert "debian/changelog" in str(err), err
        else:
            raise AssertionError("an unknown placeholder was accepted")
    print("error (bad tag, bad series, other version, unknown placeholder): OK")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"ppasource: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
