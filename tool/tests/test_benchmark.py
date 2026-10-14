"""Prove the benchmark rebuild never destroys the corpus it replaces.

`presetrun.py` (and through it `fullcheck.sh`) rebuilds
`target/bench/<baseline hash>/corpus_<preset>.urna` with `benchtime.build_variant`,
which builds under a temporary name in the same directory, validates, and
renames over the old file only at the end, so an interrupted gate never
leaves the corpus gone:

- happy path: an existing corpus is replaced by a valid one, no temporary
  left behind;
- error path: a build that fails (an unknown preset) leaves the existing
  corpus byte-for-byte and no temporary;
- edge: a temporary a killed run left behind is removed by the next build,
  one of a run still building is kept, and a first build with no corpus
  yet creates it; the variants of two baselines land in two directories,
  so `--reuse` never measures one corpus's builds against another's.

Run: .venv/bin/python tool/tests/test_benchmark.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rust" / "bridge" / "python"))
sys.path.insert(0, str(REPO / "tool" / "bench"))

from baselines import OUT_DIR, out_dir
from benchtime import build_variant

import urna

DIM = 8
META = {
    "embedding_model": "test/bench-runner",
    "embedding_dim": DIM,
    "chunker_version": "t/1",
    "model_hash": "sha256:" + "ab" * 32,
}


def _chunks() -> list[dict]:
    out = []
    for i in range(6):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        text = f"bench chunk {i}"
        out.append(
            {
                "canonical_text": text,
                "source_uri": "test://bench",
                "byte_start": 0,
                "byte_end": len(text),
                "embedding": v,
            }
        )
    return out


def _temps(d: Path) -> list[Path]:
    return sorted(d.glob(".*.tmp.urna"))


def test_replaces_a_corpus_with_a_valid_one(d: Path) -> None:
    out = d / "corpus_exact.urna"
    out.write_bytes(b"the old corpus")
    build_variant(_chunks(), META, "exact", out)
    assert urna.open(str(out)).validate() is True
    assert _temps(d) == [], _temps(d)
    print("happy path (old corpus replaced by a validated build, no temporary): OK")


def test_a_failed_build_keeps_the_old_corpus(d: Path) -> None:
    out = d / "corpus_bogus.urna"
    out.write_bytes(b"the old corpus")
    try:
        build_variant(_chunks(), META, "bogus-preset", out)
    except Exception:
        pass
    else:
        raise AssertionError("an unknown preset must fail the build")
    assert out.read_bytes() == b"the old corpus"
    assert _temps(d) == [], _temps(d)
    print("error path (failed build: old corpus byte-for-byte, no temporary): OK")


def _dead_pid() -> int:
    """The pid of a process that has already exited."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_stale_temporaries_go_and_a_first_build_creates(d: Path) -> None:
    out = d / "corpus_tiny.urna"
    stale = d / f".corpus_tiny.{_dead_pid()}.tmp.urna"
    stale.write_bytes(b"half a build from a killed run")
    assert not out.exists()
    build_variant(_chunks(), META, "exact", out)
    assert out.exists() and urna.open(str(out)).validate() is True
    assert not stale.exists()
    assert _temps(d) == [], _temps(d)
    print("edge (dead run's temporary removed, first build creates the corpus): OK")


def test_a_running_build_keeps_its_temporary(d: Path) -> None:
    # another presetrun still building the same preset: its temporary
    # is in use and must survive this build. the stand-in process blocks on
    # its stdin until the test closes it.
    other = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        out = d / "corpus_tiny.urna"
        busy = d / f".corpus_tiny.{other.pid}.tmp.urna"
        busy.write_bytes(b"another run, mid-build")
        build_variant(_chunks(), META, "exact", out)
        assert busy.read_bytes() == b"another run, mid-build"
        assert urna.open(str(out)).validate() is True
    finally:
        other.stdin.close()
        other.wait()
    print("edge (a running build's temporary kept): OK")


def test_each_baseline_has_its_own_variants(_d: Path) -> None:
    a = out_dir("sha256:" + "4bcb8e0fc38a1c13" + "0" * 48)
    b = out_dir("sha256:" + "ea86ab5097646122" + "0" * 48)
    assert a == OUT_DIR / "4bcb8e0fc38a1c13", a
    assert a != b and a.parent == b.parent == OUT_DIR, (a, b)
    assert out_dir("sha256:" + "4bcb8e0fc38a1c13" + "f" * 48) == a  # 16 hex name it
    print("edge (each baseline's variants in their own directory): OK")


def main() -> None:
    for test in (
        test_replaces_a_corpus_with_a_valid_one,
        test_a_failed_build_keeps_the_old_corpus,
        test_stale_temporaries_go_and_a_first_build_creates,
        test_a_running_build_keeps_its_temporary,
        test_each_baseline_has_its_own_variants,
    ):
        with tempfile.TemporaryDirectory(prefix="urna-bench-") as tmp:
            test(Path(tmp))
    print("all bench runner tests passed")


if __name__ == "__main__":
    main()
