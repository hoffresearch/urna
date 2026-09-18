"""Drive the release binary's space surface (RFC-4): search-space happy path
and typed errors, the stats spaces block, inspect --json spaces[], and
benchmark --space. Corpus built via urna.build spaces= (as test_space_bridge).

Run: .venv/bin/python tests/test_cli_space.py  (needs target/release/urna)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
CLI = REPO / "target" / "release" / "urna"
if not CLI.exists():
    raise SystemExit("build the CLI first: cargo build --release --workspace")

import urna

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def build_corpus(out: str) -> None:
    chunks = [
        {
            "canonical_text": f"chunk {i}",
            "source_uri": "test://s",
            "byte_start": i,
            "byte_end": i + 1,
            "embedding": [1.0 if j == i else 0.0 for j in range(4)],
        }
        for i in range(3)
    ]
    spaces = [
        {
            "name": "vision",
            "model_hash": HASH_B,
            "dtype": "float32",
            "vectors": [[1.0 if j == (2 - i) else 0.0 for j in range(8)] for i in range(3)],
        },
    ]
    urna.build(
        out,
        "test-model",
        4,
        "t/1",
        HASH_A,
        chunks,
        preset="exact",
        reproducible=True,
        spaces=spaces,
    )


def run(args: list[str]) -> tuple[int, str, str]:
    p = subprocess.run([str(CLI), *args], capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout, p.stderr


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = str(Path(tmp) / "s.urna")
        build_corpus(corpus)

        qv = "[0,0,1,0,0,0,0,0]"
        rc, out, _ = run(["search-space", corpus, qv, "--space", "vision", "-k", "2"])
        assert rc == 0 and "space:        vision" in out, out
        assert "score=1.000000" in out, "unit query must hit its own vector at cosine 1"
        print("case 1 (search-space happy): OK")

        rc, _, err = run(["search-space", corpus, qv, "--space", "nope", "-k", "2"])
        assert rc != 0 and "not found" in err, err
        print("case 2 (unknown space typed error): OK")

        rc, _, err = run(["search-space", corpus, "[1,0]", "--space", "vision", "-k", "2"])
        assert rc != 0 and ("dim" in err.lower() or "dimension" in err.lower()), err
        print("case 3 (dim mismatch typed error): OK")

        rc, _, err = run(
            [
                "search-space",
                corpus,
                "[0,0,1,0,0,0,0,0]",
                "--space",
                "vision",
                "-k",
                "2",
                "--expect-model-hash",
                HASH_A,
            ]
        )
        assert rc != 0 and "model" in err.lower(), err
        print("case 4 (expect-model-hash mismatch): OK")

        rc, out, _ = run(["stats", corpus])
        assert rc == 0 and "spaces:       1" in out and "vision" in out, out
        print("case 5 (stats spaces block): OK")

        rc, out, _ = run(["inspect", "--json", corpus])
        doc = json.loads(out)
        assert [s["name"] for s in doc["spaces"]] == ["vision"]
        assert doc["spaces"][0]["dim"] == 8 and doc["spaces"][0]["model_hash"] == HASH_B
        print("case 6 (inspect --json spaces): OK")

        rc, out, _ = run(["benchmark", corpus, "-q", "5", "-k", "2", "--space", "vision"])
        assert rc == 0 and "Space 'vision'" in out and "p50" in out, out
        print("case 7 (benchmark --space): OK")

        # no-spaces file: verbs degrade with typed errors, never a fallback
        plain = str(Path(tmp) / "plain.urna")
        urna.build(
            plain,
            "test-model",
            4,
            "t/1",
            HASH_A,
            [
                {
                    "canonical_text": "x",
                    "source_uri": "s",
                    "byte_start": 0,
                    "byte_end": 1,
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                }
            ],
            preset="exact",
            reproducible=True,
        )
        rc, _, err = run(["search-space", plain, "[1,0,0,0]", "--space", "vision", "-k", "1"])
        assert rc != 0, "no-spaces file must be a typed error"
        rc, out, _ = run(["stats", plain])
        assert rc == 0 and "spaces:" not in out
        print("case 8 (no-spaces degradation): OK")

    print("all cli space tests passed")


if __name__ == "__main__":
    main()
