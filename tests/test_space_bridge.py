"""bridge coverage for the multimodal space bands (0x15 + 0x20).

real artifacts only: builds a tiny text+vision .urna through urna.build
(spaces kwarg), opens it through urna.open, and checks

- search_space scores the vision band with real cosine and honors the
  per-space model_hash gate;
- ISOLATION: the text path never scores the vision band and a text-dim
  query against the vision space (and vice versa) raises;
- content_hash is identical with and without the multimodal sections;
- row-count and dim mismatches in the spaces kwarg are rejected.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import _urna  # noqa: E402

TEXT_DIM = 4
VIS_DIM = 2
VIS_HASH = "sha256:" + "1" * 64


def _chunk(i: int) -> dict:
    emb = [0.0] * TEXT_DIM
    emb[i % TEXT_DIM] = 1.0
    return {
        "canonical_text": f"chunk {i}",
        "source_uri": "doc.txt",
        "byte_start": i * 10,
        "byte_end": (i + 1) * 10,
        "embedding": emb,
    }


def _vision_space() -> dict:
    return {
        "name": "vision",
        "model_hash": VIS_HASH,
        "vectors": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    }


def _build(path: Path, with_spaces: bool) -> None:
    kwargs = {"spaces": [_vision_space()]} if with_spaces else {}
    _urna.build(
        str(path),
        "demo-model",
        TEXT_DIM,
        "demo-chunker/1",
        "sha256:" + "0" * 64,
        [_chunk(i) for i in range(3)],
        reproducible=True,
        **kwargs,
    )


class TestSpaceBridge(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="urna-space-bridge-"))
        self.plain = self.tmp / "plain.urna"
        self.multi = self.tmp / "multi.urna"
        _build(self.plain, False)
        _build(self.multi, True)

    def test_search_space_scores_vision_band(self) -> None:
        f = _urna.UrnaFile.open(str(self.multi))
        self.assertTrue(f.has_spaces)
        self.assertEqual(f.space_names, ["vision"])
        hits = f.search_space("vision", [1.0, 0.0], 3, expected_model_hash=VIS_HASH)
        self.assertEqual(len(hits), 3)
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)
        self.assertLess(hits[2].score, -0.99)
        # per-space honesty gate: a wrong expected hash raises.
        with self.assertRaises(ValueError):
            f.search_space("vision", [1.0, 0.0], 3, expected_model_hash="sha256:" + "9" * 64)

    def test_spaces_are_isolated(self) -> None:
        f = _urna.UrnaFile.open(str(self.multi))
        # text path works and only sees the 0x04 slab.
        hits = f.search([1.0, 0.0, 0.0, 0.0], 3)
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)
        # a text-dim query against the vision space raises (and vice versa).
        with self.assertRaises(ValueError):
            f.search_space("vision", [1.0, 0.0, 0.0, 0.0], 3)
        with self.assertRaises(ValueError):
            f.search([1.0, 0.0], 3)
        # unknown space raises, never a silent fallback to text.
        with self.assertRaises(ValueError):
            f.search_space("audio", [1.0, 0.0], 3)

    def test_content_hash_stable_with_spaces(self) -> None:
        a = _urna.UrnaFile.open(str(self.plain))
        b = _urna.UrnaFile.open(str(self.multi))
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertNotEqual(a.file_hash, b.file_hash)
        self.assertFalse(a.has_spaces)
        doc = b.inspect()
        names = {s["name"] for s in doc["sections"]}
        self.assertIn("space_table", names)
        self.assertIn("space_embeddings", names)
        self.assertTrue(doc["manifest"]["capabilities_ext"]["supports_multimodal"])

    def test_space_row_count_mismatch_rejected(self) -> None:
        bad = self.tmp / "bad.urna"
        space = _vision_space()
        space["vectors"] = [[1.0, 0.0]]  # 1 row for 3 chunks
        with self.assertRaises(ValueError):
            _urna.build(
                str(bad),
                "demo-model",
                TEXT_DIM,
                "demo-chunker/1",
                "sha256:" + "0" * 64,
                [_chunk(i) for i in range(3)],
                spaces=[space],
            )

    def test_space_dtype_float16_band(self) -> None:
        path = self.tmp / "multi16.urna"
        space = _vision_space()
        space["dtype"] = "float16"
        _urna.build(
            str(path),
            "demo-model",
            TEXT_DIM,
            "demo-chunker/1",
            "sha256:" + "0" * 64,
            [_chunk(i) for i in range(3)],
            spaces=[space],
        )
        f = _urna.UrnaFile.open(str(path))
        hits = f.search_space("vision", [1.0, 0.0], 3)
        self.assertAlmostEqual(hits[0].score, 1.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
