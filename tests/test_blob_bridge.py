"""bridge coverage for the media blob pair (0x14 blob_refs + 0x16 overlay).

real artifacts only: builds two tiny .urna files through urna.build (one
with the blob pair, one without), opens them through urna.open, and checks

- content_hash is IDENTICAL with and without the blob pair (citations
  stable), while file_hash legitimately differs;
- the blob table round-trips through the bridge (blob_refs(), has_blobs,
  inspect()["blobs"]);
- the overlay replaces the 0x03 placeholder spans on hits;
- chunk_blob_spans with the wrong length is rejected with a typed error.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import _urna  # noqa: E402


def _chunk(i: int) -> dict:
    emb = [0.0] * 4
    emb[i % 4] = 1.0
    return {
        "canonical_text": f"frame {i}",
        "source_uri": "frames",
        "byte_start": i,
        "byte_end": i + 1,
        "embedding": emb,
    }


def _build(path: Path, with_blobs: bool) -> None:
    kwargs = {}
    if with_blobs:
        media_bytes = b"fake-av1-stream"
        digest = hashlib.sha256(media_bytes).hexdigest()
        kwargs["blob_refs"] = [
            {
                "content_hash": f"sha256:{digest}",
                "original_uri": "media/corpus.av1",
                "byte_len": len(media_bytes),
                "inlined": True,
            }
        ]
        kwargs["chunk_blob_spans"] = [
            {"blob_ref_index": 0, "byte_start": 0, "byte_end": 7},
            {"blob_ref_index": 0, "byte_start": 7, "byte_end": 15},
            {"blob_ref_index": None, "byte_start": 0, "byte_end": 0},
        ]
    _urna.build(
        str(path),
        "demo-model",
        4,
        "demo-chunker/1",
        "sha256:" + "0" * 64,
        [_chunk(i) for i in range(3)],
        reproducible=True,
        **kwargs,
    )


class TestBlobBridge(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="urna-blob-bridge-"))
        self.plain = self.tmp / "plain.urna"
        self.blobs = self.tmp / "blobs.urna"
        _build(self.plain, False)
        _build(self.blobs, True)

    def test_content_hash_stable_with_blobs(self) -> None:
        a = _urna.UrnaFile.open(str(self.plain))
        b = _urna.UrnaFile.open(str(self.blobs))
        self.assertEqual(
            a.content_hash,
            b.content_hash,
            "the 0x14/0x16 pair must not move content_hash (citations stable)",
        )
        self.assertNotEqual(a.file_hash, b.file_hash)

    def test_blob_table_roundtrips(self) -> None:
        f = _urna.UrnaFile.open(str(self.blobs))
        self.assertTrue(f.has_blobs)
        refs = f.blob_refs()
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["original_uri"], "media/corpus.av1")
        self.assertTrue(refs[0]["content_hash"].startswith("sha256:"))
        self.assertTrue(refs[0]["inlined"])
        doc = f.inspect()
        self.assertEqual(doc["blobs"][0]["original_uri"], "media/corpus.av1")
        names = {s["name"] for s in doc["sections"]}
        self.assertIn("blob_refs", names)
        self.assertIn("blob_span_overlay", names)

    def test_overlay_replaces_placeholder_spans(self) -> None:
        f = _urna.UrnaFile.open(str(self.blobs))
        hits = f.search([1.0, 0.0, 0.0, 0.0], 3)
        by_uri = {h.source_uri: h for h in hits}
        self.assertIn("media/corpus.av1", by_uri)
        self.assertIn("frames", by_uri)  # BLOB_REF_NONE keeps the 0x03 span
        blob_hit = [h for h in hits if h.source_uri == "media/corpus.av1" and h.offset_end == 7]
        self.assertEqual(len(blob_hit), 1)

    def test_plain_file_has_no_blobs(self) -> None:
        f = _urna.UrnaFile.open(str(self.plain))
        self.assertFalse(f.has_blobs)
        self.assertEqual(f.blob_refs(), [])
        self.assertIsNone(f.inspect()["blobs"])

    def test_span_count_mismatch_rejected(self) -> None:
        bad = self.tmp / "bad.urna"
        with self.assertRaises(ValueError):
            _urna.build(
                str(bad),
                "demo-model",
                4,
                "demo-chunker/1",
                "sha256:" + "0" * 64,
                [_chunk(0)],
                chunk_blob_spans=[
                    {"blob_ref_index": 0, "byte_start": 0, "byte_end": 1},
                    {"blob_ref_index": 0, "byte_start": 1, "byte_end": 2},
                ],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
