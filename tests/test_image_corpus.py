#!/usr/bin/env python3
"""End-to-end tests for the image corpus pipeline.

These run against a dataset generated inside the test, not against a path
on the author's disk: a test that skips everywhere except one laptop is not
coverage. The vision model is deliberately NOT used here. The invariants
worth guarding are the pipeline's (frame alignment, cache correctness,
portability, the model_hash gate), and none of them need 600 MB of
checkpoint to prove. A stub embedder that reduces pixels to a small
deterministic vector exercises every one of them.

Needs ffmpeg with libsvtav1 for the compressed cases; those skip cleanly
when it is absent. Run: python tests/test_image_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

STUB_DIM = 64


def have_ffmpeg() -> bool:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return "libsvtav1" in out.stdout


def have_avif() -> bool:
    return bool(shutil.which("avifenc") and shutil.which("avifdec"))


class StubEmbedder:
    """Deterministic 64-dim embedder: 8x8 grayscale, flattened, normalized.

    Crude, but a real function of the pixels, so neighbourhood structure
    survives compression the way a learned embedder's would, and identical
    input provably yields identical vectors.
    """

    model_id = "stub-pixel-8x8"
    batch_size = 8

    def __init__(self, salt: str = ""):
        self.salt = salt

    @property
    def dim(self) -> int:
        return STUB_DIM

    @property
    def model_hash(self) -> str:
        payload = f"{self.model_id}|{STUB_DIM}|{self.salt}"
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def _vector(self, image) -> np.ndarray:
        small = image.convert("L").resize((8, 8))
        vec = np.asarray(small, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else np.ones(STUB_DIM, dtype=np.float32) / np.sqrt(STUB_DIM)

    def embed_paths(self, paths) -> np.ndarray:
        from PIL import Image

        out = []
        for path in paths:
            with Image.open(path) as img:
                out.append(self._vector(img))
        return np.vstack(out)

    def embed_arrays(self, frames) -> np.ndarray:
        from PIL import Image

        return np.vstack([self._vector(Image.fromarray(arr)) for arr in frames])

    def embed_one(self, path) -> np.ndarray:
        return self.embed_paths([path])[0]


def make_dataset(root: Path, count: int = 12) -> Path:
    """Distinct, codec-survivable images: large flat colour blocks.

    Noise would be destroyed by any lossy codec and would make the test
    measure the codec's noise handling instead of the pipeline.
    """
    from PIL import Image, ImageDraw

    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        hue = (i * 97) % 256
        img = Image.new("RGB", (320, 240), (hue, (hue * 3) % 256, (255 - hue)))
        draw = ImageDraw.Draw(img)
        # a per-image block so the 8x8 reduction separates them
        draw.rectangle([20 + (i % 4) * 60, 20 + (i % 3) * 50, 120 + (i % 4) * 60, 140], fill=0)
        img.save(root / f"img{i:03d}.png")
    return root


class ImageCorpusTest(unittest.TestCase):
    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not available")
        self.tmp = Path(tempfile.mkdtemp(prefix="urna-image-test-"))
        self.src = make_dataset(self.tmp / "src")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(
        self,
        name: str,
        *,
        compress: bool,
        scratch_db=None,
        labels=None,
        dataset=None,
        sample=None,
        all_intra=False,
        backend="av1",
        control=False,
        gop_policy="auto",
        shard_size=None,
        order_similarity=False,
        dtype=None,
        preset="compressed",
        src=None,
        speed=8,
    ) -> dict:
        from tools import urna_build_image_corpus as builder

        return builder.build_corpus(
            all_intra=all_intra,
            speed=speed,
            input_dir=src or self.src,
            output_path=self.tmp / name / f"{name}.urna",
            dataset_name=dataset or name,
            embedder=StubEmbedder(),
            compress=compress,
            labels=labels,
            scratch_db=scratch_db,
            sample=sample,
            width=256,
            backend=backend,
            control=control,
            gop_policy=gop_policy,
            shard_size=shard_size,
            order_similarity=order_similarity,
            dtype=dtype,
            preset=preset,
        )

    # ---- frame alignment: the bug class that silently shifts every uri ----

    def test_compressed_frames_align_with_items(self):
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        import urna

        result = self._build("aligned", compress=True)
        media = result["media"]
        self.assertEqual(media["frame_count"], result["n_items"])
        self.assertTrue(media["media_sha256"].startswith("sha256:"))

        manifest = json.loads(Path(result["manifest"]).read_text())
        for item in manifest["items"]:
            self.assertEqual(item["source_uri"], f"media://aligned-av1.mp4#frame={item['ordinal']}")

        db = urna.open(result["urna"])
        self.assertEqual(db.n_embeddings, result["n_items"])
        db.validate()

    def test_all_intra_stays_frame_aligned(self):
        """keyint=1 changes the gop, so the alignment guard is re-checked.

        The lever exists because inter-frame prediction finds nothing on a
        corpus of unrelated images, but it must not buy size by losing a
        frame.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        import urna

        result = self._build("intra", compress=True, all_intra=True)
        self.assertEqual(result["media"]["keyint"], 1)
        self.assertEqual(result["media"]["frame_count"], result["n_items"])
        db = urna.open(result["urna"])
        self.assertEqual(db.n_embeddings, result["n_items"])
        db.validate()

    def test_probe_counts_every_encoded_frame(self):
        """The alignment guard leans on this count, so it is checked directly."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_decode, image_encode, image_media

        paths = sorted(self.src.glob("*.png"))
        canvas = image_media.canvas_size(paths, 256)
        out = self.tmp / "probe.mp4"
        info = image_encode.encode_av1(paths, out, canvas=canvas)
        self.assertEqual(info["frame_count"], len(paths))
        self.assertEqual(image_media.probe_frame_count(out), len(paths))
        decoded = sum(len(b) for b in image_decode.decode_frames(out, canvas, batch_size=5))
        self.assertEqual(decoded, len(paths), "decode yielded a different frame count")

    def test_sampling_renumbers_ordinals_densely(self):
        """A sampled corpus must renumber, not keep the original ordinals.

        The ordinal is also the frame number of the stream encoded from the
        sampled list. If sampling kept the source ordinals, item 7 of 12
        would claim `#frame=7` in a 6-frame stream: either a decode error or,
        worse, someone else's frame.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_media

        result = self._build("sampled", compress=True, sample=6)
        manifest = json.loads(Path(result["manifest"]).read_text())

        self.assertEqual(result["n_items"], 6)
        self.assertEqual(result["media"]["frame_count"], 6)
        self.assertEqual([i["ordinal"] for i in manifest["items"]], list(range(6)))
        for item in manifest["items"]:
            _, frame = image_media.parse_media_uri(item["source_uri"])
            self.assertEqual(frame, item["ordinal"])
        # the subset is drawn from the source and keeps the sorted order
        origins = [i["origin"] for i in manifest["items"]]
        self.assertEqual(origins, sorted(origins))
        self.assertEqual(len(set(origins)), 6)

    def test_sampling_is_seed_deterministic(self):
        """Same seed, same subset: otherwise a rebuild is a different corpus."""
        first = self._build("s1", compress=False, sample=6, dataset="sampletest")
        second = self._build("s2", compress=False, sample=6, dataset="sampletest")
        self.assertEqual(Path(first["urna"]).read_bytes(), Path(second["urna"]).read_bytes())

    # ---- the cache bug: a warm scratch db must not reshuffle vectors ----

    def test_partly_warm_cache_produces_identical_corpus(self):
        """A scattered cache miss set must still map to the right vectors.

        Pipeline hands the embedder ONLY the chunks the cache missed. If the
        builder answers by position instead of by chunk_id, every miss gets
        some other image's vector, and nothing anywhere reports an error: the
        corpus is simply wrong. Half the cache is dropped here so the miss
        set is scattered, which is the case a fully cold or fully warm run
        never reaches.
        """
        import sqlite3

        cache = str(self.tmp / "scratch.sqlite")
        # same dataset name on both, so the ONLY difference between the runs
        # is which vectors came from the cache.
        cold = self._build("cold", compress=False, scratch_db=cache, dataset="cachetest")
        # `with sqlite3.connect(...)` commits but does not close, so the
        # connection is closed explicitly rather than left to the finalizer.
        conn = sqlite3.connect(cache)
        try:
            with conn:
                conn.execute("DELETE FROM embeddings_v2 WHERE rowid % 2 = 0")
                remaining = conn.execute("SELECT count(*) FROM embeddings_v2").fetchone()[0]
        finally:
            conn.close()
        self.assertGreater(remaining, 0, "cache emptied; the test would prove nothing")

        warm = self._build("warm", compress=False, scratch_db=cache, dataset="cachetest")
        self.assertEqual(
            Path(cold["urna"]).read_bytes(),
            Path(warm["urna"]).read_bytes(),
            "partly-warm rebuild diverged: cached vectors were mismatched to chunks",
        )

    def test_search_finds_the_query_itself(self):
        import urna

        result = self._build("selftest", compress=False)
        manifest = json.loads(Path(result["manifest"]).read_text())
        target = manifest["items"][5]
        vec = StubEmbedder().embed_one(target["render_path"]).tolist()
        hits = urna.open(result["urna"]).search(vec, 3)
        self.assertEqual(hits[0].offset_start, target["ordinal"])

    # ---- portability: a corpus must survive being moved ----

    def test_corpus_is_relocatable(self):
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_decode, image_media

        result = self._build("portable", compress=True)
        manifest = json.loads(Path(result["manifest"]).read_text())
        for item in manifest["items"]:
            self.assertNotIn("/", item["source_uri"].split("://")[1].split("#")[0])
            self.assertNotIn(str(self.tmp), item["source_uri"])

        moved = self.tmp / "elsewhere"
        moved.mkdir()
        shutil.copy(result["urna"], moved / "portable.urna")
        shutil.copytree(image_media.media_dir_for(Path(result["urna"])), moved / "portable.media")
        name, ordinal = image_media.parse_media_uri(manifest["items"][3]["source_uri"])
        frame = image_decode.decode_frame(
            image_media.media_dir_for(moved / "portable.urna") / name,
            tuple(manifest["media"]["canvas"]),
            ordinal,
        )
        self.assertEqual(frame.shape, (*reversed(manifest["media"]["canvas"]), 3))

    # ---- the manifest gate has to hold for image corpora too ----

    def test_wrong_model_hash_is_rejected(self):
        import urna

        result = self._build("gated", compress=False)
        db = urna.open(result["urna"])
        vec = StubEmbedder().embed_one(sorted(self.src.glob("*.png"))[0]).tolist()
        db.retrieve(vec, 3, expected_model_hash=StubEmbedder().model_hash)
        with self.assertRaises(ValueError):
            db.retrieve(vec, 3, expected_model_hash=StubEmbedder(salt="other").model_hash)

    # ---- labels and canvas geometry ----

    def test_labels_reach_the_manifest(self):
        labels = {f"img{i:03d}": "benign" if i % 2 else "malignant" for i in range(12)}
        result = self._build("labelled", compress=False, labels=labels)
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["items"][1]["label"], "benign")
        self.assertEqual(manifest["items"][0]["label"], "malignant")

    def test_pdf_pages_keep_their_provenance(self):
        """A pdf hit is only useful if the corpus can say which page it was.

        The rendered pages are build-time temporaries, so the durable path
        back to the pixels is the source pdf plus the page number. The eval
        harness has to be able to re-render from that.
        """
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not available")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import urna_image_eval as ev

        from tools import urna_build_image_corpus as builder

        pdf_dir = self.tmp / "pdfs"
        pdf_dir.mkdir()
        doc = fitz.open()
        for page in range(3):
            doc.new_page().insert_text((72, 144), f"page number {page}", fontsize=48)
        doc.save(str(pdf_dir / "guide.pdf"))
        doc.close()

        result = builder.build_corpus(
            input_dir=pdf_dir,
            output_path=self.tmp / "pdfcorpus" / "guide.urna",
            dataset_name="guide",
            embedder=StubEmbedder(),
            is_pdf=True,
            compress=False,
            width=256,
        )
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(result["n_items"], 3)
        self.assertEqual(manifest["input_kind"], "pdf")
        self.assertEqual([i["page"] for i in manifest["items"]], [0, 1, 2])
        self.assertEqual(manifest["items"][2]["origin"], "guide.pdf")

        # the build-time render is gone; the harness must still find pixels.
        self.assertFalse(Path(manifest["items"][1]["render_path"]).exists())
        rendered = ev.query_image(manifest["items"][1], manifest, self.tmp / "requery")
        self.assertTrue(rendered.exists())

    def test_canvas_is_even_and_letterbox_does_not_distort(self):
        from PIL import Image

        from forge import image_media

        paths = sorted(self.src.glob("*.png"))
        canvas = image_media.canvas_size(paths, 256)
        self.assertEqual(canvas[0] % 2, 0)
        self.assertEqual(canvas[1] % 2, 0)
        # a wildly off-aspect image must be padded into the canvas, never
        # stretched to fill it.
        tall = Image.new("RGB", (40, 400), (255, 0, 0))
        padded = image_media.letterbox(tall, canvas)
        self.assertEqual(padded.size, canvas)
        corner = padded.getpixel((1, canvas[1] // 2))
        self.assertEqual(corner, (0, 0, 0), "expected padding, got stretched content")

    def test_delta_carries_a_confidence_interval(self):
        """A delta without an interval is not a result.

        The published claim "av1 costs 1.9 points of precision@10" came from a
        point estimate whose interval crossed zero: at n=200 that difference
        was not distinguishable from noise. The harness now reports the
        interval and says so, so the same claim cannot be made twice.
        """
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import urna_image_eval as ev

        rng = np.random.default_rng(0)
        same = rng.normal(0.6, 0.2, 400)
        ci = ev.bootstrap_delta(same, same.copy(), seed=1)
        self.assertEqual(ci["mean"], 0.0)
        self.assertFalse(ci["significant"], "identical samples cannot differ")

        worse = np.clip(same - 0.30, 0, 1)
        ci = ev.bootstrap_delta(worse, same, seed=1)
        self.assertLess(ci["mean"], 0)
        self.assertLess(ci["ci95"][1], 0, "a 30-point drop must clear zero")
        self.assertTrue(ci["significant"])

        # a difference far below what this sample size resolves must NOT be
        # reported as real, which is the exact failure being guarded.
        tiny = same + rng.normal(0.002, 0.2, 400)
        self.assertFalse(ev.bootstrap_delta(tiny, same, seed=1)["significant"])

    def test_canvas_never_upscales_the_source(self):
        """`--width` is a ceiling, not a target.

        Upscaling invents no detail, so every interpolated pixel is bits the
        encoder spends on nothing and a decoder pays for on every read. On PH2
        the 1024 default sat above a 765-wide source and cost 33 percent of
        the file for zero information.
        """
        from forge import image_media

        paths = sorted(self.src.glob("*.png"))  # the fixture is 320x240
        canvas = image_media.canvas_size(paths, 4096)
        self.assertLessEqual(canvas[0], 320, "canvas widened past the source")
        self.assertLessEqual(canvas[1], 240)
        # below the source it still downscales, which is a real size lever.
        self.assertEqual(image_media.canvas_size(paths, 160)[0], 160)

    def test_canvas_cap_survives_mixed_source_sizes(self):
        """A few large images must not drag the whole canvas up.

        The cap follows the median source width for the same reason the
        aspect does: one outlier should not set the geometry for the corpus.
        """
        from PIL import Image

        from forge import image_media

        mixed = self.tmp / "mixed"
        mixed.mkdir()
        for i in range(9):
            Image.new("RGB", (200, 150), (i * 20, 0, 0)).save(mixed / f"s{i}.png")
        Image.new("RGB", (4000, 3000), (0, 0, 255)).save(mixed / "huge.png")
        canvas = image_media.canvas_size(sorted(mixed.glob("*.png")), 1024)
        self.assertLessEqual(canvas[0], 200, f"one outlier set the canvas: {canvas}")

    # ---- F2.1: random access is seek, and it returns the same frame ----

    def test_seek_decode_matches_sequential_decode(self):
        """`decode_frame` must return the frame the ordinal names, by seek.

        Measured on PH2 (fase 0, CP-0.4): input seeking is <= the legacy
        select scan at every ordinal and much cheaper away from keyframes
        (gop161 ordinal 199: 67 ms against 189 ms; all-intra 100/199: ~42 ms
        against 97-154 ms), and the frames are byte-identical. The seek path
        is only worth having if that identity holds, so it is the assertion.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_decode

        for name, intra in (("seekgop", False), ("seekintra", True)):
            result = self._build(name, compress=True, all_intra=intra)
            media = result["media"]
            video = self.tmp / name / f"{name}.media" / f"{name}-av1.mp4"
            canvas = tuple(media["canvas"])
            sequential = [f for batch in image_decode.decode_frames(video, canvas) for f in batch]
            for ordinal in (0, 5, 11):
                sought = image_decode.decode_frame(video, canvas, ordinal)
                np.testing.assert_array_equal(
                    sought, sequential[ordinal], f"{name} frame {ordinal} diverged"
                )

    def test_decode_frames_at_returns_the_requested_frames(self):
        """One invocation resolves k ordinals, and the frames are the right ones."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_decode

        result = self._build("batch", compress=True)
        video = self.tmp / "batch" / "batch.media" / "batch-av1.mp4"
        canvas = tuple(result["media"]["canvas"])
        # two spans: one that ends on the last frame and one that stops
        # early. the second one is the real-corpus case (k hits out of
        # 38627 frames, measured 2026-09-13): ffmpeg still has frames to
        # write when the reader is done, and closing the pipe must not
        # turn into a raised broken-pipe error.
        for wanted in ([1, 5, 11], [1, 5]):
            frames = image_decode.decode_frames_at(video, canvas, wanted)
            self.assertEqual(len(frames), len(wanted))
            for ordinal, frame in zip(wanted, frames, strict=True):
                single = image_decode.decode_frame(video, canvas, ordinal)
                np.testing.assert_array_equal(frame, single, f"batch frame {ordinal} diverged")

    # ---- F2.2: the codec toolchain is part of provenance ----

    def test_toolchain_provenance_is_recorded_and_sensitive(self):
        """Decoded pixels depend on ffmpeg + encoder + params; that is recorded.

        Another encoder version produces other pixels, other embeddings, and
        the index silently stops matching the media. The manifest now carries
        the toolchain and a provenance hash, and the hash must change when a
        parameter changes.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        first = self._build("prov1", compress=True)
        media = first["media"]
        self.assertIn("toolchain", media)
        self.assertIn("ffmpeg", media["toolchain"])
        self.assertTrue(media["provenance_sha256"].startswith("sha256:"))

        from tools import urna_build_image_corpus as builder

        second = builder.build_corpus(
            input_dir=self.src,
            output_path=self.tmp / "prov2" / "prov2.urna",
            dataset_name="prov2",
            embedder=StubEmbedder(),
            compress=True,
            width=256,
            crf=40,
        )
        self.assertNotEqual(
            media["provenance_sha256"],
            second["media"]["provenance_sha256"],
            "provenance did not change when the codec parameters changed",
        )

    # ---- F2.3: per-frame hashes catch reordering, not just loss ----

    def test_frame_hashes_verify_and_detect_tampering(self):
        """Counting frames catches loss, not reordering; hashes catch both."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_decode, image_media

        result = self._build("hashed", compress=True)
        manifest = json.loads(Path(result["manifest"]).read_text())
        hashes = manifest["frame_sha256"]
        self.assertEqual(len(hashes), result["n_items"])
        self.assertTrue(all(h.startswith("sha256:") for h in hashes))

        video = image_media.media_dir_for(Path(result["urna"])) / "hashed-av1.mp4"
        canvas = tuple(manifest["media"]["canvas"])
        image_decode.verify_frame_hashes(video, canvas, hashes)

        tampered = list(hashes)
        tampered[3] = tampered[5]
        with self.assertRaises(ValueError):
            image_decode.verify_frame_hashes(video, canvas, tampered)

    # ---- F2.4: pix_fmt is verified, and 444 really carries more chroma ----

    def test_pix_fmt_silent_fallback_is_rejected(self):
        """Asking for 444 from an encoder that cannot emit it must fail.

        This ffmpeg's libsvtav1 converts 444 to 420 SILENTLY and writes a
        byte-identical file (found in fase 0): a flag without a probe is a
        lie. The encoder output is probed and the lie raises.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_encode, image_media

        paths = sorted(self.src.glob("*.png"))
        canvas = image_media.canvas_size(paths, 256)
        with self.assertRaises(RuntimeError):
            image_encode.encode_av1(paths, self.tmp / "x444.mp4", canvas=canvas, pix_fmt="yuv444p")

    def test_444_preserves_more_chroma_than_420(self):
        """Red edge on black: chroma subsampling blurs it, 444 blurs less."""
        if not have_avif():
            self.skipTest("avifenc/avifdec not available")
        from PIL import Image

        from forge import image_decode, image_encode_still

        pattern = self.tmp / "edge.png"
        img = Image.new("RGB", (128, 128), (0, 0, 0))
        for x in range(64, 128):
            for y in range(128):
                img.putpixel((x, y), (255, 0, 0))
        img.save(pattern)
        src = np.asarray(img, dtype=np.int16)

        err = {}
        for yuv in ("420", "444"):
            out_dir = self.tmp / f"avif{yuv}"
            image_encode_still.encode_avif([pattern], out_dir, quality=40, yuv=yuv)
            avif = sorted(out_dir.glob("*.avif"))[0]
            decoded = image_decode.decode_avif(avif)
            err[yuv] = float(np.abs(decoded.astype(np.int16) - src).mean())
            # the uncompressed-png intermediate must hand back the same
            # pixels avifdec's default png does
            ref_png = self.tmp / f"ref{yuv}.png"
            subprocess.run(["avifdec", str(avif), str(ref_png)], check=True, capture_output=True)
            with Image.open(ref_png) as ref:
                np.testing.assert_array_equal(decoded, np.asarray(ref.convert("RGB")))
        self.assertLess(err["444"], err["420"])

    def test_avif_jobs_are_pinned_and_recorded(self):
        """The worker count is part of the recipe, not of the machine.

        libaom writes different bytes with one worker than with two or
        more (measured 2026-09-13, libavif 1.4.2 / aom 3.15.0, 256 cards:
        every file differed between -j 1 and -j 2; 2, 3, 4, 8, 16 and all
        were byte-identical). avifenc defaults to "all", which made the
        file_hash a function of the build machine's core count. The backend
        pins `-j`, records it, and its bytes equal a direct `-j 2` encode.
        """
        if not have_avif():
            self.skipTest("avifenc/avifdec not available")
        from PIL import Image

        from forge import image_encode_still

        png = self.tmp / "pinned.png"
        rng = np.random.default_rng(3)
        Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(png)
        out_dir = self.tmp / "pinned-avif"
        media = image_encode_still.encode_avif([png], out_dir, quality=40, yuv="420", speed=9)
        self.assertEqual(media["toolchain"]["params"]["jobs"], image_encode_still.AVIF_JOBS)
        ref = self.tmp / "pinned-j2.avif"
        # fmt: off
        subprocess.run(
            ["avifenc", "-j", "2", "-q", "40", "--speed", "9", "--yuv", "420", str(png), str(ref)],
            check=True, capture_output=True,
        )
        # fmt: on
        self.assertEqual((out_dir / "pinned.avif").read_bytes(), ref.read_bytes())

    def test_encode_avif_records_the_source_bytes_it_is_given(self):
        """`source_bytes` describes the ORIGINAL files, not the encoder input.

        The backend feeds avifenc letterboxed pngs from a tempdir; summing
        those recorded 21.45 GB for a 3.975 GB jpeg corpus (ratio 18.61x
        instead of 3.45x). The record must echo the caller's number and keep
        the encoder-input sum apart; without a caller number both are the
        input sum.
        """
        if not have_avif():
            self.skipTest("avifenc/avifdec not available")
        from forge import image_encode_still

        pattern = sorted(self.src.glob("*.png"))[0]
        png_bytes = pattern.stat().st_size

        given = image_encode_still.encode_avif(
            [pattern], self.tmp / "avif-given", quality=40, yuv="420", source_bytes=123
        )
        self.assertEqual(given["source_bytes"], 123)
        self.assertEqual(given["letterboxed_input_bytes"], png_bytes)
        self.assertEqual(given["compression_ratio"], round(123 / given["output_bytes"], 2))

        plain = image_encode_still.encode_avif(
            [pattern], self.tmp / "avif-plain", quality=40, yuv="420"
        )
        self.assertEqual(plain["source_bytes"], png_bytes)
        self.assertEqual(plain["letterboxed_input_bytes"], png_bytes)
        self.assertEqual(plain["compression_ratio"], round(png_bytes / plain["output_bytes"], 2))

    # ---- F2.7: the avif backend is a first-class corpus ----

    def test_avif_backend_roundtrip_and_relocates(self):
        if not have_avif():
            self.skipTest("avifenc/avifdec not available")
        import urna

        result = self._build("avicorpus", compress=True, backend="avif", speed=9)
        manifest = json.loads(Path(result["manifest"]).read_text())
        media = manifest["media"]
        self.assertEqual(media["backend"], "avif")
        # `speed` reaches avifenc (the stills profile pins it; it used to be
        # dropped on the way to the per-image backend)
        self.assertEqual(media["speed"], 9)
        self.assertEqual(media["toolchain"]["params"]["speed"], 9)
        # the manifest accounts the ORIGINAL sources, like av1 and jxl do; the
        # letterboxed pngs avifenc actually read are a different number
        # (320x240 sources onto the 256-wide canvas) and are kept apart
        expected = sum(p.stat().st_size for p in sorted(self.src.glob("*.png")))
        self.assertEqual(media["source_bytes"], expected)
        self.assertEqual(media["compression_ratio"], round(expected / media["output_bytes"], 2))
        self.assertNotEqual(media["letterboxed_input_bytes"], expected)
        for item in manifest["items"]:
            self.assertTrue(item["source_uri"].startswith("media://avicorpus-avif/"))
            self.assertNotIn(str(self.tmp), item["source_uri"])

        db = urna.open(result["urna"])
        self.assertEqual(db.n_embeddings, result["n_items"])
        db.validate()

        # a hit resolves to real pixels out of the per-image media
        from tools import urna_search_image as search

        out_dir = self.tmp / "hits"
        saved = search.save_frame(
            Path(result["urna"]), manifest, manifest["items"][2]["source_uri"], out_dir
        )
        self.assertTrue(saved and Path(saved).exists())

    # ---- F2.8: the letterbox-lossless control is a build mode ----

    def test_control_corpus_indexes_letterboxed_lossless_pixels(self):
        """The control the fase 0 measurements need, as a first-class mode.

        CP-0.1 decomposed the codec cost against a control that is the corpus
        canvas applied LOSSLESSLY. That used to be scratch; it is now a build
        flag, and the index it emits must describe exactly those pixels.
        """
        if not have_ffmpeg():
            self.skipTest("PIL decode only, but keep the skip symmetric")
        import urna
        from forge import image_media

        result = self._build("ctrlcorpus", compress=True, control=True)
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["media"]["backend"], "png-lossless")
        media_dir = image_media.media_dir_for(Path(result["urna"]))
        pngs = sorted(media_dir.rglob("*.png"))
        self.assertEqual(len(pngs), result["n_items"])

        # the index answers the letterboxed query with itself
        from PIL import Image

        canvas = tuple(manifest["media"]["canvas"])
        with Image.open(manifest["items"][4]["render_path"]) as img:
            query = image_media.letterbox(img, canvas)
        vec = StubEmbedder()._vector(query).tolist()
        hits = urna.open(result["urna"]).search(vec, 3)
        self.assertEqual(hits[0].offset_start, 4)

    # ---- F2.6: the text tower is plumbed through the search cli ----

    def test_query_text_and_query_image_are_exclusive(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import urna_search_image as search

        with self.assertRaises(SystemExit):
            search.parse_args(
                [
                    "--index",
                    "x.urna",
                    "--query-image",
                    "a.png",
                    "--query-text",
                    "blue nevus",
                ]
            )
        args = search.parse_args(["--index", "x.urna", "--query-text", "blue nevus"])
        self.assertEqual(args.query_text, "blue nevus")

    # ---- fase 5: gop policy probe, similarity order, sharding, dtype ladder ----

    def test_auto_gop_policy_records_the_probe_and_applies_the_decision(self):
        """The probe runs at build time and the keyint follows its bytes.

        Fase 0 (CP-0.5) showed embedding cosine does not separate the
        regimes, so the policy is a measured encode decision, and the probe
        statistics go to the manifest with it. The decision itself is not
        asserted on this synthetic set (flat colour blocks are a regime
        where either side can win on bytes); what is asserted is the
        wiring: the probe is recorded whole, and the encoded stream obeys
        it. The decision's quality is guarded by the redundancy test below
        and measured on real corpora in fase 6.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_encode

        result = self._build("gopauto", compress=True)
        media = result["media"]
        gop = media["gop"]
        self.assertEqual(gop["policy"], "auto")
        self.assertIn(gop["decision"], ("intra", "inter"))
        self.assertGreater(gop["intra_bytes"], 0)
        self.assertGreater(gop["inter_bytes"], 0)
        self.assertGreater(gop["n_samples"], 1)
        expected_keyint = 1 if gop["decision"] == "intra" else image_encode.INTER_KEYINT
        self.assertEqual(media["keyint"], expected_keyint)

    def test_auto_gop_policy_finds_redundancy(self):
        """Near-identical frames: inter wins, and the probe must say so."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from PIL import Image, ImageDraw

        src = self.tmp / "redundant"
        src.mkdir()
        for i in range(10):
            img = Image.new("RGB", (320, 240), (200, 100, 50))
            draw = ImageDraw.Draw(img)
            draw.rectangle([40 + i, 60, 100 + i, 120], fill=(0, 0, 0))
            img.save(src / f"frame{i:03d}.png")

        from forge import image_encode

        result = self._build("gopred", compress=True, src=src)
        gop = result["media"]["gop"]
        self.assertEqual(gop["decision"], "inter")
        self.assertLess(gop["inter_bytes"], gop["intra_bytes"])
        # inter ships with the bounded gop (INTER_KEYINT), never the encoder
        # default: measured 2026-08-31, and it caps random-access decode cost.
        self.assertEqual(result["media"]["keyint"], image_encode.INTER_KEYINT)

    def test_gop_policy_inter_forces_the_default_gop(self):
        """`inter` is the lever for material the probe has not seen."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_encode

        result = self._build("gopinter", compress=True, gop_policy="inter")
        self.assertEqual(result["media"]["keyint"], image_encode.INTER_KEYINT)
        self.assertEqual(result["media"]["gop"]["decision"], "inter")
        self.assertEqual(result["media"]["frame_count"], result["n_items"])

    def test_similarity_order_preserves_item_mapping(self):
        """A permuted stream must still hand every item its own frame.

        The stream is encoded in greedy nearest-neighbour order, but uris,
        vectors, and frame hashes are un-permuted back to item order. The
        assertion is the strongest form of that contract: decode the frame
        each uri names and its content hash must equal the manifest hash
        recorded for THAT item.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_decode, image_media

        result = self._build("ordered", compress=True, order_similarity=True)
        manifest = json.loads(Path(result["manifest"]).read_text())
        media = manifest["media"]
        media_dir = image_media.media_dir_for(Path(result["urna"]))
        canvas = tuple(media["canvas"])
        frame_ids = []
        for item in manifest["items"]:
            name, ordinal = image_media.parse_media_uri(item["source_uri"])
            frame_ids.append(ordinal)
            frame = image_decode.decode_frame(media_dir / name, canvas, ordinal)
            self.assertEqual(
                image_decode.frame_sha256(frame),
                manifest["frame_sha256"][item["ordinal"]],
                f"item {item['ordinal']} resolved to another item's frame",
            )
        self.assertEqual(
            sorted(frame_ids),
            list(range(result["n_items"])),
            "frame pointers are not a permutation of the stream",
        )

    def test_sharding_splits_the_stream_and_every_uri_resolves(self):
        """Segments of ~`shard_size` frames, indexed in the manifest."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        import urna
        from forge import image_decode, image_media

        result = self._build("sharded", compress=True, shard_size=5)
        manifest = json.loads(Path(result["manifest"]).read_text())
        media = manifest["media"]
        segments = media["segments"]
        self.assertEqual(len(segments), 3, "12 frames at shard_size 5")
        self.assertEqual(sum(s["n_frames"] for s in segments), result["n_items"])
        self.assertEqual(media["frame_count"], result["n_items"])
        media_dir = image_media.media_dir_for(Path(result["urna"]))
        canvas = tuple(media["canvas"])
        for seg in segments:
            self.assertTrue((media_dir / seg["uri"]).exists(), seg["uri"])
            self.assertTrue(seg["media_sha256"].startswith("sha256:"))
        for item in manifest["items"]:
            name, ordinal = image_media.parse_media_uri(item["source_uri"])
            frame = image_decode.decode_frame(media_dir / name, canvas, ordinal)
            self.assertEqual(
                image_decode.frame_sha256(frame),
                manifest["frame_sha256"][item["ordinal"]],
                f"item {item['ordinal']} resolved to another item's frame",
            )
        db = urna.open(result["urna"])
        self.assertEqual(db.n_embeddings, result["n_items"])
        db.validate()

    def test_order_and_sharding_compose(self):
        """Both levers together: the mapping contract must still hold."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_decode, image_media

        result = self._build("both", compress=True, shard_size=5, order_similarity=True)
        manifest = json.loads(Path(result["manifest"]).read_text())
        media = manifest["media"]
        self.assertEqual(len(media["segments"]), 3)
        media_dir = image_media.media_dir_for(Path(result["urna"]))
        canvas = tuple(media["canvas"])
        seen = []
        for item in manifest["items"]:
            name, ordinal = image_media.parse_media_uri(item["source_uri"])
            seen.append((name, ordinal))
            frame = image_decode.decode_frame(media_dir / name, canvas, ordinal)
            self.assertEqual(
                image_decode.frame_sha256(frame),
                manifest["frame_sha256"][item["ordinal"]],
            )
        self.assertEqual(len(set(seen)), result["n_items"])

    def test_auto_gop_is_probed_per_segment_when_sharded(self):
        """gop=auto + sharding: one probe per shard, recorded per segment.

        A single global probe averages regimes away: with cluster ordering
        the near-duplicate runs concentrate in a few segments, and those
        are exactly where inter pays while unique segments keep O(1)
        all-intra access (RFC-2). The decision direction is not asserted on
        this synthetic set; the wiring is: every shard gets its own probe
        and its encoded keyint obeys it.
        """
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_encode

        result = self._build("gopseg", compress=True, shard_size=5)
        media = result["media"]
        gop = media["gop"]
        self.assertEqual(gop["policy"], "auto")
        self.assertTrue(gop["per_segment"])
        self.assertIn(gop["decision"], ("intra", "inter", "mixed"))
        self.assertEqual(len(gop["segments"]), len(media["segments"]))
        for seg, probe in zip(media["segments"], gop["segments"], strict=True):
            expected = 1 if probe["decision"] == "intra" else image_encode.INTER_KEYINT
            self.assertEqual(seg["keyint"], expected)
            self.assertGreater(probe["intra_bytes"], 0)
            self.assertGreater(probe["inter_bytes"], 0)
            # source order is not engineered: the probe samples spaced frames
            self.assertEqual(probe["sampling"], "evenly-spaced")
            # the probe is quality-aware (or says out loud that it is not)
            self.assertTrue("intra_ssim2" in probe or "quality" in probe)

    def test_still_tune_is_dropped_for_inter_gop(self):
        """SVT's IQ tune is all-intra only: an inter segment must drop it
        (recorded as tune_resolved=None), never hard-error the encode."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from forge import image_encode

        paths = sorted(self.src.glob("*.png"))[:4]
        info = image_encode.encode_av1(
            paths,
            self.tmp / "interstill.mp4",
            canvas=(256, 256),
            preset=12,
            keyint=image_encode.INTER_KEYINT,
            tune="still",
        )
        self.assertIsNone(info["toolchain"]["params"]["tune_resolved"])
        self.assertEqual(info["keyint"], image_encode.INTER_KEYINT)
        intra = image_encode.encode_av1(
            paths,
            self.tmp / "intrastill.mp4",
            canvas=(256, 256),
            preset=12,
            keyint=1,
            tune="still",
        )
        self.assertIsNotNone(intra["toolchain"]["params"]["tune_resolved"])

    def test_probe_samples_contiguous_windows_for_engineered_order(self):
        """order=cluster/similarity puts the redundancy between neighbours;
        an evenly-spaced probe would erase the signal the ordering created,
        so the probe must record contiguous-window sampling there."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        result = self._build("contigprobe", compress=True, shard_size=5, order_similarity=True)
        gop = result["media"]["gop"]
        self.assertTrue(gop["per_segment"])
        for probe in gop["segments"]:
            self.assertEqual(probe["sampling"], "contiguous-windows")

    def test_per_segment_probe_finds_redundancy_in_every_shard(self):
        """Near-identical frames in every shard: each probe must say inter."""
        if not have_ffmpeg():
            self.skipTest("ffmpeg with libsvtav1 not available")
        from PIL import Image, ImageDraw

        src = self.tmp / "redundantsharded"
        src.mkdir()
        for i in range(10):
            img = Image.new("RGB", (320, 240), (200, 100, 50))
            draw = ImageDraw.Draw(img)
            draw.rectangle([40 + i, 60, 100 + i, 120], fill=(0, 0, 0))
            img.save(src / f"frame{i:03d}.png")

        from forge import image_encode

        result = self._build("gopredshard", compress=True, src=src, shard_size=5)
        media = result["media"]
        self.assertEqual(len(media["segments"]), 2)
        for seg in media["segments"]:
            self.assertEqual(seg["keyint"], image_encode.INTER_KEYINT)
        self.assertEqual(media["keyint"], image_encode.INTER_KEYINT)
        self.assertEqual(media["gop"]["decision"], "inter")
        self.assertTrue(media["gop"]["per_segment"])

    def test_dtype_override_reaches_the_built_corpus(self):
        """The dtype lever (F5.3) is a build kwarg, not a preset swap."""
        import urna

        result = self._build("int8corpus", compress=False, dtype="int8", preset="exact")
        db = urna.open(result["urna"])
        self.assertEqual(db.n_embeddings, result["n_items"])
        db.validate()

    def test_sweep_dtype_ladder_and_gop_kinds(self):
        """`dtype:` variants isolate quantization; av1 kinds pin the policy."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import urna_image_sweep as sweep

        variants = sweep.parse_variants("dtype:f16,int8,int4")
        self.assertEqual([v["dtype"] for v in variants], ["float16", "int8", "int4"])
        with self.assertRaises(ValueError):
            sweep.parse_variants("dtype:fp8")
        pair = sweep.parse_variants("av1-inter:30;av1-intra:35;av1-order:35")
        self.assertEqual(pair[0]["gop_policy"], "inter")
        self.assertEqual(pair[1]["gop_policy"], "intra")
        self.assertTrue(pair[2]["order_similarity"])
        # ordering is invisible to all-intra, so the measurement cell pins inter
        self.assertEqual(pair[2]["gop_policy"], "inter")

    # ---- fase 3: the measurement battery beyond the bootstrap ----

    def test_sign_test_pairs_with_the_bootstrap(self):
        """The bootstrap answers "how big"; the sign test answers "how often".

        A paired sign test is assumption-free: identical samples must give
        p=1, and a consistent 30-point drop on every query must clear 0.05.
        """
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import _image_metrics as met

        rng = np.random.default_rng(0)
        same = rng.normal(0.6, 0.2, 200)
        self.assertEqual(met.sign_test(same, same.copy())["p_value"], 1.0)
        worse = np.clip(same - 0.3, 0, 1)
        self.assertLess(met.sign_test(worse, same)["p_value"], 0.05)

    def test_ranking_agreement_reads_identity_and_reversal(self):
        """overlap@k counts shared hits; kendall tau-b reads their order."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import _image_metrics as met

        a = [[3, 1, 4, 0, 2], [0, 1, 2, 3, 4]]
        same = met.ranking_agreement(a, a, k=3)
        self.assertEqual(same["overlap@3"], 1.0)
        self.assertEqual(same["kendall_tau_b"], 1.0)
        b = [[2, 0, 4, 1, 3], [4, 3, 2, 1, 0]]
        rev = met.ranking_agreement(a, b, k=5)
        self.assertLess(rev["kendall_tau_b"], 0.0)

    def test_cosine_drift_distribution(self):
        """Per-image drift between source-pixel and decoded-frame vectors."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import _image_metrics as met

        rng = np.random.default_rng(0)
        base = rng.normal(size=(50, 32)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        identical = met.cosine_drift(base, base.copy())
        self.assertEqual(identical["median"], 1.0)
        drifted = met.cosine_drift(base, -base)
        self.assertEqual(drifted["median"], -1.0)
        self.assertIn("p05", drifted)

    def test_per_class_floor_catches_a_collapsed_class(self):
        """CP-0.6: a mean can hide one destroyed class; the floor cannot."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import _image_metrics as met

        labels = ["nev"] * 80 + ["mel"] * 20
        rng = np.random.default_rng(0)
        control = rng.normal(0.6, 0.1, 100)
        sample = control.copy()
        sample[80:] -= 0.3  # melanoma collapses, the mean barely moves
        out = met.per_class_delta(sample, control, labels, seed=1)
        self.assertTrue(out["mel"]["significant"])
        self.assertFalse(out["nev"]["significant"])
        self.assertFalse(met.class_floor_ok(out, floor=0.05))
        self.assertTrue(met.class_floor_ok({"nev": out["nev"]}, floor=0.05))

    def test_sweep_runs_end_to_end_on_a_tiny_corpus(self):
        """One model load, a control and two variants, the full battery out."""
        if not have_avif() or not have_ffmpeg():
            self.skipTest("sweep needs ffmpeg+libsvtav1 and avifenc/avifdec")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "tools"))
        import argparse

        import urna_image_sweep as sweep

        labels_csv = self.tmp / "labels.csv"
        labels_csv.write_text(
            "image_id,label\n"
            + "\n".join(f"img{i:03d},{'even' if i % 2 == 0 else 'odd'}" for i in range(12))
            + "\n"
        )
        args = argparse.Namespace(
            input_dir=self.src,
            dataset="sweeptest",
            out_dir=self.tmp / "sweep",
            variants="av1-intra:35;avif:35",
            labels=labels_csv,
            pdf=False,
            sample=None,
            queries=None,
            seed=42,
            width=256,
            preset="compressed",
            class_floor=0.05,
            k=[1, 5, 10],
        )
        report = sweep.run_sweep(args, StubEmbedder())
        self.assertIn("control", report)
        self.assertEqual(set(report["variants"]), {"av1-intra-35", "avif-35"})
        # the shared ruler: the control's lossless media size, and every
        # variant's index size, must be reported or a published number
        # cannot be reproduced.
        self.assertGreater(report["control"]["media_bytes"], 0)
        self.assertGreater(report["control"]["urna_bytes"], 0)
        for variant in report["variants"].values():
            self.assertIn("bootstrap", variant["delta_vs_control"]["precision@10"])
            self.assertIn("sign_test", variant["delta_vs_control"]["precision@10"])
            self.assertIn("even", variant["delta_vs_control"]["per_class"])
            self.assertIn("kendall_tau_b", variant["delta_vs_control"]["ranking"])
            self.assertIn("median", variant["drift"])
            self.assertGreater(variant["media_bytes"], 0)
            self.assertGreater(variant["urna_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
