"""end-to-end test of the Python ingestion pipeline."""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import urna
from builder import BuildConfig, EmbeddingCache, Pipeline, chunk_text


def _toy_embed(specs):
    """Toy embedder: hash-based deterministic 8-dim vector."""
    out = []
    for spec in specs:
        h = hash(spec.canonical_text)
        v = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(8)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def test_chunk_text_byte_spans_round_trip():
    text = "Olá mundo. Esta é uma frase com acentuação e emoji 🚀."
    chunks = chunk_text(text, "doc.txt", max_chars=10, overlap=0)
    encoded = text.encode("utf-8")
    # Concatenated chunk bytes must equal the original encoding.
    rebuilt = b"".join(c.canonical_text.encode("utf-8") for c in chunks)
    assert rebuilt == encoded, (rebuilt, encoded)
    # Byte spans must point to the right place in the original encoding.
    for c in chunks:
        assert encoded[c.byte_start : c.byte_end] == c.canonical_text.encode("utf-8")


def test_pipeline_emits_validated_urna_file():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pipe.urna")
        cfg = BuildConfig(
            output_path=out,
            embedding_model="toy",
            embedding_dim=8,
            chunker_version="char/512",
            model_hash="sha256:" + "0" * 64,
            reproducible=True,
        )
        pipe = Pipeline(cfg, embedder=_toy_embed, scratch_db=os.path.join(d, "cache.db"))
        for source, text in [
            ("a.txt", "uma frase em português com acentuação"),
            ("b.txt", "outra frase, completamente diferente da primeira"),
        ]:
            pipe.add_many(chunk_text(text, source, max_chars=20))
        pipe.emit()
        pipe.close()

        db = urna.open(out)
        assert db.embedding_dim == 8
        assert db.n_embeddings >= 2
        hits = db.search([1.0] + [0.0] * 7, 1)
        assert len(hits) == 1
        assert hits[0].source_uri in ("a.txt", "b.txt")


def test_cache_skips_re_embedding_on_second_run():
    """If the scratch DB has the embedding, the embedder should not be
    invoked for that chunk on the second run."""
    with tempfile.TemporaryDirectory() as d:
        scratch = os.path.join(d, "cache.db")

        call_count = {"n": 0}

        def counting_embed(specs):
            call_count["n"] += len(specs)
            return _toy_embed(specs)

        text = "frase um. frase dois. frase tres."
        out_a = os.path.join(d, "a.urna")
        out_b = os.path.join(d, "b.urna")
        for out in (out_a, out_b):
            cfg = BuildConfig(
                output_path=out,
                embedding_model="toy",
                embedding_dim=8,
                chunker_version="char/8",
                model_hash="sha256:" + "0" * 64,
                reproducible=True,
            )
            pipe = Pipeline(cfg, embedder=counting_embed, scratch_db=scratch)
            pipe.add_many(chunk_text(text, "doc.txt", max_chars=8))
            pipe.emit()
            pipe.close()

        with open(out_a, "rb") as fa, open(out_b, "rb") as fb:
            assert fa.read() == fb.read(), "reproducible builds via cache must match"
        # First run embedded everything; second run should embed zero new chunks.
        # Hence call_count["n"] == n_chunks (from the first run only).
        n_chunks = len(chunk_text(text, "doc.txt", max_chars=8))
        assert call_count["n"] == n_chunks, (
            f"expected {n_chunks} embed calls, got {call_count['n']}"
        )


def _toy_embed_dim(specs, dim):
    """Deterministic `dim`-length vector from a chunk's text hash."""
    out = []
    for spec in specs:
        h = hash(spec.canonical_text)
        v = [(((h >> (i % 60)) & 0xFF) + i) % 251 / 251.0 for i in range(dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def test_mrl_truncation_sets_embedding_dim_and_query_stride():
    """build with mrl_dim=128 over a 384-dim corpus: the opened file reports
    embedding_dim==128, a length-128 query works, and a length-384 (full-dim)
    query is rejected with a dimension mismatch."""
    full_dim = 384
    mrl_dim = 128
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "mrl.urna")
        specs = []
        for source, text in [
            ("a.txt", "uma frase em portugues com acentuacao para o teste matryoshka"),
            ("b.txt", "outra frase completamente diferente da primeira para variar"),
            ("c.txt", "terceira frase distinta com outras palavras e tokens"),
        ]:
            specs.extend(chunk_text(text, source, max_chars=20))
        embs = _toy_embed_dim(specs, full_dim)
        chunks = [
            dict(
                canonical_text=s.canonical_text,
                source_uri=s.source_uri,
                byte_start=s.byte_start,
                byte_end=s.byte_end,
                embedding=e,
            )
            for s, e in zip(specs, embs, strict=False)
        ]
        urna.build(
            output_path=out,
            embedding_model="toy",
            embedding_dim=full_dim,
            chunker_version="char/512",
            model_hash="sha256:" + "0" * 64,
            chunks=chunks,
            reproducible=True,
            mrl_dim=mrl_dim,
        )

        db = urna.open(out)
        db.validate()
        assert db.embedding_dim == mrl_dim, db.embedding_dim

        # a query at the prefix dim works.
        hits = db.search([1.0] + [0.0] * (mrl_dim - 1), 1)
        assert len(hits) == 1

        # a query at the full source dim is rejected (runtime strides by the
        # stored prefix dim, not the source dim).
        raised = False
        try:
            db.search([1.0] + [0.0] * (full_dim - 1), 1)
        except ValueError as e:
            raised = True
            assert "dimension mismatch" in str(e).lower(), str(e)
        assert raised, "full-dim query must raise a dimension mismatch"


def test_mrl_dim_validation_rejects_oversized_and_zero():
    """mrl_dim must satisfy 0 < mrl_dim <= embedding_dim."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "bad.urna")
        chunks = [
            dict(
                canonical_text="x",
                source_uri="a.txt",
                byte_start=0,
                byte_end=1,
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
        ]
        for bad in (0, 8):  # 0 and > embedding_dim(4)
            raised = False
            try:
                urna.build(
                    output_path=out,
                    embedding_model="toy",
                    embedding_dim=4,
                    chunker_version="char/512",
                    model_hash="sha256:" + "0" * 64,
                    chunks=chunks,
                    reproducible=True,
                    mrl_dim=bad,
                )
            except ValueError:
                raised = True
            assert raised, f"mrl_dim={bad} must be rejected"


def test_cache_keyed_by_model_no_stale_reuse():
    """EmbeddingCache must not hand back a DIFFERENT model's vectors for the
    same chunk_id (audit finding P2). chunk_id is model-independent, so a
    cache keyed on chunk_id alone would silently reuse stale vectors."""
    with tempfile.TemporaryDirectory() as d:
        scratch = os.path.join(d, "cache.db")
        a = EmbeddingCache(scratch, model_key="modelA")
        a.put("chunk1", [0.1, 0.2, 0.3, 0.4])
        got = a.get("chunk1", 4)
        assert got is not None and abs(got[0] - 0.1) < 1e-6, got
        a.close()

        # a different model must MISS, even though the chunk_id is identical.
        b = EmbeddingCache(scratch, model_key="modelB")
        assert b.get("chunk1", 4) is None, "different model must not reuse vectors"
        b.close()

        # the original model still hits.
        a2 = EmbeddingCache(scratch, model_key="modelA")
        again = a2.get("chunk1", 4)
        assert again is not None and len(again) == 4
        a2.close()


def test_retrieve_model_hash_gate():
    """The Python flagship retrieve binding enforces the model_hash honesty
    gate when the caller passes expected_model_hash (audit finding S4)."""
    h1 = "sha256:" + "ab" * 32
    wrong = "sha256:" + "cd" * 32
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "gate.urna")
        chunks = [
            dict(
                canonical_text="alpha",
                source_uri="a.txt",
                byte_start=0,
                byte_end=5,
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            dict(
                canonical_text="beta",
                source_uri="b.txt",
                byte_start=5,
                byte_end=9,
                embedding=[0.0, 1.0, 0.0, 0.0],
            ),
        ]
        urna.build(
            output_path=out,
            embedding_model="toy",
            embedding_dim=4,
            chunker_version="char/512",
            model_hash=h1,
            chunks=chunks,
            reproducible=True,
        )
        db = urna.open(out)
        assert db.model_hash == h1, db.model_hash
        q = [1.0, 0.0, 0.0, 0.0]
        assert db.retrieve(q, 1, expected_model_hash=h1), "matching hash must succeed"
        assert db.retrieve(q, 1), "no expected hash must stay backward-compatible"
        raised = False
        try:
            db.retrieve(q, 1, expected_model_hash=wrong)
        except ValueError as e:
            raised = True
            assert "model_hash mismatch" in str(e), str(e)
        assert raised, "a mismatched expected_model_hash must raise"


if __name__ == "__main__":
    test_chunk_text_byte_spans_round_trip()
    test_pipeline_emits_validated_urna_file()
    test_cache_skips_re_embedding_on_second_run()
    test_cache_keyed_by_model_no_stale_reuse()
    test_retrieve_model_hash_gate()
    test_mrl_truncation_sets_embedding_dim_and_query_stride()
    test_mrl_dim_validation_rejects_oversized_and_zero()
    print("builder tests OK")
