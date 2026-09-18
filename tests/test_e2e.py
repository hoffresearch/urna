"""end-to-end test of the Python (PyO3) path.

the CLI binary is exhaustively tested in `crates/urna-cli/tests/cli_e2e.rs`.
This file stays on a single Python entry point: PyO3 only. No subprocess
shell-out — `urna validate / stats / search / cite / inspect` all have
in-process equivalents through `urna.UrnaFile`.
"""

import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import urna


def _unit_vec(rng: random.Random, dim: int) -> list[float]:
    v = [rng.random() for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def make_urna(path: str, dim: int, n: int, *, reproducible: bool = False, seed: int = 0):
    rng = random.Random(seed)
    chunks = []
    cursor = 0
    for i in range(n):
        text = f"chunk_{i}"
        chunks.append(
            dict(
                canonical_text=text,
                source_uri="doc.txt",
                byte_start=cursor,
                byte_end=cursor + len(text),
                embedding=_unit_vec(rng, dim),
            )
        )
        cursor += len(text)
    urna.build(
        output_path=path,
        embedding_model="test-model",
        embedding_dim=dim,
        chunker_version="test-chunker/1",
        model_hash="sha256:" + "0" * 64,
        chunks=chunks,
        reproducible=reproducible,
    )


def test_python_build_then_python_search():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".urna") as f:
        path = f.name
    try:
        make_urna(path, dim=8, n=10)
        db = urna.open(path)
        assert db.embedding_dim == 8
        assert db.n_embeddings == 10
        assert db.file_hash.startswith("sha256:")
        assert db.content_hash.startswith("sha256:")

        q = [1.0] + [0.0] * 7
        hits = db.search(q, 3)
        assert len(hits) == 3
        h = hits[0]
        assert h.score_type == "cosine"
        assert h.index_type == "exact"
        assert h.reranked is False
        assert h.file_hash == db.file_hash
        assert h.content_hash == db.content_hash
        assert h.citation_id.startswith(f"urna://{db.content_hash}/")
        print("python build/search OK:", path)
    finally:
        os.unlink(path)


def test_validate_via_pyo3():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".urna") as f:
        path = f.name
    try:
        make_urna(path, dim=4, n=5)
        db = urna.open(path)
        assert db.validate() is True
        info = db.inspect()
        assert info["magic"] == "URNA"
        assert info["n_chunks"] == 5
        assert info["manifest"]["dtype"] == "float32"
        assert info["manifest"]["metric"] == "ip"
        names = {s["name"] for s in info["sections"]}
        assert names == {
            "chunk_ids",
            "chunks_canonical",
            "chunks_original_spans",
            "embeddings",
            "provenance",
            "search_contract",
        }
        for s in info["sections"]:
            assert s["offset"] % 64 == 0
            assert s["encoding"] == 0
        print("pyo3 validate/inspect OK")
    finally:
        os.unlink(path)


def test_reproducible_builds_match_byte_for_byte():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.urna")
        b = os.path.join(d, "b.urna")
        make_urna(a, dim=4, n=3, reproducible=True, seed=7)
        make_urna(b, dim=4, n=3, reproducible=True, seed=7)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            data_a = fa.read()
            data_b = fb.read()
        assert data_a == data_b, "reproducible builds diverged"

        # And the file_hash from a third in-process open must match too.
        ha = urna.open(a).file_hash
        hb = urna.open(b).file_hash
        assert ha == hb
        print("reproducible build OK:", len(data_a), "bytes,", ha[:32])


def test_search_hit_carries_full_contract():
    """Every required SearchHit field in the documented contract is populated and stable."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".urna") as f:
        path = f.name
    try:
        make_urna(path, dim=4, n=3)
        db = urna.open(path)
        h = db.search([1.0, 0.0, 0.0, 0.0], 1)[0]

        # Required fields
        assert isinstance(h.chunk_id, str) and h.chunk_id.startswith("sha256:")
        assert isinstance(h.score, float)
        assert h.score_type == "cosine"
        assert isinstance(h.source_uri, str) and h.source_uri
        assert isinstance(h.offset_start, int) and h.offset_start >= 0
        assert isinstance(h.offset_end, int) and h.offset_end >= h.offset_start
        assert h.embedding_model == "test-model"
        assert h.index_type == "exact"
        assert h.reranked is False
        assert h.file_hash.startswith("sha256:")
        assert h.content_hash.startswith("sha256:")
        assert h.citation_id == f"urna://{h.content_hash}/{h.chunk_id}"
        print("search hit contract OK")
    finally:
        os.unlink(path)


def test_retrieve_score_equals_search_score_exactly():
    """The flagship-is-a-lie guard in python: UrnaFile.retrieve(q, k) must
    return cited spans whose `score` equals UrnaFile.search(q, k) byte-for-byte
    (the retrieve score IS the exact rerank value, never a candidate proxy),
    and each hit carries the tier-1 text + a well-formed urna:// citation."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".urna") as f:
        path = f.name
    try:
        make_urna(path, dim=8, n=12, seed=3)
        db = urna.open(path)
        q = _unit_vec(random.Random(99), 8)

        search_hits = db.search(q, 5)
        retrieve_hits = db.retrieve(q, 5)
        assert len(retrieve_hits) == len(search_hits)
        for r, s in zip(retrieve_hits, search_hits, strict=False):
            assert r.chunk_id == s.chunk_id
            # the load-bearing guard: identical bits, not "close".
            assert r.score == s.score, (r.score, s.score)
            assert r.score_type == "cosine"
            assert r.citation_id == f"urna://{db.content_hash}/{r.chunk_id}"
            assert r.content_hash == db.content_hash
            assert r.file_hash == db.file_hash
            # tier-1 stored canonical text is attached and non-empty.
            assert isinstance(r.text, str) and r.text
            assert r.rerank_source == "full_precision"  # f32 corpus
        print("retrieve score == search score (byte-identical) OK")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_python_build_then_python_search()
    test_validate_via_pyo3()
    test_reproducible_builds_match_byte_for_byte()
    test_search_hit_carries_full_contract()
    test_retrieve_score_equals_search_score_exactly()
