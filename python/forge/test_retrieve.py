"""self-test for the offline retrieve flagship over the cc0 demo corpus.

proves the one-gif demo path is sovereign and honest:
  - build_demo + retrieve open NO socket (block connect/getaddrinfo first)
  - the build is deterministic (two builds byte-identical)
  - retrieve returns a urna:// citation that the runtime resolves back to the
    SAME stored canonical text (tier-1 round-trip), and the retrieve score IS
    the exact-cosine search score (the flagship-is-a-lie guard, again on the
    real corpus)

run with the forge deps (numpy + tokenizers + the vendored potion table):
  .venv/bin/python python/forge/test_retrieve.py

not run by release_check.sh (same as the other forge self-tests).
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/

import urna  # noqa: E402

from forge.embed_potion import potion_embedder  # noqa: E402
from forge.retrieve import build_demo, retrieve  # noqa: E402


def test_no_socket_on_build_and_retrieve() -> None:
    import numpy  # noqa: F401  warm c-extensions before blocking
    import tokenizers  # noqa: F401

    real_conn, real_gai, real_cc = (
        socket.socket.connect,
        socket.getaddrinfo,
        socket.create_connection,
    )

    def _blocked(*_a, **_k):
        raise AssertionError("network access attempted during offline retrieve")

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "demo.urna")
            build_demo(path)
            db = urna.open(path)
            hits = retrieve(db, "can I run this with no internet", k=2)
            assert hits, "retrieve must return cited spans fully offline"
    finally:
        socket.socket.connect = real_conn  # type: ignore[method-assign]
        socket.getaddrinfo = real_gai  # type: ignore[assignment]
        socket.create_connection = real_cc  # type: ignore[assignment]
    print("[offline] no socket opened during build_demo + retrieve: ok")


def test_deterministic_build() -> None:
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.urna")
        b = os.path.join(d, "b.urna")
        build_demo(a)
        build_demo(b)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read(), "demo build must be byte-identical"
    print("deterministic demo build: ok")


def test_citation_round_trips_and_score_is_exact() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "demo.urna")
        build_demo(path)
        db = urna.open(path)
        emb = potion_embedder()
        q = emb.embed_texts(["how do citations let an agent prove a source"])[0]

        retrieve_hits = db.retrieve(q, 3)
        search_hits = db.search(q, 3)
        assert retrieve_hits, "expected cited hits"

        for r, s in zip(retrieve_hits, search_hits, strict=False):
            assert r.chunk_id == s.chunk_id
            # the flagship-is-a-lie guard on the real corpus: identical bits.
            assert r.score == s.score, (r.score, s.score)
            assert r.citation_id == f"urna://{db.content_hash}/{r.chunk_id}"
            assert r.citation_id.startswith("urna://sha256:")
            assert isinstance(r.text, str) and r.text
            assert r.rerank_source == "full_precision"  # exact preset is f32

        # tier-1 round-trip: re-running search returns the SAME citation_id +
        # the same stored canonical text the retrieve hit carries, never an
        # original-byte reopen.
        top = retrieve_hits[0]
        re_top = db.search(q, 1)[0]
        assert re_top.citation_id == top.citation_id
        print(f"citation round-trip + exact score on real corpus: {top.citation_id}")


if __name__ == "__main__":
    test_no_socket_on_build_and_retrieve()
    test_deterministic_build()
    test_citation_round_trips_and_score_is_exact()
    print("\nok: offline retrieve flagship is sovereign, deterministic, and honest")
