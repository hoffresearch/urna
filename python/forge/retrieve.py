"""offline retrieve convenience + the one-gif flagship demo over demo_corpus.

this is the agent-native flagship end to end, OFFLINE and deterministic:

  1. embed the query with the default potion static table (no torch, no socket)
  2. UrnaFile.retrieve(qvec, k) routes by manifest capability and returns cited
     spans whose `score` IS the exact-cosine rerank value
  3. each hit carries the tier-1 stored canonical text + a urna:// citation that
     `urna cite` resolves back to the same bytes

two entry points:

  - retrieve(urnafile, query, k, embedder=None): the convenience wrapper. embeds
    `query` with potion, calls UrnaFile.retrieve, returns the RetrieveHit list.
  - build_demo(out_path) + main(): build a .urna from python/forge/demo_corpus
    with the potion embedder, ask a question, print the cited answer + citation.
    run: python python/forge/retrieve.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/

import urna  # noqa: E402
from builder import BuildConfig, Pipeline, chunk_text  # noqa: E402

from forge.embed_potion import potion_embedder  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_corpus")


def retrieve(urnafile, query: str, k: int = 5, embedder=None, verify_model: bool = True):
    """embed `query` OFFLINE with potion, then UrnaFile.retrieve. the returned
    hits' `score` is the exact-cosine rerank value; each carries tier-1 text +
    a urna:// citation. `embedder` defaults to the vendored potion table.

    honesty gate (on by default): the embedder's `model_hash` is passed to
    UrnaFile.retrieve, which raises if it does not match the corpus manifest —
    so a query embedded by a DIFFERENT model can never silently return
    cosine-valid, wrong hits (the invariant the CLI enforces, now on the
    flagship Python surface too). Set `verify_model=False` to bypass."""
    emb = embedder or potion_embedder()
    qvec = emb.embed_texts([query])[0]
    expected = emb.model_hash() if (verify_model and hasattr(emb, "model_hash")) else None
    return urnafile.retrieve(qvec, k, expected_model_hash=expected)


def build_demo(out_path: str) -> str:
    """build a .urna from the cc0 demo corpus with the potion embedder.
    deterministic + offline: same docs + same table => byte-identical file."""
    emb = potion_embedder()
    cfg = BuildConfig(
        output_path=out_path,
        embedding_model=emb.embedding_model,
        embedding_dim=emb.embedding_dim,
        chunker_version="forge-demo/1",
        model_hash=emb.model_hash(),
        preset="exact",
        reproducible=True,
    )
    pipe = Pipeline(cfg, embedder=emb)
    for fn in sorted(os.listdir(CORPUS)):
        if not fn.endswith(".md") or fn == "README.md":
            continue
        with open(os.path.join(CORPUS, fn), encoding="utf-8") as fh:
            text = fh.read()
        for spec in chunk_text(text, source_uri=fn):
            pipe.add(spec)
    pipe.emit()
    return out_path


def main() -> None:
    import tempfile

    out = os.path.join(tempfile.mkdtemp(prefix="urna-demo-"), "demo_corpus.urna")
    build_demo(out)
    db = urna.open(out)
    print(f"built {out} ({db.n_embeddings} chunks, dim={db.embedding_dim}, dtype={db.dtype})")

    query = "can I use this with no internet, everything on my own machine"
    print(f'\nask: "{query}"\n')
    hits = retrieve(db, query, k=2)
    for h in hits:
        print(h.text.strip())
        print(f"  -- {h.citation_id}")
        print(f"     score={h.score:.4f} ({h.rerank_source})  source={h.source_uri}\n")

    # the citation round-trips: cite resolves it to the SAME stored canonical text.
    top = hits[0]
    cited = db.search(potion_embedder().embed_texts([query])[0], 1)[0]
    assert cited.citation_id == top.citation_id, "retrieve/search citation disagree"
    print(f"citation round-trips through search: {top.citation_id}")


if __name__ == "__main__":
    main()
