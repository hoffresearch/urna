"""build a .urna from a handful of paragraphs, ask it a question, get a cited
answer. offline, deterministic, no torch. the same flow as `urna build` +
`urna ask`, on the python surface.

run from the repo root:   python examples/quickstart/quickstart.py
after `pip install "urna[embed]"` the two imports at the top are all it needs.
"""

from __future__ import annotations

import json
import os
import sys

try:  # installed wheel: `pip install "urna[embed]"`
    import urna
    from urna.embed_potion import potion_embedder
except ImportError:  # dev checkout: python/urna.py + python/_urna.so + python/forge/
    _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    sys.path.insert(0, os.path.join(_repo, "python"))
    import urna
    from forge.embed_potion import potion_embedder

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "quickstart.urna")

# 1. the embedder. potion is a static table bundled with the wheel: it turns
#    text into a 256-d unit vector with no model download and no network.
#    its model_hash is written into the file and checked on every query.
emb = potion_embedder()

# 2. the chunks. each one is a dict: the text, where it came from, and its
#    embedding. byte offsets point back into the source document.
with open(os.path.join(HERE, "docs.jsonl"), encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh if line.strip()]
vectors = emb.embed_texts([r["text"] for r in rows])
chunks = []
for row, vec in zip(rows, vectors, strict=True):
    chunks.append(
        {
            "canonical_text": row["text"],
            "source_uri": row["source_uri"],
            "byte_start": 0,
            "byte_end": len(row["text"].encode("utf-8")),
            "embedding": vec,
        }
    )

# 3. build. `reproducible=True` makes the file byte-identical on every run.
os.makedirs(os.path.dirname(OUT), exist_ok=True)
urna.build(
    output_path=OUT,
    embedding_model=emb.embedding_model,
    embedding_dim=emb.embedding_dim,
    chunker_version="quickstart/1",
    model_hash=emb.model_hash(),
    chunks=chunks,
    reproducible=True,
    preset="exact",
)

# 4. open and ask. the query goes through the SAME embedder, and
#    `expected_model_hash` refuses a corpus built with another model.
db = urna.open(OUT)
assert db.validate() is True
print(f"built {OUT}: {db.n_embeddings} chunks, dim={db.embedding_dim}, dtype={db.dtype}\n")

question = "can I use this with no internet, everything on my own machine"
qvec = emb.embed_texts([question])[0]
hits = db.retrieve(qvec, 3, expected_model_hash=emb.model_hash())

print(f"ask: {question}\n")
for h in hits:
    print(h.text)
    print(f"  -- {h.citation_id}")
    print(f"     score={h.score:.4f} ({h.rerank_source})  source={h.source_uri}\n")
