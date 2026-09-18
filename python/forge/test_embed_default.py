"""Self-test for the forge default static embedder (#04).

run: python python/forge/test_embed_default.py

a plain script (no pytest), matching the repo's tests/ convention. proves the
embedder is deterministic, offline (stdlib only), f32-stable, normalized,
carries a stable config fingerprint, and that the demo corpus ships.
"""

from __future__ import annotations

import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/

from forge.embed_default import DEFAULT_DIM, StaticEmbedder, default_embedder, embed_one


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    emb = default_embedder()

    # deterministic: same text -> identical vector, every time.
    v1 = embed_one("vacina contra a covid")
    v2 = embed_one("vacina contra a covid")
    assert v1 == v2, "embedding must be deterministic"

    # correct dim, ~unit norm.
    assert len(v1) == DEFAULT_DIM, f"dim {len(v1)} != {DEFAULT_DIM}"
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-5, f"expected ~unit norm, got {norm}"

    # f32-stable: every value round-trips through float32 unchanged, so a
    # cache-backed rebuild (float32 cache) is byte-identical to a fresh build.
    for x in v1:
        assert struct.unpack("<f", struct.pack("<f", x))[0] == x, "values must be f32-stable"

    # lexical signal: overlapping text scores closer than disjoint text.
    base = embed_one("the sovereign urna file is a single file database")
    near = embed_one("the urna file is a sovereign single-file database")
    far = embed_one("completely unrelated words about turtles and the weather")
    assert _cos(base, near) > _cos(base, far), "shared tokens must raise cosine"

    # empty text -> deterministic, non-zero (so the zero-norm guard never trips).
    e1 = embed_one("")
    e2 = embed_one("")
    assert e1 == e2 and any(e1), "empty text embeds deterministically and non-zero"

    # builder.Pipeline-compatible: callable over ChunkSpec-likes and raw strings.
    class _Spec:
        canonical_text = "hello world"

    out = emb([_Spec(), "hello world"])
    assert len(out) == 2 and out[0] == out[1], "__call__ reads canonical_text and strings alike"

    # fingerprint: stable, sha256-shaped, sensitive to config.
    assert emb.model_hash() == default_embedder().model_hash(), "model_hash must be stable"
    assert emb.model_hash().startswith("sha256:"), "model_hash must be sha256:<hex>"
    assert StaticEmbedder(dim=128).model_hash() != emb.model_hash(), "dim must change the hash"
    assert StaticEmbedder(seed="other").model_hash() != emb.model_hash(), "seed must change it"

    # demo corpus ships at least one document.
    corpus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_corpus")
    docs = [f for f in os.listdir(corpus) if f.endswith((".md", ".txt")) and f != "README.md"]
    assert docs, "demo_corpus must ship at least one doc"

    print(
        f"ok: static embedder deterministic, f32-stable, dim={DEFAULT_DIM}, "
        f"model={emb.embedding_model}, model_hash={emb.model_hash()[:23]}..., demo_docs={len(docs)}"
    )


if __name__ == "__main__":
    main()
