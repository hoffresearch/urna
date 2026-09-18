"""Default static embedder for forge: offline, deterministic, no network.

the flagship is a pip-installable retrieve() that must work offline by
construction. the existing search-text path resolves a sentence-transformers
model from the hugging face cache (a network round-trip on first use), which
breaks offline-by-construction for a brand-new user. this module is the
DEFAULT embedder beside forge: a model2vec-style STATIC embedder that needs no
model download, no torch, no network -- only the standard library -- and is
byte-identical-deterministic, so reproducible builds are trivial.

it is a quality FLOOR, not a ceiling. each token maps to a fixed
pseudo-random vector derived deterministically from its bytes, and a text
embeds to the l2-normalized mean of its token vectors, so cosine captures
lexical overlap (shared tokens raise similarity). power users bring a stronger
sentence-transformers model (the brought-model path); whichever embedder
produced a corpus is recorded by its model_hash, so a build is reproducible
and a query embedded by a different embedder fails the manifest model_hash
gate loudly instead of returning cosine-valid garbage.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence

MODEL_ID = "urna-forge-static"
STATIC_EMBEDDER_VERSION = "1"
DEFAULT_DIM = 256
# the seed is part of the fingerprint: changing it changes the model_hash, so
# two corpora built with different seeds are never silently confused.
DEFAULT_SEED = "urna-forge-static/v1"


def _tokenize(text: str) -> list[str]:
    """lUnicode alphanumeric word tokenizer, lowercased. correct for latin,
    cyrillic, greek, devanagari; degrades for cjk (each char a token), same
    tradeoff the bm25 tokenizer documents. deterministic by construction."""
    tokens: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur).lower())
            cur = []
    if cur:
        tokens.append("".join(cur).lower())
    return tokens


def _token_vector(token: str, dim: int, seed: str) -> list[float]:
    """lA fixed pseudo-random vector for a token, derived deterministically
    from its bytes by expanding a seeded sha256 counter stream. same token +
    seed -> same vector on every machine; no stored table, no network."""
    need = dim * 4
    buf = bytearray()
    counter = 0
    base = f"{seed}\x00{token}".encode()
    while len(buf) < need:
        buf.extend(hashlib.sha256(base + counter.to_bytes(8, "little")).digest())
        counter += 1
    vec: list[float] = []
    for j in range(dim):
        u = int.from_bytes(buf[j * 4 : j * 4 + 4], "little")
        vec.append(u / 2147483648.0 - 1.0)  # uint32 -> [-1, 1)
    return vec


def _f32(x: float) -> float:
    """lRound to float32 precision so a fresh build and a cache-backed rebuild
    (the builder caches embeddings as float32) produce byte-identical .urna."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def embed_one(text: str, dim: int = DEFAULT_DIM, seed: str = DEFAULT_SEED) -> list[float]:
    """lEmbed one text: l2-normalized mean of its token vectors, f32-stable."""
    tokens = _tokenize(text)
    if not tokens:
        # deterministic non-zero fallback for text with no alphanumerics, so
        # the runtime's zero-norm guard never trips on a built chunk.
        tokens = ["\x00empty"]
    acc = [0.0] * dim
    for tok in tokens:
        tv = _token_vector(tok, dim, seed)
        for j in range(dim):
            acc[j] += tv[j]
    inv_n = 1.0 / len(tokens)
    pooled = [a * inv_n for a in acc]
    norm = math.sqrt(sum(p * p for p in pooled)) or 1.0
    return [_f32(p / norm) for p in pooled]


class StaticEmbedder:
    """lThe offline default embedder. callable with ChunkSpec-like objects (it
    reads `.canonical_text`) or raw strings, so it drops into builder.Pipeline
    as the `embedder`."""

    def __init__(self, dim: int = DEFAULT_DIM, seed: str = DEFAULT_SEED):
        self.dim = dim
        self.seed = seed

    @property
    def embedding_model(self) -> str:
        return f"{MODEL_ID}/v{STATIC_EMBEDDER_VERSION}"

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def fingerprint(self) -> dict:
        """lThe inference-relevant config. there are no model files to hash, so
        the fingerprint is over this config (mirroring model_fingerprint.py's
        idea: identify exactly what produced the embeddings)."""
        return {
            "embedder": MODEL_ID,
            "version": STATIC_EMBEDDER_VERSION,
            "embedding_dim": self.dim,
            "seed": self.seed,
            "tokenizer": "unicode-alnum-lower",
            "pooling": "mean",
            "normalize": "l2",
            "dtype": "float32-stable",
        }

    def model_hash(self) -> str:
        """l`sha256:<hex>` of the canonical-json fingerprint, the SAME
        convention as model_fingerprint.fingerprint_to_model_hash, so the
        manifest model_hash gate treats it uniformly. recorded in provenance
        so byte-identical builds are provable."""
        canonical = json.dumps(self.fingerprint(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [embed_one(t, self.dim, self.seed) for t in texts]

    def __call__(self, specs: Sequence[object]) -> list[list[float]]:
        texts = [getattr(s, "canonical_text", s) for s in specs]
        return self.embed_texts(texts)  # type: ignore[arg-type]


def default_embedder() -> StaticEmbedder:
    """lThe offline default: model2vec-style static embedder, no network."""
    return StaticEmbedder()
