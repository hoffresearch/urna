"""Self-test for the forge REAL static embedder (#10, model2vec/potion-base-8M).

run: python tool/tests/test_potiontab.py   (or: pytest this file)

unlike the floor's test_lexifloor.py (stdlib only), this one needs numpy +
tokenizers and the vendored table under model/potionb8m/ (git-lfs). it
proves the decisive claim: potion separates synonyms from unrelated words by
MEANING (car ~ automobile >> car ~ banana) where the lexical floor cannot, and
that the embedder is deterministic, f32-stable, normalized, offline (no socket
at embed time), and carries a stable, config-sensitive model_hash.
"""

from __future__ import annotations

import math
import os
import socket
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rust", "bridge", "python"))

from urna.embed.lexifloor import default_embedder as lexical_embedder
from urna.embed.potiontab import MODEL_DIR, PotionEmbedder, potion_embedder

_EMB = potion_embedder()


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _e(text: str) -> list[float]:
    return _EMB.embed_texts([text])[0]


def test_dim_norm_determinism() -> None:
    v1 = _e("the file is the database")
    v2 = _e("the file is the database")
    assert v1 == v2, "embedding must be deterministic (byte-identical f32)"
    assert len(v1) == _EMB.embedding_dim == 256, f"dim {len(v1)} != 256"
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-5, f"expected ~unit norm, got {norm}"


def test_f32_stable() -> None:
    for x in _e("offline by construction"):
        assert struct.unpack("<f", struct.pack("<f", x))[0] == x, "values must be f32-stable"


def test_semantic_jump() -> None:
    """the decisive acceptance: synonyms cluster, unrelated words do not."""
    syn = {
        "automobile": _cos(_e("car"), _e("automobile")),
        "vehicle": _cos(_e("car"), _e("vehicle")),
    }
    unrel = {w: _cos(_e("car"), _e(w)) for w in ("banana", "fruit", "dog")}
    print("\n[semantic] english, anchor=car")
    for k, v in {**syn, **unrel}.items():
        print(f"  car ~ {k:11s} = {v:+.4f}")
    assert min(syn.values()) > 0.6, "synonyms must be strongly close"
    assert max(unrel.values()) < 0.25, "unrelated words must be far"
    assert min(syn.values()) > max(unrel.values()) + 0.4, "clear synonym/unrelated gap"

    # secondary, honest: potion-base-8M is english; portuguese rides english
    # subwords, so the jump is real but weak. assert only the direction.
    pt_syn = _cos(_e("carro"), _e("automovel"))
    pt_unrel = _cos(_e("carro"), _e("banana"))
    print(
        f"[semantic] portuguese (honest, weak): carro~automovel={pt_syn:+.4f} "
        f"> carro~banana={pt_unrel:+.4f}"
    )
    assert pt_syn > pt_unrel, "even on portuguese the synonym must beat the unrelated word"


def test_beats_the_lexical_floor() -> None:
    """side by side: the floor cannot tell a synonym from an unrelated word
    (disjoint single tokens -> near-zero random cosine); potion can."""
    floor = lexical_embedder()

    def fcos(a: str, b: str, emb) -> float:
        return _cos(emb.embed_texts([a])[0], emb.embed_texts([b])[0])

    p_syn, p_unrel = fcos("car", "automobile", _EMB), fcos("car", "banana", _EMB)
    f_syn, f_unrel = fcos("car", "automobile", floor), fcos("car", "banana", floor)
    print("\n[floor vs potion] car~automobile (syn) vs car~banana (unrelated)")
    print(f"  floor : syn={f_syn:+.4f}  unrel={f_unrel:+.4f}  separation={f_syn - f_unrel:+.4f}")
    print(f"  potion: syn={p_syn:+.4f}  unrel={p_unrel:+.4f}  separation={p_syn - p_unrel:+.4f}")
    assert (p_syn - p_unrel) > (f_syn - f_unrel) + 0.3, "potion must separate far better than floor"


def test_no_network_at_embed() -> None:
    """fail-closed: prove no socket is opened while loading the table or
    embedding. block connect/getaddrinfo, then a FRESH load+embed must work."""
    import numpy  # noqa: F401  warm the c-extensions before blocking
    import tokenizers  # noqa: F401

    from urna.embed import potiontab

    potiontab._load_table.cache_clear()  # force a fresh table load under the block

    real_conn, real_gai, real_cc = (
        socket.socket.connect,
        socket.getaddrinfo,
        socket.create_connection,
    )

    def _blocked(*_a, **_k):
        raise AssertionError("network access attempted during offline embed")

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        fresh = PotionEmbedder(MODEL_DIR)
        out = fresh.embed_texts(["a sovereign single file database"])
        assert len(out) == 1 and len(out[0]) == 256, "embed must work fully offline"
    finally:
        socket.socket.connect = real_conn  # type: ignore[method-assign]
        socket.getaddrinfo = real_gai  # type: ignore[assignment]
        socket.create_connection = real_cc  # type: ignore[assignment]
    print("\n[offline] no socket opened during load+embed: ok")


def test_interface_parity() -> None:
    """same surface as the floor's StaticEmbedder, so it drops into the pipeline."""
    assert _EMB.embedding_model.startswith("minishlab/potion-base-8M")
    assert _EMB.embedding_dim == 256
    assert _EMB.model_hash().startswith("sha256:"), "model_hash must be sha256:<hex>"

    class _Spec:
        canonical_text = "hello world"

    out = _EMB([_Spec(), "hello world"])
    assert len(out) == 2 and out[0] == out[1], "__call__ reads canonical_text and strings alike"


def test_model_hash_stable_and_distinct() -> None:
    assert _EMB.model_hash() == potion_embedder().model_hash(), "model_hash must be stable"
    assert _EMB.model_hash() != lexical_embedder().model_hash(), "must differ from the floor"
    assert PotionEmbedder(MODEL_DIR, normalize=False).model_hash() != _EMB.model_hash(), (
        "config (normalize) must change the hash"
    )


def main() -> None:
    test_dim_norm_determinism()
    test_f32_stable()
    test_semantic_jump()
    test_beats_the_lexical_floor()
    test_no_network_at_embed()
    test_interface_parity()
    test_model_hash_stable_and_distinct()
    print(
        f"\nok: potion embedder semantic, deterministic, f32-stable, offline, "
        f"dim={_EMB.embedding_dim}, model={_EMB.embedding_model}, "
        f"model_hash={_EMB.model_hash()[:23]}..."
    )


if __name__ == "__main__":
    main()
