"""Self-test for the forge REAL static embedder (#10, model2vec/potion-base-8M).

the embedder runs in Rust through urna-cbinds when the library is built
(urna/_cabi.py), else numpy + tokenizers; test_native_matches_the_reference
holds the two to the same ids and vectors.

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
    from urna.embed import potiontab

    if _EMB.backend == "reference":
        import numpy  # noqa: F401  warm the c-extensions before blocking
        import tokenizers  # noqa: F401
    else:
        from urna import _cabi

        _cabi.library()  # load the library before blocking
    potiontab._load_table.cache_clear()  # force a fresh table load under the block
    potiontab._native_embedder.cache_clear()

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


PARITY_TEXTS = [
    "Lightning Bolt deals 3 damage to any target.",
    "Relâmpago: cause 3 pontos de dano a qualquer alvo.",
    "x" * 150,
    "😀 日本語 한국어 Ωμέγα",
    "zzzzzq qqqzz",
    "hello,world!!!don't(stop)",
    "Café   naïve\tTAB\nnewline",
    "nul\0inside",
    "",
]


def test_native_matches_the_reference() -> None:
    """the Rust path (urna-neural through the C ABI) against numpy + tokenizers:
    the same token ids, vectors within 1e-6, the same model_hash. skipped when
    either side is missing, unless URNA_REQUIRE_NATIVE=1 (the CI job)."""
    import importlib.util

    from urna import _cabi

    have_ref = all(importlib.util.find_spec(m) for m in ("numpy", "tokenizers"))
    if _cabi.library() is None or not have_ref:
        assert os.environ.get("URNA_REQUIRE_NATIVE") != "1", (
            "native parity required: no urna-cbinds library or no numpy + tokenizers"
        )
        print("\n[parity] skipped: no urna-cbinds library or no numpy + tokenizers")
        return
    from tokenizers import Tokenizer

    native = _cabi.Embedder(MODEL_DIR)
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    os.environ["URNA_POTION_REFERENCE"] = "1"
    try:
        ref = PotionEmbedder(MODEL_DIR)
        assert ref.backend == "reference"
        theirs = ref.embed_texts(PARITY_TEXTS)
        ref_hash = ref.model_hash()
    finally:
        del os.environ["URNA_POTION_REFERENCE"]
    assert native.model_hash == ref_hash, (native.model_hash, ref_hash)
    worst = 0.0
    for text, b in zip(PARITY_TEXTS, theirs, strict=True):
        ids = tok.encode(text, add_special_tokens=False).ids
        assert native.tokenize(text) == ids, f"token ids differ on {text!r}"
        a = native.embed(text)
        assert len(a) == len(b) == 256, text
        d = max(abs(x - y) for x, y in zip(a, b, strict=True))
        worst = max(worst, d)
        assert d <= 1e-6, f"{text!r} differs from the reference by {d:e}"
    print(f"\n[parity] native = reference on {len(PARITY_TEXTS)} texts (worst {worst:e}): ok")


def main() -> None:
    test_native_matches_the_reference()
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
