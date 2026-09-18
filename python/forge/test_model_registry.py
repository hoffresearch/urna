"""Prove the model registry contract without heavy ML deps: preset table
integrity, the RFC-0 gates (fake env, remote-code opt-in, pinned hashes,
heavy flag), deterministic fake embeddings, and slice_renorm equivalence
with the engine's mrl_dim truncate-then-renormalize.

Run: .venv/bin/python python/forge/test_model_registry.py
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from forge import model_registry as mr


def test_preset_table() -> None:
    for name in ("potion", "clip-vit-b32", "siglip2", "wemm-2b", "wemm-4b", "wemm-9b"):
        assert name in mr.PRESETS, name
    assert mr.PRESETS["wemm-4b"].executable is False
    assert mr.PRESETS["wemm-9b"].executable is False
    names = [p.embedding_model for p in mr.PRESETS.values()]
    assert len(names) == len(set(names)), "embedding_model must be unique (reverse lookup)"
    assert mr.PRESETS["clip-vit-b32"].mrl.supported is False
    assert mr.PRESETS["wemm-2b"].mrl.method == "prefix_slice_l2"


def test_unknown_preset_lists_valid_names() -> None:
    try:
        mr.get_preset("nope")
    except mr.RegistryError as e:
        assert "potion" in str(e) and "wemm-2b" in str(e)
    else:
        raise AssertionError("unknown preset must raise")


def test_fake_preset_is_env_gated() -> None:
    os.environ.pop("URNA_ENABLE_FAKE_PRESET", None)
    try:
        mr.get_preset("fake-test")
    except mr.RegistryError:
        pass
    else:
        raise AssertionError("fake-test must require URNA_ENABLE_FAKE_PRESET=1")
    os.environ["URNA_ENABLE_FAKE_PRESET"] = "1"
    assert mr.get_preset("fake-test").kind == "fake"


def test_fake_adapter_is_deterministic() -> None:
    os.environ["URNA_ENABLE_FAKE_PRESET"] = "1"
    a = mr.create_embedder("fake-test")
    b = mr.create_embedder("fake-test")
    va, vb = a.embed_texts(["hello", "world"]), b.embed_texts(["hello", "world"])
    assert np.array_equal(va, vb)
    assert va.shape == (2, 8)
    assert np.allclose(np.linalg.norm(va, axis=1), 1.0, atol=1e-6)
    assert not np.array_equal(va[0], va[1])


def test_remote_code_and_heavy_gates() -> None:
    try:
        mr.create_embedder("wemm-2b")
    except mr.RegistryError as e:
        assert "allow_remote_code" in str(e)
    else:
        raise AssertionError("remote-code preset must require opt-in")
    try:
        mr.create_embedder("wemm-4b", allow_remote_code={"wemm-4b"})
    except mr.RegistryError as e:
        assert "heavy" in str(e)
    else:
        raise AssertionError("executable=False must require allow_heavy")


def test_pinned_hash_refuses_altered_code() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / "modeling_x.py"
        code.write_text("print('v1')")
        good = hashlib.sha256(code.read_bytes()).hexdigest()
        preset = mr.PRESETS["wemm-2b"].__class__(
            **{**mr.PRESETS["wemm-2b"].__dict__, "remote_code_hashes": (("modeling_x.py", good),)}
        )
        mr.verify_remote_code(preset, Path(tmp))  # matching pin passes
        code.write_text("print('v2')")
        try:
            mr.verify_remote_code(preset, Path(tmp))
        except mr.RegistryError as e:
            assert "refusing" in str(e)
        else:
            raise AssertionError("altered pinned code must be refused")


def test_resolve_model_dir_precedence() -> None:
    preset = mr.PRESETS["wemm-2b"]
    assert mr.resolve_model_dir(preset, "/explicit/x") == Path("/explicit/x")
    os.environ["URNA_MODEL_DIR_WEMM_2B"] = "/from/env"
    try:
        assert mr.resolve_model_dir(preset) == Path("/from/env")
    finally:
        del os.environ["URNA_MODEL_DIR_WEMM_2B"]


def test_potion_adapter_is_text_only() -> None:
    emb = mr.create_embedder("potion")
    try:
        emb.embed_paths(["/tmp/x.jpg"])
    except mr.CapabilityError:
        pass
    else:
        raise AssertionError("potion must refuse image input")


def test_slice_renorm_matches_engine_mrl() -> None:
    import urna

    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((3, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sliced = mr.slice_renorm(vecs, 4)
    assert np.allclose(np.linalg.norm(sliced, axis=1), 1.0, atol=1e-6)
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "mrl.urna")
        chunks = [
            {
                "canonical_text": f"chunk {i}",
                "source_uri": "test://mrl",
                "byte_start": i,
                "byte_end": i + 1,
                "embedding": vecs[i].tolist(),
            }
            for i in range(3)
        ]
        urna.build(
            out,
            "test-model",
            8,
            "test/1",
            "sha256:" + "1" * 64,
            chunks,
            preset="exact",
            mrl_dim=4,
            reproducible=True,
        )
        db = urna.open(out)
        for i in range(3):
            hits = db.search(sliced[i].tolist(), k=1)
            assert hits[0].offset_start == i, "sliced query must retrieve its own chunk"
            assert hits[0].score > 0.9999, f"expected cosine ~1.0, got {hits[0].score}"


def main() -> None:
    for fn in sorted(k for k in globals() if k.startswith("test_")):
        globals()[fn]()
        print(f"{fn}: OK")


if __name__ == "__main__":
    main()
