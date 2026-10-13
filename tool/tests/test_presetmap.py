# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""Prove the model registry contract without heavy ML deps: preset table
integrity, the RFC-0 gates (fake env, remote-code opt-in, pinned hashes,
heavy flag), the pinned hub snapshot of an open_clip preset (the revision,
never refs/main; a missing file named), deterministic fake embeddings, and
slice_renorm equivalence with the engine's mrl_dim truncate-then-renormalize.

Run: .venv/bin/python tool/tests/test_presetmap.py
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rust", "bridge", "python"))

import numpy as np

from urna.model import presetmap as mr


def test_preset_table() -> None:
    for name in ("potion", "clip-vit-b32", "siglip2", "wemm-2b", "qwen3-embedding-0.6b"):
        assert name in mr.PRESETS, name
    # more than a default machine and a standard ci runner hold (#534).
    assert "wemm-4b" not in mr.PRESETS and "wemm-9b" not in mr.PRESETS
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
    # no preset is flagged heavy now; the gate still holds for one that is.
    import dataclasses

    heavy = dataclasses.replace(
        mr.PRESETS["wemm-2b"], name="heavy-test", embedding_model="test/heavy", executable=False
    )
    mr.PRESETS["heavy-test"] = heavy
    try:
        mr.create_embedder("heavy-test", allow_remote_code={"heavy-test"})
    except mr.RegistryError as e:
        assert "heavy" in str(e)
    else:
        raise AssertionError("executable=False must require allow_heavy")
    finally:
        del mr.PRESETS["heavy-test"]


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


def _fake_hf_home(tmp: str, files: tuple[str, ...]) -> Path:
    """An hf cache holding siglip2's pinned snapshot with `files`, plus a newer
    snapshot that refs/main points to: the layout `hf download` leaves."""
    preset = mr.PRESETS["siglip2"]
    cache = Path(tmp) / "hub" / f"models--{preset.hf_repo.replace('/', '--')}"
    pinned = cache / "snapshots" / preset.revision
    newer = cache / "snapshots" / ("f" * 40)
    for snap, names in ((pinned, files), (newer, preset.snapshot_files)):
        snap.mkdir(parents=True)
        for name in names:
            (snap / name).write_text(name)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("f" * 40)
    return pinned


def _snapshot_missing(preset, model_path=None) -> str:
    try:
        mr.pinned_snapshot(preset, model_path)
    except mr.SnapshotMissing as e:
        assert e.repo == preset.hf_repo
        return str(e)
    raise AssertionError("a missing snapshot file must raise SnapshotMissing")


def test_siglip2_pins_a_revision() -> None:
    p = mr.PRESETS["siglip2"]
    assert len(p.revision) == 40 and all(c in "0123456789abcdef" for c in p.revision)
    assert p.weights_file in p.snapshot_files
    assert {"tokenizer.json", "tokenizer_config.json"} <= set(p.snapshot_files)
    assert "transformers" in {m for m, _ in p.requires}


def test_pinned_snapshot_reads_the_revision_not_refs_main() -> None:
    preset = mr.PRESETS["siglip2"]
    with tempfile.TemporaryDirectory() as tmp:
        pinned = _fake_hf_home(tmp, preset.snapshot_files)
        os.environ["HF_HOME"] = tmp
        try:
            assert mr.pinned_snapshot(preset) == pinned
        finally:
            del os.environ["HF_HOME"]


def test_pinned_snapshot_names_a_missing_file() -> None:
    preset = mr.PRESETS["siglip2"]
    os.environ.pop("URNA_ALLOW_DOWNLOAD", None)
    with tempfile.TemporaryDirectory() as tmp:
        files = tuple(f for f in preset.snapshot_files if f != "tokenizer.json")
        _fake_hf_home(tmp, files)
        os.environ["HF_HOME"] = tmp
        try:
            msg = _snapshot_missing(preset)
        finally:
            del os.environ["HF_HOME"]
    assert "tokenizer.json missing" in msg, msg
    assert f"--revision {preset.revision}" in msg and "URNA_ALLOW_DOWNLOAD=1" in msg, msg
    # the newer snapshot under refs/main has every file and is still not used
    with tempfile.TemporaryDirectory() as tmp:
        msg = _snapshot_missing(preset, tmp)
    assert f"missing from {tmp}" in msg and preset.weights_file in msg, msg


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
