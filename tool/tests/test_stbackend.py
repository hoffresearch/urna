"""Prove the st_multimodal backend: the pinned model_hash of a synthetic
snapshot, which needs numpy only and always runs, then the model cases for
each st_multimodal preset chosen (dim/norm contracts, model_hash stability
across constructions without a load, subprocess isolation: the jina+wemm
dynamic-module collision is the reason the adapter owns a worker).

Which presets load is a choice, since each is a real model (WeMM-2B is 2B
parameters): `--models a,b`, `--models all`, `--models none` or
URNA_TEST_MODELS. without one, a terminal is asked; anything else runs the
pinned case only. fullcheck.sh and gatecheck.yml pass `--models none`, so
they never load a model, whatever is installed. a preset added to the
registry with kind st_multimodal is offered here with no change to this file.

Run: .venv/bin/python tool/tests/test_stbackend.py [--models NAME[,NAME]|all|none]
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rust", "bridge", "python"))

import numpy as np

from urna.model import presetmap as mr

ST_PRESETS = [p for p in mr.PRESETS.values() if p.kind == "st_multimodal" and p.executable]


def has_deps(preset: mr.ModelPreset) -> bool:
    return all(importlib.util.find_spec(m) is not None for m, _ in preset.requires)


def where(preset: mr.ModelPreset) -> str:
    """Where the weights would load from, for the prompt."""
    if not has_deps(preset):
        return "deps missing"
    if preset.local_dir is None:
        return "hf cache"
    return "snapshot found" if Path(preset.local_dir).is_dir() else "snapshot missing"


# the model_hash of a fixed synthetic snapshot, as the st_multimodal backend
# computed it before the move (306e65fd, python/forge/embed_st.py). the key
# names of the fingerprint dict are data that goes into the hash: a renamed
# key changes every corpus's model_hash.
PINNED_MODEL_HASH = "sha256:9885c42ea691ef1dc8fffd08c4403d8580ada6eb3cc20f763e9c0bb8d8b341cb"


def test_model_hash_is_pinned() -> None:
    import hashlib
    import json
    import tempfile
    from types import SimpleNamespace

    from urna.embed.stbackend import fingerprint_for

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "acme-fixture"
        d.mkdir()
        (d / "config.json").write_text('{"hidden_size": 8, "_name_or_path": "acme/fixture"}')
        (d / "tokenizer.json").write_text('{"version": "1.0"}')
        (d / "modules.json").write_text("[]")
        (d / "modeling_fixture.py").write_text("X = 1\n")
        fp = fingerprint_for(SimpleNamespace(model_id="acme/fixture"), d, True, "float32")
    blob = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    assert "model_fingerprint" in fp, sorted(fp)
    assert "sha256:" + hashlib.sha256(blob.encode()).hexdigest() == PINNED_MODEL_HASH, blob


def test_hash_without_load(name: str) -> None:
    a = mr.create_embedder(name, allow_remote_code={name})
    b = mr.create_embedder(name, allow_remote_code={name})
    assert a.model_hash == b.model_hash and a.model_hash.startswith("sha256:")
    assert a._proc is None, "model_hash must not load the model"
    if mr.PRESETS[name].trust_remote_code:
        assert a.fingerprint()["remote_code_sha256"], "remote-code files must be fingerprinted"
    a.close()
    b.close()


def test_embed_contracts(name: str) -> None:
    preset = mr.PRESETS[name]
    emb = mr.create_embedder(name, allow_remote_code={name}, batch_size=2)
    tv = emb.embed_texts(["counter target spell", "a red dragon"], role="query")
    assert tv.shape[0] == 2 and (not preset.default_dim or tv.shape[1] == preset.default_dim)
    assert np.allclose(np.linalg.norm(tv, axis=1), 1.0, atol=1e-3)
    if "image" in preset.modalities:
        iv = emb.embed_arrays([np.zeros((64, 64, 3), dtype=np.uint8)])
        assert iv.shape == (1, tv.shape[1])
    emb.close()


def test_worker_survives_multi_model(name: str) -> None:
    """The measured failure: jina loaded first broke wemm image embeds in
    one process. Through subprocess adapters both must work. runs with
    wemm-2b, since that is the pair it was measured on."""
    if name != "wemm-2b":
        return
    try:
        j = mr.create_embedder(
            "jina-v5-omni-nano", allow_remote_code={"jina-v5-omni-nano"}, batch_size=2
        )
        j.embed_texts(["warm"])
    except mr.RegistryError:
        print("test_worker_survives_multi_model: SKIP (jina snapshot not cached)")
        return
    w = mr.create_embedder("wemm-2b", allow_remote_code={"wemm-2b"}, batch_size=2)
    iv = w.embed_arrays([np.zeros((64, 64, 3), dtype=np.uint8)])
    assert iv.shape[0] == 1
    j.close()
    w.close()


def parse(value: str) -> list[str]:
    value = value.strip().lower()
    if value in ("", "none"):
        return []
    if value == "all":
        return [p.name for p in ST_PRESETS if has_deps(p)]
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in {p.name for p in ST_PRESETS}]
    if unknown:
        sys.exit(f"unknown st_multimodal preset: {', '.join(unknown)}")
    return names


def ask() -> list[str]:
    """Lists the st_multimodal presets and asks which to load."""
    print("st_multimodal presets (each case loads the real model):")
    for i, p in enumerate(ST_PRESETS, 1):
        print(f"  {i}. {p.name:<20} {where(p)}")
    try:
        answer = input("run which? numbers or names, comma-separated; all; Enter for none: ")
    except EOFError:
        answer = ""
    picked = []
    for item in answer.split(","):
        item = item.strip()
        if item.isdigit() and 1 <= int(item) <= len(ST_PRESETS):
            picked.append(ST_PRESETS[int(item) - 1].name)
        elif item:
            picked.extend(parse(item))
    return picked


def chosen(argv: list[str]) -> list[str]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", help="NAME[,NAME], all or none (default: ask on a terminal)")
    args = ap.parse_args(argv)
    value = args.models if args.models is not None else os.environ.get("URNA_TEST_MODELS")
    if value is not None:
        return parse(value)
    return ask() if sys.stdin.isatty() else []


class _Recorder:
    """A sentence-transformers model that records how it was asked."""

    def __init__(self, prompts: dict):
        self.prompts = prompts
        self.calls: list[tuple[str, dict]] = []

    def _out(self, items):
        return [[1.0, 0.0] for _ in items]

    def encode(self, items, **kw):
        self.calls.append(("encode", kw))
        return self._out(items)

    def encode_query(self, items, **kw):
        self.calls.append(("encode_query", kw))
        return self._out(items)

    def encode_document(self, items, **kw):
        self.calls.append(("encode_document", kw))
        return self._out(items)


def test_roles_reach_the_model_the_way_it_takes_them() -> None:
    """jina v5 picks its adapter by `task`, so a role goes in as its prompt
    (encode_query would pass task="query" over it); harrier names its query
    prompt otherwise than `query`; a plain model gets encode_query."""
    from urna.embed import stbackend

    def asked(name: str, role: str) -> tuple[str, dict]:
        emb = stbackend.STMultimodalEmbedder(mr.PRESETS[name], device="cpu")
        emb._model = _Recorder({"query": "Q: ", "document": "D: "})
        emb.embed_texts(["x"], role=role)
        return emb._model.calls[0]

    how, kw = asked("jina-v5-text-nano", "query")
    assert how == "encode" and kw["task"] == "retrieval" and kw["prompt_name"] == "query"
    how, kw = asked("jina-v5-text-nano", "document")
    assert how == "encode" and kw["prompt_name"] == "document"
    how, kw = asked("harrier-270m", "query")
    assert how == "encode_query" and kw["prompt_name"] == "web_search_query"
    how, kw = asked("qwen3-embedding-0.6b", "query")
    assert how == "encode_query" and "prompt_name" not in kw
    os.environ.pop("URNA_ST_DTYPE", None)
    assert stbackend._default_dtype("mps", mr.PRESETS["harrier-270m"]) == "bfloat16"
    assert stbackend._default_dtype("mps", mr.PRESETS["qwen3-embedding-0.6b"]) == "float16"
    assert stbackend._default_dtype("cpu", mr.PRESETS["harrier-270m"]) == "float32"


def main() -> None:
    picked = chosen(sys.argv[1:])
    # needs no model and no sentence-transformers: always runs.
    test_model_hash_is_pinned()
    print("test_model_hash_is_pinned: OK")
    test_roles_reach_the_model_the_way_it_takes_them()
    print("test_roles_reach_the_model_the_way_it_takes_them: OK")
    for name in picked:
        preset = mr.PRESETS[name]
        if where(preset) in ("deps missing", "snapshot missing"):
            print(f"{name}: SKIP ({where(preset)})")
            continue
        for fn in (test_hash_without_load, test_embed_contracts, test_worker_survives_multi_model):
            fn(name)
            print(f"{fn.__name__}[{name}]: OK")
    if not picked:
        print("model cases: none chosen (--models NAME[,NAME]|all, or run on a terminal)")
    print("all stbackend tests passed")


if __name__ == "__main__":
    main()
