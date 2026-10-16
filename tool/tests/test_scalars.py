"""chunk_scalars through the Python path: `urna.build(scalar_rows=,
schema=, dataset=, media=)` writes the columns and the declarations into
the file, the file validates, the manifest carries the schema, the dataset
and the media derivation, and the content_hash is the one a build without
the columns gives. PyO3 only, no subprocess.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rust", "bridge", "python"))

import urna

DIM = 4
MODEL_HASH = "sha256:" + "1" * 64

SCHEMA = [
    {"name": "name", "kind": "text", "roles": ["label", "display"], "prefix": "n"},
    {"name": "year", "kind": "int", "roles": ["filter", "rank"], "prefix": "y"},
    {"name": "tag", "kind": "enum", "roles": ["filter", "facet"], "prefix": "t"},
    {"name": "colors", "kind": "enum_list", "roles": ["filter"], "prefix": "c"},
    {"name": "power", "kind": "enum", "numeric": True, "roles": ["filter"], "prefix": "pow"},
    {"name": "legal", "kind": "enum_map", "roles": ["filter", "display"], "prefix": "f"},
    {"name": "rules", "kind": "chunk", "unit": "1", "roles": ["display"], "display": "text"},
]


def chunks():
    out = []
    for i in range(4):
        emb = [0.0] * DIM
        emb[i % DIM] = 1.0
        out.append(
            dict(
                canonical_text=f"item {i}\nrules of item {i}",
                source_uri="docs",
                byte_start=i,
                byte_end=i + 1,
                embedding=emb,
            )
        )
    return out


def rows():
    return [
        {
            "name": "alpha",
            "year": 2001,
            "tag": "law",
            "colors": ["R", "G"],
            "power": "3",
            "legal": {"pauper": "legal"},
        },
        {
            "name": "beta",
            "year": None,
            "tag": "art",
            "colors": [],
            "power": "*",
            "legal": {"modern": "banned"},
        },
        {"name": "gamma", "year": 1999, "tag": "law", "colors": ["U"], "power": None, "legal": {}},
        {
            "name": "delta",
            "year": 2020,
            "tag": "art",
            "colors": ["R"],
            "power": "1",
            "legal": {"pauper": "legal", "modern": "legal"},
        },
    ]


def build(path, **extra):
    urna.build(
        output_path=path,
        embedding_model="demo",
        embedding_dim=DIM,
        chunker_version="demo/1",
        model_hash=MODEL_HASH,
        chunks=chunks(),
        reproducible=True,
        preset="exact",
        **extra,
    )
    db = urna.open(path)
    assert db.validate()
    return db


def test_build_writes_columns_schema_dataset_and_media_without_moving_the_content_hash():
    with tempfile.TemporaryDirectory() as tmp:
        plain = build(os.path.join(tmp, "plain.urna"))
        full = build(
            os.path.join(tmp, "full.urna"),
            scalar_rows=rows(),
            schema=SCHEMA,
            dataset={"name": "demo bulk", "license": "CC0-1.0", "sha256": "abc"},
            media={"codec": "avif", "quality": 48, "lossy": True},
        )
        assert full.content_hash == plain.content_hash
        assert full.file_hash != plain.file_hash
        info = full.inspect()
        manifest = info["manifest"]
        names = [f["name"] for f in manifest["schema"]]
        assert names == [
            "name",
            "year",
            "tag",
            "colors",
            "power",
            "legal",
            "legal.modern",
            "legal.pauper",
            "rules",
        ]
        legal_pauper = next(f for f in manifest["schema"] if f["name"] == "legal.pauper")
        assert legal_pauper["kind"] == "enum" and legal_pauper["roles"] == ["filter"]
        assert manifest["dataset"] == {"name": "demo bulk", "license": "CC0-1.0", "sha256": "abc"}
        assert manifest["media"] == {"codec": "avif", "quality": 48, "lossy": True}
        assert manifest["capabilities_ext"]["scalars_present"] is True
        assert any(s["name"] == "chunk_scalars" for s in info["sections"])
        assert "schema" not in plain.inspect()["manifest"]


def test_rows_and_schema_go_together_and_bad_values_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        for kwargs in (
            {"scalar_rows": rows()},
            {"schema": SCHEMA},
            {"scalar_rows": rows()[:2], "schema": SCHEMA},
            {"scalar_rows": [dict(r, year="two thousand") for r in rows()], "schema": SCHEMA},
        ):
            try:
                build(os.path.join(tmp, "bad.urna"), **kwargs)
            except ValueError as e:
                assert "scalar" in str(e) or "column" in str(e), str(e)
            else:
                raise AssertionError(f"{list(kwargs)} was accepted")


def test_spec_declares_scalars_and_dataset_and_refuses_a_nameless_field():
    from urna.specs.specparse import SpecError, _parse

    base = {
        "corpus": {"name": "demo", "chunker_version": "demo/1"},
        "source": {"kind": "jsonl", "path": "rows.jsonl", "order_by": ["id"]},
        "models": [{"preset": "potion", "text": "default"}],
    }
    spec = _parse(
        {
            **base,
            "scalars": {"fields": SCHEMA},
            "dataset": {"name": "demo bulk", "license": "CC0-1.0", "sha256": "auto"},
        },
        "spec.toml",
    )
    assert [f["name"] for f in spec.scalars.fields] == [f["name"] for f in SCHEMA]
    assert spec.dataset is not None and spec.dataset.sha256 == "auto"
    assert _parse(base, "spec.toml").scalars.fields == []
    try:
        _parse({**base, "scalars": {"fields": [{"kind": "int"}]}}, "spec.toml")
    except SpecError as e:
        assert "scalars.fields[0]" in str(e)
    else:
        raise AssertionError("a field without a name was accepted")
    try:
        _parse({**base, "scalars": {"columns": []}}, "spec.toml")
    except SpecError as e:
        assert "unknown key" in str(e)
    else:
        raise AssertionError("an unknown scalars key was accepted")


if __name__ == "__main__":
    test_build_writes_columns_schema_dataset_and_media_without_moving_the_content_hash()
    test_rows_and_schema_go_together_and_bad_values_are_refused()
    test_spec_declares_scalars_and_dataset_and_refuses_a_nameless_field()
    print("test_scalars: ok")
