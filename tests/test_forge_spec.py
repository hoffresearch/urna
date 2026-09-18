"""Prove the declarative build contract (RFC-1) with the fake preset:
spec validation errors name their key; total ordering is enforced; the
fake e2e build emits valid multi-space files in all three output modes
with dedup'd media, shared blob spans and citation-consistent chunk_ids;
the N2 triad invalidates caches on any content change; --rebuild-only is
byte-identical (L3); a corrupted cache is recomputed, never reused;
provenance = "minimal" writes compact items (key + ordinal, items_compact,
the dedup map) and readers refuse dropped fields with a clear error;
${VAR} in spec paths expands strictly (unset names the key); the
embed cache is content-addressed under one shared root (URNA_CACHE_DIR or
xdg), so two specs with the same rows share one potion table, a media
knob change adds an entry instead of overwriting, `[output] cache_dir`
wins over the env var, the same root via another override source still
claims L3 (the lock never records the cache location), `<out>/.cache` is
never created, a conflicting model_hash probe is corrected by the loaded
model, and an unusable cache root is a SpecError naming the setting.

Run: .venv/bin/python tests/test_forge_spec.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
os.environ["URNA_ENABLE_FAKE_PRESET"] = "1"

import numpy as np
import urna
from forge.build_spec import SpecError, load_spec, validate
from forge.forge_pipeline import build

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
CACHE_ROOT = Path()  # set by main(): the suite's own URNA_CACHE_DIR, inside its tmp dir


def _listing(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")} if root.is_dir() else set()


def _fixture(
    base: Path,
    with_media: bool = True,
    mode: str = "both",
    embed_media: bool = False,
    salt: str = "",
    provenance: str = "",
    spec_name: str = "spec.toml",
    out_dir: str = "out",
) -> Path:
    from PIL import Image

    rng = np.random.default_rng(1)
    rows = []
    for i in range(12):
        img = base / f"img{i:02d}.png"
        arr = rng.integers(0, 255, (48 + i, 64, 3), dtype=np.uint8)
        if i in (5, 9):  # byte-identical duplicates of img01 -> dedup
            arr = np.array(Image.open(base / "img01.png"))
        Image.fromarray(arr).save(img)
        rows.append(
            {
                "id": f"k{i:02d}",
                "title": f"Item {i}",
                "body": f"body text {i}{salt}" if i % 3 else "",
                "img": str(img.resolve()),
            }
        )
    (base / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    media = (
        """
[media]
backend = "av1"
crf = 40
speed = 10
shard_size = 8
"""
        if with_media
        else ""
    )
    image_block = (
        """
[source.image]
path_template = "{img}"
label_template = "{title}"
"""
        if with_media
        else ""
    )
    fake_image = 'image = "space"' if with_media else ""
    spec = base / spec_name
    spec.write_text(f"""
[corpus]
name = "faketest"
chunker_version = "fake/1"

[source]
kind = "jsonl"
path = "{(base / "rows.jsonl").resolve()}"
order_by = ["id"]
[source.text]
template = \"\"\"
{{title}}
{{body}}
\"\"\"
{image_block}
{media}
[[models]]
preset = "potion"
text = "default"

[[models]]
preset = "fake-test"
text = "space"
{fake_image}
dims = [4]
space_dtype = "float32"

[build]
preset = "hybrid"
dtype = "int8"

[output]
mode = "{mode}"
dir = "{base / out_dir}"
{"embed_media = true" if embed_media else ""}
{f'provenance = "{provenance}"' if provenance else ""}
""")
    return spec


def _expect_spec_error(toml_text: str, needle: str, base: Path) -> None:
    p = base / "bad.toml"
    p.write_text(toml_text)
    try:
        validate(load_spec(p))
    except SpecError as e:
        assert needle in str(e), f"expected '{needle}' in: {e}"
    else:
        raise AssertionError(f"expected SpecError containing '{needle}'")


MINIMAL = """
[corpus]
name = "x"
chunker_version = "v"
[source]
kind = "jsonl"
path = "/tmp/none.jsonl"
order_by = ["id"]
{models}
[output]
dir = "/tmp/o"
{extra}
"""


def test_validation_errors(base: Path) -> None:
    _expect_spec_error(MINIMAL.format(models="", extra=""), "at least one", base)
    two = (
        '[[models]]\npreset="potion"\ntext="default"\n'
        '[[models]]\npreset="fake-test"\ntext="default"\n'
    )
    _expect_spec_error(MINIMAL.format(models=two, extra=""), "exactly one", base)
    bad_dims = '[[models]]\npreset="fake-test"\ntext="default"\ndims=[3]\n'
    _expect_spec_error(MINIMAL.format(models=bad_dims, extra=""), "validated ladder", base)
    heavy = '[[models]]\npreset="wemm-4b"\ntext="default"\n'
    _expect_spec_error(MINIMAL.format(models=heavy, extra="[output.x]"), "unknown key", base)
    _expect_spec_error(MINIMAL.format(models=heavy, extra=""), "allow-heavy", base)
    remote = '[[models]]\npreset="wemm-2b"\ntext="default"\n'
    _expect_spec_error(MINIMAL.format(models=remote, extra=""), "allow_remote_code", base)
    print("test_validation_errors: OK")


def test_media_profiles(base: Path) -> None:
    """media.profile resolves measured knob defaults; explicit keys win."""
    from forge.build_spec import MEDIA_PROFILES

    models = '[[models]]\npreset="potion"\ntext="default"\n'

    def parse(extra: str):
        p = base / "prof.toml"
        p.write_text(MINIMAL.format(models=models, extra=extra))
        return load_spec(p)

    m = parse('[media]\nprofile = "near-dup"').media
    assert (m.profile, m.order, m.gop, m.tune) == ("near-dup", "cluster", "auto", "still")
    m = parse('[media]\nprofile = "near-dup"\norder = "none"').media
    assert m.order == "none" and m.tune == "still", "explicit key must win over the profile"
    m = parse('[media]\nprofile = "archive"').media
    assert m.backend == "jxl-transcode"
    m = parse('[media]\nprofile = "stills"').media
    assert (m.backend, m.crf, m.speed) == ("avif", 48, 8), "stills = one avif per image"
    assert (m.gop, m.tune) == ("auto", "still"), "stream knobs stay at the schema defaults"
    m = parse("[media]\n").media
    assert m.tune == "still", "the bare default is the still tune, like every av1 profile"
    m = parse('[media]\ntune = "default"').media
    assert m.tune == "default", "svt-av1's own tune stays reachable"
    m = parse('[media]\nprofile = "stills"\ncrf = 40').media
    assert m.crf == 40 and m.backend == "avif", "explicit crf wins over the profile"
    m = parse('[media]\nprofile = "stills-av1"').media
    assert (m.backend, m.gop, m.tune, m.crf) == ("av1", "intra", "still", 35), "old stills"
    m = parse('[media]\nprofile = "retrieval"').media
    assert (m.gop, m.tune, m.speed, m.crf) == ("intra", "still", 6, 50)
    assert m.backend == "av1" and m.quality.drift_floor_p10 == 0.95, "gate defaults untouched"
    m = parse('[media]\nprofile = "retrieval"\ncrf = 45').media
    assert m.crf == 45 and m.speed == 6, "explicit crf wins, the rest of the profile stays"
    m = parse('[media]\nprofile = "retrieval-auto"').media
    assert (m.gop, m.tune, m.speed, m.crf) == ("intra", "still", 6, "auto")
    q = m.quality
    assert (q.drift_floor_p10, q.utility_floor_hit1, q.utility_tol) == (-1.0, 0.0, 0.02)
    assert q.visual_floor_p10 == -1e9 and q.crf_ladder == [40, 45, 50, 55, 60]
    assert q.sample_per_bucket == 12 and q.utility_query_template == "{label}", "schema defaults"
    m = parse('[media]\nprofile = "retrieval-auto"\n[media.quality]\nutility_tol = 0.05').media
    assert m.quality.utility_tol == 0.05, "explicit quality key wins"
    assert m.quality.drift_floor_p10 == -1.0 and m.crf == "auto", "profile quality keys stay"
    m = parse("[media]").media
    assert m.profile == "" and m.gop == "auto", "no profile keeps the schema defaults"
    assert m.quality.utility_floor_hit1 == -1.0, "utility leg is off by default"
    assert set(MEDIA_PROFILES) == {
        "near-dup",
        "stills",
        "stills-av1",
        "archive",
        "retrieval",
        "retrieval-auto",
    }
    gated = (
        '[[models]]\npreset="potion"\ntext="default"\n[[models]]\npreset="fake-test"\nimage="space"\n'
        '[source.image]\npath_template = "{id}.png"\n'
    )
    auto = '[media]\nprofile = "retrieval-auto"\n[media.quality]\n'
    for line, key in (
        ("utility_floor_hit1 = 1.5", "utility_floor_hit1"),
        ("utility_queries = -1", "utility_queries"),
        ('utility_query_template = "x"', "utility_query_template"),
        ("utility_tol = 2", "utility_tol"),
    ):
        _expect_spec_error(MINIMAL.format(models=gated, extra=auto + line), key, base)
    p = base / "prof-ok.toml"
    p.write_text(MINIMAL.format(models=gated, extra='[media]\nprofile = "retrieval-auto"'))
    validate(load_spec(p))
    assert load_spec(p).media.quality.utility_floor_hit1 == 0.0
    _expect_spec_error(
        MINIMAL.format(models=gated, extra='[media]\nprofile = "stills"\ncrf = "auto"'),
        "media.crf",
        base,
    )
    _expect_spec_error(
        MINIMAL.format(models=models, extra='[media]\nprofile = "cards"'),
        "media.profile",
        base,
    )
    print("test_media_profiles: OK")


def test_quality_defaults(base: Path) -> None:
    """the gate floors default to values a real corpus reaches (the mtg cards
    at 488x680: crf30 p10 65.3 passes, crf35 p10 55.7 fails), and a spec
    still overrides each floor and the ladder under [media.quality]."""
    from forge.build_spec import QualitySpec

    q = QualitySpec()
    assert (q.visual_floor_p10, q.visual_floor_min, q.drift_floor_p10) == (60.0, 45.0, 0.95)
    assert q.crf_ladder == [25, 30, 35, 40, 45, 50]
    assert 55.7 < q.visual_floor_p10 <= 65.3, "crf30 must pass and crf35 fail on p10"
    assert q.visual_floor_min <= 45.3 and q.drift_floor_p10 <= 0.965, "crf35 fails on p10 only"
    models = '[[models]]\npreset="potion"\ntext="default"\n'
    p = base / "floors.toml"
    p.write_text(MINIMAL.format(models=models, extra="[media]"))
    assert load_spec(p).media.quality == QualitySpec(), "no [media.quality] = the defaults"
    p.write_text(
        MINIMAL.format(
            models=models,
            extra="[media]\n[media.quality]\nvisual_floor_p10 = 85\nvisual_floor_min = 72\n"
            "drift_floor_p10 = 0.98\ncrf_ladder = [30, 35, 40, 45]",
        )
    )
    q = load_spec(p).media.quality
    assert (q.visual_floor_p10, q.visual_floor_min, q.drift_floor_p10) == (85.0, 72.0, 0.98)
    assert q.crf_ladder == [30, 35, 40, 45], "the old floors are one override away"
    one = "[media]\n[media.quality]\ndrift_floor_p10 = -1"
    p.write_text(MINIMAL.format(models=models, extra=one))
    q = load_spec(p).media.quality
    assert q.drift_floor_p10 == -1 and q.visual_floor_p10 == 60.0, "one key, the rest stays"
    print("test_quality_defaults: OK")


def test_env_expansion(base: Path) -> None:
    prior = os.environ.get("URNA_FORGE_TEST_DATA")
    try:
        _env_expansion_body(base)
    finally:
        if prior is None:
            os.environ.pop("URNA_FORGE_TEST_DATA", None)
        else:
            os.environ["URNA_FORGE_TEST_DATA"] = prior
    print("test_env_expansion: OK")


def _env_expansion_body(base: Path) -> None:
    d = base / "env"
    (d / "data").mkdir(parents=True)
    rows = [{"id": f"r{i}", "title": f"Row {i}"} for i in range(3)]
    (d / "data" / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    spec_p = d / "env.toml"
    spec_p.write_text(f"""
[corpus]
name = "envtest"
chunker_version = "v"
[source]
kind = "jsonl"
path = "${{URNA_FORGE_TEST_DATA}}/rows.jsonl"
order_by = ["id"]
[source.text]
template = "{{title}} costs $5"
[[models]]
preset = "potion"
text = "default"
[output]
dir = "{d / "out"}"
""")
    os.environ["URNA_FORGE_TEST_DATA"] = str(d / "data")
    spec = load_spec(spec_p)
    validate(spec)
    assert spec.source.path == str(d / "data" / "rows.jsonl"), "braced var must expand"
    assert spec.source.text.template == "{title} costs $5", (
        "bare $5 and {col} placeholders must survive"
    )
    result = build(spec)
    assert result["n_items"] == 3
    urna.open(result["outputs"]["envtest.urna"]["file"]).validate()
    # the lock stores the expanded path; a rebuild under a moved data root
    # (same rows, another location) must still claim L3 under --strict-env
    shutil.copytree(d / "data", d / "data-b")
    os.environ["URNA_FORGE_TEST_DATA"] = str(d / "data-b")
    again = build(load_spec(spec_p), rebuild_only=True, strict_env=True)
    assert (
        again["outputs"]["envtest.urna"]["file_hash"]
        == (result["outputs"]["envtest.urna"]["file_hash"])
    ), "rebuild-only under another data root must stay byte-identical"
    for state, setup in (("is not set", None), ("is empty", "")):
        if setup is None:
            os.environ.pop("URNA_FORGE_TEST_DATA")
        else:
            os.environ["URNA_FORGE_TEST_DATA"] = setup
        try:
            load_spec(spec_p)
        except SpecError as e:
            msg = str(e)
            assert "source.path" in msg and "URNA_FORGE_TEST_DATA" in msg, msg
            assert state in msg and "export" in msg, msg
        else:
            raise AssertionError(f"a ${{VAR}} that {state} must be a SpecError, never a silent '$'")
    # no expanduser mid-string, bare $ untouched, and ~/ still expands
    from forge.spec_paths import expand_paths

    os.environ["URNA_FORGE_TEST_DATA"] = "/x"
    out = expand_paths(
        {"source": {"db": "${URNA_FORGE_TEST_DATA}/~x", "query": "cost > $5"}},
    )
    assert out == {"source": {"db": "/x/~x", "query": "cost > $5"}}, out
    assert expand_paths({"text": {"template": "{t} $5 ${URNA_FORGE_TEST_DATA}"}}) == {
        "text": {"template": "{t} $5 /x"}
    }, "the braced var expands inside a template while {col} and $5 survive"
    assert expand_paths(["~/x"]) == [os.path.expanduser("~/x")]
    os.environ["URNA_FORGE_TEST_DATA"] = "~/via-var"
    assert expand_paths("${URNA_FORGE_TEST_DATA}/y") == os.path.expanduser("~/via-var/y"), (
        "a variable holding ~/ expands too"
    )
    try:
        expand_paths({"models": [{"preset": "p"}, {"model_path": "${URNA_FORGE_UNSET_X}"}]})
    except SpecError as e:
        assert str(e).startswith("models[1].model_path:"), str(e)
    else:
        raise AssertionError("dotted key path must reach into lists")
    # the module must import on its own in a fresh interpreter (build_spec
    # imports it at its bottom; SpecError is resolved lazily at raise time)
    probe = subprocess.run(
        [sys.executable, "-c", "from forge.spec_paths import expand_paths"],
        cwd=REPO / "python",
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_total_ordering(base: Path) -> None:
    d = base / "dupes"
    d.mkdir()
    (d / "rows.jsonl").write_text('{"id": "a", "t": "x"}\n{"id": "a", "t": "y"}')
    spec_p = d / "s.toml"
    spec_p.write_text(f"""
[corpus]
name = "d"
chunker_version = "v"
[source]
kind = "jsonl"
path = "{d / "rows.jsonl"}"
order_by = ["id"]
[source.text]
template = "{{t}}"
[[models]]
preset = "potion"
text = "default"
[output]
dir = "{d / "out"}"
""")
    try:
        build(load_spec(spec_p))
    except SpecError as e:
        assert "total order" in str(e)
    else:
        raise AssertionError("duplicate order_by keys must fail")
    print("test_total_ordering: OK")


def test_e2e_fake(base: Path) -> None:
    if not HAVE_FFMPEG:
        print("test_e2e_fake: SKIP (no ffmpeg)")
        return
    d = base / "e2e"
    d.mkdir()
    spec = load_spec(_fixture(d, with_media=True, mode="both"))
    result = build(spec)
    assert result["n_items"] == 12 and result["n_unique_frames"] == 10, "dedup must collapse dupes"
    out = d / "out"
    dbs = {name: urna.open(str(out / name)) for name in result["outputs"]}
    for db in dbs.values():
        db.validate()
    single = dbs["faketest.urna"]
    assert single.space_names == ["fake-test@4", "fake-test-text@4"]
    hits = single.search_space("fake-test@4", [0.5, 0.5, 0.5, 0.5], 3)
    assert len(hits) == 3 and hits[0].score >= hits[2].score
    # citation consistency: same chunk_ids in the single and per-model files
    q = [1.0] + [0.0] * 255
    ids_single = {h.chunk_id for h in single.search(q, k=12)}
    ids_per = {h.chunk_id for h in dbs["faketest-fake-test.urna"].search(q, k=12)}
    assert ids_single == ids_per, "chunk_ids must be identical across output modes"
    # duplicate rows share the same blob span (N chunks -> 1 frame)
    manifest = json.loads((out / "faketest.manifest.json").read_text())
    assert manifest["manifest_schema_version"] == 1
    uris = [it["media_uri"] for it in manifest["items"]]
    assert uris[1] == uris[5] == uris[9], "dup rows must map to one frame"
    assert manifest["media"]["dedup"]["n_unique_frames"] == 10
    # L3: rebuild from caches is byte-identical
    h1 = (out / "faketest.urna").read_bytes()
    build(load_spec(d / "spec.toml"), rebuild_only=True)
    assert (out / "faketest.urna").read_bytes() == h1, "rebuild-only must be byte-identical"
    print("test_e2e_fake: OK")


def test_provenance_minimal(base: Path) -> None:
    """provenance = "minimal" writes items[] as key + ordinal only, flagged
    by items_compact, keeps the dedup map when it is not the identity, and
    the .urna is byte-for-byte the standard build's (the manifest is a
    sidecar, never part of the file)."""
    if not HAVE_FFMPEG:
        print("test_provenance_minimal: SKIP (no ffmpeg)")
        return
    from forge.forge_manifest import frame_resolver, manifest_items

    d = base / "prov"
    d.mkdir()
    build(load_spec(_fixture(d, mode="single")))
    build(
        load_spec(
            _fixture(d, mode="single", provenance="minimal", spec_name="min.toml", out_dir="min")
        )
    )
    full = json.loads((d / "out" / "faketest.manifest.json").read_text())
    compact = json.loads((d / "min" / "faketest.manifest.json").read_text())
    assert full["items_compact"] is False and compact["items_compact"] is True
    assert compact["provenance_mode"] == "minimal" and "sql" not in compact
    assert all(set(it) == {"key", "ordinal"} for it in compact["items"]), compact["items"][0]
    assert [it["key"] for it in compact["items"]] == [it["key"] for it in full["items"]]
    assert {"image_path", "label", "media_uri"} <= set(full["items"][0]), "standard keeps full"
    assert "frame_of_row" not in full, "full items carry media_uri; no dedup map"
    assert compact["frame_of_row"][1] == compact["frame_of_row"][5] == compact["frame_of_row"][9]
    assert len(set(compact["frame_of_row"])) == 10, "10 unique frames for 12 rows"
    # the derived frame of every compact item equals the full item's frame
    rf, rc = frame_resolver(full), frame_resolver(compact)
    assert [rf(it) for it in full["items"]] == [rc(it) for it in compact["items"]]
    assert rc(compact["items"][5]) == rc(compact["items"][1]), "dup rows share one frame"
    # the file is the same: the manifest is a sidecar
    a = urna.open(str(d / "out" / "faketest.urna"))
    b = urna.open(str(d / "min" / "faketest.urna"))
    assert a.file_hash == b.file_hash, "provenance mode must not touch the .urna"
    assert (d / "out" / "faketest.manifest.json").stat().st_size > (
        d / "min" / "faketest.manifest.json"
    ).stat().st_size
    # readers: key + ordinal never raise; a dropped field is a clear error
    assert len(manifest_items(compact)) == 12
    assert len(manifest_items(compact, need=("key", "ordinal"))) == 12
    assert len(manifest_items(full, need=("image_path", "label"))) == 12
    for field in ("image_path", "label", "media_uri"):
        try:
            manifest_items(compact, need=(field,))
        except ValueError as e:
            msg = str(e)
            assert field in msg and 'provenance = "standard"' in msg and "minimal" in msg, msg
        else:
            raise AssertionError(f"compact items must refuse need={field!r}")
    # the ui bridge browses both: same blob uri + frame per ordinal, the
    # compact title falls back to the key (no label recorded)
    pages = []
    for out in ("out", "min"):
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO / "python" / "tools" / "urna_ui_bridge.py"),
                str(d / out / "faketest.urna"),
                "browse",
                "--limit",
                "12",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "URNA_ENABLE_FAKE_PRESET": "1"},
        )
        assert proc.returncode == 0, proc.stderr
        pages.append(json.loads(proc.stdout))

    def strip(page: dict) -> list[tuple]:
        return [(i["ordinal"], i["uri"], i["frame"], i["chunk_id"]) for i in page["items"]]

    assert strip(pages[0]) == strip(pages[1]) and pages[1]["total"] == 12
    assert pages[0]["items"][3]["title"] == "Item 3" and pages[1]["items"][3]["title"] == "k03"
    assert pages[1]["items"][5]["frame"] == pages[1]["items"][1]["frame"]
    print("test_provenance_minimal: OK")


def test_embed_media(base: Path) -> None:
    if not HAVE_FFMPEG:
        print("test_embed_media: SKIP (no ffmpeg)")
        return
    d = base / "embed"
    d.mkdir()
    spec = load_spec(_fixture(d, with_media=True, mode="single", embed_media=True))
    build(spec)
    out = d / "out"
    db = urna.open(str(out / "faketest.urna"))
    db.validate()
    refs = db.blob_refs()
    assert refs and all(r["inlined"] for r in refs), "embed_media must inline every blob"
    media_bytes = sum(p.stat().st_size for p in (out / "faketest.media").glob("*") if p.is_file())
    urna_bytes = (out / "faketest.urna").stat().st_size
    assert urna_bytes > media_bytes, "the self-contained file must carry the media bytes"
    manifest = json.loads((out / "faketest.manifest.json").read_text())
    assert manifest["media"]["embedded"] is True
    # the sidecar-mode twin of the same corpus keeps blobs out-of-line
    d2 = base / "embed-side"
    d2.mkdir()
    build(load_spec(_fixture(d2, with_media=True, mode="single")))
    side = urna.open(str(d2 / "out" / "faketest.urna"))
    assert all(not r["inlined"] for r in side.blob_refs())
    print("test_embed_media: OK")


def test_triad_invalidation(base: Path) -> None:
    d = base / "triad"
    d.mkdir()
    spec_p = _fixture(d, with_media=False, mode="single")
    r1 = build(load_spec(spec_p))
    rows = (d / "rows.jsonl").read_text().replace("body text 1", "body text 1 EDITED")
    (d / "rows.jsonl").write_text(rows)
    try:
        build(load_spec(spec_p), rebuild_only=True)
    except Exception as e:
        assert "triad" in str(e) or "stale" in str(e) or "missing" in str(e)
    else:
        raise AssertionError("content change must invalidate the cache under --rebuild-only")
    r2 = build(load_spec(spec_p))
    assert r1["corpus_input_hash"] != r2["corpus_input_hash"]
    print("test_triad_invalidation: OK")


def _new_entries(preset_dir: Path, before: set[str]) -> set[Path]:
    return {p for p in preset_dir.glob("*.npz") if p.name not in before}


def test_corrupt_cache_recomputed(base: Path) -> None:
    """a torn entry is recomputed, never reused: corrupt the very npz the
    build wrote (its own salt keeps the triad apart from every other
    fixture, so the first build is a miss and writes exactly one entry),
    then prove the second build rewrote it and its sidecar is valid again."""
    import hashlib

    d = base / "corrupt"
    d.mkdir()
    spec_p = _fixture(d, with_media=False, mode="single", salt=" torn")
    potion = CACHE_ROOT / "embed" / "potion"
    before = {p.name for p in potion.glob("*.npz")} if potion.is_dir() else set()
    build(load_spec(spec_p))
    assert not (d / "out" / ".cache").exists(), "the embed cache must not live in the output dir"
    (cache,) = _new_entries(potion, before)  # exactly one entry, the one this build wrote
    sidecar = cache.with_suffix(".npz.sha256")
    assert sidecar.read_text().strip() == hashlib.sha256(cache.read_bytes()).hexdigest()
    mtime = cache.stat().st_mtime_ns
    cache.write_bytes(cache.read_bytes()[:-7])  # torn write: sidecar no longer matches
    result = build(load_spec(spec_p))  # must recompute, not crash or reuse
    urna.open(result["outputs"]["faketest.urna"]["file"]).validate()
    assert cache.stat().st_mtime_ns != mtime, "a torn entry must be rewritten, not reused"
    assert sidecar.read_text().strip() == hashlib.sha256(cache.read_bytes()).hexdigest(), (
        "the recompute must leave a valid checksum sidecar"
    )
    assert _new_entries(potion, before) == {cache}, "the recompute reuses the same entry name"
    assert build(load_spec(spec_p), rebuild_only=True)["timings"]["embed.potion"] == 0.0
    print("test_corrupt_cache_recomputed: OK")


def test_cache_shared_across_specs(base: Path) -> None:
    """content-addressed entries under one root: same rows in two output
    dirs read one potion table; a media knob change adds a clip-side entry
    instead of overwriting; the spec's cache_dir beats URNA_CACHE_DIR."""
    if not HAVE_FFMPEG:
        print("test_cache_shared_across_specs: SKIP (no ffmpeg)")
        return
    potion = CACHE_ROOT / "embed" / "potion"
    fake = CACHE_ROOT / "embed" / "fake-test"
    before_potion = {p.name for p in potion.glob("*.npz")} if potion.is_dir() else set()
    before_fake = {p.name for p in fake.glob("*.npz")} if fake.is_dir() else set()

    def fixture_rows(d: Path) -> Path:
        # byte-identical rows + images across dirs: only the paths differ, and
        # paths never enter the triad (corpus_input_hash is over content). the
        # salt keeps this test's triad apart from the earlier fixtures, which
        # already share their own entries in the same root.
        d.mkdir()
        return _fixture(d, with_media=True, mode="single", salt=" shared")

    # timings alone cannot tell a hit from a sub-millisecond compute on 12
    # rows (both round to 0.0), so a hit is proven by the entry's mtime: a
    # recompute rewrites the content-addressed file, a hit never touches it.
    d1, d2 = base / "shared-a", base / "shared-b"
    r1 = build(load_spec(fixture_rows(d1)))
    new_potion = {p.name for p in potion.glob("*.npz")} - before_potion
    new_fake = {p.name for p in fake.glob("*.npz")} - before_fake
    assert len(new_potion) == 1, f"one potion entry for one triad, got {new_potion}"
    assert len(new_fake) == 1, f"one image-space entry for one triad, got {new_fake}"
    assert r1["corpus_input_hash"], "the manifest must carry the shared key"
    potion_entry = potion / next(iter(new_potion))
    fake_entry = fake / next(iter(new_fake))
    potion_mtime, fake_mtime = potion_entry.stat().st_mtime_ns, fake_entry.stat().st_mtime_ns

    r2 = build(load_spec(fixture_rows(d2)))
    assert r2["corpus_input_hash"] == r1["corpus_input_hash"], "same rows, same triad key"
    assert r2["timings"]["embed.potion"] == 0.0, "second output dir must hit the shared entry"
    assert r2["timings"]["embed.fake-test"] == 0.0, "same media knobs must hit the image entry"
    assert potion_entry.stat().st_mtime_ns == potion_mtime, "a hit must not rewrite the entry"
    assert fake_entry.stat().st_mtime_ns == fake_mtime, "a hit must not rewrite the entry"
    assert {p.name for p in potion.glob("*.npz")} - before_potion == new_potion, (
        "a second spec with the same rows must not add a second potion table"
    )
    for d in (d1, d2):
        assert not (d / "out" / ".cache").exists(), "no per-output .cache dir"
        assert (d / "out" / ".forge-state").is_dir(), ".forge-state stays transactional in out"

    # same corpus name, changed [media] crf: the text-only potion recipe is
    # unchanged (no decoder fingerprint), the image-space recipe is not, so a
    # SECOND fake-test entry appears beside the first, nothing is overwritten.
    spec_p = d2 / "spec.toml"
    spec_p.write_text(spec_p.read_text().replace("crf = 40", "crf = 45"))
    r3 = build(load_spec(spec_p))
    assert r3["timings"]["embed.potion"] == 0.0, "text-only recipe must not change with crf"
    assert potion_entry.stat().st_mtime_ns == potion_mtime, "potion entry must be untouched"
    assert fake_entry.stat().st_mtime_ns == fake_mtime, "the old image entry must be untouched"
    after_fake = {p.name for p in fake.glob("*.npz")} - before_fake
    assert new_fake < after_fake and len(after_fake) == 2, (
        f"crf change must add an image entry, not overwrite: {after_fake}"
    )
    for npz in after_fake:
        assert (fake / (npz + ".sha256")).is_file(), "every entry keeps its checksum sidecar"

    # [output] cache_dir in the spec wins over URNA_CACHE_DIR
    local = base / "local-cache"
    spec_p.write_text(spec_p.read_text().replace("[output]", f'[output]\ncache_dir = "{local}"'))
    build(load_spec(spec_p))
    assert len(list((local / "embed" / "potion").glob("*.npz"))) == 1, (
        "a fresh root has no entry to hit: the potion table must be written under cache_dir"
    )
    assert list((local / "models").glob("model_hash.potion.*.json")), "probe joins the root"
    assert {p.name for p in potion.glob("*.npz")} - before_potion == new_potion, (
        "the env root must be untouched when the spec overrides it"
    )

    # the cache location is not identity: the same root supplied through
    # URNA_CACHE_DIR instead of the spec must still claim L3 under --strict-env,
    # and the lock never records where the cache lived.
    spec_p.write_text(spec_p.read_text().replace(f'cache_dir = "{local}"\n', ""))
    assert "cache_dir" not in spec_p.read_text()
    env_root = os.environ["URNA_CACHE_DIR"]
    os.environ["URNA_CACHE_DIR"] = str(local)
    try:
        r5 = build(load_spec(spec_p), rebuild_only=True, strict_env=True)
    finally:
        os.environ["URNA_CACHE_DIR"] = env_root
    assert r5["timings"]["embed.potion"] == 0.0, "same root via env must hit the spec's entries"
    lock = json.loads(Path(r5["build_lock"]).read_text())
    assert "cache_dir" not in lock["resolved_spec"]["output"], "cache root is not in the lock"
    assert str(local) not in Path(r5["build_lock"]).read_text(), "no cache path in the lock"
    print("test_cache_shared_across_specs: OK")


def test_probe_conflict(base: Path) -> None:
    """the model_hash probe is keyed by the knobs that enter the fingerprint,
    and a planted probe that disagrees with the loaded model is corrected,
    not fatal: the second spec still builds and the entry is keyed by the
    real hash."""
    from forge import forge_recipe
    from forge.build_spec import ModelSpec

    a = ModelSpec(preset="potion", text="default")
    b = ModelSpec(preset="potion", text="default", normalize=False)
    c = ModelSpec(preset="potion", text="default", dtype="float16")
    paths = {forge_recipe.probe_path(CACHE_ROOT, m) for m in (a, b, c)}
    assert len(paths) == 3, "normalize and dtype must select different probes"
    assert all(p.name.startswith("model_hash.potion.") for p in paths)

    d = base / "probe"
    d.mkdir()
    spec_p = _fixture(d, with_media=False, mode="single", salt=" probe")
    probe = forge_recipe.probe_path(CACHE_ROOT, a)
    assert probe.is_file(), "earlier builds must have written the default-knob potion probe"
    real = json.loads(probe.read_text())["model_hash"]
    # plant a conflicting probe with a matching dir fingerprint (potion is
    # vendored: no model dir, fingerprint None), as another spec's stale probe
    probe.write_text(json.dumps({"model_hash": "sha256:" + "0" * 64, "dir_fingerprint": None}))
    result = build(load_spec(spec_p))
    urna.open(result["outputs"]["faketest.urna"]["file"]).validate()
    lock = json.loads(Path(result["build_lock"]).read_text())
    assert lock["models"]["potion"] == real, "the loaded model is the ground truth"
    assert json.loads(probe.read_text())["model_hash"] == real, "the probe must be corrected"
    assert build(load_spec(spec_p), rebuild_only=True)["timings"]["embed.potion"] == 0.0, (
        "the entry must be keyed by the real hash so the rebuild hits it"
    )
    print("test_probe_conflict: OK")


def test_cache_root_errors(base: Path) -> None:
    """a cache root that cannot be a directory is a typed SpecError naming
    the setting, not an OSError traceback out of the cli."""
    d = base / "badroot"
    d.mkdir()
    spec_p = _fixture(d, with_media=False, mode="single")
    not_a_dir = base / "cache-as-file"
    not_a_dir.write_text("x")
    env_root = os.environ["URNA_CACHE_DIR"]
    os.environ["URNA_CACHE_DIR"] = str(not_a_dir)
    try:
        build(load_spec(spec_p))
    except SpecError as e:
        assert "output.cache_dir" in str(e) and "URNA_CACHE_DIR" in str(e), str(e)
    else:
        raise AssertionError("a file as cache root must be a SpecError")
    finally:
        os.environ["URNA_CACHE_DIR"] = env_root
    spec_p.write_text(
        spec_p.read_text().replace("[output]", f'[output]\ncache_dir = "{not_a_dir}"')
    )
    try:
        build(load_spec(spec_p))
    except SpecError as e:
        assert "output.cache_dir" in str(e)
    else:
        raise AssertionError("a file as [output] cache_dir must be a SpecError")
    print("test_cache_root_errors: OK")


def main() -> None:
    global CACHE_ROOT
    user_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "urna"
    user_before = _listing(user_cache)
    with tempfile.TemporaryDirectory(prefix="urna-forge-spec-") as tmp:
        base = Path(tmp)
        # every build in this suite writes its embed cache here, never ~/.cache
        CACHE_ROOT = base / "xdg-cache"
        os.environ["URNA_CACHE_DIR"] = str(CACHE_ROOT)
        test_validation_errors(base)
        test_media_profiles(base)
        test_quality_defaults(base)
        test_env_expansion(base)
        test_total_ordering(base)
        test_e2e_fake(base)
        test_provenance_minimal(base)
        test_embed_media(base)
        test_triad_invalidation(base)
        test_corrupt_cache_recomputed(base)
        test_cache_shared_across_specs(base)
        test_probe_conflict(base)
        test_cache_root_errors(base)
        assert (CACHE_ROOT / "embed" / "potion").is_dir(), "builds must have used the env root"
    assert _listing(user_cache) == user_before, "the suite must never touch the user's cache"
    print("all forge spec tests passed")


if __name__ == "__main__":
    main()
