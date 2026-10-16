"""Build the card fields file the core reads next to the `.urna`.

One row per ordinal, from the dataset's Parquet (the fields the chunks were
rendered from) joined by oracle_id to Scryfall's oracle-cards bulk (colors,
cmc, legalities, popularity). The output is zstd json, bound to the `.urna`
by its content_hash, and records the bulk it came from.

    python tools/build_fields.py DATASET_DIR ORACLE_CARDS_JSONL_GZ URNA_FILE OUT
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

PARQUET_COLUMNS = [
    "ordinal", "oracle_id", "name", "pt_name", "mana_cost", "type_line",
    "rarity", "set_code", "oracle_text", "art_series",
]


def scryfall_index(path: Path) -> dict[str, dict]:
    """oracle_id -> card, faces included (reversible cards keep theirs there)."""
    out: dict[str, dict] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            c = json.loads(line)
            ids = {c.get("oracle_id")} | {face.get("oracle_id") for face in c.get("card_faces") or []}
            for i in ids - {None}:
                out.setdefault(i, c)
    return out


def kind(row: dict, sf: dict | None) -> str:
    if row["art_series"]:
        return "art_series"
    t = row["type_line"] or ""
    if "Emblem" in t:
        return "emblem"
    if "Token" in t:
        return "token"
    if sf is None and (row["name"] or "").startswith("A-"):
        return "alchemy"
    if sf is not None and not any(v != "not_legal" for v in (sf.get("legalities") or {}).values()):
        return "unsanctioned"
    return "card"


def face_field(sf: dict, key: str):
    """power, toughness and loyalty live on the faces of a two-faced card."""
    if key in sf:
        return sf[key]
    vals = [f[key] for f in sf.get("card_faces") or [] if key in f]
    return " // ".join(vals) if vals else None


def card_row(row: dict, sf: dict | None) -> dict:
    out = {
        "ordinal": row["ordinal"],
        "oracle_id": row["oracle_id"],
        "name": row["name"],
        "pt_name": row["pt_name"] or None,
        "mana_cost": row["mana_cost"] or "",
        "type_line": row["type_line"] or "",
        "rarity": row["rarity"] or "",
        "set_code": row["set_code"] or "",
        "oracle_text": row["oracle_text"] or "",
        "kind": kind(row, sf),
    }
    if sf is not None:
        out.update({
            "colors": sf.get("colors") or sorted({c for f in sf.get("card_faces") or [] for c in f.get("colors") or []}),
            "color_identity": sf.get("color_identity") or [],
            "cmc": sf.get("cmc"),
            "power": face_field(sf, "power"),
            "toughness": face_field(sf, "toughness"),
            "loyalty": face_field(sf, "loyalty"),
            "keywords": sf.get("keywords") or [],
            "layout": sf.get("layout"),
            "legal": {f: s for f, s in (sf.get("legalities") or {}).items() if s != "not_legal"},
            "edhrec_rank": sf.get("edhrec_rank"),
            "games": sf.get("games") or [],
        })
    return out


def content_hash(urna: Path) -> str:
    meta = json.loads(subprocess.run(["urna", "inspect", "--json", str(urna)], check=True, capture_output=True).stdout)
    return meta["content_hash"]


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    dataset, bulk, urna, out = (Path(a) for a in sys.argv[1:])
    files = sorted((dataset / "data").glob("cards-*.parquet"))
    rows = pa.concat_tables([pq.read_table(f, columns=PARQUET_COLUMNS) for f in files]).sort_by("ordinal").to_pylist()
    if [r["ordinal"] for r in rows] != list(range(len(rows))):
        print("error: ordinals are not 0..n-1", file=sys.stderr)
        return 1
    index = scryfall_index(bulk)
    cards = [card_row(r, index.get(r["oracle_id"])) for r in rows]
    doc = {
        "format": "pasta-fields/1",
        "content_hash": content_hash(urna),
        "n_cards": len(cards),
        "scryfall_bulk": {"file": bulk.name, "sha256": hashlib.sha256(bulk.read_bytes()).hexdigest()},
        "cards": cards,
    }
    raw = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()
    out.write_bytes(zstandard.ZstdCompressor(level=19).compress(raw))
    kinds: dict[str, int] = {}
    for c in cards:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"{out}: {len(cards)} cards, {len(raw)} bytes json, {out.stat().st_size} bytes zstd, kinds {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
