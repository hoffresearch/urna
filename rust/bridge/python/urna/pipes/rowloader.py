"""Row loaders for declarative builds: sqlite/csv/jsonl/image_dir.

Every loader returns the same shape: an ordered list of Row with the item
input_hash (RFC-0 N2) computed at load time - sha256 over canonical text,
source image bytes, label and chunker_version - so any content change
invalidates caches by construction. The image byte hash is computed once
here and reused by media dedup (RFC-1).

Total ordering (N1): sqlite/csv/jsonl require `order_by`; the resolved sort
key must be UNIQUE across rows or loading fails telling the operator to
append a unique column. `ordinal` is presentation order, never identity (N3).

source_uri: a row column named `source_uri` wins; otherwise the stable form
`item://<corpus>/<key>` - deterministic, survives media re-encoding.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import string
from dataclasses import dataclass
from pathlib import Path

from urna.specs.specparse import CorpusSpec, SpecError


@dataclass
class Row:
    ordinal: int
    key: str
    canonical_text: str
    source_uri: str
    image_path: Path | None
    image_sha256: str | None  # sha256 of source image bytes; reused by dedup
    label: str | None
    item_input_hash: str


_DERIVE_RE = re.compile(r"^(\w+)\((\w+)\)$")


def _derive_value(expr: str, row: dict) -> str:
    m = _DERIVE_RE.match(expr.strip())
    if not m:
        raise SpecError(f"source.derive: unsupported expression '{expr}' (use helper(column))")
    helper, col = m.groups()
    value = str(row.get(col, ""))
    if helper == "basename_stem":
        return Path(value.split("?")[0]).name.rsplit(".", 1)[0]
    if helper == "lower":
        return value.lower()
    raise SpecError(f"source.derive: unknown helper '{helper}' (valid: basename_stem, lower)")


class _Blank(dict):
    def __missing__(self, key):
        return ""


def render_template(template: str, row: dict) -> str:
    """Format a text template, dropping lines whose placeholders ALL resolve
    empty, then cleaning empty punctuation artifacts (`()`, dangling `-`)."""
    lines = []
    for line in template.strip().splitlines():
        fields = [f for _, f, _, _ in string.Formatter().parse(line) if f]
        names = [f.split(".")[0].split("[")[0] for f in fields]
        if names and all(not str(row.get(n, "")).strip() for n in names):
            continue
        rendered = line.format_map(_Blank(row))
        rendered = re.sub(r"\(\s*\)", "", rendered)
        rendered = re.sub(r"\[\s*,?\s*\]", "", rendered)
        rendered = re.sub(r"(^\s*—\s*)|(\s*—\s*$)", "", rendered)
        rendered = re.sub(r",\s*\]", "]", rendered)
        rendered = re.sub(r"\s{2,}", " ", rendered).strip()
        if rendered:
            lines.append(rendered)
    return "\n".join(lines)


def _item_input_hash(
    text: str, image_sha256: str | None, label: str | None, chunker_version: str
) -> str:
    h = hashlib.sha256()
    h.update(b"text:" + hashlib.sha256(text.encode()).digest())
    h.update(b"image:" + (image_sha256 or "none").encode())
    h.update(b"label:" + (label or "").encode())
    h.update(b"chunker:" + chunker_version.encode())
    return "sha256:" + h.hexdigest()


def corpus_input_hash(rows: list[Row]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(row.item_input_hash.encode())
    return "sha256:" + h.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _rows_from_dicts(spec: CorpusSpec, raw: list[dict]) -> list[Row]:
    src = spec.source
    for row in raw:
        for name, expr in src.derive.items():
            row[name] = _derive_value(expr, row)
    if not src.order_by:
        raise SpecError("source.order_by: required")
    keyed = [(tuple(str(r.get(c, "")) for c in src.order_by), r) for r in raw]
    keys = [k for k, _ in keyed]
    if len(keys) != len(set(keys)):
        dupe = next(k for k in keys if keys.count(k) > 1)
        raise SpecError(
            f"source.order_by {src.order_by} is not a total order (duplicate key {dupe}); "
            "append a unique column (RFC-0 N1)"
        )
    keyed.sort(key=lambda kv: kv[0])

    rows: list[Row] = []
    for ordinal, (key, r) in enumerate(keyed):
        text = render_template(src.text.template, r) if src.text.template else ""
        image_path = image_hash = None
        if src.image.path_template:
            p = Path(src.image.path_template.format_map(_Blank(r)))
            if not p.is_file():
                raise SpecError(f"source.image: file missing for key {key}: {p}")
            image_path, image_hash = p, _sha256_file(p)
        label = (
            src.image.label_template.format_map(_Blank(r)).strip()
            if src.image.label_template
            else None
        )
        if not text:
            text = label or "|".join(key)  # identity-only text (RFC-0 N6)
        uri = str(r["source_uri"]) if r.get("source_uri") else f"item://{spec.name}/{'|'.join(key)}"
        rows.append(
            Row(
                ordinal,
                "|".join(key),
                text,
                uri,
                image_path,
                image_hash,
                label,
                _item_input_hash(text, image_hash, label, spec.chunker_version),
            )
        )
    return rows


def _load_sqlite(spec: CorpusSpec) -> list[Row]:
    src = spec.source
    con = sqlite3.connect(f"file:{src.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    raw = [dict(r) for r in con.execute(src.query)]
    for join in src.joins:
        lookup = {}
        for r in con.execute(join.query):
            d = dict(r)
            if join.on not in d:
                raise SpecError(f"source.joins: join query has no column '{join.on}'")
            lookup[d[join.on]] = d
        hits = 0
        for row in raw:
            extra = lookup.get(row.get(join.on))
            if extra:
                hits += 1
                for k, v in extra.items():
                    row.setdefault(k, v)
        # a join that matches NOTHING is a typo or a type mismatch (TEXT vs
        # INTEGER keys), never intent; partial misses are legitimate
        # (sparse localizations) but the operator should see the count.
        if raw and lookup and hits == 0:
            raise SpecError(
                f"source.joins on '{join.on}': 0 of {len(raw)} rows matched - "
                "check the key name and its sqlite type on both sides"
            )
        if raw and hits < len(raw):
            print(f"[forge] join on '{join.on}': {hits}/{len(raw)} rows matched")
    con.close()
    return _rows_from_dicts(spec, raw)


def _load_csv_jsonl(spec: CorpusSpec) -> list[Row]:
    path = Path(spec.source.path)
    if spec.source.kind == "csv":
        with path.open(newline="") as fh:
            raw = [dict(r) for r in csv.DictReader(fh)]
    else:
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return _rows_from_dicts(spec, raw)


def _load_image_dir(spec: CorpusSpec) -> list[Row]:
    from urna.image import discovery

    src = spec.source
    labels = discovery.load_labels(Path(src.labels)) if src.labels else None
    items = discovery.collect_images(Path(src.input_dir), labels)
    rows = []
    for item in items:
        p = Path(item.render_path)
        image_hash = _sha256_file(p)
        text = item.canonical_text()  # identity-only (RFC-0 N6)
        rows.append(
            Row(
                item.ordinal,
                item.origin,
                text,
                f"item://{spec.name}/{item.origin}",
                p,
                image_hash,
                item.label or None,
                _item_input_hash(text, image_hash, item.label or None, spec.chunker_version),
            )
        )
    return rows


def load_rows(spec: CorpusSpec, sample: int | None = None, seed: int = 42) -> list[Row]:
    kind = spec.source.kind
    if kind == "sqlite":
        rows = _load_sqlite(spec)
    elif kind in ("csv", "jsonl"):
        rows = _load_csv_jsonl(spec)
    elif kind == "image_dir":
        rows = _load_image_dir(spec)
    else:
        raise SpecError(f"source.kind: unknown '{kind}'")
    if sample and sample < len(rows):
        step = len(rows) / sample  # evenly spaced: deterministic, seed-independent
        rows = [rows[int(i * step)] for i in range(sample)]
        for ordinal, row in enumerate(rows):
            row.ordinal = ordinal
    return rows
