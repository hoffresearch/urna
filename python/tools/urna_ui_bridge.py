"""JSON bridge between the urna UI and the engine: browse + search in every
mode, with hits enriched (label, media frame) so the browser renders cards,
never terminal text.

usage:
  urna_ui_bridge.py FILE browse --offset N --limit M
  urna_ui_bridge.py FILE search --query Q --k K [--space NAME]
                    [--mode exact|ann|hybrid|graph] [--hops N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import _urna  # noqa: E402
from forge.forge_manifest import frame_resolver, manifest_items  # noqa: E402


def load_manifest(index: Path) -> dict:
    for cand in (
        index.with_suffix(".manifest.json"),
        index.parent / ".build" / f"{index.stem}.manifest.json",
    ):
        if cand.is_file():
            return json.loads(cand.read_text())
    return {}


def media_resolver(db, manifest: dict):
    """global frame -> (internal blob uri, local frame). The manifest's uri
    strings may be renamed; blobs are matched by sha256, never by name."""
    refs = db.blob_refs()
    segs = (manifest.get("media") or {}).get("segments") or []
    by_hash = {r["content_hash"]: r["original_uri"] for r in refs}
    table = []
    for i, s in enumerate(segs):
        uri = by_hash.get(s.get("media_sha256"))
        if uri is None and i < len(refs):
            uri = refs[i]["original_uri"]
        table.append((s.get("start_frame", 0), s.get("n_frames", 0), uri))
    n_frames = (manifest.get("media") or {}).get("frame_count", 0)
    per_image = len(refs) == n_frames and n_frames > 0

    def resolve(frame: int) -> tuple[str, int]:
        if per_image:
            return refs[frame]["original_uri"], 0
        for start, n, uri in table:
            if uri and start <= frame < start + n:
                return uri, frame - start
        if len(refs) == 1:
            return refs[0]["original_uri"], frame
        return "", frame

    return resolve


def cmd_browse(db, manifest, args) -> dict:
    # a compact manifest (provenance minimal) has no label: the title falls
    # back to the key, and the frame is derived from the ordinal.
    items = manifest_items(manifest)
    resolve = media_resolver(db, manifest)
    item_frame = frame_resolver(manifest)
    ids = db.chunk_ids()
    page = []
    for it in items[args.offset : args.offset + args.limit]:
        uri, frame = resolve(item_frame(it))
        page.append(
            {
                "ordinal": it["ordinal"],
                "title": it.get("label") or it.get("key") or str(it["ordinal"]),
                "uri": uri,
                "frame": frame,
                "chunk_id": ids[it["ordinal"]] if it["ordinal"] < len(ids) else "",
            }
        )
    return {"total": len(items), "offset": args.offset, "items": page}


def embed_query(preset_name: str, dim: int, query: str):
    import os

    from forge import model_registry

    allowed = frozenset(
        p.strip() for p in os.environ.get("URNA_ALLOW_REMOTE_CODE", "").split(",") if p.strip()
    )
    adapter = model_registry.create_embedder(
        preset_name, allow_remote_code=allowed, allow_heavy=True, batch_size=4
    )
    vec = adapter.embed_texts([query], role="query")
    if dim:
        vec = model_registry.slice_renorm(vec, dim)
    return vec[0].tolist()


def cmd_search(db, manifest, args) -> dict:
    items = manifest_items(manifest)
    by_ordinal = {it["ordinal"]: it for it in items}
    ids = db.chunk_ids()
    ord_of = {cid: i for i, cid in enumerate(ids)}

    if args.space:
        preset = args.space.split("@")[0].removesuffix("-text")
        dim = int(args.space.split("@")[1]) if "@" in args.space else 0
        vec = embed_query(preset, dim, args.query)
        hits = db.search_space(args.space, vec, args.k)
    else:
        vec = embed_query("potion", 0, args.query)
        if args.mode == "ann":
            hits = db.search_ann(vec, args.k, 100)
        elif args.mode == "hybrid":
            hits = db.search_hybrid(vec, args.query, args.k, 100)
        elif args.mode == "graph":
            hits = db.search_graph(vec, args.k, args.hops)
        else:
            hits = db.retrieve(vec, args.k)

    out = []
    for h in hits:
        ordinal = ord_of.get(h.chunk_id, -1)
        it = by_ordinal.get(ordinal, {})
        text = getattr(h, "text", "") or ""
        title = text.split("\n")[0] if text else (it.get("label") or h.chunk_id[:18])
        out.append(
            {
                "chunk_id": h.chunk_id,
                "citation_id": h.citation_id,
                "score": round(float(h.score), 4),
                "score_type": h.score_type,
                "title": title,
                "text": text,
                "uri": h.source_uri,
                "frame": h.offset_start,
                "ordinal": ordinal,
            }
        )
    return {"mode": args.space or args.mode, "hits": out}


def cmd_blob(db, args) -> dict:
    """Write ONE inlined blob to --out, addressed by its media:// uri.
    Single-asset extraction: a 3.6 GB store never leaves the file just
    to render one thumbnail."""
    want = args.uri
    for i, r in enumerate(db.blob_refs()):
        if r["original_uri"] == want:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_bytes(db.blob_bytes(i))
            tmp.replace(out)
            return {"written": str(out), "bytes": out.stat().st_size, "index": i}
    raise SystemExit(f"error: no blob with uri '{want}'")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file")
    sub = p.add_subparsers(dest="op", required=True)
    b = sub.add_parser("browse")
    b.add_argument("--offset", type=int, default=0)
    b.add_argument("--limit", type=int, default=60)
    s = sub.add_parser("search")
    s.add_argument("--query", required=True)
    s.add_argument("--k", type=int, default=24)
    s.add_argument("--space", default="")
    s.add_argument("--mode", default="exact", choices=["exact", "ann", "hybrid", "graph"])
    s.add_argument("--hops", type=int, default=1)
    bl = sub.add_parser("blob")
    bl.add_argument("--uri", required=True)
    bl.add_argument("--out", required=True)
    args = p.parse_args()

    index = Path(args.file)
    db = _urna.UrnaFile.open(str(index))
    if args.op == "blob":
        result = cmd_blob(db, args)
    else:
        manifest = load_manifest(index)
        result = (
            cmd_browse(db, manifest, args)
            if args.op == "browse"
            else cmd_search(db, manifest, args)
        )
    json.dump(result, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
