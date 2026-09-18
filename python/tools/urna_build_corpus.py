"""urna_build_corpus.py — build a deterministic .urna from the seven
PT-BR fake-news datasets under `dat/demo/`.

Steps:
  1. read each source (loaders live in `_corpus_sources.py`)
  2. normalize to (text, label, source, title, url)
  3. filter empty/short text (len > 20)
  4. dedupe by sha256(text), keep first
  5. embed each row with the model declared in EMBED_MODEL
  6. call builder.Pipeline (existing tool) to chunk + cache + emit
    7. write dat/<name>.urna
  8. shell out to `urna validate`
  9. print report: original counts vs post-dedup vs in .urna

Run:
  python3 python/tools/urna_build_corpus.py
    python3 python/tools/urna_build_corpus.py --out dat/foo.urna
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Force HF/sentence-transformers OFFLINE by default before the (lazy) ST
# import at embed time. A corpus build touches source text; it must not make
# an unexpected outbound fetch. Opt into a first-time download with
# URNA_ALLOW_DOWNLOAD=1 (then re-run offline for reproducibility).
if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_k, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "python"))

import pandas as pd  # noqa: E402
from _corpus_sources import MIN_TEXT_LEN, SOURCES  # noqa: E402
from builder import BuildConfig, ChunkSpec, Pipeline  # noqa: E402

DATA = REPO / "dat"
DB = REPO / "dat" / "demo"
URNA_BIN = REPO / "target" / "release" / "urna"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384


def _filter_and_dedupe(combined: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop short rows; dedupe by sha256(text) keeping first occurrence.
    Returns (deduped_df, n_before, n_after_filter)."""
    n_before = len(combined)
    combined = combined[combined["text"].str.len() > MIN_TEXT_LEN].copy()
    n_after_filter = len(combined)
    combined["_h"] = combined["text"].apply(
        lambda t: hashlib.sha256(t.encode("utf-8", errors="replace")).hexdigest()
    )
    combined = combined.drop_duplicates(subset="_h", keep="first").drop(columns="_h")
    return combined, n_before, n_after_filter


def _resolve_model_hash():
    """Compute the corpus' `model_hash` from the local sentence-transformers
    snapshot. Refuses to write the legacy placeholder."""
    from model_fingerprint import (
        compute_model_fingerprint,
        fingerprint_to_model_hash,
        resolve_model_dir,
    )

    model_dir = resolve_model_dir(EMBED_MODEL)
    fp = compute_model_fingerprint(model_dir, model_id=EMBED_MODEL)
    return fingerprint_to_model_hash(fp), fp.to_dict()


def _build_specs(combined: pd.DataFrame):
    """Convert the deduped frame into (specs, sources, labels, titles, urls)
    parallel arrays. `source_uri` follows `corpus-next://<source>/<hash16>`."""
    specs, sources, labels, titles, urls = [], [], [], [], []
    for _, r in combined.iterrows():
        ub = r["text"].encode("utf-8")
        h = hashlib.sha256(ub).hexdigest()[:16]
        specs.append(
            ChunkSpec(
                canonical_text=r["text"],
                source_uri=f"corpus-next://{r['source']}/{h}",
                byte_start=0,
                byte_end=len(ub),
            )
        )
        sources.append(r["source"])
        labels.append(r["label"])
        titles.append(r.get("title", "") or "")
        urls.append(r.get("url", "") or "")
    return specs, sources, labels, titles, urls


def _print_source_report(raw: dict, sources: list[str], labels: list[str]) -> None:
    in_urna = Counter(sources)
    print("\n" + "=" * 90)
    print(f"{'source':50s} {'original':>10s} {'in .urna':>10s} {'∆':>10s}")
    print("=" * 90)
    total_orig = total_urna = 0
    for name, _ in SOURCES:
        orig = len(raw[name])
        urna_n = in_urna.get(name, 0)
        delta = urna_n - orig
        total_orig += orig
        total_urna += urna_n
        print(f"{name:50s} {orig:>10,} {urna_n:>10,} {delta:>+10,}")
    print("-" * 90)
    print(f"{'TOTAL':50s} {total_orig:>10,} {total_urna:>10,} {total_urna - total_orig:>+10,}")
    print(f"\nlabels in .urna: {dict(Counter(labels))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(DATA / "corpus_next.v1.urna"),
        help="output .urna path (default: dat/corpus_next.v1.urna)",
    )
    ap.add_argument(
        "--cache",
        default=str(DB / "corpus-next" / "embed_cache.sqlite"),
        help="sqlite scratch path for embedding cache",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)

    print("→ reading sources")
    raw = {}
    for name, loader in SOURCES:
        df = loader()
        raw[name] = df
        print(f"   {name:50s} {len(df):>7,} rows")

    combined = pd.concat([raw[n] for n, _ in SOURCES], ignore_index=True)
    combined, n_before, n_after_filter = _filter_and_dedupe(combined)
    n_after_dedup = len(combined)
    print(
        f"   raw rows={n_before:,}  after-filter={n_after_filter:,}  after-dedup={n_after_dedup:,}"
    )

    print(f"\n→ loading {EMBED_MODEL}")
    from sentence_transformers import SentenceTransformer  # local import

    model = SentenceTransformer(EMBED_MODEL)
    print(f"   device: {model.device}")

    def embedder(specs):
        embs = model.encode(
            [s.canonical_text for s in specs],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return [e.astype("float32").tolist() for e in embs]

    specs, sources, labels, titles, urls = _build_specs(combined)

    real_model_hash, fp_dict = _resolve_model_hash()
    print(f"model fingerprint: {fp_dict}")
    print(f"model_hash: {real_model_hash}")

    cfg = BuildConfig(
        output_path=str(out_path),
        embedding_model=EMBED_MODEL,
        embedding_dim=EMBED_DIM,
        chunker_version="corpus-next/v0.1.0",
        model_hash=real_model_hash,
        title="corpus-next",
        version="v0.1.0",
        description=(
            "Unified PT-BR fake-news corpus combining 7 public datasets, "
            "deduplicated by SHA-256 of text."
        ),
        license="mixed (per-source)",
        reproducible=True,
    )

    pipe = Pipeline(cfg, embedder=embedder, scratch_db=args.cache)
    pipe.add_many(specs)
    pipe.emit(
        provenance={
            "legacy_source": "corpus-next (unified PT-BR fake-news corpus)",
            "legacy_version": "v0.1.0",
            "datasets": [n for n, _ in SOURCES],
            "labels": labels,
            "sources": sources,
            "titles": titles,
            "urls": urls,
            "note": "row index matches the post-dedup corpus",
        }
    )
    pipe.close()
    size = out_path.stat().st_size
    print(f"\n✓ .urna → {out_path}  ({size / 1e6:.2f} MB)")

    print("\n→ urna validate")
    if URNA_BIN.exists():
        subprocess.run([str(URNA_BIN), "validate", str(out_path)], check=True)
    else:
        print(f"   (skipped: {URNA_BIN} not found — run `cargo build --release -p urna-cli`)")

    _print_source_report(raw, sources, labels)


if __name__ == "__main__":
    main()
