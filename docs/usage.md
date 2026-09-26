# usage

`urna` is a single-file binary container for distributing semantic knowledge bases. one file: chunks, canonical text, byte-spans, embeddings, search contract, hashes. copy it, share it, search it.

this guide covers the commands you'll actually use: the agent verbs `ask`, `retrieve` and `build` (the front door; they shell out to the offline python embedder or the forge), and the engine subcommands beneath them (validate, stats, inspect, media, search/search-ann/search-graph/search-space/search-text, benchmark, cite, doctor), which take a file and a vector and never run python. `urna --help` lists them in the same two groups. getting the binary onto a machine (every install channel, verification, offline notes, the maintainer checklist) is the reference section at the end of this document; the short form is `curl -sSf https://raw.githubusercontent.com/hoffresearch/urna/main/scripts/install.sh | sh` followed by `urna doctor`.

## quickstart

five verbs cover the whole loop. what each one is for:

| verb | what it does | in one line |
|------|--------------|-------------|
| `build` | creates the base | rows + embedding model in, one `.urna` out |
| `ask` | queries it from the terminal | text in, one cited answer out |
| `retrieve` | hands results to another program | json/jsonl of cited spans, `score` is the exact rerank |
| `cite` | resolves the source | a `urna://` citation back to the stored text and its hashes |
| `validate` | proves the file | every checksum, every hash, the manifest contract |

`examples/quickstart/` ships a corpus that needs nothing downloaded: twelve paragraphs of cc0 prose about urna (`docs.jsonl`) and the smallest spec that builds them (`corpus.toml`, one jsonl source, the bundled potion model). from the repo root:

```sh
urna build --spec examples/quickstart/corpus.toml
urna ask examples/quickstart/out/quickstart.urna "can I use this offline" -k 1
urna retrieve examples/quickstart/out/quickstart.urna "how do citations work" -k 2 --format jsonl
urna cite examples/quickstart/out/quickstart.urna 'urna://<content_hash>/<chunk_id>'   # a citation_id from ask or retrieve
urna validate examples/quickstart/out/quickstart.urna
```

`build` runs the forge in `python/`, so it needs the repo checkout (the installed payload carries the query embedder only); the other four verbs work with the installed binary alone. the python version of the same loop, with the embedder in plain sight, is `python examples/quickstart/quickstart.py`. to build your own corpus, swap `docs.jsonl` for your rows (jsonl, csv, sqlite, an image dir) in the spec: §13 has the full contract.

## 1. build a `.urna` from chunks

the python pipeline owns chunking, embedding, caching, and the final emit. the rust writer owns reproducibility, hashing, and deterministic byte layout. the one thing every build needs that is easy to miss: the embedder's `model_hash`, written into the file so a query embedded by another model fails loudly instead of returning plausible wrong hits. the bundled potion embedder carries its own (`emb.model_hash()`); for a sentence-transformers model, §7 has the fingerprint.

```python
import sys

sys.path.insert(0, "python")
from builder import BuildConfig, Pipeline, chunk_text
from forge.embed_potion import potion_embedder

emb = potion_embedder()  # offline static table; swap for python/embed_query.py's ST path if you need a bigger model

cfg = BuildConfig(
    output_path="my_corpus.urna",
    embedding_model=emb.embedding_model,
    embedding_dim=emb.embedding_dim,
    chunker_version="my-chunker/v1",
    model_hash=emb.model_hash(),  # a zero placeholder is rejected at write time
    preset="exact",  # see §6 for preset choices
    reproducible=True,
)
pipe = Pipeline(cfg, embedder=emb, scratch_db="cache.sqlite")
for source_uri, text in documents:  # your (uri, text) pairs
    for spec in chunk_text(text, source_uri):
        pipe.add(spec)
pipe.emit()
```

the embedder is any callable that takes the chunk specs and returns one l2-normalized vector per spec; `potion_embedder()` is one, and a sentence-transformers wrapper is a few lines (`m.encode([s.canonical_text for s in specs], normalize_embeddings=True).tolist()`). for real-world examples: `python/convert_legacy.py` (SQLite to `.urna`), `python/tools/urna_build_corpus.py` (7 PT-BR datasets to a unified `.urna`), and `examples/quickstart/quickstart.py` (the shortest complete build, on `urna.build` directly).

### image and pdf corpora

image corpora live in the forge tooling layer because a vision tower needs torch, which the sovereign runtime does not take. the `.urna` they emit is an ordinary `.urna`, served by the same rust runtime from mmap.

the media travels inside the file. the encoded stream is stored as content-addressed blobs (section 0x14), each chunk carries the exact byte span it was embedded from (overlay 0x16), and the image vectors sit in their own named space (registry 0x15, slab in the 0x20-0x2F band) behind the `supports_multimodal` capability, gated by their own `model_hash` in isolation. these sections are excluded from `content_hash`, so adding media never moves an existing citation.

`python/tools/urna_build_image_corpus.py` letterboxes every image onto one canvas, encodes the sequence, embeds the DECODED frames, and writes one chunk per image or pdf page. embedding the decoded frames rather than the source pixels is deliberate: the index has to describe what a reader can actually get back.

```sh
.venv/bin/python python/tools/urna_build_image_corpus.py \
    --input-dir /path/to/dermoscopy_images \
    --dataset my-derm \
    --output corpora/my-derm.urna \
    --labels labels.csv
```

a corpus is one file. `corpora/my-derm.urna` carries the index, the media blobs, the span overlay, and the space registry; copying it moves the corpus intact. provenance (ordinals, origins, labels, media digests) rides inside as well.

`--width` is a ceiling, not a target: the canvas is clamped to the dataset's median source width, so a corpus is never upscaled. lower it to trade quality for size; raising it above the source does nothing but make the encoder pay for interpolated pixels.

`--gop-policy auto` (the default) probes a spaced sample of frames and lets the bytes decide between all-intra and inter coding. on every corpus measured so far (ph2, ham10000, wsi tiles, scanned pdf) the probe chose intra: unrelated images give inter-frame prediction nothing to find. inter stays available for genuinely sequential media. `--all-intra` forces every frame a keyframe and overrides the probe.

`--shard-size N` splits the stream into consecutive segments of about N frames, one blob per shard, which caps decode memory and improves cold seek on large corpora. `--order-similarity` tries a greedy nearest-neighbour frame order before encoding; measured on 1210 wsi tiles it cost 0.15 percent instead of helping, so it stays off by default.

`--backend av1` (the default) won the size-matched matrix; `--backend avif` writes one avif per image and is the only backend that accepts `--pix-fmt yuv444p`. `--crf` (default 35) sets the av1 rate, `--avif-quality` (default 35) the avif one. `--control` builds the letterbox-lossless png control corpus that codec cost is measured against. for every backend the manifest's `media.source_bytes` is the byte size of the original source files and `compression_ratio` is that over `output_bytes`; the avif path records the letterboxed pngs it actually feeds avifenc apart as `letterboxed_input_bytes`, so the ratio is comparable across av1, avif and jxl.

`--dtype float32|float16|int8|int4` overrides the preset's vector dtype for the image space (int4 needs the dim divisible by 64). measured: quantization was not the driver of quality loss (the melanoma delta is identical at f16 and int8, and similar at int4), while the vectors themselves shrink 214 KB to 112.6 KB to 63.8 KB on ph2.

add `--pdf` to render pdf pages as the images; page numbers are kept in the manifest and in the citable text. for a non-dermatology domain pass `--model ViT-B-32 --pretrained openai`. the pretrained tag is required for bare architecture names, because open_clip answers a missing tag with random weights.

search with a query image or a clinical description, and optionally decode the matched frames back out:

```sh
.venv/bin/python python/tools/urna_search_image.py \
    --index corpora/my-derm.urna --query-image lesion.jpg -k 10 \
    --letterbox-query --save-frames hits/
```

`--query-text "..."` searches with a clinical description instead of an image, and `--letterbox-query` normalizes the query onto the corpus canvas before embedding. queries route through `search_space`, so the image space's own `model_hash` is checked against the manifest, in isolation from the default text space, before anything is scored. `--skip-model-check` bypasses that gate explicitly.

### measuring an image corpus

`python/tools/urna_image_eval.py` reports two rulers and keeps them apart, because they answer different questions:

- `identity` asks whether a source image retrieves its own frame. it measures rank stability under the codec and is inflated by construction, since the corpus contains the answer. on an uncompressed index it returns 1.000 by definition.
- `label` removes the query's own frame and scores how many of the remaining neighbours share its label. nothing in the corpus is the answer, so this is the one that reports retrieval quality. it is printed next to the random-pick baseline for the same label distribution, without which the number cannot be read.

neither means much alone. pass `--baseline` with the uncompressed control index (`--control` at build time) to get the delta, which is what the codec actually cost:

```sh
.venv/bin/python python/tools/urna_image_eval.py \
    --index corpora/my-derm.urna \
    --baseline corpora/my-derm-control.urna \
    -k 1 5 10 --out eval.json
```

measured in phase 6 (full matrix and intervals in `docs/CHANGELOG`): on ph2 (n=200) av1-intra crf35 compresses the media 86x for a mean label `precision@10` delta of -3.4 to -4.7 points whose interval crosses zero, but the melanoma class alone drops 16.9 points with a significant interval ([-25, -10]); on ham10000 (2000-sample) the media shrinks 151x for a mean delta of -1.5 [-3.5, +0.6], again with a significant melanoma cost (-10.7). the text-to-image ruler is harsher and honest: 44/60 correct top-10 clinical queries on the control falls to 22/60 at crf35, and the loss does not recover with rate. per-class floors matter more than the mean: report the interval and the worst class, not just the point.

`python/tools/urna_image_sweep.py` runs the variant matrix for you (av1-intra crf ladder, avif444, control, `dtype:` rungs, `av1-order`), records `urna_bytes` and the control's `media_bytes` per variant, and writes one consolidated comparison json:

```sh
.venv/bin/python python/tools/urna_image_sweep.py \
    --input-dir /path/to/images --dataset my-derm \
    --variants av1-intra-crf35,av1-intra-crf40,control,dtype:int8 \
    --labels labels.csv --out-dir sweep/ --out sweep/summary.json
```

direct API (no chunker): `urna.build(output_path, embedding_model, embedding_dim, chunker_version, model_hash, chunks, preset="exact", reproducible=True)`.

## 2. validate

full integrity check: magic, header checksum, every section's SHA-256 (over physical bytes), footer hash (over the whole file), manifest schema, contract cross-check against the manifest, NaN/Inf walk over the embeddings.

```sh
urna validate my_corpus.urna
```

failure modes are typed (`SectionChecksumMismatch(0x04)`, `UnsupportedDType("bfloat16")`, etc.), never "best effort".

## 3. stats

sizes, dim, dtype, model, hashes, per-section bytes, the SIMD backend the runtime selected.

```sh
urna stats my_corpus.urna
```

## 4. inspect

header bytes, full section table, manifest as JSON. use `--json` for programmatic consumers (CI dashboards, drift detection):

```sh
urna inspect my_corpus.urna             # human-readable
urna inspect my_corpus.urna --json | jq # structured
```

schema: `{magic, version_major, version_minor, format_version, schema_version, embedding_dim, n_chunks, n_embeddings, file_size, manifest, sections[], file_hash, content_hash, simd_backend}`.

## 5. search

### exact path (vector input)

pass a query vector directly as a JSON array. recall = 1.0 by construction.

```sh
urna search my_corpus.urna "[0.1, 0.2, ...]" -k 10
```

### search by text

embed the query with the same model the corpus was built with (the manifest declares it), then route to the declared `index_type` (exact, hnsw, hybrid). the runtime cross-checks the embedder's `model_hash` against the manifest before running search and refuses on mismatch. see §7.

```sh
urna search-text my_corpus.urna "vacina contra covid funciona" -k 5
```

for tuning the candidate set: `--candidates N` (default `4*k`, min 64).

### force the ANN path

useful for debugging or measuring `ef_search` curves. falls back to exact if the file has no HNSW section.

```sh
urna search-ann my_corpus.urna "[0.1, 0.2, ...]" -k 10 --ef 200
```

### graph search (chunk-to-chunk)

seeds from the exact-cosine top-`ef`, expands a bounded breadth-first walk over the chunk-to-chunk graph (`--hops`), then exact-reranks the union. the graph only generates candidates; the returned score is real cosine (recall is not computed, the rerank guarantees the score). falls back to exact if the file has no `graph_adjacency` (0x0C) section. build a graph-carrying file with `urna.build(..., with_graph=True)` (default off); the section is additive and excluded from content_hash, so adding a graph never changes a citation.

```sh
urna search-graph my_corpus.urna "[0.1, 0.2, ...]" -k 10 --hops 2 --ef 100
```

### search a named space (multimodal)

`search-space` runs the per-space exact search over one named vector band (0x15 + 0x20+): image spaces, extra text spaces, mrl-sliced spaces. the query vector must be embedded with the space's model at the space's dim; an unknown space, a wrong dim, or (with `--expect-model-hash`) a wrong model are typed errors; never a silent fallback to the text path. the space names come from `urna stats` (the `spaces:` block) or `inspect --json` (the `spaces[]` array).

```sh
urna search-space my_corpus.urna "[0.1, ...]" --space "wemm-2b@256" -k 5
urna benchmark my_corpus.urna -q 100 -k 10 --space "wemm-2b@256"
```

### the flagship: ask and retrieve

`ask` and `retrieve` are the agent-native front door: text query in, cited answer out, no flags needed. they embed the query OFFLINE and route the embedder BY THE MANIFEST MODEL: a potion corpus keeps the potion static table (`python/forge/embed_query_potion.py`, the unchanged fast path), and a corpus whose default text space is any registry model (wemm, jina, clip; see §12) goes through `python/forge/embed_query_model.py`, which encodes the query with that model's query route and, for an mrl-truncated default space, slices + renormalizes to the manifest dim (`--mrl-dim`, passed automatically). both paths validate the embedder's `model_hash` against the manifest exactly like `search-text`, and route by manifest capability (exact if only embeddings, hnsw/hybrid/graph as the file advertises). every printed score IS the exact-cosine rerank value.

`ask` prints one low-cognitive-load cited answer:

```sh
urna ask my_corpus.urna "can I use this offline" -k 3
```

`--disclose answer` (default) prints the cited canonical text and a `urna://` citation, nothing else. `--disclose explain` ALSO prints the rerank-source honesty line: `real cosine` when the score is full precision, `real cosine at stored precision` for a lossy stored slab (float16/int8/int4) with no full-precision source, plus the route and per-path candidate counts.

`retrieve` is the agent-shaped surface: a json/jsonl answer-pack of cited spans.

```sh
urna retrieve my_corpus.urna "can I use this offline" -k 5 --format jsonl
```

each hit is `{chunk_id, score, score_type=cosine, source_uri, offset_start, offset_end, citation_id, text, file_hash, content_hash, rerank_source}`. the `score` is the exact rerank value (never a candidate-generator proxy), `text` is the tier-1 stored canonical text, and `citation_id` round-trips through `urna cite`. `--format json` emits a single pretty array instead of one object per line.

the embedder picks its interpreter in a fixed order: `URNA_PYTHON` if set, else the repo's `.venv/bin/python` (which carries the forge deps: numpy + tokenizers + the vendored potion table) discovered by walking up from the cwd, else `python3` on PATH. so the repo `.venv` is used automatically; set `URNA_PYTHON` only to force a specific interpreter. the selected interpreter is printed to stderr; and since discovery executes the nearest ancestor `.venv/bin/python`, set `URNA_PYTHON` explicitly if you run `urna` from inside an untrusted directory tree. point `--model-path` at a copied potion table dir for a fully sealed offline run.

the python convenience is `python python/forge/retrieve.py`: it builds a `.urna` from the cc0 demo corpus with the potion embedder, asks a question, and prints the cited answer with a `urna://` citation, all offline and deterministic (the one-gif demo).

## 6. presets

`preset=` selects a (text encoding, embedding dtype, optional ANN, optional BM25) bundle. per-knob overrides win, see `BuildConfig.text_encoding`, `.dtype`, `.with_hnsw`, `.with_bm25`, `.mrl_dim`.

| preset       | text encoding | embeddings  | ANN | BM25 | size_ratio | recall@10 |
|--------------|---------------|-------------|-----|------|-----------:|----------:|
| `exact`      | raw           | float32     | no  | no   |      1.000 |    1.0000 |
| `compressed` | zstd          | float16     | no  | no   |      0.339 |    1.0000 |
| `tiny`       | zstd          | int8        | yes | no   |      0.256 |    0.9920 |
| `micro`      | zstd          | mrl256-int8 | yes | no   |      0.223 |    0.8100 |
| `nano`       | zstd          | int4        | yes | no   |      0.209 |    0.9130 |
| `hybrid`     | zstd          | float32     | yes | yes  |      0.609 |    1.0000 |

numbers measured on the project's PT-BR fake-news corpus (n=30,725, dim=384), 100 queries, k=10 vs the float32 exact baseline (the published ladder `data/measure/ladder.json`, gated against `data/measure/baseline.json`). RULER CAVEAT: these `recall@10` figures use a SELF-PERTURBATION ruler (each query is a corpus vector plus tiny noise), so they measure rank-stability under quantization, NOT real-query retrieval, and are likely inflated; see the `ruler` field in `ladder.json`/`baseline.json` and the pending real-query (mteb-style) ruler (gate-zero). these are the honest current sizes after the text-codec repack (intpack chunk_ids/spans, bitpacked hnsw/bm25 payloads) shrank the indexed presets below the v0.2 figures: `tiny` 0.283 -> 0.256, `compressed` 0.350 -> 0.339, `hybrid` 0.668 -> 0.609. latency ranges (NEON, hot cache): exact p50 ~3.1 ms, tiny p50 ~1.2 ms, micro p50 ~0.8 ms, nano p50 ~2.1 ms, hybrid p50 ~4.0 ms.

the `exact`/`compressed`/`tiny`/`nano`/`hybrid` rows are direct `preset=` values; `micro` is the published name for the matryoshka size lever (the documented honest point `mrl256-int8`), built with `urna.build(text_encoding="zstd", dtype="int8", mrl_dim=256, with_hnsw=True)` and emitted by `measure_presets.py --variants ...,micro,...`.

pick `nano` for the smallest distributable file with recall above the nano floor: int4 block-64 embeddings (per-64-dim-group f16 absmax scales + packed 4-bit codes) take the embeddings section from int8's 11.92 MB down to 6.27 MB (~1.9x over int8, ~7.5x over float32). `nano`/`micro` require the effective `embedding_dim` divisible by 64. every sub-int8 preset (`micro`/`nano` and the whole mrl curve) is STORED-PRECISION: the 0x09 `embeddings_fp` rerank source is not wired, so the net-of-fp ratio equals the stored ratio and `score`/`recall@10` are real cosine AT THE STORED PRECISION (int4/int8), disclosed via `dtype` (and `mrl_dim`/`full_dim` for `micro`) in `urna stats` and on every result, never a bare-slab ratio. `micro` trades recall for size on this non-mrl MiniLM baseline (0.810 recall@10 at 0.223 ratio, see the curve below); pick it only when raw size beats the last ~10 recall points or once a real mrl-trained model lands. pick `tiny` when you want a smaller file than `compressed` with recall still above 0.99, `compressed` when you need lossless cosine + 3x compression, `hybrid` when queries include rare terms, proper nouns, or siglas that pure embeddings underweight, and `exact` when storage isn't the bottleneck and you want the recall=1.0 ground truth.

### matryoshka prefix truncation (`mrl_dim`)

`urna.build(..., mrl_dim=K)` (or `BuildConfig.mrl_dim`) slices each l2-normalized vector to its first `K` components and re-l2-normalizes the prefix BEFORE quantization (Qwen3/ST/BGE truncate-then-renormalize). this is the dimension axis: orthogonal to and multiplicative with the dtype levers. the stored `embedding_dim` becomes `K`, the source dim is recorded as `full_dim`, and both appear in `urna stats`. queries are striped at `K` too, so a full-dim query against a truncated file is a dimension mismatch; slice + renorm the query to `K` first. truncation is a pure deterministic op, so builds stay byte-identical; `content_hash` is over the truncated embeddings, so a citation is tied to its `mrl_dim` (never claimed stable across dims). int4 still needs the effective dim divisible by 64, so `mrl_dim` in {256, 192, 128} works with int4 but 96 does not (use int8/f16/f32 at 96).

matryoshka pays off on a model trained for it (information front-loads into the prefix). the shipped MiniLM corpus is NOT mrl-trained, so truncation costs real recall@10 there; the published ladder in `data/measure/ladder.json` (100 queries, k=10) reports the honest curve (same self-perturbation ruler as above, see the RULER CAVEAT) and `python/tools/measure_presets.py` emits it (the default `--variants` are `compressed,tiny,micro,nano,hybrid` plus `mrl256/192/128-int8`, `mrl96-int8`, `mrl256/192/128-int4`):

| ladder        | size ratio | recall@10 |
|---------------|-----------:|----------:|
| `mrl256-int8` (`micro`) |     0.223  |   0.810   |
| `mrl192-int8` |     0.207  |   0.733   |
| `mrl128-int8` |     0.190  |   0.659   |
| `mrl96-int8`  |     0.182  |   0.574   |
| `mrl256-int4` |     0.191  |   0.777   |
| `mrl192-int4` |     0.183  |   0.713   |
| `mrl128-int4` |     0.174  |   0.627   |

on that baseline `nano` (full-dim int4) still beats every truncated point on recall, so reach for `mrl_dim` (the `micro` rung is `mrl256-int8`) when raw size matters more than the last ~10 recall points, or once an mrl-trained embedder is in play. `compare_measure.py` gates `micro`/`nano`/`mrl256-int8` conditionally (size_ratio <= 0.25; recall >= 0.78 for micro/mrl256-int8, >= 0.85 for nano), only when the run includes them.

## 7. model_hash and offline operation (`--model-path`)

`search-text` cross-checks three things before running search:

1. `manifest.embedding_model` (name) matches the embedder's report.
2. `manifest.embedding_dim` matches `len(vector)`.
3. `manifest.model_hash` matches the embedder's reproducible fingerprint.

layer 3 is the only one that catches the silent failure mode "same name + same dim + different snapshot, cosine-valid garbage". the fingerprint hashes a fixed list of inference-relevant files (`config.json`, `tokenizer.json`, `model.safetensors`, `1_Pooling/config.json`, etc.). see `python/model_fingerprint.py`.

build with a real fingerprint:

```python
from model_fingerprint import (
    compute_model_fingerprint,
    fingerprint_to_model_hash,
    resolve_model_dir,
)

md = resolve_model_dir("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
fp = compute_model_fingerprint(
    md, model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
cfg.model_hash = fingerprint_to_model_hash(fp)
```

### fully offline search

distribute the model directory alongside the `.urna` (e.g. on a USB stick or in a sealed Docker image), then point `--model-path` at it on every search:

```sh
urna search-text my_corpus.urna "vacina contra covid" -k 5 \
    --model-path /mnt/models/paraphrase-multilingual-MiniLM-L12-v2
```

no HuggingFace cache hits, no network. the fingerprint is recomputed locally and verified against the manifest.

### pre-phase-3 corpora

files built with `model_hash = sha256:0...0` (the legacy placeholder) fail the strict gate by design. two options:

- rebuild with a real fingerprint (recommended).
- pass `--skip-model-hash-check` to proceed at your own risk. the search is still cosine-valid if you genuinely use the same embedding model, but there is no guarantee.

## 8. benchmark

random-query latency stats (mean, p50, p95, p99). with `--ann`, also runs ANN against the same queries and computes `recall@k (ANN vs exact)`. with `--madvise-cold`, runs an extra pass calling `posix_madvise(MADV_DONTNEED)` between queries: upper bound on cold-cache latency, not absolute cold (see `MmapUrnaFile::madvise_cold` docs).

```sh
urna benchmark my_corpus.urna -q 100 -k 10 --ann 100 --madvise-cold
```

typical output (n=30,725, dim=384, neon, int8):

```
Exact (100 queries, dim=384, dtype=int8, simd=neon) [hot]:
  p50: 1.28 ms  p95: 1.68 ms
Exact ... [madvise-cold]:
  p50: 1.95 ms  p95: 2.40 ms
ANN ef=100 (100 queries) [hot]:
  p50: 0.44 ms  p95: 0.62 ms
  recall@10 (ANN vs exact): 0.9920
```

recall@10 here is ANN-vs-exact rank-stability (the ANN index against the exact-cosine top-k on the same queries), NOT real-query retrieval quality, and the printed value mirrors the published tiny ladder number; see the RULER CAVEAT in section 6.

## 9. citations

every search hit carries a stable `citation_id` of the form `urna://<content_hash>/<chunk_id>`. resolve it back to the canonical text and original byte span:

```sh
urna cite my_corpus.urna 'urna://sha256:1aa9.../sha256:8f314...'
```

`content_hash` is hashed over the **decoded** bytes, so a corpus stored with `text_encoding=zstd` produces the same `content_hash` as the same logical content stored raw. citations are stable across wire encodings.

`cite` is tier-1: it returns the stored canonical text plus the verifying hashes (`file_hash`, `content_hash`) and the byte span. it does NOT reopen the original source bytes; original-byte reopen with a blob-digest verify is net-new tier-2 work that belongs to catalog mode, not the flagship. `ask` and `retrieve` print the same tier-1 stored canonical text, so the answer you get is exactly what `cite` resolves.

## 10. release verification

```sh
./scripts/release_check.sh
```

runs the full pipeline: cargo test, clippy, fmt, all 3 python test suites, ruff, `measure_presets.py`, `compare_measure.py` against the committed baseline. exits non-zero on any failure.

## 11. install health check (`urna doctor`)

`doctor` takes no file. it validates the install surface after a one-liner / tarball install (the channels are in the reference section below): urna and format versions, the detected simd backend, the python interpreter the embedder will run under, the numpy + tokenizers deps, the potion embedder script, the potion table (a git-lfs pointer is rejected), and one real offline embed of a fixed probe string.

```sh
urna doctor
```

the exit code is typed so installers and ci branch on codes, not text: `0` ok, `2` python interpreter missing, `3` python deps missing, `4` potion embedder script not found, `5` potion table missing or a git-lfs pointer, `6` embedder run failed. a scalar simd fallback prints a warning but still exits `0`. the embedder check opens no socket, so doctor itself stays offline-by-construction.

the embedder script resolves in this order: the repo layout (`python/forge/embed_query_potion.py`, dev checkout), then `${XDG_DATA_HOME:-~/.local/share}/urna/forge/` (one-liner installs), then `<exe>/../share/urna/forge/` (tarball and homebrew-style layouts).

## 12. model registry and multi-model spaces

embedding models are DATA, not per-project code: `python/forge/model_registry.py` holds named presets, each declaring what the model is (ids, dim, the VALIDATED matryoshka ladder), what it needs (deps with the exact pip fix line), and how it is used by default (the asymmetric query/document contract). the build spec (§13) selects one or several presets per build.

| preset | kind | dim | mrl dims | modalities |
|---|---|---|---|---|
| `potion` | static table (offline, no torch) | 256 |; | text |
| `clip-vit-b32` | open_clip ViT-B-32/openai | 512 |; | text, image |
| `siglip2` | open_clip ViT-B-16-SigLIP2/webli | 768 |; | text, image |
| `jina-v5-omni-nano` / `-small` | sentence-transformers | 768 / 1024 | 32–768 / 32–1024 | text, image, video |
| `wemm-2b` | sentence-transformers | 2048 | 128–2048 | text, image, video |
| `wemm-4b` / `wemm-9b` | sentence-transformers | 2560 / 4096 | idem | registered; `--allow-heavy` required |

three rules the registry enforces, loudly:

- **mrl is a ladder, not a slider.** `dims=[256]` is accepted only when the preset's model card validates 256 (`mrl.method="prefix_slice_l2"`). slicing at an unvalidated dim is refused; mathematically possible is not semantically supported.
- **remote code is an opt-in plus a pin.** presets with `trust_remote_code` load only when the spec lists them in `output.allow_remote_code` AND every model-repo code file matches the pinned sha256 allowlist. a hash identifies a version; the opt-in is the consent. build in an isolated environment when the model dir is not fully trusted. the QUERY side has the same rule: a manifest is data, never an authorization, so `ask`/`retrieve`/`search-text` over a remote-code corpus (and `urna_model_bench.py`) refuse to load the model until the operator opts in with `URNA_ALLOW_REMOTE_CODE="<preset>[,<preset>]"` in the environment.
- **three hashes, never conflated.** `model_hash` identifies the model (weights + tokenizer + processor + remote code + pooling/normalize/dtype policy). the per-item `input_hash` identifies the content (canonical text ⊕ image bytes ⊕ label ⊕ chunker). the `embedding_recipe_hash` identifies the usage (prompts, query/document modes, preprocess version, `image_max_side`, device class, decoder fingerprint when embedding decoded media). the embed cache key is the triad, so a retranslated text or a re-exported image invalidates exactly what changed.

known limitation: the siglip2 TEXT tower resolves its hf tokenizer through transformers' AutoTokenizer, which probes optional files that 404 online; a fresh process in strict offline mode can fail that probe even with the snapshot cached. the image tower and every other preset are unaffected; for a sealed offline run either query siglip2 spaces by image, or use the wemm/jina text towers.

model dirs resolve explicit `model_path` > `URNA_MODEL_DIR_<PRESET>` env > the preset's `local_dir` > the hf cache; a hub download requires `URNA_ALLOW_DOWNLOAD=1` explicitly. dtype defaults are measured, not assumed: bf16 on cuda, fp16 on mps (wemm-2b image embeds 0.5s vs 23s in fp32 on this class of machine), fp32 on cpu; override with `dtype=` in the spec or `URNA_ST_DTYPE`.

## 13. declarative corpus builds (`urna build --spec`)

one toml describes the whole corpus; nobody writes a build script per project. `urna build` is a launcher over `python/tools/urna_forge.py` (the build is officially a python frontend; torch and ffmpeg live there).

```sh
urna build --spec corpus.toml --dry-run     # plan + dep status, loads nothing
urna build --spec corpus.toml --sample 1500
```

a complete working spec: a sqlite table with per-row images, two models, media behind the dual quality gate, one self-contained output file:

```toml
[corpus]
name = "cards"
chunker_version = "cards/1"     # changes ⇒ every chunk_id (citation) changes

[source]
kind = "sqlite"
db = "${MTG_DATA}/mtg.sqlite"   # ${VAR} is strict: unset => SpecError naming the key
query = "SELECT id, name, body, image_uri FROM cards WHERE image_uri IS NOT NULL"
order_by = ["id"]               # must be a TOTAL order (verified)

[source.text]
template = """
{name}
{body}
"""

[source.image]
path_template = "${MTG_DATA}/images/{id}.jpg"
label_template = "{name}"

[media]
profile = "stills"              # one avif per image at q48; explicit keys still win
# profile = "stills-av1"        # the av1 stream instead, and the only pairing for
# crf = "auto"                  # the dual gate (it probes the av1 ladder)

[[models]]
preset = "potion"
text = "default"                # space 0 of every emitted file

[[models]]
preset = "clip-vit-b32"
image = "space"                 # a named vector band over the artwork

[output]
mode = "single"
dir = "out/cards"
embed_media = true              # media inlined via 0x17: ONE file serves it all
```

the contract highlights:

- **source**: `sqlite` (query, `[[source.joins]]`, `[source.derive]` helpers, text template whose lines drop when ALL their placeholders are empty, `path_template`/`label_template` for images) | `csv` | `jsonl` | `image_dir`. `order_by` must be a TOTAL order; the composite key's uniqueness is verified against the loaded rows, because `ORDER BY x` with duplicate x is not deterministic. path rule, applied to every string in the spec: `${VAR}` expands from the environment (only the braced form, so a bare `$` in `query` or `template` and the `{col}` placeholders survive), an unset or empty `${VAR}` is a SpecError naming the key (`source.db: ${MTG_DATA} is not set; export MTG_DATA=/path`, never a silent `$` or an empty root), then a leading `~/` expands to home (also when the variable itself holds `~/...`); anything else stays relative to the CWD. the build lock records the expanded strings, and the L3 compare ignores where the spec, the outputs and the inputs live (`spec_path`, `output.dir`, `source.path`, `source.db`, `source.image.path_template`; row content is guarded per item by `input_hash`), so `--rebuild-only` under another data root stays claimable.
- **models**: each `[[models]]` names a preset and its role; `text = "default" | "space" | "none"` (exactly ONE default; it is space 0 of every emitted file, never injected implicitly), `image = "space" | "none"`, `dims = [256, 512]` (one named space per dim: `wemm-2b@256`), `space_dtype`, plus the recipe fields (`image_prompt`, `text_query_mode`, `image_max_side`, `encode_kwargs`).
- **media** (§14 for the levers): `profile` (a measured recipe resolved into knob defaults; explicit keys always win, and an explicit `[media.quality]` key wins over the profile's quality table key by key), `backend`, `crf` (int or `"auto"`), `tune`, `speed`, `fps`, `gop`, `order`, `shard_size`, `dedup` (identical source images stored once; duplicate rows share the frame through the 0x16 overlay).
- **embedding.image_input**: `mode = "decoded_media"` (default with media: the index describes what the file serves) | `"source"` (measures the model, not the codec). the decoder fingerprint joins the recipe hash in decoded mode; the two modes answer different questions and are never mixed.
- **output**: `mode = "single" | "per-model" | "both"` (one media encode, one embed pass per model, shared across outputs; chunk_ids are content-addressed so citations agree across modes), `provenance = "minimal" | "standard" | "full"` (path/sql/label redaction. `standard` writes image paths relative to the spec dir and drops nothing else; `full` keeps absolute paths and the sql; `minimal` drops the sql and writes `items[]` compact: `key` + `ordinal` per item, `items_compact = true` at the top level, and a `frame_of_row` list only when dedup collapsed rows. readers get the media frame from `forge.forge_manifest.frame_resolver` and the items from `manifest_items(manifest, need=(...))`, which refuses a dropped field (`image_path`, `label`, `media_uri`) with an error naming `provenance = "standard"` instead of a KeyError; `urna_model_bench.py` needs `standard` or `full` for its source-image queries. motivation: 38627 items with all five fields were 12 MB of a 13 to 16 MB manifest. the mode never touches the `.urna` itself, the manifest is a sidecar), `allow_remote_code`, `embed_media = true|false` (inline the encoded media into the `.urna` itself, section 0x17, so the corpus is ONE self-contained file with no media sidecar at read time; `urna media <file>` lists the blobs, `urna media <file> --export DIR` writes them back out hash-verified, and `urna validate` proves every inlined blob against its `blob_refs` sha256. the sidecar `.media/` dir remains on disk as the build cache; peak build memory is roughly twice the media bytes, so prefer sidecar mode for very large corpora), `cache_dir` (root of the shared embed cache, see the transactional paragraph below; default `URNA_CACHE_DIR`, else `${XDG_CACHE_HOME:-~/.cache}/urna`).

every build emits `<name>.manifest.json` (`manifest_schema_version = 1`, canonical serialization; a versioned contract, not an ad-hoc log) and `<name>.build.lock.json` (package versions, tool binaries with sha256, model hashes, the materialized spec). reproduction has three declared levels: L1 = same top-k anywhere; L2 = per-vector cosine within 1e-5 on the same device class; L3 = byte-identical `file_hash`, claimable ONLY under a matching lock (`--rebuild-only` re-emits from the triad-keyed caches and compares the lock; `--strict-env` turns divergence into an error). builds are transactional: per-stage state under `<out>/.forge-state/`, outputs staged in `<out>/.tmp/` and committed by atomic rename (same filesystem, so both stay in the output dir), `--resume` continues from the last intact stage. embed caches live OUTSIDE the output dir, under `${XDG_CACHE_HOME:-~/.cache}/urna/embed/<preset>/<triad>.npz` (override with `[output] cache_dir` in the spec, `--cache-dir` on `urna build` or `urna_forge.py`, or `URNA_CACHE_DIR`; `--dry-run` prints the resolved root; the location is not identity, so it never enters the lock and any override source claims L3 against the same root): the file name is a hash of the triad plus the arrays the spec needs, so two specs with the same rows and model read one entry whatever their output dir, a changed knob adds a sibling entry instead of overwriting, and a text-only model's entry does not change with the media crf (the decoder fingerprint enters only image-space recipes). the `model_hash` probes sit beside them under `models/`, keyed by preset plus the knobs that enter the fingerprint (normalize, dtype, device, model_path), and the loaded model corrects a probe that disagrees. caches are flock'd with checksum sidecars, and a torn cache is recomputed, never reused. output dirs built before this layout keep an orphaned `<out>/.cache/` that nothing reads; delete it by hand. the `<name>.media/` sidecar is not a cache in sidecar mode: it is the served media, and it stays beside the `.urna`.

## 14. dataset compression levers and the dual quality gate

the media section is where the compression research became knobs. all decisions land in the manifest and provenance:

- `tune = "still"` (the default since 2026-09-13; `"default"` keeps svt-av1's own tune): svt-av1's still-picture tune, PROBED against the local encoder (the numeric value varies by version); unsupported ⇒ stderr warning + recorded fallback, never a silently ignored flag. measured on 2048 cards at crf 35: ssimulacra2 p50 62.7 against 51.8 for the default tune, for +10% bytes. the tune is part of the media state key, so an av1 build that relied on the old implicit default re-encodes once on `--resume`; a spec that sets `tune` explicitly is unaffected. `speed = 6` buys quality per byte over the default 8 at ~2x encode time. `fps` changes playback timestamps only (frames are 1:1 with items; verified, no duplication).
- `crf = "auto"`: the dual gate. a stratified sample (deterministic, versioned bucket heuristics: resolution / entropy / has_text / alpha / source_format) is encoded at every ladder crf and must clear BOTH floors: ssimulacra2 per-bucket p10 ≥ `visual_floor_p10` and global min ≥ `visual_floor_min` (a global average would let one whole stratum degrade), and embedding drift p10 ≥ `drift_floor_p10` measured by the declared `gate_model`; an image can look fine to humans and still move in retrieval space, which is what the corpus actually serves. the largest passing crf wins; none passing ⇒ smallest + a loud warning. the defaults are floors a real corpus reaches: `visual_floor_p10 = 60`, `visual_floor_min = 45`, `drift_floor_p10 = 0.95`, `crf_ladder = [25, 30, 35, 40, 45, 50]`, set from the mtgdataset cards at 488x680 yuv420 (2048 sample, av1 still speed 6): crf30 measures ssimulacra2 p10 65.3, min 58.6, drift p10 0.967 and passes; crf35 measures p50 62.7, p10 55.7, min 45.3, drift p10 0.965 and fails on p10, so the default picks crf30 instead of warning. the earlier floors (p10 85, min 72, drift 0.98, ladder [30, 35, 40, 45]) were set for large photos and no rung reached them on the cards, so the gate always fell back to the smallest crf; they are one `[media.quality]` override away. the gate encodes the ladder with the av1 stream, so `crf = "auto"` is a SpecError naming `media.crf` on the avif and jxl backends (an av1 crf is not an avifenc -q). full retrieval recall lives in the sweep, outside this loop, where it costs O(1) per variant. a negative `drift_floor_p10` disables the drift leg (recorded as `drift_pass = true` in the report).
- `[media.quality]` task-utility floor, the third leg for corpora that only serve retrieval: `utility_floor_hit1` (negative = off, the default; 0..1 enables it), `utility_queries` (how many sampled items become queries, 0 = every sampled item, evenly spaced so every stratum keeps a share), `utility_query_template` (rendered per item with `{label}`, the row's image label or its text), `utility_tol` (allowed hit@1 loss against the lossless source). one text query per item is embedded once by the gate model's text tower (a gate model without one is a SpecError naming `utility_floor_hit1`), and at every ladder crf text-to-image hit@1 is measured against the decoded frames of the same sample; the same hit@1 against the source frames is the lossless reference. a rung passes the leg when `hit1_decoded >= max(utility_floor_hit1, hit1_source - utility_tol)`. the manifest's `media.crf_auto.ladder` carries `hit1`, `visual_pass`, `drift_pass`, `utility_pass` per rung next to `drift_p10`, and `media.crf_auto.utility` carries `hit1_source`, the threshold and the query count, so a refused rung says which leg refused it. motivation, measured 2026-09-12 on 38627 cards (experiment 13 of mtg-urna-benchmark): clip cosine drift p10 falls 0.932 at crf40 to 0.829 at crf60 and the default drift floor vetoes every rung, while text-to-image hit@1 on 100 queries does not move up to crf50. drift is a stability signal; utility is the floor a retrieval profile needs.
- `dedup = true`: content-hash dedup of source images; n rows → one frame via the span overlay, zero format change. on the full scryfall printings set the potential is ~48% of the media (100,452 printings, 51,870 unique arts).
- `order = "cluster"`: greedy cosine clustering (deterministic tie-breaks) makes near-duplicates adjacent so per-segment inter coding has something to predict; measured before recommended; on a 1-per-card corpus the honest expectation is ~0, and on the same-artwork reprint corpus it is -29% (2026-08-31, g=16 + scd=0 vs all-intra).
- `gop = "auto"` with sharding probes PER SEGMENT: each `shard_size` chunk runs its own intra-vs-inter probe encode and ships its own keyint (recorded per segment in the manifest, `gop.per_segment = true`). a single global probe averages regimes away; with `order = "cluster"` the near-duplicate runs concentrate in a few segments, which decide inter (bounded gop, keyint=16, scene-change detection off), while unique segments keep O(1) all-intra access. forced `gop = "intra" | "inter"` still applies to every segment alike.
- `profile`: dataset-type presets resolved BEFORE explicit keys (an explicit key always wins, so no other use case is closed off). `"near-dup"` = cluster ordering + per-segment gop + still tune (visually similar corpora: card reprints, video frames, scans); `"stills"` = `backend = "avif"` + `crf = 48` + `speed = 8`, one libaom avif per image (unique images: O(1) per-image access with no video decode; measured 2026-09-12 on 38627 cards, experiment 11 of brennercruvinel/mtg-urna-benchmark: 1195973116 B against 1374431484 B for the all-intra av1 stream, 13% less at matched ssimulacra2 mean 61.96 on the 2048 sample; the cost is a 4 to 10x slower clip embed at build time because frames are decoded one avif at a time); `"stills-av1"` = the previous stills recipe, all-intra + still tune on the av1 stream (one file per shard, ffmpeg decode, and the profile to pair with `crf = "auto"`); `"archive"` = jxl-transcode (byte-reversible, for corpora where loss is not acceptable); `"retrieval"` = all-intra + still tune + `speed = 6` + fixed `crf = 50`, for a corpus that only serves search and never shows its pixels (measured 2026-09-03 on 38627 cards: 532671548 B self-contained, 7.46x vs the jpeg source, no measurable txt@1 loss on 100 queries; the default drift floor at p10 0.942 would have vetoed it, which is why the profile pins crf instead of running the gate); `"retrieval-auto"` = the same recipe with `crf = "auto"` gated by task utility alone: visual and drift floors disabled (`-1e9` / `-1.0`), `utility_floor_hit1 = 0.0`, `utility_tol = 0.02`, ladder `[40, 45, 50, 55, 60]`, so the largest crf whose hit@1 stays within 0.02 of the lossless source wins (the gate model needs a text tower: clip, siglip2, wemm, jina). the resolved knobs and the profile name both land in the manifest.
- `backend = "jxl"` / `"jxl-transcode"`: the ONLY truly lossless modes. `jxl` is lossless of the source pixels; `jxl-transcode` repacks jpegs reversibly (~20% smaller, round-trip verified by reconstructing the jpeg and comparing sha256). non-transcodable inputs follow `on_unsupported_jpeg = error | copy-source | lossless-jxl`, per-file decisions recorded. preservation contract: decoded pixels (jxl) / original jpeg bytes (verified transcode); exif/icc/xmp only with `keep_metadata`; timestamps and filenames live in the manifest. needs `cjxl`/`djxl` (`brew install jpeg-xl`, which also ships `ssimulacra2` for the gate).

measure everything with `python/tools/urna_image_sweep.py` (variants now include `av1-tune`, `jxl`, `jxl-transcode`) and compare models with the three-tier `python/tools/urna_model_bench.py`: T1 pipeline stability (identity self-retrieval, inflated by construction and labeled as such), T2 codec cost (embedding drift), T3 task utility (label-template text→image as declared weak ground truth, plus `--queries-file` with real operator queries: hit@k, mrr, negative leakage). the tiers answer different questions and are never aggregated into one number.

## 15. media blobs (`urna media`)

a corpus built with `[output] embed_media = true` (§13) carries its encoded media inside the file (section 0x17, an offset table parallel to the `blob_refs` records plus the raw bytes). `urna media` is the read side:

```sh
urna media corpus.urna                 # one line per blob: index, sha256, byte length, inlined or sidecar, original uri
urna media corpus.urna --export DIR    # write every inlined blob to DIR, verifying each against its blob_refs sha256
```

`--export` fails on the first blob whose bytes do not hash to the recorded `content_hash`; `urna validate` performs the same proof over every inlined blob without writing anything. the python side reads one blob without exporting the store: `UrnaFile.blob_bytes(i)`. the section is content_hash-excluded, so an embedded corpus and its sidecar twin carry the same citations.

## reference

every way to get `urna` onto a machine, what each channel lays down, how to verify what you got, and what a maintainer has to set up once before a release can feed these channels. each item is collapsed; open the one you need.

status: the release pipeline (`.github/workflows/release.yml` via cargo-dist, `.github/workflows/pypi.yml`, `.github/workflows/install-test.yml`) serves from `v0.5.0` (2026-09-26) on; `v0.4.0` was never tagged and `v0.3.0` predates the pipeline, so neither carries artifacts. the maintainer checklist below is what each channel needs on the account side; a channel whose prerequisite is missing fails its own job and leaves the github release intact.

the product is offline by construction: the installers are the only thing that ever opens a socket. after install, `urna doctor` validates the surface without network.

### environment variables

every `URNA_*` variable read anywhere in the codebase (installers, cli, forge, dev scripts), in one place. channel sections below mention the ones relevant to that channel; this table is the source of truth.

| variable | scope | default | what it does |
|---|---|---|---|
| `URNA_RELEASE_BASE` | install | github release url | url prefix `install.sh` / `install.ps1` fetch the four release files from (`file://` works for air-gapped installs) |
| `URNA_BIN_DIR` | install | `~/.local/bin` (`~\.local\bin` on windows) | where the installer puts the `urna` binary |
| `URNA_DATA_DIR` | install | `${XDG_DATA_HOME:-~/.local/share}` (`%LOCALAPPDATA%` on windows) | parent dir for the embedder payload the installer extracts |
| `URNA_PYTHON` | runtime, dev | `python3`, or `./.venv/bin/python` if present (`release_check.sh`) | python interpreter the cli, `urna doctor`, and the dev scripts shell out to; must carry the forge deps (numpy, tokenizers) |
| `URNA_FORCE_SCALAR` | runtime | unset | forces the scalar simd kernel over avx2 / neon, for a/b benchmarking |
| `URNA_ALLOW_DOWNLOAD` | runtime, build | unset (offline) | lets `search-text`, `embed_query.py`, `model_fingerprint.py`, and the corpus builder fetch a sentence-transformers model instead of failing offline |
| `URNA_ALLOW_REMOTE_CODE` | runtime, build | unset (empty) | comma-separated preset names allowed to load `trust_remote_code` model-repo code (`ask` / `retrieve` routing, `urna_model_bench.py`, `urna_ui_bridge.py`) |
| `URNA_ALLOW_HEAVY` | runtime | unset | allows an executable / heavy embedder preset in `embed_query_model.py` |
| `URNA_CACHE_DIR` | build | `${XDG_CACHE_HOME:-~/.cache}/urna` | forge's triad-addressed embed cache root (declarative builds, section 13) |
| `URNA_ST_DEVICE` | build | `auto` | sentence-transformers device override (`cpu`, `mps`, `cuda`) for the forge embed workers |
| `URNA_ST_DTYPE` | build | unset (model default) | sentence-transformers dtype override for the forge embed workers |
| `URNA_MODEL_DIR_<NAME>` | build | unset | local dir override for a registry preset, e.g. `URNA_MODEL_DIR_WEMM_2B`; wins over the hf cache, loses to an explicit `--model-path` |
| `URNA_ENABLE_FAKE_PRESET` | test-only | unset | unlocks the `fake-test` model preset used by the registry's own test suite |
| `URNA_MUTATION_ITERS` | dev | `1500` | iteration count for the mutation-fuzz harness; raise for a soak run |
| `URNA_FUZZ_SEED_DIR` | dev | unset | seed corpus dir override for the mutation-fuzz harness |
| `URNA_FUZZ_TARGETS` | dev | `urna-view section-decoders runtime-indexes mmap-open-search` | space-separated cargo-fuzz targets `scripts/fuzz_soak.sh` runs |
| `URNA_BASELINE` | dev | `data/measure/baseline.json` | regression baseline `release_check.sh` compares against |
| `URNA_QUERIES` | dev | `100` | query count `measure_presets.py` uses via `release_check.sh` |
| `URNA_K` | dev | `10` | top-k `measure_presets.py` uses via `release_check.sh` |
| `URNA_OUT` | dev | `/tmp/release_check_post.json` | where `release_check.sh` writes the post-run measurement json |

<details>
<summary>one-liner (linux, macos)</summary>

```sh
curl -sSf https://raw.githubusercontent.com/hoffresearch/urna/main/scripts/install.sh | sh
urna doctor
```

`scripts/install.sh` (posix sh; needs `curl`, `tar`, and `sha256sum` or `shasum`):

1. detects the platform and maps it to a release target: `x86_64` / `aarch64` times `unknown-linux-musl` / `apple-darwin`.
2. downloads four files from the github release: `urna-cli-<target>.tar.xz`, its `.sha256`, `urna-embedder-payload.tar.gz`, its `.sha256`.
3. verifies both sha256 sums before anything touches the install dirs. a mismatch aborts with the two hashes printed.
4. installs the binary to `~/.local/bin/urna` and extracts the payload to `${XDG_DATA_HOME:-~/.local/share}/urna/forge/` (the potion embedder script plus its vendored table).
5. warns if `~/.local/bin` is not on `PATH`.

| flag | effect |
|---|---|
| `--version vX.Y.Z` | pin a release (default: latest). a bare `X.Y.Z` gets the `v` prepended |
| `--uninstall` | remove the binary and the payload dir |

`URNA_RELEASE_BASE`, `URNA_BIN_DIR`, and `URNA_DATA_DIR` override the install paths; see the environment variables reference above.

the linux binaries are static musl, so they run on any distro and inside `scratch` containers. `urna doctor` needs a python 3.12+ interpreter with `numpy` and `tokenizers` for the offline embed probe (`URNA_PYTHON` selects the interpreter); everything else in the cli runs without python.

</details>

<details>
<summary>windows</summary>

```powershell
irm https://raw.githubusercontent.com/hoffresearch/urna/main/scripts/install.ps1 | iex
urna doctor
```

`scripts/install.ps1` mirrors the shell installer: `-Version vX.Y.Z`, `-Uninstall`, the same `URNA_RELEASE_BASE` / `URNA_BIN_DIR` / `URNA_DATA_DIR` overrides. the binary goes to `~\.local\bin\urna.exe`, the payload to `%LOCALAPPDATA%\urna\forge\`. the payload is a `.tar.gz`; `tar` ships with windows 10 1803+. the windows archive is a `.zip`.

</details>

<details>
<summary>python package `urna`</summary>

```sh
pip install "urna[embed]"            # library + offline potion embedding
uvx --from urna urna validate file.urna
```

one `cp312-abi3` wheel per platform: linux x86_64, linux aarch64, macos universal2, windows amd64. python 3.12+. the wheel carries `urna._urna` (the pyo3 extension), the `urna` package, and the bundled potion table (~30 mb) under `urna/models/potion-base-8M/`. the table is bundled on purpose: the installed package embeds offline by construction, so there is no lazy-fetch path to fail in an air-gapped environment. the `embed` extra adds `numpy` and `tokenizers`; the core surface (`urna.open`, `search`, `retrieve`, `validate`, `inspect`) needs nothing beyond the wheel.

the console entry point `urna` installed by the wheel is the read-only subset (`validate`, `inspect`, `stats`, `search`) over the library api, which is what makes `uvx --from urna urna ...` work. the full cli (`ask`, `retrieve`, `build`, `doctor`, `media`, the ann/graph/space searches) is the rust binary from the one-liner, windows, homebrew and binstall items.

the wheel is staged by `scripts/stage_wheel.py` into `packaging/staging/` (gitignored) from `packaging/pyproject.toml`, and built by maturin in `pypi.yml`. the dev flow (`cargo build` + copy the `.so`) is unchanged and does not install a package.

</details>

<details>
<summary>homebrew</summary>

```sh
brew install hoffresearch/urna/urna
```

the formula lives in the `hoffresearch/homebrew-urna` tap and is generated by cargo-dist on every release (`installers = ["homebrew"]` in `Cargo.toml`). known gap, tracked in `.contracts/.agents/AGENTS.md`: the generated formula installs the binary only. a brew-installed `urna` reports exit `4` from `urna doctor` until the payload is laid down, either by running the one-liner (it overwrites nothing brew owns) or by copying `python/forge/` from a checkout into `${XDG_DATA_HOME:-~/.local/share}/urna/forge/`. a custom formula that ships the payload is deferred until the tap sees real use.

</details>

<details>
<summary>npm</summary>

```sh
npm install -g @urna/cli
npx @urna/cli validate file.urna
```

the `@urna/cli` package is generated by cargo-dist on every release (`installers = ["npm"]`, `npm-scope = "@urna"` in `Cargo.toml`, `npm-package = "cli"` in `crates/urna-cli/Cargo.toml`). it carries no binary of its own: on install it fetches the release archive for the platform from the github release and exposes it as the `urna` command. same payload gap as homebrew: run the one-liner (or copy `python/forge/`) before `urna doctor`.

</details>

<details>
<summary>crates.io and cargo binstall</summary>

```sh
cargo binstall urna-cli     # prebuilt binary from the github release
cargo install urna-cli      # compile from crates.io
```

`urna-format`, `urna-runtime` and `urna-cli` are published to crates.io on every release by `.github/workflows/publish-crates.yml`, in that order; `urna-python` ships as the wheel and is not a crate. a rust project that reads or writes `.urna` files depends on `urna-format` (container) and `urna-runtime` (search). `[package.metadata.binstall]` in `crates/urna-cli/Cargo.toml` maps the crate to the cargo-dist archive names (`.tar.xz`, `.zip` on windows), so binstall downloads the released binary instead of compiling. same payload gap as homebrew. to build the unreleased tree instead: `cargo install --git https://github.com/hoffresearch/urna urna-cli`.

</details>

<details>
<summary>docker</summary>

```sh
docker build --platform=linux/amd64 -f docker/Dockerfile -t urna .
docker run --rm -v "$PWD/data:/data:ro" urna validate /data/corpus_next.v1.urna
```

`docker/Dockerfile` builds the static musl binary in a throwaway toolchain stage and copies it into `scratch`: no shell, no package manager, no network at runtime. the corpus arrives as a mounted volume, so the same image serves air-gapped hosts. on apple silicon build the aarch64 variant natively (`--build-arg TARGET=aarch64-unknown-linux-musl`); qemu user emulation crashes rustc mid-build. the image has no python, so `ask` / `retrieve` are not available inside it; the engine verbs (file + vector in) are.

</details>

<details>
<summary>dev build</summary>

```sh
cargo build --release --workspace
cargo build --release -p urna-python --features pyo3/extension-module
cp target/release/lib_urna.dylib python/_urna.so   # macos (.so on linux)
```

rust edition 2024 (`rustc >= 1.85`), python 3.12+. the potion table is git-lfs: `git lfs pull` before `urna doctor` or any `ask` / `retrieve`, a pointer file is rejected with exit `5`. setup details, hooks and the merge gate are in `docs/CONTRIBUTING.md`.

</details>

<details>
<summary>verification</summary>

every release artifact ships with a per-file `<name>.sha256` and the release carries a combined `sha256.sum`. the installers verify before writing; by hand:

```sh
sha256sum -c urna-cli-x86_64-unknown-linux-musl.tar.xz.sha256
gh attestation verify urna-cli-x86_64-unknown-linux-musl.tar.xz --repo hoffresearch/urna
```

the attestation is sigstore keyless provenance produced in the release job (`attestations: write`, `actions/attest`), binding the artifact digest to the workflow, the commit, and the tag. the pypi wheels carry pep 740 attestations produced by trusted publishing (no stored token), visible on the file's pypi page; the first release (`0.5.0`) went out with a bootstrap token and carries none (maintainer checklist step 4).

every release also carries a cyclonedx sbom per built package (`urna-cli.cdx.xml`, generated by `cargo cyclonedx` in the build job and attested like the binaries), and the binaries are built with `cargo auditable`, so the dependency tree can be read back out of the executable:

```sh
gh attestation verify urna-cli.cdx.xml --repo hoffresearch/urna
cargo audit bin ~/.local/bin/urna
```

commits on `main` are ssh-signed and the branch ruleset requires verified signatures. release tags are annotated and ssh-signed too: `.github/workflows/tag-verify.yml` checks the tag against `.github/allowed_signers` in the plan phase of the release and before the wheels build, so an unsigned tag, a lightweight tag, or a signature from a key not on that list stops the release before anything is built.

`.github/workflows/install-test.yml` runs after every published release and installs the product the way a user does: the one-liner against the release url on linux x86_64 / aarch64, macos arm64 / x86_64, windows, then `urna validate` on the golden fixture and `urna doctor`; a second job pip-installs the published wheel and runs the `uvx` entry point. a failure there means the release is broken for users: yank and re-cut.

</details>

<details>
<summary>offline and air-gapped notes</summary>

- after install nothing opens a socket: `ask`, `retrieve`, `doctor`, and the python `urna.embed_potion` all resolve the vendored potion table locally. sentence-transformers presets from the model registry are the exception and download only with `URNA_ALLOW_DOWNLOAD=1` (section 12).
- the cli finds the embedder in this order: the repo layout (`python/forge/embed_query_potion.py`), then `${XDG_DATA_HOME:-~/.local/share}/urna/forge/`, then `<exe>/../share/urna/forge/`. `urna doctor` prints which one it picked and validates the table is real bytes, not an lfs pointer.
- air-gapped install: fetch the four files the one-liner downloads on a connected machine, copy them over, and run the installer with `URNA_RELEASE_BASE=file:///path/to/dir`. for the wheel, `pip download urna[embed]` on the connected side and `pip install --no-index --find-links` on the other.
- `urna doctor` exit codes are typed so provisioning scripts branch on them: `0` ok, `2` python missing, `3` numpy/tokenizers missing, `4` embedder script missing, `5` table missing or lfs pointer, `6` embed run failed. a scalar simd fallback warns but exits `0`.

</details>

<details>
<summary>maintainer checklist (one-time, before the first tagged release)</summary>

the release workflows assume external state that a fresh org does not have. as of 2026-09-26 (repo secrets of `hoffresearch/urna`: `CARGO_REGISTRY_TOKEN`, `NPM_TOKEN` and `HOMEBREW_TAP_TOKEN` set):

1. **homebrew tap**: the public repo `hoffresearch/homebrew-urna` exists with an empty `Formula/` directory (created 2026-09-26); the `publish-homebrew-formula` job in `release.yml` checks it out and pushes `Formula/urna.rb` to it. the job authenticates with the `HOMEBREW_TAP_TOKEN` secret, a fine-grained token (`urna-homebrew-tap`, resource owner `hoffresearch`) with contents read and write on that repo only, expiring 2027-09-26 (the org caps fine-grained tokens at 365 days). github has no api that mints one: rotate it by hand in the account settings and replace the secret. without it the job fails and the rest of the release stands.
2. **npm**: `@urna/cli` publishes under the `urna` npm org with the `NPM_TOKEN` secret, a granular token with package and org write on that org (user `notlikedev`). granular tokens expire: the current one expires 2026-12-11, rotate it before a release that falls after that date. the name `@urna/cli` is unclaimed until the first publish.
3. **crates.io**: `publish-crates.yml` publishes with the `CARGO_REGISTRY_TOKEN` secret (crates.io user `brennercruvinel`). the three names are unclaimed until the first publish. after it, the token can be swapped for crates.io trusted publishing (github oidc, no stored token, configured per crate on crates.io) the same way pypi works; that change replaces the secret with `rust-lang/crates-io-auth-action` in the workflow.
4. **pypi**: the `pypi` environment exists in the github repo (created 2026-09-26) and carries a `PYPI_API_TOKEN` secret, a bootstrap token that publishes the first release because a trusted publisher can only be attached once the project exists (a pending publisher needs a web login). releases published with it carry no pep 740 attestations. after the first publish: on pypi.org add the trusted publisher to the `urna` project (owner `hoffresearch`, repository `urna`, workflow `pypi.yml`, environment `pypi`; the publish log prints a direct link), delete the `PYPI_API_TOKEN` environment secret, and revoke the token on pypi.org. with the secret gone the empty `password` input selects OIDC trusted publishing again and no api token is stored anywhere.
5. **git-lfs**: release and wheel builds pull the potion table (`.github/dist-build-setup.yml`, `lfs: true` in `pypi.yml`). check the lfs bandwidth quota before a release; five targets plus four wheels each fetch the ~30 mb table.
6. **attestations**: nothing to configure. `release.yml` already requests `attestations: write`, `pypi.yml` requests `id-token: write`.
7. **short url**: `get.hoffresearch.com` is not registered (nxdomain). the scripts and the README use the raw github url. if the short form is wanted, point the dns at a 302 to the raw script and update the README plus both script headers in the same change.
8. **cutting a release**: bump `version` in `Cargo.toml` (the workspace version tracks the latest tag) together with the `version` of the `urna-format` / `urna-runtime` entries in `[workspace.dependencies]` (crates.io resolves those, the path only serves the workspace), move the `[Unreleased]` block in `docs/CHANGELOG` under the new version, merge to `main`, then `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z` (annotated and signed; `git config tag.gpgsign true` makes `-s` the default with the ssh key already used for commits). the tag drives `release.yml` (github release, homebrew, npm, crates.io) and `pypi.yml`, both of which verify the signature first; the published release triggers `install-test.yml`. if that trigger does not fire, run it by hand with `workflow_dispatch` and the tag.
9. **release signers**: `.github/allowed_signers` lists the keys allowed to sign release tags (one line per principal). a new maintainer key is a pull request that appends a line there; the verify step reads the file from the tagged commit.
10. **changing the dist config**: after editing `[workspace.metadata.dist]` run `dist generate` and commit the regenerated `release.yml`; never hand-edit it. `pr-run-mode = "plan"` keeps pull requests on the plan step only.

</details>
