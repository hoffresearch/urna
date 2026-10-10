---
project: urna
audience: users and integrators
status: active
last-updated: 2026-10-10
domain: usage
---

# Usage

`urna` is a single-file binary container for distributing semantic knowledge bases. One file: chunks, canonical text, byte-spans, embeddings, search contract, hashes. Copy it, share it, search it.

This guide covers the commands you'll actually use: the agent verbs `ask`, `retrieve` and `build` (the front door; they shell out to the offline Python embedder or the forge), and the engine subcommands beneath them (validate, stats, inspect, media, search/search-ann/search-graph/search-space/search-text, benchmark, cite, doctor), which take a file and a vector; two of them run Python (`search-text`, for its sentence-transformers embedder, and `doctor`, which probes the Python env), the other ten never do. `urna --help` lists them in the same two groups. Getting the binary onto a machine (every install channel, verification, offline notes, the maintainer checklist) is the reference section at the end of this document; the short form is `curl -sSf https://raw.githubusercontent.com/hoffresearch/urna/main/tool/tasks/installer.sh | sh` (or brew, npm, cargo) followed by `urna setup`, the interactive installer that completes every channel (section 16).

## Quickstart

Five verbs cover the whole loop. What each one is for:

| Verb | What it does | In one line |
|------|--------------|-------------|
| `build` | Creates the base | Rows + embedding model in, one `.urna` out |
| `ask` | Queries it from the terminal | Text in, one cited answer out |
| `retrieve` | Hands results to another program | JSON/JSONL of cited spans, `score` is the exact rerank |
| `cite` | Resolves the source | A `urna://` citation back to the stored text and its hashes |
| `validate` | Proves the file | Every checksum, every hash, the manifest contract |

`demo/starter/` ships a corpus that needs nothing downloaded: twelve paragraphs of CC0 prose about Urna (`docs.jsonl`) and the smallest spec that builds them (`corpus.toml`, one JSONL source, the bundled potion model). From the repo root:

```sh
urna build --spec demo/starter/corpus.toml
urna ask demo/starter/out/quickstart.urna "can I use this offline" -k 1
urna retrieve demo/starter/out/quickstart.urna "how do citations work" -k 2 --format jsonl
urna cite demo/starter/out/quickstart.urna 'urna://<content_hash>/<chunk_id>'   # a citation_id from ask or retrieve
urna validate demo/starter/out/quickstart.urna
```

`build` runs the forge in `rust/bridge/python/urna/`, so it needs the repo checkout (the installed payload carries the two query embedders only: the potion script and the registry script for wemm, CLIP and Jina corpora, whose model deps you add to the setup venv when you need them, section 12); the other four verbs work with the installed binary alone. The Python version of the same loop, with the embedder in plain sight, is `python demo/starter/quickstart.py`. To build your own corpus, swap `docs.jsonl` for your rows (JSONL, CSV, SQLite, an image dir) in the spec: §13 has the full contract.

## 1. Build a `.urna` from chunks

The Python pipeline owns chunking, embedding, caching, and the final emit. The Rust writer owns reproducibility, hashing, and deterministic byte layout. The one thing every build needs that is easy to miss: the embedder's `model_hash`, written into the file so a query embedded by another model fails loudly instead of returning plausible wrong hits. The bundled potion embedder carries its own (`emb.model_hash()`); for a sentence-transformers model, §7 has the fingerprint.

```python
import sys

sys.path.insert(0, "rust/bridge/python")
from urna.pipes.buildfile import BuildConfig, Pipeline, chunk_text
from urna.embed.potiontab import potion_embedder

emb = potion_embedder()  # offline static table; swap for rust/bridge/python/urna/embed/searchtxt.py's ST path if you need a bigger model

cfg = BuildConfig(
    output_path="my_corpus.urna",
    embedding_model=emb.embedding_model,
    embedding_dim=emb.embedding_dim,
    chunker_version="my-chunker/v1",
    model_hash=emb.model_hash(),  # the zero placeholder is refused here, at write time
    preset="exact",  # see §6 for preset choices
    reproducible=True,
)
pipe = Pipeline(cfg, embedder=emb, scratch_db="cache.sqlite")
for source_uri, text in documents:  # your (uri, text) pairs
    for spec in chunk_text(text, source_uri):
        pipe.add(spec)
pipe.emit()
```

The embedder is any callable that takes the chunk specs and returns one l2-normalized vector per spec; `potion_embedder()` is one, and a sentence-transformers wrapper is a few lines (`m.encode([s.canonical_text for s in specs], normalize_embeddings=True).tolist()`). For real-world examples: `rust/bridge/python/urna/entry/oldformat.py` (SQLite to `.urna`), the [fakenews-ptbr-urna-benchmark](https://github.com/brennercruvinel/fakenews-ptbr-urna-benchmark) (seven pt-BR datasets to `.urna` files with three example embedders, on the published wheel), and `demo/starter/quickstart.py` (the shortest complete build, on `urna.build` directly).

### Image and PDF corpora

Image corpora live in the forge tooling layer because a vision tower needs torch, which the sovereign runtime does not take. The `.urna` they emit is an ordinary `.urna`, served by the same Rust runtime from mmap.

Two tools build them. The declarative build (`urna build --spec`, section 13, with a `[media]` table) is the one that puts the media inside the file: the encoded frames as content-addressed blobs (section 0x14, the bytes themselves with `embed_media = true`, section 0x17), the byte span each chunk was embedded from (overlay 0x16), and each model's image vectors in their own named space (table 0x15, slab in the 0x20 to 0x2F band) behind the `supports_multimodal` capability, gated by their own `model_hash`. Those sections are excluded from `content_hash`, so adding media never moves an existing citation.

The older standalone tool, `rust/bridge/python/urna/entry/imgcorpus.py`, is the sweep and measurement path: it letterboxes every image onto one canvas, encodes the sequence, embeds the DECODED frames with one open_clip model, and writes one chunk per image or PDF page whose vectors are the file's default space. Its output is a directory, `corpus.urna` next to `corpus.media/` (the encoded stream, referenced by `media://` URIs in the chunks) and `corpus.manifest.json`; it writes no blob sections and no named spaces. Embedding the decoded frames rather than the source pixels is deliberate in both tools: the index has to describe what a reader can actually get back.

```sh
.venv/bin/python rust/bridge/python/urna/entry/imgcorpus.py \
    --input-dir /path/to/dermoscopy_images \
    --dataset my-derm \
    --output corpora/my-derm.urna \
    --labels labels.csv
```

`corpora/my-derm.urna` carries the index and the citable text; `corpora/my-derm.media/` carries the encoded frames and `corpora/my-derm.manifest.json` the provenance (ordinals, origins, labels, media digests). Move the three together. For one self-contained file, build the same images with a spec (`[source] kind = "image_dir"`, `[media]`, `embed_media = true`).

`--width` is a ceiling, not a target: the canvas is clamped to the dataset's median source width, so a corpus is never upscaled. Lower it to trade quality for size; raising it above the source does nothing but make the encoder pay for interpolated pixels.

`--gop-policy auto` (the default) probes a spaced sample of frames and lets the bytes decide between all-intra and inter coding. On every corpus measured so far (PH2, HAM10000, WSI tiles, scanned PDF) the probe chose intra: unrelated images give inter-frame prediction nothing to find. Inter stays available for genuinely sequential media. `--all-intra` forces every frame a keyframe and overrides the probe.

`--shard-size N` splits the stream into consecutive segments of about N frames, one blob per shard, which caps decode memory and improves cold seek on large corpora. `--order-similarity` tries a greedy nearest-neighbour frame order before encoding; measured on 1210 WSI tiles it cost 0.15 percent instead of helping, so it stays off by default.

`--backend av1` (the default) won the size-matched matrix; `--backend avif` writes one AVIF per image and is the only backend that accepts `--pix-fmt yuv444p`. `--crf` (default 35) sets the AV1 rate, `--avif-quality` (default 35) the AVIF one. `--control` builds the letterbox-lossless PNG control corpus that codec cost is measured against. For every backend the manifest's `media.source_bytes` is the byte size of the original source files and `compression_ratio` is that over `output_bytes`; the AVIF path records the letterboxed PNGs it actually feeds avifenc apart as `letterboxed_input_bytes`, so the ratio is comparable across AV1, AVIF and JXL.

`--dtype float32|float16|int8|int4` overrides the preset's vector dtype for the image space (int4 needs the dim divisible by 64). Measured: quantization was not the driver of quality loss (the melanoma delta is identical at f16 and int8, and similar at int4), while the vectors themselves shrink 214 KB to 112.6 KB to 63.8 KB on PH2.

Add `--pdf` to render PDF pages as the images; page numbers are kept in the manifest and in the citable text. For a non-dermatology domain pass `--model ViT-B-32 --pretrained openai`. The pretrained tag is required for bare architecture names, because open_clip answers a missing tag with random weights.

Search with a query image or a clinical description, and optionally decode the matched frames back out:

```sh
.venv/bin/python rust/bridge/python/urna/entry/imgsearch.py \
    --index corpora/my-derm.urna --query-image lesion.jpg -k 10 \
    --letterbox-query --save-frames hits/
```

`--query-text "..."` searches with a clinical description instead of an image, and `--letterbox-query` normalizes the query onto the corpus canvas before embedding. Queries go through `UrnaFile.retrieve` with the open_clip embedder's `model_hash`, so a corpus built with another model is refused before anything is scored; `--skip-model-check` bypasses that gate explicitly. A spec-built multimodal corpus is queried per space instead (`urna search-space`, `UrnaFile.search_space`, section 5).

### Measuring an image corpus

`tool/bench/imageeval.py` reports two rulers and keeps them apart, because they answer different questions:

- `identity` asks whether a source image retrieves its own frame. It measures rank stability under the codec and is inflated by construction, since the corpus contains the answer. On an uncompressed index it returns 1.000 by definition.
- `label` removes the query's own frame and scores how many of the remaining neighbours share its label. Nothing in the corpus is the answer, so this is the one that reports retrieval quality. It is printed next to the random-pick baseline for the same label distribution, without which the number cannot be read.

Neither means much alone. Pass `--baseline` with the uncompressed control index (`--control` at build time) to get the delta, which is what the codec actually cost:

```sh
.venv/bin/python tool/bench/imageeval.py \
    --index corpora/my-derm.urna \
    --baseline corpora/my-derm-control.urna \
    -k 1 5 10 --out eval.json
```

Measured in phase 6 (full matrix and intervals in `docs/CHANGELOG`): on PH2 (n=200) av1-intra crf35 compresses the media 86x for a mean label `precision@10` delta of -3.4 to -4.7 points whose interval crosses zero, but the melanoma class alone drops 16.9 points with a significant interval ([-25, -10]); on HAM10000 (2000-sample) the media shrinks 151x for a mean delta of -1.5 [-3.5, +0.6], again with a significant melanoma cost (-10.7). The text-to-image ruler is harsher and honest: 44/60 correct top-10 clinical queries on the control falls to 22/60 at crf35, and the loss does not recover with rate. Per-class floors matter more than the mean: report the interval and the worst class, not just the point.

`tool/bench/imagerate.py` runs the variant matrix for you (`av1-intra` and `av1-inter` CRF ladders, `avif` qualities, `dtype:` rungs, `av1-order`; kinds separated by `;`, values by `,`; the PNG control is always built), records `urna_bytes` and the control's `media_bytes` per variant, and writes one consolidated comparison JSON:

```sh
.venv/bin/python tool/bench/imagerate.py \
    --input-dir /path/to/images --dataset my-derm \
    --variants "av1-intra:35,40;dtype:int8" \
    --labels labels.csv --out-dir sweep/ --out sweep/summary.json
```

Direct API (no chunker): `urna.build(output_path, embedding_model, embedding_dim, chunker_version, model_hash, chunks, preset="exact", reproducible=True)`.

## 2. Validate

Full integrity check: magic, the header checksum (the first 8 bytes of the SHA-256 over the header), every section's checksum (the same 8-byte prefix over its physical bytes), the footer hash (the full SHA-256 of everything before the footer), manifest schema, contract cross-check against the manifest, a NaN/Inf walk over the default embeddings, and, when the file inlines its media (section 0x17), every blob against its `blob_refs` digest. The `file_hash` it prints is the SHA-256 of the whole file, footer included.

```sh
urna validate my_corpus.urna
```

Failure modes are typed (`SectionChecksumMismatch(0x04)`, `UnsupportedDType("bfloat16")`, etc.), never "best effort".

## 3. Stats

Sizes, dim, dtype, model, hashes, per-section bytes, the SIMD backend the runtime selected.

```sh
urna stats my_corpus.urna
```

## 4. Inspect

Header bytes, full section table, manifest as JSON. Use `--json` for programmatic consumers (CI dashboards, drift detection):

```sh
urna inspect my_corpus.urna             # human-readable
urna inspect my_corpus.urna --json | jq # structured
```

Schema: `{magic, version_major, version_minor, format_version, schema_version, embedding_dim, n_chunks, n_embeddings, file_size, manifest, sections[], blobs, spaces, file_hash, content_hash, simd_backend}`; `blobs` is the 0x14 table (or null) and `spaces` the 0x15 table (or null).

## 5. Search

### Exact path (vector input)

Pass a query vector directly as a JSON array. Recall = 1.0 by construction.

```sh
urna search my_corpus.urna "[0.1, 0.2, ...]" -k 10
```

### Search by text

Embed the query with the same model the corpus was built with (the manifest declares it), then route by what the file carries: BM25 plus vectors (the hybrid path) when the file has a BM25 section, HNSW when it has an HNSW section, exact otherwise. The manifest's `index_type` names the vector index only, so a `hybrid` preset file (it declares `hnsw` and carries BM25 as the `supports_bm25` capability) takes the hybrid path. The runtime cross-checks the embedder's `model_hash` against the manifest before running search and refuses on mismatch. See §7.

```sh
urna search-text my_corpus.urna "vacina contra covid funciona" -k 5
```

For tuning the candidate set: `--candidates N` (default `4*k`, min 64).

### Force the ANN path

Useful for debugging or measuring `ef_search` curves above the file's floor. The beam that runs is `max(--ef, k, ef_construction)`, where `ef_construction` is the build's (400 for `urna.build` and the forge), so an `--ef` below 400 changes nothing on those files; the `candidates:` line of the output prints the beam that ran. Falls back to exact if the file has no HNSW section.

```sh
urna search-ann my_corpus.urna "[0.1, 0.2, ...]" -k 10 --ef 800
```

### Graph search (chunk-to-chunk)

Seeds from the exact-cosine top-`ef`, expands a bounded breadth-first walk over the chunk-to-chunk graph (`--hops`), then exact-reranks the union. The graph only generates candidates; the returned score is real cosine (recall is not computed, the rerank guarantees the score). Falls back to exact if the file has no `graph_adjacency` (0x0C) section. Build a graph-carrying file with `urna.build(..., with_graph=True)` (default off); the section is additive and excluded from content_hash, so adding a graph never changes a citation.

```sh
urna search-graph my_corpus.urna "[0.1, 0.2, ...]" -k 10 --hops 2 --ef 100
```

### Search a named space (multimodal)

`search-space` runs the per-space exact search over one named vector band (0x15 + 0x20+): image spaces, extra text spaces, MRL-sliced spaces. The query vector must be embedded with the space's model at the space's dim; an unknown space, a wrong dim, or (with `--expect-model-hash`) a wrong model are typed errors; never a silent fallback to the text path. The space names come from `urna stats` (the `spaces:` block) or `inspect --json` (the `spaces[]` array).

```sh
urna search-space my_corpus.urna "[0.1, ...]" --space "wemm-2b@256" -k 5
urna benchmark my_corpus.urna -q 100 -k 10 --space "wemm-2b@256"
```

### The flagship: ask and retrieve

`ask` and `retrieve` are the agent-native front door: text query in, cited answer out, no flags needed. They embed the query OFFLINE and route the embedder BY THE MANIFEST MODEL: a potion corpus keeps the potion static table (`rust/bridge/python/urna/embed/potionqry.py`, the unchanged fast path), and a corpus whose default text space is any registry model (wemm, Jina, CLIP; see §12) goes through `rust/bridge/python/urna/embed/presetqry.py`, which encodes the query with that model's query route and, for an MRL-truncated default space, slices + renormalizes to the manifest dim (`--mrl-dim`, passed automatically). The pt-BR MiniLM demo corpus resolves to the `minilm-multilingual` preset, and any other sentence-transformers model no preset names takes the same path: the query is embedded by `rust/bridge/python/urna/embed/searchtxt.py`, the `search-text` embedder, with the `model_hash` the corpus was built with. It needs `sentence-transformers` in the Python urna runs and the model in the local HF cache (`URNA_ALLOW_DOWNLOAD=1` fetches it once); a missing one is named with its fix. Both paths validate the embedder's `model_hash` against the manifest exactly like `search-text`, and route by what the file carries: the hybrid path (BM25 candidates plus the vector shortlist, fused, then the exact rerank) when the file has a BM25 section, HNSW when it has an HNSW section, exact otherwise. The graph is never routed to automatically; `search-graph` is its verb. Every printed score IS the exact-cosine rerank value, so on the hybrid path the BM25 leg widens the candidate set and the cosine orders it: a lexical match reaches the answer when its cosine earns a place in the top-k.

`ask` prints one low-cognitive-load cited answer:

```sh
urna ask my_corpus.urna "can I use this offline" -k 3
```

`--disclose answer` (default) prints the cited canonical text and a `urna://` citation, nothing else. `--disclose explain` ALSO prints the rerank-source honesty line: `real cosine` when the score is full precision, `real cosine at stored precision` for a lossy stored slab (float16/int8/int4) with no full-precision source, plus the route and per-path candidate counts.

`retrieve` is the agent-shaped surface: a JSON/JSONL answer-pack of cited spans.

```sh
urna retrieve my_corpus.urna "can I use this offline" -k 5 --format jsonl
```

Each hit is `{chunk_id, score, score_type=cosine, source_uri, offset_start, offset_end, citation_id, text, file_hash, content_hash, rerank_source}`. The `score` is the exact rerank value (never a candidate-generator proxy), `text` is the tier-1 stored canonical text, and `citation_id` round-trips through `urna cite`. `--format json` emits a single pretty array instead of one object per line.

The embedder picks its interpreter in a fixed order (the same ladder section 11 states for `doctor`): `URNA_PYTHON` if set, else the venv `urna setup` built (`<data root>/urna/venv`), else the nearest `.venv/bin/python` walking up from the CWD (the repo's carries the embed deps: numpy + tokenizers + the vendored potion table), else `python3` on PATH. So the setup venv, or the repo `.venv` in a checkout, is used automatically; set `URNA_PYTHON` only to force a specific interpreter. The selected interpreter is printed to stderr; and since discovery executes the nearest ancestor `.venv/bin/python`, set `URNA_PYTHON` explicitly if you run `urna` from inside an untrusted directory tree. Point `--model-path` at a copied potion table dir for a fully sealed offline run.

The Python convenience is `python rust/bridge/python/urna/reads/retrieval.py`: it builds a `.urna` from the CC0 demo corpus with the potion embedder, asks a question, and prints the cited answer with a `urna://` citation, all offline and deterministic (the one-GIF demo).

## 6. Presets

`preset=` selects a (text encoding, embedding dtype, optional ANN, optional BM25) bundle. Per-knob overrides win, see `BuildConfig.text_encoding`, `.dtype`, `.with_hnsw`, `.with_bm25`, `.mrl_dim`.

| Preset       | Text encoding | Embeddings  | ANN | BM25 | size_ratio | recall@10 |
|--------------|---------------|-------------|-----|------|-----------:|----------:|
| `exact`      | raw           | float32     | No  | No   |      1.000 |    1.0000 |
| `compressed` | zstd          | float16     | No  | No   |      0.339 |    1.0000 |
| `tiny`       | zstd          | int8        | Yes | No   |      0.256 |    0.9920 |
| `micro`      | zstd          | mrl256-int8 | Yes | No   |      0.223 |    0.8100 |
| `nano`       | zstd          | int4        | Yes | No   |      0.209 |    0.9130 |
| `hybrid`     | zstd          | float32     | Yes | Yes  |      0.609 |    1.0000 |

Numbers measured on the project's pt-BR fake-news corpus (n=30,725, dim=384), 100 queries, k=10 vs the float32 exact baseline. RULER CAVEAT: these `recall@10` figures use a SELF-PERTURBATION ruler (each query is a corpus vector plus tiny noise), so they measure rank-stability under quantization, NOT real-query retrieval, and are likely inflated; see the `ruler` field in `ladder.json`/`baseline.json` and the pending real-query (MTEB-style) ruler (gate-zero). These are the honest current sizes after the text-codec repack (intpack chunk_ids/spans, bitpacked HNSW/BM25 payloads) shrank the indexed presets below the v0.2 figures: `tiny` 0.283 -> 0.256, `compressed` 0.350 -> 0.339, `hybrid` 0.668 -> 0.609. Latency ranges (NEON, hot cache): exact p50 ~3.1 ms, tiny p50 ~1.2 ms, micro p50 ~0.8 ms, nano p50 ~2.1 ms, hybrid p50 ~4.0 ms.

The `exact`/`compressed`/`tiny`/`nano`/`hybrid` rows are direct `preset=` values; `micro` is the published name for the matryoshka size lever (the documented honest point `mrl256-int8`), built with `urna.build(text_encoding="zstd", dtype="int8", mrl_dim=256, with_hnsw=True)` and emitted by `presetrun.py --variants ...,micro,...`.

Pick `nano` for the smallest distributable file with recall above the nano floor: int4 block-64 embeddings (per-64-dim-group f16 absmax scales + packed 4-bit codes) take the embeddings section from int8's 11.92 MB down to 6.27 MB (~1.9x over int8, ~7.5x over float32). `nano`/`micro` require the effective `embedding_dim` divisible by 64. Every sub-int8 preset (`micro`/`nano` and the whole MRL curve) is STORED-PRECISION: the 0x09 `embeddings_fp` rerank source is not wired, so the net-of-fp ratio equals the stored ratio and `score`/`recall@10` are real cosine AT THE STORED PRECISION (int4/int8), disclosed via `dtype` (and `mrl_dim`/`full_dim` for `micro`) in `urna stats` and on every result, never a bare-slab ratio. `micro` trades recall for size on this non-MRL MiniLM baseline (0.810 recall@10 at 0.223 ratio, see the curve below); pick it only when raw size beats the last ~10 recall points or once a real MRL-trained model lands. Pick `tiny` when you want a smaller file than `compressed` with recall still above 0.99, `compressed` when you need lossless cosine + 3x compression, `hybrid` when queries include rare terms, proper nouns, or siglas that pure embeddings underweight (the BM25 leg puts the chunks carrying those words into the candidate set that the exact cosine then orders; it widens the shortlist, it does not re-score, so the match wins the top-k only when its cosine earns it), and `exact` when storage isn't the bottleneck and you want the recall=1.0 ground truth. A `hybrid` file declares `index_type = "hnsw"` and `supports_bm25 = true` in its manifest; `ask`, `retrieve` and `search-text` see the capability and take the hybrid path.

### Matryoshka prefix truncation (`mrl_dim`)

`urna.build(..., mrl_dim=K)` (or `BuildConfig.mrl_dim`) slices each l2-normalized vector to its first `K` components and re-l2-normalizes the prefix BEFORE quantization (Qwen3/ST/BGE truncate-then-renormalize). This is the dimension axis: orthogonal to and multiplicative with the dtype levers. The stored `embedding_dim` becomes `K`, the source dim is recorded as `full_dim`, and both appear in `urna stats`. Queries are striped at `K` too, so a full-dim query against a truncated file is a dimension mismatch; slice + renorm the query to `K` first. Truncation is a pure deterministic op, so builds stay byte-identical; `content_hash` is over the truncated embeddings, so a citation is tied to its `mrl_dim` (never claimed stable across dims). int4 still needs the effective dim divisible by 64, so `mrl_dim` in {256, 192, 128} works with int4 but 96 does not (use int8/f16/f32 at 96).

Matryoshka pays off on a model trained for it (information front-loads into the prefix). The shipped MiniLM corpus is NOT MRL-trained, so truncation costs real recall@10 there; the published ladder (100 queries, k=10) reports the honest curve (same self-perturbation ruler as above, see the RULER CAVEAT) and `tool/bench/presetrun.py` emits it (the default `--variants` are `compressed,tiny,micro,nano,hybrid` plus `mrl256/192/128-int8`, `mrl96-int8`, `mrl256/192/128-int4`):

| Ladder        | Size ratio | recall@10 |
|---------------|-----------:|----------:|
| `mrl256-int8` (`micro`) |     0.223  |   0.810   |
| `mrl192-int8` |     0.207  |   0.733   |
| `mrl128-int8` |     0.190  |   0.659   |
| `mrl96-int8`  |     0.182  |   0.574   |
| `mrl256-int4` |     0.191  |   0.777   |
| `mrl192-int4` |     0.183  |   0.713   |
| `mrl128-int4` |     0.174  |   0.627   |

On that baseline `nano` (full-dim int4) still beats every truncated point on recall, so reach for `mrl_dim` (the `micro` rung is `mrl256-int8`) when raw size matters more than the last ~10 recall points, or once an MRL-trained embedder is in play. `benchgate.py` gates `micro`/`nano`/`mrl256-int8` conditionally (size_ratio <= 0.25; recall >= 0.78 for micro/mrl256-int8, >= 0.85 for nano), only when the run includes them.

## 7. model_hash and offline operation (`--model-path`)

`search-text` cross-checks three things before running search:

1. `manifest.embedding_model` (name) matches the embedder's report.
2. `manifest.embedding_dim` matches `len(vector)`.
3. `manifest.model_hash` matches the embedder's reproducible fingerprint.

Layer 3 is the only one that catches the silent failure mode "same name + same dim + different snapshot, cosine-valid garbage". The fingerprint hashes a fixed list of inference-relevant files (`config.json`, `tokenizer.json`, `model.safetensors`, `1_Pooling/config.json`, etc.). See `rust/bridge/python/urna/model/modelhash.py`.

Build with a real fingerprint:

```python
from urna.model.modelhash import (
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

### Fully offline search

Distribute the model directory alongside the `.urna` (e.g. on a USB stick or in a sealed Docker image), then point `--model-path` at it on every search:

```sh
urna search-text my_corpus.urna "vacina contra covid" -k 5 \
    --model-path /mnt/models/paraphrase-multilingual-MiniLM-L12-v2
```

No HuggingFace cache hits, no network. The fingerprint is recomputed locally and verified against the manifest.

### Pre-phase-3 corpora

Files built with `model_hash = sha256:0...0` (the legacy placeholder) fail the strict gate by design; `urna.build` no longer writes one unless asked (`allow_placeholder_model_hash=True`, for fixtures). For a file you already have, two options:

- Rebuild with a real fingerprint (recommended).
- Pass `--skip-model-hash-check` to proceed at your own risk. The search is still cosine-valid if you genuinely use the same embedding model, but there is no guarantee. The flag covers the placeholder only: a corpus with a real fingerprint that disagrees with the embedder fails whatever flags are passed.

`search-text` on a corpus built with `mrl_dim` (section 6) passes `--mrl-dim <embedding_dim>` to the embedder when the manifest records `full_dim`, so the query is sliced and renormalized to the file's dim the way `ask` and `retrieve` already did; `rust/bridge/python/urna/embed/searchtxt.py` takes the same flag by hand.

## 8. Benchmark

Random-query latency stats (mean, p50, p95, p99). With `--ann`, also runs ANN against the same queries and computes `recall@k (ANN vs exact)`. With `--madvise-cold`, runs an extra pass calling `posix_madvise(MADV_DONTNEED)` between queries: upper bound on cold-cache latency, not absolute cold (see `MmapUrnaFile::madvise_cold` docs).

```sh
urna benchmark my_corpus.urna -q 100 -k 10 --ann 100 --madvise-cold
```

Typical output (n=30,725, dim=384, NEON, int8):

```
Exact (100 queries, dim=384, dtype=int8, simd=neon) [hot]:
  p50: 1.28 ms  p95: 1.68 ms
Exact ... [madvise-cold]:
  p50: 1.95 ms  p95: 2.40 ms
ANN ef=100 (beam 400, 100 queries) [hot]:
  p50: 0.44 ms  p95: 0.62 ms
  recall@10 (ANN vs exact): 0.9920
```

`beam` is the candidate width that ran, `max(--ann, k, the file's ef_construction)`: on a file built with the default `ef_construction = 400`, `--ann 100` searches 400 candidates, so the latency and recall curves only move above that floor. recall@10 here is ANN-vs-exact rank-stability (the ANN index against the exact-cosine top-k on the same queries), NOT real-query retrieval quality, and the printed value mirrors the published tiny ladder number; see the RULER CAVEAT in section 6.

## 9. Citations

Every search hit carries a stable `citation_id` of the form `urna://<content_hash>/<chunk_id>`. Resolve it back to the canonical text and original byte span:

```sh
urna cite my_corpus.urna 'urna://sha256:1aa9.../sha256:8f314...'
```

`content_hash` is hashed over the **decoded** bytes, so a corpus stored with `text_encoding=zstd` produces the same `content_hash` as the same logical content stored raw. Citations are stable across wire encodings.

`cite` is tier-1: it returns the stored canonical text plus the verifying hashes (`file_hash`, `content_hash`) and the byte span. On a media corpus the span is the one search hits and `retrieve` report: the blob URI and the byte range from the span overlay (0x16), not the row ordinal the forge stores in 0x03. It does NOT reopen the original source bytes; original-byte reopen with a blob-digest verify is net-new tier-2 work that belongs to catalog mode, not the flagship. `ask` and `retrieve` print the same tier-1 stored canonical text, so the answer you get is exactly what `cite` resolves.

## 10. Release verification

```sh
./tool/tasks/fullcheck.sh
```

Runs the full pipeline: the name check (`namecheck.py` and its test), the release build, the Python extension rebuilt with `pyo3/extension-module` (before the Rust suite, whose CLI end-to-end tests load it), the Rust suite in release, Clippy, cargo fmt, the 639-line guard, the nineteen Python suites in `tool/tests/` (pythonapi, ingestion, hashguard, imagepipe, specrules, mediagate, clispaces, askrouter, embedpack, stbackend, nonetwork, catalogue, modelpull, benchmark, preflight, pypiindex, rehearsal, chanprobe, releasepr; `test_modelpull.py` fetches an 18 MB hub test model and skips that case when huggingface.co does not answer), ruff when importable, `presetrun.py` and `benchgate.py` against the committed baseline. Exits non-zero on any failure. `test_stbackend.py` runs its pinned model_hash case everywhere and skips the model cases without the local WeMM-2B snapshot. What it does not run: the flagship e2e tests that need the embed deps (`cli_e2e.rs` skips without them), `test_mediablob.py`, `test_spaceband.py`, the self-tests of the embedders (`test_lexifloor.py`, `test_potiontab.py`, `test_presetmap.py`, `test_clipsnaps.py`, `test_retrieval.py`), and the checks CI adds on top (cargo-deny, cargo-semver-checks, the Windows job, the fuzz smoke).

A change to `.github/workflows/` or `.github/zizmor.yml` also runs `lintworks.yml` in CI, which reads the workflows without running them. The same two checks run locally from the root (`brew install actionlint zizmor`, no Docker):

```sh
SHELLCHECK_OPTS="-S warning" actionlint -ignore 'property "output" is not defined in object type'
zizmor .
```

actionlint is the gate: syntax, `${{ }}` expressions, `needs`, runner labels and the `run:` scripts through shellcheck. The ignored finding is cargo-dist's `steps.cargo-cyclonedx.output.paths` in `release.yml` and `rehearsal.yml`, which are generated and never hand-edited. zizmor is the security audit (template injection, wide permissions, persisted credentials, unpinned actions) and gates the pull request too, forks included: any finding fails it, with an annotation on the line. The findings kept on purpose are in `.github/zizmor.yml`, each with its reason; a new one is fixed, or added there with its reason in the same pull request. On `main`, CI uploads the results to code scanning instead, which closes the alerts a merge fixed.

## 11. Install health check (`urna doctor`)

`doctor` takes no file. It validates the install surface after a one-liner / tarball install (the channels are in the reference section below): urna and format versions, the detected SIMD backend, the Python interpreter the embedder will run under, the numpy + tokenizers deps, the potion embedder script, the potion table (a git-lfs pointer is rejected), and one real offline embed of a fixed probe string.

```sh
urna doctor
```

The exit code is typed so installers and CI branch on codes, not text: `0` ok, `2` Python interpreter missing, `3` Python deps missing, `4` potion embedder script not found, `5` potion table missing or a git-lfs pointer, `6` embedder run failed. A scalar SIMD fallback prints a warning but still exits `0`. The embedder check opens no socket, so doctor itself stays offline-by-construction.

The embedder script resolves in this order: the repo layout (`rust/bridge/python/urna/embed/potionqry.py`, dev checkout), then `<root>/urna/python/urna/` for each data root in turn: `URNA_DATA_DIR`, `XDG_DATA_HOME`, `~/.local/share`, `%LOCALAPPDATA%` (where `installer.ps1` and `urna setup` on Windows put it), `<exe>/../share` (tarball layouts). The interpreter resolves in this order: `URNA_PYTHON`, the venv `urna setup` builds (`<root>/urna/venv`), the nearest `.venv` walking up from the working directory, then `python3` on `PATH`.

A failing doctor names its fix: every code except `6` prints `next: urna setup`, which lays down the payload and the Python env and then re-runs these checks (section 16).

## 12. Model registry and multi-model spaces

Embedding models are DATA, not per-project code: `rust/bridge/python/urna/model/presetmap.py` holds named presets, each declaring what the model is (ids, dim, the VALIDATED matryoshka ladder), what it needs (deps with the exact pip fix line), and how it is used by default (the asymmetric query/document contract). The build spec (§13) selects one or several presets per build.

| Preset | Kind | Dim | MRL dims | Modalities |
|---|---|---|---|---|
| `potion` | Static table (offline, no torch) | 256 | None | Text |
| `clip-vit-b32` | open_clip ViT-B-32/openai | 512 | None | Text, image |
| `siglip2` | open_clip ViT-B-16-SigLIP2/webli | 768 | None | Text, image |
| `jina-v5-omni-nano` / `-small` | sentence-transformers | 768 / 1024 | 32, 64, 128, 256, 512, 768 (small adds 1024) | Text, image, video |
| `minilm-multilingual` | sentence-transformers text, in-process (`rust/bridge/python/urna/embed/searchtxt.py`'s path) | 384 | None | Text |
| `wemm-2b` | sentence-transformers | 2048 | 128, 256, 512, 1024, 2048 | Text, image, video |
| `wemm-4b` / `wemm-9b` | sentence-transformers | 2560 / 4096 | 128, 256, 512, 1024, native | Registered; `--allow-heavy` required |

The MRL column is the validated ladder (`dims=` accepts exactly those values, not a range).

On an installed binary the potion route works out of the box; the other presets need their deps in the setup venv and a local model snapshot. A preset whose install was validated end to end (a pinned hub revision, the exact files fetched at it and the `model_hash` they fingerprint to) is in the model catalog the payload ships, and `urna setup --model <name>` or the explorer's install panel installs it (section 16). `python rust/bridge/python/urna/model/catalogue.py` prints the catalog: today `minilm-multilingual`, and every other preset with the reason it is left out (heavy, model-repo code not reviewed and pinned, no pinned hub revision). For a preset outside the catalog, `urna ask` says what is missing: the packages with the exact install line for the interpreter it ran (`uv pip install --python <venv python> sentence-transformers`), or the model missing from the local cache, fetched once by running the same query with `URNA_ALLOW_DOWNLOAD=1`. `minilm-multilingual` is the model of the pt-BR demo corpus: it embeds through the same functions `search-text` uses, so a corpus built with it before the preset existed keeps its `model_hash`.

Three rules the registry enforces, loudly:

- **MRL is a ladder, not a slider.** `dims=[256]` is accepted only when the preset's model card validates 256 (`mrl.method="prefix_slice_l2"`). Slicing at an unvalidated dim is refused; mathematically possible is not semantically supported.
- **Remote code is an opt-in plus a pin.** Presets with `trust_remote_code` load only when the spec lists them in `output.allow_remote_code` AND every model-repo code file matches the pinned SHA256 allowlist. A hash identifies a version; the opt-in is the consent. Build in an isolated environment when the model dir is not fully trusted. The QUERY side has the same rule: a manifest is data, never an authorization, so `ask`/`retrieve`/`search-text` over a remote-code corpus (and `modelrank.py`) refuse to load the model until the operator opts in with `URNA_ALLOW_REMOTE_CODE="<preset>[,<preset>]"` in the environment.
- **Three hashes, never conflated.** `model_hash` identifies the model (weights + tokenizer + processor + remote code + pooling/normalize/dtype policy). The per-item `input_hash` identifies the content (canonical text ⊕ image bytes ⊕ label ⊕ chunker). The `embedding_recipe_hash` identifies the usage (prompts, query/document modes, preprocess version, `image_max_side`, device class, decoder fingerprint when embedding decoded media). The embed cache key is the triad, so a retranslated text or a re-exported image invalidates exactly what changed.

`siglip2` is pinned to one hub revision of `timm/ViT-B-16-SigLIP2` (the preset's `revision` and `snapshot_files`): the weights and the tokenizer load from `snapshots/<revision>` of the HF cache, never through `refs/main` and never by the hub name, so its text tower runs offline. By name it could not: transformers' AutoTokenizer looks up the repo's absent `config.json` without a commit hash and fails offline even with every file cached. Fetch the pinned files once with `hf download timm/ViT-B-16-SigLIP2 open_clip_model.safetensors open_clip_config.json tokenizer.json tokenizer_config.json special_tokens_map.json --revision eee10eff6dd8cabae2d7f379d4e8cfcd352030aa`, or run the first query with `URNA_ALLOW_DOWNLOAD=1`. A missing file is named, with that line, and `ask` reports it as weights missing. The weights go into open_clip's built-in architecture with the `webli` tag's preprocess, so the `model_hash` is the one a load by tag gives; the tokenizer is outside the fingerprint and comes from the same snapshot. `urna setup --model` does not offer SigLIP2 yet: its installer verifies a file fingerprint, and an open_clip `model_hash` hashes the loaded tensors. `clip-vit-b32` still loads by tag.

Model dirs resolve explicit `model_path` > `URNA_MODEL_DIR_<PRESET>` env > the preset's `local_dir` > the HF cache; a hub download requires `URNA_ALLOW_DOWNLOAD=1` explicitly. Dtype defaults are measured, not assumed: bf16 on CUDA, fp16 on MPS (wemm-2b image embeds 0.5s vs 23s in fp32 on this class of machine), fp32 on CPU; override with `dtype=` in the spec or `URNA_ST_DTYPE`.

## 13. Declarative corpus builds (`urna build --spec`)

One TOML describes the whole corpus; nobody writes a build script per project. `urna build` is a launcher over `rust/bridge/python/urna/entry/specbuild.py` (the build is officially a Python frontend; torch and FFmpeg live there).

```sh
urna build --spec corpus.toml --dry-run     # plan + dep status, loads nothing
urna build --spec corpus.toml --sample 1500
```

`urna build` forwards `--spec` (`.toml`, `.json`, or `.yaml` with pyyaml installed), `--sample N`, `--models a,b` (a subset of the spec's presets; it rewrites the outputs in place, so point a pilot at `--out-dir`), `--out-dir`, `--cache-dir`, `--resume`, `--rebuild-only`, `--dry-run` and `--allow-heavy`. Three flags exist only on the Python tool, `python rust/bridge/python/urna/entry/specbuild.py`: `--strict-env` (a lock divergence under `--rebuild-only` becomes an error), `--seed` and `--json` (the dry-run plan as JSON). `--dry-run` validates the spec and prints the plan without opening the source, so a missing file, a bad query or a non-total `order_by` surface on the real run.

A complete working spec: a SQLite table with per-row images, two models, media behind the dual quality gate, one self-contained output file:

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

The contract highlights:

- **corpus**: `name` and `chunker_version` (required; a new chunker version changes every chunk_id), `title`, `version` (default `0.1.0`, written to the manifest), `reproducible` (default true: the `created` field is pinned so the same inputs give a byte-identical file).
- **source**: `kind` is `sqlite` (`db`, `query`, `[[source.joins]]`, `[source.derive]` helpers, text template whose lines drop when ALL their placeholders are empty, `path_template`/`label_template` for images) | `csv` | `jsonl` (both take `path`) | `image_dir` (`input_dir`, an optional `labels` CSV; every image becomes one row whose text is its identity, and a spec model with `image = "space"` embeds the pixels). PDF pages are not a source kind: render them with `rust/bridge/python/urna/entry/imgcorpus.py --pdf`, or point `image_dir` at the rendered pages. `order_by` must be a TOTAL order; the composite key's uniqueness is verified against the loaded rows, because `ORDER BY x` with duplicate x is not deterministic. Path rule, applied to every string in the spec: `${VAR}` expands from the environment (only the braced form, so a bare `$` in `query` or `template` and the `{col}` placeholders survive), an unset or empty `${VAR}` is a SpecError naming the key (`source.db: ${MTG_DATA} is not set; export MTG_DATA=/path`, never a silent `$` or an empty root), then a leading `~/` expands to home (also when the variable itself holds `~/...`); anything else stays relative to the CWD. The build lock records the expanded strings, and the L3 compare ignores where the spec, the outputs and the inputs live (`spec_path`, `output.dir`, `source.path`, `source.db`, `source.image.path_template`; row content is guarded per item by `input_hash`), so `--rebuild-only` under another data root stays claimable.
- **models**: each `[[models]]` names a preset and its role; `text = "default" | "space" | "none"` (exactly ONE default; it is space 0 of every emitted file, never injected implicitly), `image = "space" | "none"`, `dims = [256, 512]` (one named space per dim: `wemm-2b@256`, each dim on the preset's validated ladder), `space_dtype`, the loading knobs (`model_path`, `device`, `batch_size`, `dtype`), plus the recipe fields (`image_prompt`, `text_query_mode`, `text_corpus_mode`, `image_mode`, `image_max_side`, `normalize`, `preprocess_version`, `encode_kwargs`). The recipe fields enter the embedding recipe hash; `dtype`, `device` and `normalize` also enter the model_hash of a sentence-transformers preset, so a corpus built with a spec `dtype` is asked with `URNA_ST_DTYPE` set the same way.
- **build**: the engine options of the default space, passed to `urna.build` at emit: `preset` (default `"hybrid"`: zstd text, float32 vectors, HNSW and BM25; the other values are section 6's), `dtype` (overrides the preset's vector dtype), `with_graph` (default true here, unlike `urna.build`'s false), `graph_top_m` (8), `graph_space` (only `"default"`), `mrl_dim` (0 = off; it truncates space 0 and is checked only as `0 < K <= dim` and `K % 64 == 0` for int4, not against the model's ladder; for a ladder-validated slice use `dims` on a `[[models]]` space instead). `preset` and `dtype` are validated by `urna.build` after the embed stage, not by `--dry-run`.
- **media** (§14 for the levers): `profile` (a measured recipe resolved into knob defaults; explicit keys always win, and an explicit `[media.quality]` key wins over the profile's quality table key by key), `backend`, `crf` (int or `"auto"`), `tune`, `speed`, `fps`, `gop`, `order`, `shard_size`, `width` (a ceiling on the canvas, default 1024), `pix_fmt`, `dedup` (identical source images stored once; duplicate rows share the frame through the 0x16 overlay); `[media.quality]` (the gate: the floors of §14 plus `strategy`, `buckets`, `sample_per_bucket`, `gate_model`, `crf_ladder`), `[media.cluster]` (`space`, the model that orders frames for `order = "similarity" | "cluster"`, and `threshold`), `[media.jxl_transcode]` (`on_unsupported_jpeg`, `verify_roundtrip`).
- **embedding.image_input**: `mode = "decoded_media"` (default with media: the index describes what the file serves) | `"source"` (measures the model, not the codec). The decoder fingerprint joins the recipe hash in decoded mode; the two modes answer different questions and are never mixed.
- **output**: `mode = "single" | "per-model" | "both"` (one media encode, one embed pass per model, shared across outputs; chunk_ids are content-addressed so citations agree across modes), `provenance = "minimal" | "standard" | "full"` (path/SQL/label redaction. `standard` writes image paths relative to the spec dir and drops nothing else; `full` keeps absolute paths and the SQL; `minimal` drops the SQL and writes `items[]` compact: `key` + `ordinal` per item, `items_compact = true` at the top level, and a `frame_of_row` list only when dedup collapsed rows. Readers get the media frame from `urna.pipes.manifests.frame_resolver` and the items from `manifest_items(manifest, need=(...))`, which refuses a dropped field (`image_path`, `label`, `media_uri`) with an error naming `provenance = "standard"` instead of a KeyError; `modelrank.py` needs `standard` or `full` for its source-image queries. Motivation: 38627 items with all five fields were 12 MB of a 13 to 16 MB manifest. The mode never touches the `.urna` itself, the manifest is a sidecar), `allow_remote_code`, `embed_media = true|false` (inline the encoded media into the `.urna` itself, section 0x17, so the corpus is ONE self-contained file with no media sidecar at read time; `urna media <file>` lists the blobs, `urna media <file> --export DIR` writes them back out hash-verified, and `urna validate` proves every inlined blob against its `blob_refs` SHA256. The sidecar `.media/` dir remains on disk as the build cache; peak build memory is roughly twice the media bytes, so prefer sidecar mode for very large corpora), `cache_dir` (root of the shared embed cache, see the transactional paragraph below; default `URNA_CACHE_DIR`, else `${XDG_CACHE_HOME:-~/.cache}/urna`).

Every build emits `<name>.manifest.json` (`manifest_schema_version = 1`, canonical serialization; a versioned contract, not an ad-hoc log) and `<name>.build.lock.json` (package versions, tool binaries with SHA256, model hashes, the materialized spec). Reproduction has three declared levels: L1 = same top-k anywhere; L2 = per-vector cosine within 1e-5 on the same device class; L3 = byte-identical `file_hash`, claimable ONLY under a matching lock (`--rebuild-only` re-emits from the triad-keyed caches and compares the lock, which it never rewrites: a divergent rebuild prints the divergence as a warning and writes its own lock beside it as `<name>.build.lock.rebuild.json`, so the two can be diffed; `specbuild.py --strict-env` turns the divergence into an error instead). `--resume` reuses the media stage when its state file matches; rows are reloaded and the embed caches consulted on every run. Builds are transactional: per-stage state under `<out>/.forge-state/`, outputs staged in `<out>/.tmp/` and committed by atomic rename (same filesystem, so both stay in the output dir), `--resume` continues from the last intact stage. Embed caches live OUTSIDE the output dir, under `${XDG_CACHE_HOME:-~/.cache}/urna/embed/<preset>/<triad>.npz` (override with `[output] cache_dir` in the spec, `--cache-dir` on `urna build` or `specbuild.py`, or `URNA_CACHE_DIR`; `--dry-run` prints the resolved root; the location is not identity, so it never enters the lock and any override source claims L3 against the same root): the file name is a hash of the triad plus the arrays the spec needs, so two specs with the same rows and model read one entry whatever their output dir, a changed knob adds a sibling entry instead of overwriting, and a text-only model's entry does not change with the media CRF (the decoder fingerprint enters only image-space recipes). The `model_hash` probes sit beside them under `models/`, keyed by preset plus the knobs that enter the fingerprint (normalize, dtype, device, model_path), and the loaded model corrects a probe that disagrees. Caches are flock'd with checksum sidecars, and a torn cache is recomputed, never reused. Output dirs built before this layout keep an orphaned `<out>/.cache/` that nothing reads; delete it by hand. The `<name>.media/` sidecar is not a cache in sidecar mode: it is the served media, and it stays beside the `.urna`.

## 14. Dataset compression levers and the dual quality gate

The media section is where the compression research became knobs. All decisions land in the manifest and provenance:

- `tune = "still"` (the default since 2026-09-13; `"default"` keeps SVT-AV1's own tune): SVT-AV1's still-picture tune, PROBED against the local encoder (the numeric value varies by version); unsupported ⇒ stderr warning + recorded fallback, never a silently ignored flag. Measured on 2048 cards at CRF 35: ssimulacra2 p50 62.7 against 51.8 for the default tune, for +10% bytes. The tune is part of the media state key, so an AV1 build that relied on the old implicit default re-encodes once on `--resume`; a spec that sets `tune` explicitly is unaffected. `speed = 6` buys quality per byte over the default 8 at ~2x encode time. `fps` changes playback timestamps only (frames are 1:1 with items; verified, no duplication).
- `crf = "auto"`: the dual gate. A stratified sample (deterministic, versioned bucket heuristics: resolution / entropy / has_text / alpha / source_format) is encoded at every ladder CRF and must clear BOTH floors: ssimulacra2 per-bucket p10 ≥ `visual_floor_p10` and global min ≥ `visual_floor_min` (a global average would let one whole stratum degrade), and embedding drift p10 ≥ `drift_floor_p10` measured by the declared `gate_model`; an image can look fine to humans and still move in retrieval space, which is what the corpus actually serves. The largest passing CRF wins; none passing ⇒ smallest + a loud warning. The defaults are floors a real corpus reaches: `visual_floor_p10 = 60`, `visual_floor_min = 45`, `drift_floor_p10 = 0.95`, `crf_ladder = [25, 30, 35, 40, 45, 50]`, set from the mtgdataset cards at 488x680 yuv420 (2048 sample, AV1 still speed 6): crf30 measures ssimulacra2 p10 65.3, min 58.6, drift p10 0.967 and passes; crf35 measures p50 62.7, p10 55.7, min 45.3, drift p10 0.965 and fails on p10, so the default picks crf30 instead of warning. The earlier floors (p10 85, min 72, drift 0.98, ladder [30, 35, 40, 45]) were set for large photos and no rung reached them on the cards, so the gate always fell back to the smallest CRF; they are one `[media.quality]` override away. The gate encodes the ladder with the AV1 stream, so `crf = "auto"` is a SpecError naming `media.crf` on the AVIF and JXL backends (an AV1 CRF is not an avifenc -q). Full retrieval recall lives in the sweep, outside this loop, where it costs O(1) per variant. A negative `drift_floor_p10` disables the drift leg (recorded as `drift_pass = true` in the report).
- `[media.quality]` task-utility floor, the third leg for corpora that only serve retrieval: `utility_floor_hit1` (negative = off, the default; 0..1 enables it), `utility_queries` (how many sampled items become queries, 0 = every sampled item, evenly spaced so every stratum keeps a share), `utility_query_template` (rendered per item with `{label}`, the row's image label, or its canonical text when the row has no label), `utility_tol` (allowed hit@1 loss against the lossless source). One text query per item is embedded once by the gate model's text tower (a gate model without one is a SpecError naming `utility_floor_hit1`), and at every ladder CRF text-to-image hit@1 is measured against the decoded frames of the same sample; the same hit@1 against the source frames is the lossless reference. A rung passes the leg when `hit1_decoded >= max(utility_floor_hit1, hit1_source - utility_tol)`. The manifest's `media.crf_auto.ladder` carries `hit1`, `visual_pass`, `drift_pass`, `utility_pass` per rung next to `drift_p10`, and `media.crf_auto.utility` carries `hit1_source`, the threshold and the query count, so a refused rung says which leg refused it. Motivation, measured 2026-09-12 on 38627 cards (experiment 13 of mtg-urna-benchmark): CLIP cosine drift p10 falls 0.932 at crf40 to 0.829 at crf60 and the default drift floor vetoes every rung, while text-to-image hit@1 on 100 queries does not move up to crf50. Drift is a stability signal; utility is the floor a retrieval profile needs.
- `dedup = true`: content-hash dedup of source images; n rows → one frame via the span overlay, zero format change. On the full Scryfall printings set the potential is ~48% of the media (100,452 printings, 51,870 unique arts).
- `order = "cluster"`: greedy cosine clustering (deterministic tie-breaks) makes near-duplicates adjacent so per-segment inter coding has something to predict; measured before recommended; on a 1-per-card corpus the honest expectation is ~0, and on the same-artwork reprint corpus it is -29% (2026-08-31, g=16 + scd=0 vs all-intra).
- `gop = "auto"` with sharding probes PER SEGMENT: each `shard_size` chunk runs its own intra-vs-inter probe encode and ships its own keyint (recorded per segment in the manifest, `gop.per_segment = true`). A single global probe averages regimes away; with `order = "cluster"` the near-duplicate runs concentrate in a few segments, which decide inter (bounded GOP, keyint=16, scene-change detection off), while unique segments keep O(1) all-intra access. Forced `gop = "intra" | "inter"` still applies to every segment alike.
- `profile`: dataset-type presets resolved BEFORE explicit keys (an explicit key always wins, so no other use case is closed off). `"near-dup"` = cluster ordering + per-segment GOP + still tune (visually similar corpora: card reprints, video frames, scans); `"stills"` = `backend = "avif"` + `crf = 48` + `speed = 8`, one libaom AVIF per image (unique images: O(1) per-image access with no video decode; measured 2026-09-12 on 38627 cards, experiment 11 of brennercruvinel/mtg-urna-benchmark: 1195973116 B against 1374431484 B for the all-intra AV1 stream, 13% less at matched ssimulacra2 mean 61.96 on the 2048 sample; the cost is a 4 to 10x slower CLIP embed at build time because frames are decoded one AVIF at a time); `"stills-av1"` = the previous stills recipe, all-intra + still tune on the AV1 stream (one file per shard, FFmpeg decode, and the profile to pair with `crf = "auto"`); `"archive"` = jxl-transcode (byte-reversible, for corpora where loss is not acceptable); `"retrieval"` = all-intra + still tune + `speed = 6` + fixed `crf = 50`, for a corpus that only serves search and never shows its pixels (measured 2026-09-03 on 38627 cards: 532671548 B self-contained, 7.46x vs the JPEG source, no measurable txt@1 loss on 100 queries; the default drift floor at p10 0.942 would have vetoed it, which is why the profile pins CRF instead of running the gate); `"retrieval-auto"` = the same recipe with `crf = "auto"` gated by task utility alone: visual and drift floors disabled (`-1e9` / `-1.0`), `utility_floor_hit1 = 0.0`, `utility_tol = 0.02`, ladder `[40, 45, 50, 55, 60]`, so the largest CRF whose hit@1 stays within 0.02 of the lossless source wins (the gate model needs a text tower: CLIP, SigLIP2, wemm, Jina). The resolved knobs land in the manifest's `media` block; the profile name is in `<name>.build.lock.json` (`resolved_spec.media`).
- `backend = "jxl"` / `"jxl-transcode"`: the ONLY truly lossless modes. `jxl` is lossless of the source pixels; `jxl-transcode` repacks JPEGs reversibly (about 10% smaller: 1.10x on the 38627-card corpus, round-trip verified by reconstructing the JPEG and comparing SHA256). Non-transcodable inputs follow `on_unsupported_jpeg = error | copy-source | lossless-jxl`, per-file decisions recorded. Preservation contract: decoded pixels (JXL) / original JPEG bytes (verified transcode); EXIF/ICC/XMP only with `keep_metadata`; timestamps and filenames live in the manifest. Needs `cjxl`/`djxl` (`brew install jpeg-xl`, which also ships `ssimulacra2` for the gate).

Measure everything with `tool/bench/imagerate.py` (variants now include `av1-tune`, `jxl`, `jxl-transcode`) and compare models with the three-tier `tool/bench/modelrank.py`: T1 pipeline stability (identity self-retrieval, inflated by construction and labeled as such), T2 codec cost (embedding drift), T3 task utility (label-template text→image as declared weak ground truth, plus `--queries-file` with real operator queries: hit@k, MRR, negative leakage). The tiers answer different questions and are never aggregated into one number.

## 15. Media blobs (`urna media`)

A corpus built with `[output] embed_media = true` (§13) carries its encoded media inside the file (section 0x17, an offset table parallel to the `blob_refs` records plus the raw bytes). `urna media` is the read side:

```sh
urna media corpus.urna                 # one line per blob: index, sha256, byte length, inlined or sidecar, original uri
urna media corpus.urna --export DIR    # write every inlined blob to DIR, verifying each against its blob_refs sha256
```

`--export` fails on the first blob whose bytes do not hash to the recorded `content_hash`; `urna validate` performs the same proof over every inlined blob without writing anything, and runs it before it prints its `OK:` line: a blob that fails leaves nothing on stdout and exits 1. The Python side reads one blob without exporting the store: `UrnaFile.blob_bytes(i)`. The section is content_hash-excluded, so an embedded corpus and its sidecar twin carry the same citations.

## 16. Setup and the terminal explorer (`urna setup`, `urna tui`)

The package channels (Homebrew, npm, crates.io) ship the bare binary; the binary cannot carry the ~30 MB embedder payload or a Python env. `urna setup` lays both down and proves the install:

```sh
urna setup
```

On a terminal it runs inline, in four steps with the screen left in the scrollback when it ends:

1. Scan: how urna got here (Homebrew, npm, cargo, install script, dev build), the data dir, whether the payload and a Python with numpy + tokenizers are present, and which tools the steps can use (`curl`, `uv`, a `python3` with `venv`). Read-only.
2. Plan: one checkbox per step (payload, Python env, models, verify). A satisfied step starts unticked (ticking it reinstalls); a step this machine cannot run is shown blocked with the reason. The payload counts as satisfied only when setup's own data dir holds one from the release it wants (this binary's, or `--version`), read from `<data root>/urna/VERSION`: a payload from another release, or from before the stamp (0.5.1 and older), is replaced, so upgrading the binary and running `urna setup` upgrades the scripts it runs. A payload missing a required file is reinstalled the same way, whatever its stamp. The venv next to it is never touched.
3. Install: the payload (`urna-embedder-payload.tar.gz` from the release that matches the binary's version, its SHA256 checked while it streams, unpacked into a staging dir, refused if any required file is missing, then laid down as one step: `python/` (the `urna` package with the query embedders and the potion table), then `VERSION` last; if anything fails the previous payload is put back as it was, and if putting it back fails too its files are kept at `<data root>/.urna-previous-<pid>-<time>` and the error names that path) and the Python env (`<data root>/urna/venv`, built with `uv` when it is on `PATH`, else `python3 -m venv` + pip), with the tools' output live in a log.
4. Verify: the doctor checks (section 11), then what to run next, or on failure each step's error and how to retry.

```sh
urna setup --yes                     # the default plan, no questions, plain lines (ci, scripts)
urna setup --version v0.5.3          # the payload of a given release (replaces one from another)
urna setup --force                   # reinstall the payload even when it matches
urna setup --no-python               # payload only; bring your own interpreter via URNA_PYTHON
urna setup --model minilm-multilingual  # also install a catalog model (repeatable; `all` picks every offered one)
urna setup --uninstall               # remove the payload and the env (never the binary, never the hf cache)
```

Without a terminal on both ends (a pipe, CI, a postinstall hook) setup behaves as `--yes`. Exit codes extend doctor's: `0` ready, `2`..`6` a doctor check failed after the steps ran, `10` download failed, `11` the payload does not match the release checksum (nothing is installed), `12` unpack failed, `13` the Python env failed, `14` a step the machine needs is blocked here (no `curl`, no `uv` and no `python3`, no data dir, `URNA_PYTHON` pins an interpreter without the deps, or a chosen model the catalog leaves out, whose repo code was not allowed, or whose packages would have to go into an env setup does not manage), `15` a model install failed.

### Models

The payload carries `urna/model/catalogue.json`, generated from the registry (section 12): the models setup can install, each with the hub repo and the pinned revision, the exact files fetched at it, their size, the pip packages its backend needs and the `model_hash` the files must fingerprint to. On the plan screen `m` (or space on the models step) opens a picker: a checkbox per offered model with what it adds and downloads, `a` for every one, and below them every preset the catalog leaves out with the reason. After a fresh install the catalog has only just arrived, so the last screen offers `m` too. With `--yes`, `--model <name>` chooses (a catalog name or the manifest model name; repeatable; `all` is every offered model).

Choosing a model is the consent to download it: the plan (or the flag) names the size and the source before anything moves. A model that runs code from its own repo needs a second, separate consent: `r` in the picker or `--allow-remote-code <name>`; without it the step stays blocked. The install, the same one the explorer's panel runs:

1. The model's packages go into the venv setup manages and nowhere else (`uv pip install --python <data root>/urna/venv/bin/python`, or that env's pip). When models are chosen and that venv is missing, the Python step builds it, even if another Python already has numpy and tokenizers. An `URNA_PYTHON` pin elsewhere blocks the step: urna does not install into an interpreter it does not manage.
2. `urna/model/installer.py` fetches exactly the catalog's files at the pinned revision into the shared Hugging Face cache (`$HF_HOME/hub`, default `~/.cache/huggingface/hub`), the cache the query embedders read offline, reporting progress as it goes. It points the repo's `refs/main` at the pin when the cache has none; a `refs/main` that already names another revision belongs to whoever fetched it and is left alone, and the install fails naming both.
3. The files are fingerprinted: anything but the catalog's `model_hash` fails the install (and, from the explorer, anything but the corpus's own).

A failed model does not stop the next one. `--uninstall` leaves the Hugging Face cache alone: it is shared with every other tool that reads it.

The binary links no network stack: setup downloads through the system `curl` (HTTPS only, no downgrade on redirect), the same tool `installer.sh` uses, and `URNA_RELEASE_BASE` points it at a mirror or a `file://` directory for air-gapped machines.

The explorer opens a `.urna` and shows it: the manifest and the verdict of the same checks `urna validate` runs, the section table with a size bar per section, an ask tab that embeds offline through the same routed embedder and `model_hash` gate as `urna ask` and shows each hit's stored text, citation and source, and the doctor checks.

When a query fails because the corpus's model needs packages or weights this machine lacks, and the payload's catalog offers that model, the ask tab opens an install panel instead of an error: the packages it adds to the managed venv, the megabytes it fetches from huggingface.co at the pinned revision, and the `model_hash` it must come out as, checked against the corpus's. `y` installs (the download consent), `r` allows the model's repo code (the separate consent, when it runs any), `n` or `esc` declines; the install streams its steps, a progress bar and the log into the panel, a failure keeps the facts and offers a retry, and success runs the query again. When queries do not run the managed venv (an `URNA_PYTHON` pin, or no setup yet) the panel names the reason and the `urna setup --model` line instead of installing anything.

```sh
urna tui corpus.urna
urna                                 # a bare urna on a terminal opens the explorer
```

| Key | Where | Action |
|---|---|---|
| `tab` / `shift+tab`, mouse click | Everywhere | Next / previous tab |
| `o` (`ctrl+o` in ask) | Everywhere | Open a `.urna` (dirs and `.urna` files only) |
| `1`..`9` | Home | Open a `.urna` found in the working dir or one level below |
| `enter`, `↑↓`, `pgup/pgdn`, `esc` | Ask | Ask, pick a hit, scroll its text, clear the query |
| `y`, `r`, `n` / `esc` | Ask, install panel open | Install (download consent), allow repo code (separate consent), decline |
| `r` | Health | Re-run the checks |
| `s` | Everywhere but ask | Hand the terminal to `urna setup`, then come back with the same corpus open |
| `q` (`ctrl+q` in ask), `esc` on home | Everywhere | Quit |

Both screens follow the terminal: 24-bit color when the terminal advertises it (`COLORTERM=truecolor` and the known truecolor terminals), the nearest xterm-256 color otherwise (Terminal.app, tmux without `Tc`), and no color with `NO_COLOR` set or `URNA_COLOR=none`. `URNA_COLOR=truecolor|256|none` overrides the probe. `cargo install urna --no-default-features` builds the CLI without the `tui` feature: every engine and agent verb, no `setup`, no `tui`.

## Reference

Every way to get `urna` onto a machine, what each channel lays down, how to verify what you got, and what a maintainer has to set up once before a release can feed these channels. Each item is collapsed; open the one you need.

Status: the release pipeline (`.github/workflows/release.yml` via cargo-dist, which also builds the wheels through `wheelmake.yml` and publishes them through `pypiindex.yml`, plus `.github/workflows/setuptest.yml`; up to 0.5.3 these were `build-wheels.yml`, `pypi.yml` and `install-test.yml`) serves from `v0.5.0` (2026-09-26) on; `v0.4.0` was never tagged and `v0.3.0` predates the pipeline, so neither carries artifacts. The maintainer checklist below is what each channel needs on the account side; a channel whose prerequisite is missing fails its own job and leaves the GitHub release intact.

The product is offline by construction: the installers are the only thing that ever opens a socket (`urna setup` does it through a `curl` child process). After install, `urna setup` completes the channel and `urna doctor` validates the surface without network.

### Environment variables

Every `URNA_*` variable read anywhere in the codebase (installers, CLI, forge, dev scripts), in one place. Channel sections below mention the ones relevant to that channel; this table is the source of truth.

| Variable | Scope | Default | What it does |
|---|---|---|---|
| `URNA_RELEASE_BASE` | Install, setup | GitHub release URL | URL prefix `installer.sh` / `installer.ps1` fetch the four release files from, and `urna setup` the payload from (`file://` works for air-gapped installs) |
| `URNA_BIN_DIR` | Install | `~/.local/bin` (`~\.local\bin` on Windows) | Where the installer puts the `urna` binary |
| `URNA_DATA_DIR` | Install, setup, runtime | `${XDG_DATA_HOME:-~/.local/share}` (`%LOCALAPPDATA%` on Windows) | Parent dir for the embedder payload and the `urna setup` venv; the CLI searches it first (section 11) |
| `URNA_PYTHON` | Runtime, dev | The `urna setup` venv, else the nearest `.venv`, else `python3` | Python interpreter the CLI, `urna doctor`, and the dev scripts shell out to; must carry the forge deps (numpy, tokenizers). It wins over the setup venv, so setup reports it as blocking when it lacks the deps |
| `URNA_COLOR` | Runtime | Probed from the terminal | `truecolor`, `256` or `none`: color depth of `urna setup`, `urna tui` and the colored `doctor` output |
| `NO_COLOR` | Runtime | Unset | Any value turns every color off (no-color.org); modifiers stay |
| `URNA_FORCE_SCALAR` | Runtime | Unset | Forces the scalar SIMD kernel over AVX2 / NEON, for A/B benchmarking |
| `URNA_ALLOW_DOWNLOAD` | Runtime, build | Unset (offline) | Lets `search-text`, `searchtxt.py`, `modelhash.py`, and the corpus builder fetch a sentence-transformers model instead of failing offline, and `tool/tasks/benchdata.py` fetch the benchmark corpus `fullcheck.sh` measures; `urna setup --model` and the explorer's install panel set it for their own fetch only, after the download was confirmed |
| `URNA_ALLOW_REMOTE_CODE` | Runtime, build | Unset (empty) | Comma-separated preset names allowed to load `trust_remote_code` model-repo code (`ask` / `retrieve` routing, `modelrank.py`, `uibackend.py`) |
| `URNA_ALLOW_HEAVY` | Runtime | Unset | Allows an executable / heavy embedder preset in `presetqry.py` |
| `URNA_CACHE_DIR` | Build | `${XDG_CACHE_HOME:-~/.cache}/urna` | Forge's triad-addressed embed cache root (declarative builds, section 13) |
| `URNA_ST_DEVICE` | Build | `auto` | sentence-transformers device override (`cpu`, `mps`, `cuda`) for the forge embed workers |
| `URNA_ST_DTYPE` | Build | Unset (model default) | sentence-transformers dtype override for the forge embed workers |
| `URNA_MODEL_DIR_<NAME>` | Build | Unset | Local dir override for a registry preset, e.g. `URNA_MODEL_DIR_WEMM_2B`; wins over the HF cache, loses to an explicit `--model-path` |
| `URNA_ENABLE_FAKE_PRESET` | Test-only | Unset | Unlocks the `fake-test` model preset used by the registry's own test suite |
| `URNA_MUTATION_ITERS` | Dev | `1500` | Iteration count for the mutation-fuzz harness; raise for a soak run |
| `URNA_FUZZ_SEED_DIR` | Dev | Unset | Seed corpus dir override for the mutation-fuzz harness |
| `URNA_FUZZ_TARGETS` | Dev | `urna-view section-decoders runtime-indexes mmap-open-search` | Space-separated cargo-fuzz targets `tool/tasks/fuzzsweep.sh` runs |
| `URNA_QUERIES` | Dev | `100` | Query count `presetrun.py` uses via `fullcheck.sh` |
| `URNA_K` | Dev | `10` | Top-k `presetrun.py` uses via `fullcheck.sh` |
| `URNA_OUT` | Dev | `/tmp/fullcheck_post.json` | Where `fullcheck.sh` writes the post-run measurement JSON |
| `URNA_CORPUS` | Dev | The benchmark corpus | The corpus `presetrun.py` measures via `fullcheck.sh`. Unset, it is `fakenews.urna` (MiniLM, exact) of `brennercruvinel/fakenews-ptbr-urna-benchmark` at a pinned commit, in the Hugging Face cache, checked against its sha-256; `tool/tasks/benchdata.py fetch` downloads it only with `URNA_ALLOW_DOWNLOAD=1` |
| `URNA_BASELINE` | Dev | `tool/bench/reference.json` | The metrics `benchgate.py` compares against via `fullcheck.sh`, measured on the default corpus. They must come from the corpus the run measures: before any step, `fullcheck.sh` stops with exit 9 when the file is missing or unreadable and with exit 10 when its `baseline_file_hash` is not the corpus's sha-256 (`tool/tasks/benchdata.py match`); exit 3 stays the uncached corpus |
| `URNA_FRESH` | Dev | `0` | `1` makes `fullcheck.sh` run every step, ignoring the steps that already passed with the same inputs |

<details>
<summary>One-liner (Linux, macOS)</summary>

```sh
curl -sSf https://raw.githubusercontent.com/hoffresearch/urna/main/tool/tasks/installer.sh | sh
urna setup
```

`tool/tasks/installer.sh` (POSIX sh; needs `curl`, `tar`, and `sha256sum` or `shasum`):

1. Detects the platform and maps it to a release target: `x86_64` / `aarch64` times `unknown-linux-musl` / `apple-darwin`.
2. Downloads four files from the GitHub release: `urna-<target>.tar.xz`, its `.sha256`, `urna-embedder-payload.tar.gz`, its `.sha256`.
3. Verifies both SHA256 sums before anything touches the install dirs. A mismatch aborts with the two hashes printed.
4. Installs the binary to `~/.local/bin/urna` and extracts the payload to `${XDG_DATA_HOME:-~/.local/share}/urna/python/urna/` (the potion embedder script plus its vendored table).
5. Warns if `~/.local/bin` is not on `PATH`, and points at `urna setup`, which then only has the Python env left to build.

| Flag | Effect |
|---|---|
| `--version vX.Y.Z` | Pin a release (default: latest). A bare `X.Y.Z` gets the `v` prepended |
| `--uninstall` | Remove the binary and the payload dir |

`URNA_RELEASE_BASE`, `URNA_BIN_DIR`, and `URNA_DATA_DIR` override the install paths; see the environment variables reference above.

The Linux binaries are static musl, so they run on any distro and inside `scratch` containers. `urna doctor` needs a Python 3.12+ interpreter with `numpy` and `tokenizers` for the offline embed probe (`URNA_PYTHON` selects the interpreter); so do `ask`, `retrieve`, `search-text` and `build`, which spawn the embedder or the forge. The other engine verbs run without Python, which is what the Docker image serves.

</details>

<details>
<summary>Windows</summary>

```powershell
irm https://raw.githubusercontent.com/hoffresearch/urna/main/tool/tasks/installer.ps1 | iex
urna setup
```

`tool/tasks/installer.ps1` mirrors the shell installer: `-Version vX.Y.Z`, `-Uninstall`, the same `URNA_RELEASE_BASE` / `URNA_BIN_DIR` / `URNA_DATA_DIR` overrides. The binary goes to `~\.local\bin\urna.exe`, the payload to `%LOCALAPPDATA%\urna\python\urna\`. The payload is a `.tar.gz`; `tar` ships with Windows 10 1803+. The Windows archive is a `.zip`.

</details>

<details>
<summary>Python package `urna`</summary>

```sh
pip install "urna[embed]"            # library + offline potion embedding
uvx --from urna urna validate file.urna
```

One `cp312-abi3` wheel per platform: Linux x86_64, Linux aarch64, macOS universal2, Windows amd64. Python 3.12+. The wheel carries `urna._urna` (the PyO3 extension), the `urna` package, and the bundled potion table (~30 MB) under `urna/model/potionb8m/`. The table is bundled on purpose: the installed package embeds offline by construction, so there is no lazy-fetch path to fail in an air-gapped environment. The `embed` extra adds `numpy` and `tokenizers`; the core surface (`urna.open`, `search`, `retrieve`, `validate`, `inspect`) needs nothing beyond the wheel.

The console entry point `urna` installed by the wheel is the read-only subset (`validate`, `inspect`, `stats`, `search`) over the library API, which is what makes `uvx --from urna urna ...` work. The full CLI (`ask`, `retrieve`, `build`, `doctor`, `media`, the ANN/graph/space searches) is the Rust binary from the one-liner, Windows, Homebrew and binstall items.

The wheel is staged by `tool/tasks/wheelprep.py` into `pkgs/stage/` (gitignored) from `pkgs/wheel/pyproject.toml`, and built by maturin in `wheelmake.yml` inside the release run; `pypiindex.yml` publishes those same files. The dev flow (`cargo build` + copy the `.so`) is unchanged and does not install a package.

</details>

<details>
<summary>Homebrew</summary>

```sh
brew tap hoffresearch/urna
brew install urna
urna setup
```

`brew install hoffresearch/urna/urna` does the same in one line. The formula lives in the `hoffresearch/homebrew-urna` tap and is generated by cargo-dist on every release (`installers = ["homebrew"]` in `Cargo.toml`). The generated formula installs the binary only; `urna setup` lays the payload and the Python env down under the data dir (nothing brew owns is touched), so `brew upgrade` and `brew uninstall` keep working as usual. `urna setup --uninstall` removes what setup added.

</details>

<details>
<summary>npm</summary>

```sh
npm install -g @urna/cli
urna setup
npx @urna/cli validate file.urna
```

The other Node package managers install the same package:

```sh
bun add -g @urna/cli
pnpm add -g @urna/cli
yarn global add @urna/cli   # yarn 1; yarn 2+ has no global installs: yarn dlx @urna/cli
```

bun and pnpm 10 block the package's postinstall by default. Nothing to allow: when the binary is missing, the `urna` wrapper downloads it on the first run.

The `urna` command these four install is a Node script (`#!/usr/bin/env node`, the launcher cargo-dist generates), which finds the real binary and runs it as a child: every call pays a Node start (about 40 ms on a laptop against 3 ms for the binary itself), and a machine with bun but no Node gets `env: node: No such file or directory`, exit 127, from `bun add -g` as much as from the others. Two ways out: run the launcher under bun's own runtime, `bunx --bun @urna/cli <verb>`, which needs no Node; or take the binary from a channel that ships it as is (the one-liner, Homebrew, `cargo binstall urna`, the release archive). The setuptest job's bun rows pass because GitHub's runners come with Node installed.

The `@urna/cli` npm package is generated by cargo-dist on every release (`installers = ["npm"]`, `npm-scope = "@urna"` in `Cargo.toml`, `npm-package = "cli"` in `rust/clitui/Cargo.toml`). npm refuses the bare name `urna` as too similar to existing packages (URL, lerna, ora). It carries no binary of its own: on install it fetches the release archive for the platform from the GitHub release and exposes it as the `urna` command. The package ships the binary only: `urna setup` completes it (npm hides postinstall output, so setup is a step you run, not a hook).

</details>

<details>
<summary>crates.io and cargo binstall</summary>

```sh
cargo binstall urna     # prebuilt binary from the github release
cargo install urna      # compile from crates.io
```

`urna-format`, `urna-engine` and `urna` (the CLI crate, binary `urna`) are published to crates.io on every release by `.github/workflows/rustready.yml`, in that order; `urna-bridge` ships as the wheel and is not a crate. A Rust project that reads or writes `.urna` files depends on `urna-format` (container) and `urna-engine` (search); `urna-engine` replaces `urna-runtime`, whose last version is 0.5.3. `[package.metadata.binstall]` in `rust/clitui/Cargo.toml` maps the crate to the cargo-dist archive names (`.tar.xz`, `.zip` on Windows), so binstall downloads the released binary instead of compiling. Either way the binary comes alone: `urna setup` completes it. To build the unreleased tree instead: `cargo install --git https://github.com/hoffresearch/urna urna`.

</details>

<details>
<summary>Docker</summary>

```sh
docker build --platform=linux/amd64 -t urna .
```

The root `Dockerfile` builds the static musl binary in a throwaway toolchain stage and copies it into `scratch`: no shell, no package manager, no network at runtime. The corpus arrives as a mounted volume, so the same image serves air-gapped hosts. On Apple silicon build the aarch64 variant natively (`--build-arg TARGET=aarch64-unknown-linux-musl`); QEMU user emulation crashes rustc mid-build. The binary is the engine-only CLI, built with `--locked` (the dependency versions of `Cargo.lock`) and `--no-default-features` (no terminal UI, so no `setup` or `tui`), from a toolchain image pinned by tag and digest. The image has no Python, so `ask`, `retrieve`, `build`, `search-text` and `doctor`'s embed check do not run inside it; the other engine verbs (file + vector in) do.

</details>

<details>
<summary>Dev build</summary>

```sh
cargo build --release --workspace
cargo build --release -p urna-bridge --features pyo3/extension-module
cp target/release/lib_urna.dylib rust/bridge/python/urna/_urna.so   # macos (.so on linux)
```

Rust edition 2024 (`rustc >= 1.88` for the CLI crate, 1.85 for the format, runtime and Python crates), Python 3.12+. The potion table is git-lfs: `git lfs pull` before `urna doctor` or any `ask` / `retrieve`, a pointer file is rejected with exit `5`. Setup details, hooks and the merge gate are in `docs/CONTRIBUTING.md`.

</details>

<details>
<summary>Verification</summary>

What every release asset carries: the five binary archives and the embedder payload each ship a per-file `<name>.sha256`; `sha256.sum` covers the five archives and the npm package; the SBOM (`urna.cdx.xml`), the Homebrew formula and the npm package have no `.sha256` of their own. The installers verify the archive and the payload before writing; by hand:

```sh
sha256sum -c urna-x86_64-unknown-linux-musl.tar.xz.sha256
gh attestation verify urna-x86_64-unknown-linux-musl.tar.xz --repo hoffresearch/urna
```

The attestation is Sigstore keyless provenance produced in the release job (`attestations: write`, `actions/attest`), binding the archive digest to the workflow, the commit, and the tag. The five archives are attested in the per-target build job; the payload, the SBOM, the formula and the npm package are built in the global job and carry no build attestation (`gh attestation verify` answers 404 for them; `gh release verify-asset` only proves an asset is the bytes GitHub stored).

The release workflows as they stand also build the four wheels in the release run: `wheelmake.yml` attests them and attaches them to the release with a `.sha256` each (`gh attestation verify urna-<version>-cp312-abi3-<platform>.whl --repo hoffresearch/urna`; the signer workflow is `wheelmake.yml`), and `pypiindex.yml` uploads those same files after checking each against its `.sha256` and its attestation, with OIDC trusted publishing and PEP 740 provenance. v0.5.4 is the first release through that path (release run 37255482415, `pypiindex.yml` run 37255752641): its GitHub release carries the four wheels with a `.sha256` and an attestation each, and PyPI serves the same four files with PEP 740 provenance. Earlier releases predate it: the GitHub releases up to v0.5.3 carry no wheels, and 0.5.0 to 0.5.3 reached PyPI on a token with no PEP 740 provenance (the integrity endpoint answers 404 for them).

Every release also carries one CycloneDX SBOM (`urna.cdx.xml`, the CLI package's dependency tree, generated by `cargo cyclonedx` in the global job, not attested), and the binaries are built with `cargo auditable`, so the dependency tree can be read back out of the executable:

```sh
cargo audit bin ~/.local/bin/urna
```

Commits on `main` are SSH-signed and the branch ruleset requires verified signatures. Release tags are annotated and SSH-signed too: `.github/workflows/tagverify.yml` checks the tag against `.github/trustkeys` as the first job of `wheelmake.yml` inside the release run, and again before a PyPI upload. An unsigned tag, a lightweight tag, a signature from a key not on that list, or a tag that names another version than the workspace's, points off `main`, or predates the release date (`tool/tasks/preflight.py --tag`) fails that job: the wheels are not built, the host job refuses to create the GitHub release, and no publish job runs. The archive builds run beside the check and may finish, but nothing they build is released.

`.github/workflows/setuptest.yml` runs inside the release run once it is announced and installs the product the way a user does, in three jobs: `cli`, the one-liner against the release URL on Linux x86_64 / aarch64, macOS arm64 / x86_64 and Windows, then `urna validate` on the golden fixture and `urna doctor`; `channels`, eleven legs over Homebrew (macOS, Linux), npm (Linux, macOS, Windows), bun (Linux, macOS), pnpm (Linux, macOS), yarn and cargo binstall, each ending in `urna setup --yes`, `urna doctor` and `urna validate`; and `wheel`, which pip-installs the published wheel on four platforms and runs the `uvx` entry point. Before installing from a registry, each leg waits for the exact version there (`tool/tasks/chanprobe.py wait`, up to 15 minutes: npm's version document, the crates.io sparse index, PyPI's version JSON), so a version still propagating is waited for instead of failing the leg; every leg then checks that `urna --version` is exactly the tag's version, Homebrew included, and the Windows leg checks the exit code of each command on its own. A failure there means users of that channel get a broken install: read the release report below, fix the cause on `main`, and decide per channel. A crates.io version is permanent and PyPI refuses a second upload of a version, so the fix is usually the next patch release; yank only a version that harms the users who install it.

`.github/workflows/runreport.yml` writes one summary per release when the release run completes, whatever its conclusion, so a failed or cancelled publish, or a setuptest that never started, still gets one. It names the release by the triggering run's id and commit (a `workflow_run` event's own `github.sha` is `main`'s), checks the tag points at that commit, lists every job that did not succeed and the `pypiindex.yml` run of that commit, and shows what the GitHub release, crates.io, npm, Homebrew and PyPI serve for the version. An API that does not answer, answers something unreadable or lists no run shows as unavailable or absent, never as success. The summary is on the run's page, written before the job fails on anything missing. To report on a past release by hand: `GH_TOKEN=$(gh auth token) python tool/tasks/chanprobe.py report --run <release run id> --sha <tag commit> --tag vX.Y.Z`.

</details>

<details>
<summary>Offline and air-gapped notes</summary>

- After install nothing opens a socket: `ask`, `retrieve`, `doctor`, and the Python `urna.embed.potiontab` all resolve the vendored potion table locally. sentence-transformers presets from the model registry are the exception and download only with `URNA_ALLOW_DOWNLOAD=1` (section 12).
- The CLI finds the embedder scripts in the order section 11 states: the repo layout (`rust/bridge/python/urna/`), then `<root>/urna/python/urna/` for each data root (`URNA_DATA_DIR`, `XDG_DATA_HOME`, `~/.local/share`, `%LOCALAPPDATA%`, `<exe>/../share`). `urna doctor` prints which one it picked and validates the table is real bytes, not an LFS pointer.
- Air-gapped install: fetch the four files the one-liner downloads on a connected machine, copy them over, and run the installer with `URNA_RELEASE_BASE=file:///path/to/dir`. For the wheel, `pip download urna[embed]` on the connected side and `pip install --no-index --find-links` on the other.
- `urna doctor` exit codes are typed so provisioning scripts branch on them: `0` ok, `2` Python missing, `3` numpy/tokenizers missing, `4` embedder script missing, `5` table missing or LFS pointer, `6` embed run failed. A scalar SIMD fallback warns but exits `0`.

</details>

<details>
<summary>Maintainer checklist (one-time, before the first tagged release)</summary>

The release workflows assume external state that a fresh org does not have. As of 2026-10-04 (repo secrets of `hoffresearch/urna`: `CARGO_REGISTRY_TOKEN`, `NPM_TOKEN` and `HOMEBREW_TAP_TOKEN` set):

1. **Homebrew tap**: the public repo `hoffresearch/homebrew-urna` (created 2026-09-26) carries `Formula/urna.rb`, at 0.5.4 since the v0.5.4 release; the `publish-homebrew-formula` job in `release.yml` checks it out and pushes `Formula/urna.rb` to it. The job authenticates with the `HOMEBREW_TAP_TOKEN` secret, a fine-grained token (`urna-homebrew-tap`, resource owner `hoffresearch`) with contents read and write on that repo only, expiring 2027-09-26 (the org caps fine-grained tokens at 365 days). GitHub has no API that mints one: rotate it by hand in the account settings and replace the secret. Without it the job fails and the rest of the release stands.
2. **npm**: `@urna/cli` publishes under the `urna` npm org with the `NPM_TOKEN` secret, a granular token with package and org write on that org (user `notlikedev`). Granular tokens expire: the current one expires 2026-12-11, rotate it before a release that falls after that date. 0.5.0 was published by hand from the release's npm package after the bare name was refused.
3. **crates.io**: `rustready.yml` publishes with the `CARGO_REGISTRY_TOKEN` secret (crates.io user `brennercruvinel`). `urna-format` and `urna` (the CLI) are published since 0.5.0 (2026-09-26), at 0.5.4. `urna-engine`, the renamed `urna-runtime` (0.5.0 to 0.5.3, no longer released), is published since 0.5.4. A crate published for the first time needs a token allowed to publish new crates (crates.io scope `publish-new`, and a crate pattern that matches it if the token has any); the current token has it. The job publishes the next version of each and skips one crates.io already has. The token can be swapped for crates.io trusted publishing (GitHub OIDC, no stored token, configured per crate on crates.io) the same way PyPI works; that change replaces the secret with `rust-lang/crates-io-auth-action` in the workflow.
4. **PyPI**: the release run's `pypiready` job dispatches `pypiindex.yml` on the tag and waits for that run; `pypiindex.yml` is a top-level workflow because PyPI's trusted publishing does not support a reusable workflow as the publisher, and every cargo-dist custom job is one. It publishes with OIDC only, no stored token. One-time setup, in this order: (a) on pypi.org add a trusted publisher to the `urna` project (owner `hoffresearch`, repository `urna`, workflow `pypiindex.yml`, environment `pypi`; the `pypi.yml` publisher stays registered so a tag cut before the rename can still upload); (b) on test.pypi.org the same with environment `testpypi`, and create the `testpypi` environment in the GitHub repo; (c) to prove a publisher change before a tag, dispatch `pypiindex.yml` on `main` with `index=testpypi` and `source_run` set to a successful `rehearsal.yml` run on a commit of `main`; (d) v0.5.4 published to PyPI this way, with OIDC and PEP 740 attestations; (e) the `PYPI_API_TOKEN` environment secret, which no workflow read any more, was deleted on 2026-10-05; revoke the token on pypi.org as well. The environments restrict where they deploy from: `pypi` only from `v*` tags, `testpypi` only from `main`.
5. **potion table**: release and wheel builds do not use git-lfs. `tool/tasks/getpotion.sh` downloads the table from `minishlab/potion-base-8M` at the pinned revision and accepts it only when its SHA256 matches the LFS pointer in the checkout. Bumping the table means updating the pointer and `REV` in that script together.
6. **Attestations**: nothing to configure. `release.yml` requests `attestations: write` for the archives and `wheelmake.yml` for the wheels; `pypiindex.yml` requests `id-token: write` for trusted publishing and PEP 740.
7. **Short URL**: `get.hoffresearch.com` is not registered (NXDOMAIN). The scripts and the README use the raw GitHub URL. If the short form is wanted, point the DNS at a 302 to the raw script and update the README plus both script headers in the same change.
8. **Cutting a release**, in three separate steps. Prepare: `tool/tasks/releasepr.sh X.Y.Z` (it needs `cargo install cargo-release --version 1.1.6 --locked`, the version the script pins, and `gh`) works in a new worktree of `origin/main` on the branch `release-X.Y.Z`, never in the current checkout. cargo-release sets the workspace version, the `version` of the `urna-format` / `urna-engine` entries in `[workspace.dependencies]` (crates.io resolves those, the path only serves the workspace) and the lockfile, then moves the `[Unreleased]` block of `docs/CHANGELOG` under `## [X.Y.Z] - YYYY-MM-DD` and sets `version`, `date-released` (the same date), `repository-artifact` and the release identifier's URL and description in `CITATION.cff`; the replacements are listed in `rust/clitui/Cargo.toml` and the rest of `CITATION.cff` stays as it is. The script then runs `python tool/tasks/preflight.py` (CI runs it on every pull request), refuses any change outside those four files, makes a signed commit, pushes that one branch and opens the pull request. It refuses a version that is not after the current one, an empty `[Unreleased]` and another cargo-release version. Release dates are UTC days: cargo-release dates the release in UTC and the tag check compares it with the tag's UTC day, so a release prepared and tagged the same evening fits, and one dated after the tag is refused. Merge: after CI and the required `rehearsal` check. Tag, on the merge commit: `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z` (annotated and signed; `git config tag.gpgsign true` makes `-s` the default with the SSH key already used for commits). The tag drives `release.yml`: the archives, the payload and the wheels build first, and `wheelmake.yml` opens with the signature check and the preflight in tag mode on the files of the tag's own commit (the tag names that commit's version, the commit is on `main`, the release date is not after the tag's UTC day); only when every build passed does the host job create the GitHub release, and then the publish jobs run (Homebrew, npm, crates.io, and PyPI through `pypiready.yml`, which dispatches `pypiindex.yml` on the tag and fails when that run fails); `setuptest.yml` runs inside the release run once it is announced (a dist post-announce job). `release.yml` has no manual trigger; to test the installs of a released tag again, dispatch `gh workflow run setuptest.yml -f tag=vX.Y.Z`, which publishes nothing.
9. **Release signers**: `.github/trustkeys` lists the keys allowed to sign release tags (one line per principal). The `release-tags` ruleset refuses creating, moving or deleting a `v*` tag to everyone but repository and organization admins, who bypass it. A new maintainer key is a pull request that appends a line there; the verify step reads the file from the tagged commit.
10. **Changing the dist config**: after editing `[workspace.metadata.dist]` run `dist generate`, then `python tool/tasks/rehearsal.py generate`, and commit both regenerated workflows; never hand-edit either. `pr-run-mode = "plan"` keeps `release.yml` on the plan step for pull requests; the real build runs in the rehearsal (step 12). When the generator refuses a `release.yml` it does not know (a dist upgrade changed a job, a condition or a step), teach `tool/tasks/rehearsal.py` the new shape in the same pull request. Every action runs at a commit: the hand-written workflows write `@<sha> # vX.Y.Z`, and `release.yml` takes its commits from `[workspace.metadata.dist.github-action-commits]`, which `rehearsal.py` reads for the jobs it adds; to move one, change the commit there and in the hand-written workflows, then regenerate both.
11. **What the 0.5.0 tag taught** (five runs before it shipped):
    - No release job may smudge git-lfs: the repository can run out of LFS budget, and then every checkout that pulls LFS fails. The potion table comes from `tool/tasks/getpotion.sh` (in the per-target setup and inside the payload build, because dist's global job does not run `github-build-setup`), `precise-builds = true` keeps dist from building `urna-bridge`'s cdylib for musl, and `source-tarball = false` keeps dist's `git archive` from smudging the demo corpora.
    - Rehearse before tagging: the rehearsal workflow does it on every pull request that touches a release input (step 12). By hand, `dist plan`, then `dist build --artifacts=global` (it needs `cargo install cargo-auditable`) from a checkout where the table is a pointer. `git checkout -- <file>` re-smudges from the local LFS cache, so write the pointer with `git cat-file blob HEAD:<path> > <path>`.
    - Moving a tag is safe only while nothing irreversible went out, and only an admin can do it (the `release-tags` ruleset refuses anyone else) (a crates.io version is permanent, PyPI refuses a second upload of a version). Confirm the fix is on `main` (`gh pr view N --json state` says `MERGED` and `origin/main` moved), chain with `&&`, then `git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z && git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
    - PyPI publishes only from inside the release run: it published 0.5.0 and 0.5.2 on its own run while the binary release failed. `pypiready` dispatches `pypiindex.yml` once every build passed and the host released them. A PyPI-only failure is finished by rerunning the failed `pypiready` job: the upload skips a wheel PyPI already has with the same sha256 and refuses one with another, so a partial upload completes and never overwrites. `pypiindex.yml` by hand takes `index=pypi` only on a `v*` tag with the release run as `source_run`.
    - A plan job (dist's `plan-jobs`) cannot gate a release: when it fails, the builds that need it are skipped, and dist's host job runs on skipped builds. The tag check therefore lives inside `wheelmake.yml`, whose failure the host refuses.
    - A failed publish job leaves the release unannounced and `setuptest.yml` unfired: the release report (above, under Verification) still names what each channel serves; fix, then `gh run rerun <id> --failed`, and dispatch setuptest with the tag if it does not start.
    - npm refuses a bare name close to a popular package (`403 package name too similar to existing packages`) whoever owns the org; the package lives under `@urna`. A new package takes a few minutes to appear on the registry; poll `https://registry.npmjs.org/@urna%2fcli` for a 200 before testing an install.
    - A release job can fail on the runner's own tools, which no local gate sees: the 0.5.2 global job ran `tool/tasks/embedpack.py` with ubuntu-22.04's `python3` (3.10, no `tomllib`) and stopped; only PyPI, its own workflow, had published, so the fix shipped as 0.5.3 rather than a moved tag. `rehearsal.yml` now runs that global job on its own runner image before a tag, and CI's `payload-py310` job stages the payload under Python 3.10 on every pull request.
    - setuptest's npm, bun, pnpm and yarn legs ran before the registry served 0.5.3 (`ETARGET`, no matching version). Each registry leg now waits for the exact version first; a leg that still fails after the 15-minute wait names the channel's last state.
    - GitHub retires runner labels (`macos-13` queued forever): keep setuptest's matrix on the runners dist uses (`dist plan` lists them).

12. **The release rehearsal**: `.github/workflows/rehearsal.yml` runs the release's own build before a tag, publishing nothing. It is generated from `release.yml` by `tool/tasks/rehearsal.py`: the same plan, the five archives with the same runners and matrix, the wheels (`wheelmake.yml`) and the global job, with the host and publish jobs dropped, every permission read-only, no secret but the run's token and no attestation. It runs on pull requests and pushes to `main`; its `impact` job decides whether the change touches a release input (the crates, the manifests and lockfile, `pkgs/wheel/`, the staging scripts, every file the payload and the wheel copy, the model dirs, `README.md`, `LICENSE`, the release workflows), and a `workflow_dispatch` always runs it. With impact, the `assets` job checks every artifact the plan names, the four wheels and their `.sha256`, every checksum and `sha256.sum`, the payload's `VERSION`, and runs the Linux binary against the golden fixture. The `rehearsal` job is a required check of the `main` ruleset (source: GitHub Actions), so a pull request cannot merge without it: no impact passes as dispensed; with impact, a failed, cancelled or skipped build or assets job fails it. It does not rebuild the measurement corpora (that is `fullcheck.sh`), and it proves neither the publishing jobs nor the attestations; those run only on a tag. A successful rehearsal on a `main` commit is the `source_run` of the TestPyPI proof (step 4).

</details>
