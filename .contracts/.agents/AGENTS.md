# user guide

operating notes for ai agents and human contributors working in this repo. the principal author writes fast and uses voice transcription. typos, caps lock, missing accents are common. read intent, do not flag tone, do not project emotional risk.

# build and test

- `cargo build --workspace` / `cargo build --release --workspace`
- `cargo test --workspace`: all rust tests (unit + integration + golden), 371/371 on the current unreleased state (`forge-core` adds 6 more on its own manifest)
- `cargo fmt --all --check`: formatting check
- `cargo clippy --workspace --all-targets -- -D warnings`: linting (warnings are errors)
- `ruff check .` / `ruff format --check .`: python linting and formatting (config in `pyproject.toml`)
- `./scripts/release_check.sh`: full pipeline + regression gates against `data/measure/baseline.json`. single source of truth for "PR-ready". exits non-zero on any failure.
- `forge-core` (the ingestion layer) is a SEPARATE cargo workspace OUTSIDE `crates/`; the sovereign `--workspace` commands and `release_check.sh` do not touch it. build and test it on its own manifest: `cargo build --manifest-path forge-core/Cargo.toml`, `cargo test --manifest-path forge-core/Cargo.toml`, `cargo clippy --manifest-path forge-core/Cargo.toml --all-targets -- -D warnings`, `cargo fmt --manifest-path forge-core/Cargo.toml --all --check`. the <=300-line rule applies there too (release_check's guard only scans `crates/`).

# pyo3 extension

dev build is still manual:

```
cargo build --release -p urna-python --features pyo3/extension-module
cp target/release/lib_urna.dylib python/_urna.so   # macOS
cp target/release/lib_urna.so   python/_urna.so    # linux
```

the `pyo3/extension-module` feature keeps libpython out of the cdylib. without it the .so hard-links a libpython path and segfaults under statically-embedded interpreters (uv's python-build-standalone) by loading a second runtime; `release_check.sh` builds with the feature.

the published wheel is maturin, staged (never edit `packaging/staging/` by hand):

```
python scripts/stage_wheel.py
(cd packaging/staging && maturin build --release)   # wheel lands in target/wheels/
```

`packaging/pyproject.toml` is the single source for the wheel project; the staging script copies it plus `python/urna.py` (as `urna/__init__.py`), `python/urna_cli.py` (as `urna/_cli.py`), `python/forge/embed_potion.py`, and the potion table. abi3 targets python 3.12+ (not 3.14). python tests need the built `.so` first:

```
python tests/test_e2e.py
python tests/test_builder.py
python tests/test_search_text_model_hash.py
python tests/test_offline_guard.py
python tests/test_blob_bridge.py
python tests/test_space_bridge.py
python tests/test_image_corpus.py
python tests/test_forge_spec.py
python tests/test_quality_gate.py
python tests/test_cli_space.py
python tests/test_query_embedder_routing.py
```

no pytest. tests are plain scripts with `if __name__ == "__main__"`. `pytest tests/` does not work. `test_image_corpus.py` (42 cases) exercises the forge image-corpus pillar (encode/decode, gop probe, sharding, ordering) and skips cleanly when ffmpeg's av1/avif encoders are unavailable; it does not need the vision model itself. building an actual image corpus (`python/forge/embed_image.py`) needs `open_clip` + torch, not part of the default forge dependency group.

the forge build-side default embedder is now the REAL SEMANTIC one: a vendored model2vec/potion-base-8M static table (`python/forge/embed_potion.py`), offline, no torch, no network. its self-test needs numpy + tokenizers and the vendored table (git-lfs): `python python/forge/test_embed_potion.py` (run with `.venv/bin/python`; install the deps with `uv pip install numpy tokenizers`, declared in `pyproject.toml` under `[dependency-groups] forge`). it proves the semantic jump (car ~ automobile >> car ~ banana), determinism, f32-stability, and that no socket is opened at embed time; `python python/forge/recall_harness.py` shows per-query recall vs the floor. the table (`python/forge/models/potion-base-8M/model.safetensors`, ~30mb) is git-lfs; run `git lfs pull` if it is a pointer. the #04 lexical bag-of-words stays as the stdlib-only zero-dep FLOOR with its own self-test (no `.so`, no deps): `python python/forge/test_embed_default.py`. both self-fingerprint to a `model_hash` recorded in provenance; neither is run by `release_check.sh`.

# single-target commands

- `cargo test -p urna-format`: format crate only
- `cargo test -p urna-runtime`: runtime crate only
- `cargo test --release -p urna-runtime --test hnsw_recall`: HNSW recall regression (needs release; debug is too slow to run within timeout)
- `cargo test -p urna-cli`: CLI integration tests (requires release build)
- `cargo run -p urna-format --example regen_golden`: regenerate the byte-frozen golden fixture

# architecture

```
crates/urna-format    frozen v1 container: layout, manifest, sections, encodings, hashes, reader, writer
crates/urna-runtime   depends on urna-format: mmap open, SIMD dispatcher, MmapUrnaFile, ann::HnswIndex,
                       bm25::Bm25Index, graph::CsrIndex, exact/ann/graph/hybrid search with mandatory
                       exact rerank
crates/urna-cli        depends on urna-format + urna-runtime: clap binary `urna`, twelve engine
                       subcommands in cmd/*.rs (incl media, search-space, doctor) + the three
                       agent verbs ask/retrieve/build in cmd/agent/*.rs; `--help` tags and orders
                       the two groups; the clap surface lives in cli.rs, the one three-layer
                       model gate in cmd/embed_gate.rs
crates/urna-python     depends on urna-format + urna-runtime: cdylib _urna, PyO3 abi3-py312

forge-core/            SEPARATE cargo workspace at the repo root, OUTSIDE crates/ (ingestion layer,
                       FORGE-0a: the frozen .fci canonical-intermediate schema). its deps never enter
                       the sovereign crates; not in the `--workspace` set. .fci is versioned independently.

python/                writer pipeline (builder.py), model fingerprint, query embedders, forge/
                       tools incl the declarative build surface (build_spec + spec_rules + spec_paths +
                       corpus_sources + forge_pipeline + forge_media_stage + forge_recipe +
                       forge_emit + forge_cache + forge_manifest),
                       the model registry (model_registry + model_adapters + embed_st +
                       embed_st_worker) and the dual quality gate (quality_gate)
tests/                 python test scripts (plain scripts, not pytest)
docs/                  arc/ architecture pair, usage.md (with the install reference), CHANGELOG, SECURITY.md (with the data-governance posture)
data/                  corpus_next.v1.urna (LFS demo corpus), measure/ regression baselines, demo/ sources
scripts/               release_check.sh (the merge gate), pre-commit (PHI/data backstop hook),
                       install.sh / install.ps1, stage_wheel.py, stage_embedder_payload.py
packaging/             pyproject.toml, single source for the published wheel (staging/ is generated)
docker/                minimal image: static musl urna binary on scratch
examples/              quickstart (the five-verb loop on twelve cc0 paragraphs), fastapi, flask, jupyter
.contracts/.agents/    AGENTS.md, the single agent instruction source
```

key rust deps: memmap2 (mmap), rayon (parallel build), zstd / half / bytemuck (encodings), sha2 (hashing), thiserror (typed errors), clap (cli), serde / serde_json (manifest).

CLI binary: `urna`. twelve engine subcommands (file + vector in, never run python): `inspect`, `validate`, `stats`, `media` (list / export the inlined 0x17 blobs, sha256-verified), `search`, `search-ann`, `search-graph`, `search-space` (exact search over one named multimodal band), `search-text`, `benchmark` (incl `--space`), `cite`, `doctor`. plus three agent verbs layered over the same engine, under `cmd/agent/`: `build` (declarative corpus build, a launcher over `python/tools/urna_forge.py`), `ask` (text query in, cited answer out, `--disclose answer|explain`) and `retrieve` (json/jsonl answer-pack of cited spans where score IS the exact rerank value). the flagship embeds offline and routes the query embedder BY THE MANIFEST MODEL: potion corpora keep the potion script, any registry model goes through `python/forge/embed_query_model.py` (with `--mrl-dim` for truncated default spaces). one or several embedding models per build come from the preset registry (`python/forge/model_registry.py`); the build contract with every user-selectable knob is `docs/usage.md` sections 12-14 (section 13 carries a complete working spec). verb-collapse, the `urna dev` namespace, and the urna-profile crate stay deferred.

python entry: `sys.path.insert(0, "python"); import urna`. dynamic loader finds `_urna.so` or `lib_urna.dylib`.

# format and runtime contract

- rust edition 2024, resolver 3, `thiserror` for errors (never panic in library code). `repr(C)` + `bytemuck::Pod` structs for binary layout; all integers LE unsigned. every `unsafe` block needs a `// SAFETY:` comment naming the invariant it relies on, and clippy denies undocumented ones (`[workspace.lints]`: `undocumented_unsafe_blocks`, `unwrap_used`; tests exempt). read fixed-width fields through `urna_format::bytes::{le_u32, le_u64, le_f32, array32}` (typed `UnexpectedEof`, no `try_into().unwrap()`); compute header-derived sizes with checked arithmetic (`expected_embeddings_size`); write cursor bounds checks as `need > remaining`, never `pos + need > len`; sort f32 scores through `urna_runtime::order` (NaN-last total order), never `partial_cmp(..).unwrap_or(Equal)`.
- binary format v1 is frozen. v0.2 added encodings 1/2/3 (zstd, float16, int8) and optional sections 0x07 (HNSW) and 0x08 (BM25). v0.3 added encoding 7 (int4) and the graph pillar (section 0x0C). since then the media blob pillar (0x14 blob_refs, 0x16 blob_span_overlay) and the multimodal space pillar (0x15 space_table + the 0x20-0x2F embedding band) shipped, all additive and content_hash-excluded; see `docs/arc/arc.toml`'s `contract` array for the full section-id map.
- hash format: always `sha256:<64 lowercase hex>`. four hashes: `header_checksum`, per-section `checksum` (physical bytes), `file_hash` (whole file), `content_hash` (decoded canonical sections, stable across encodings). same chunks + same model fingerprint + `reproducible=True` produce byte-identical files, so the `urna://content_hash/chunk_id` citation URI points at content, not at a copy.
- `UrnaFileBuilder` is a consuming builder (`add_chunk(self) -> Self`). presets via `.text_encoding()` + `.embedding_dtype()`, or the bundled levers: `exact`, `compressed` (zstd + f16), `tiny` (int8 + hnsw), `micro` (mrl256-int8), `nano` (int4 block-64), `hybrid` (f32 + hnsw + bm25).
- matryoshka prefix truncation is a build-time kwarg (`urna.build(mrl_dim=K)` / `BuildConfig.mrl_dim`): the python builder slices each l2-normalized row to its first K components and re-l2-normalizes the prefix BEFORE quantization, sets the header/manifest `embedding_dim` to K, and records the source dim as `full_dim`. additive optional manifest fields (`mrl_dim`/`full_dim`, omitted when unset so existing files stay byte-identical). NO runtime kernel change: the reader strides by `header.embedding_dim`. int4 needs the EFFECTIVE dim %64==0, so the int4 ladder is valid only at mrl_dim in {256,192,128}. truncation is a pure deterministic slice => byte-identical builds; content_hash is over the truncated embeddings so citations are tied to a given mrl_dim. the shipped MiniLM corpus is NOT mrl-trained, so truncation costs measured recall: `measure_presets.py --variants mrl<DIM>-<dtype>` reports the curve, gated conditionally in `compare_measure.py`.
- HNSW build is deterministic given a seed. BM25 index is sorted by alphabetical term order.
- `model_hash` is a granular fingerprint over `(model_id, files_hash, tokenizer_hash, pooling_config_hash, embedding_dim, normalize_embeddings)`. zero-placeholder is rejected at write time. a mismatch between runtime model and corpus model fails loudly with a typed error.
- runtime SIMD dispatch: AVX2 (x86_64), NEON (aarch64), scalar fallback, accumulators always f32. `URNA_FORCE_SCALAR=1` forces scalar for A/B benchmarks.
- golden fixture: `crates/urna-format/tests/fixtures/golden_v1_minimal.urna` (1366 bytes, byte-frozen).
- CLI `search` takes a JSON f32 array positional arg; `search-text` shells out to `python/embed_query.py` and validates the embedder's `model_hash` against the manifest.
- python api: `urna.open(path)` returns a `UrnaFile` with `search`, `search_ann`, `search_hybrid`, `retrieve`, `validate`, `inspect`. hits carry `citation_id`, `source_uri`, offsets, and the exact-rerank `score`.
- file hygiene: every rust source file in `crates/**/src/**` and every first-party python module is at most 300 lines. test files and the `crates/urna-format/tests/roundtrip.rs` carve-out are exempt.

# repo workflow

- remote: `git@github.com:hoffresearch/urna.git`. owner: hoff research. maintainer: brenner cruvinel (`brenner@hoffresearch.com`).
- branches: `main` is the only long-lived branch. work happens on short-lived branches off `main`; every change reaches `main` through a pull request.
- PRs target `main` and are squash merged (the ruleset requires pull requests, verified ssh-signed commits, and linear history). delete the branch after merge and start the next one from `origin/main`.
- tags on `main` only (`v0.4.0` is current). `Cargo.toml` workspace version tracks the latest released tag.
- every push and pull request runs `.github/workflows/ci.yml`: fmt, clippy with the workspace deny lints, build + test on ubuntu (avx2) and macos (neon), the mutation-fuzz harnesses at a higher iteration count, the 300-line guard, forge-core's own gate, ruff via `scripts/ruff_check.sh` (the ONE python file list, shared with release_check.sh), and a bounded cargo-fuzz smoke on nightly. it is release_check.sh minus the lfs corpus measurement.
- pushing a `v*` tag on `main` runs the full release: `.github/workflows/release.yml` (cargo-dist: cli tarballs for 5 targets, checksums, sigstore attestations, homebrew formula, the embedder payload artifact) and `.github/workflows/pypi.yml` (maturin abi3 wheels for 4 platforms, OIDC trusted publishing). `.github/workflows/install-test.yml` then tests the INSTALLED product per platform. maintainer one-time setup for these channels is the maintainer checklist in the reference section of `docs/usage.md`.
- git lfs tracks `*.urna`, `*.safetensors`, datasets, and the vendored potion table (including `data/corpus_next.v1.urna`); golden fixtures under `crates/urna-format/tests/fixtures/` stay in regular git. run `git lfs pull` if a binary is a pointer.
- demo datasets under `data/demo/` are intentionally gitignored and downloaded locally from upstream sources listed in `data/demo/Instructions.md`.
- tests run without the demo datasets (the unit and golden-fixture tests avoid depend on them); only `measure_presets.py` and `release_check.sh` need the baseline corpus.
- `data/measure/corpus_*.urna` and `*.urna-*` are gitignored: regeneration artifacts, not assets. the JSON files next to them ARE tracked (regression baselines).
- `scripts/pre-commit` is a PHI/data backstop that aborts commits staging non-allow-listed data artifacts; install per clone with `cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit` (copy, not `core.hooksPath`, so git-lfs hooks keep working).

# conventions

- every change ships with real tests, no mocks: happy path, error path, one edge case minimum.
- test against real artifacts (built .urna files, golden fixtures, real corpora), never mocked interfaces.
- applies to every contributor, human or agent; nothing merges without executable proof.
- keep the docs/CHANGELOG test-surface count in sync when adding suites.
- base formatting via `.editorconfig`: utf-8, lf, 4-space indent (2 for toml/yaml/json), final newline.

# naming

directories, docs, and assets: kebab-case in english. source files: idiomatic to the language. check folder structure consistency with the project's design pattern. when proposing renames or moves, list them as `mv` commands. after renaming, test every import touched by the change and fix them. run the test suite after the operation.

build folders, files, and codebase items following apl-style 3-char tokens that read as variable syntax. that keeps the glyph atomic and context natural. high core value for neurodivergent readers.

# docs

write in diataxis style. all lowercase. no emojis. no em-dash. no decorative markdown. pragmatic, professional, objective. every doc starts with a yaml header for semantic resolution (helps llm, agentic, vector search): project, audience, status, last-updated, domain. design notes that turn out wrong get a note on top. they are not deleted.

`docs/arc/arc.toml` is the single architecture reference, machine-readable for agents and tooling: the narrative (system_view, contract, quality, risks), the file inventory, and the mermaid visual map of the build and query flows (`diagram.source`, byte-equal to a `.mmd` file; extract it verbatim to feed mermaid-cli or any mermaid live editor) all in one file. a `schema` table documents the shape (former IDE-side validation via `arc.schema.json` + a yaml-language-server modeline has no toml equivalent, so the contract is documentation, not enforced).

at task start, read `docs/arc/arc.toml` in a short pass to preserve structure and naming pattern. after any implementation, refactor, rename, or doc move that changes architecture, boundaries, data flow, module layout, public contracts, storage, or runtime behavior, update `arc.toml` in the same change (bump `last-updated`, append a dated note to `summary`). keep it concise and pragmatic. do not keep a parallel second architecture document.

# file hygiene

hard limit is 333 lines per file. operational target for new files is 220 lines. human working memory holds 4 plus or minus 1 chunks at once (cowan 2001, refining miller). neural networks also work better that way. a file that does not fit the "mental window" forces internal context switching, degrading comprehension and raising bug rates. this is unnecessary cognitive load, the same principle applied in ux.

every file created or modified in a session that exceeds 333 lines must be read in full and refactored along single-responsibility lines. the rust source carve-out (`crates/**/src/**` at 300 lines) and the test-file exemptions documented above remain in force.

# audit when finishing a task

run a full audit over every change made in the session, no summarizing, from devops, code quality, and secops angles. write a temporary manifest in markdown under your tmp folder to track tasks executed.

identify every trace of dead code, generated scripts and files no longer useful, items needing update, and items to be moved to the correct location per architecture and design pattern. if the project lacks documented conventions, create them: design notes in `docs/CHANGELOG` for architectural decisions, `.editorconfig` for stack-agnostic base formatting, and an idiomatic linter config per language used.

identify temporary scripts and possible dead-code files in incorrect folders. understand how each works, preserve application integrity, test and validate that no imports or responsibilities are left orphan. run tests after execution.

# style

documentation, comments, and commit messages follow the README's tone.

- lowercase headers throughout markdown (acronyms like `## CLI` are the only exception).
- no em-dash (`—`). use `,` `;` `.` or a regular hyphen `-`.
- no emoji.
- short paragraphs, direct voice, no marketing copy.
- commit messages in plain english, no Conventional Commits prefix. body explains the why; the diff already shows the what.

# gotchas

- **rebuild `python/_urna.so` after every rust change** that touches `urna-format`, `urna-runtime`, or `urna-python`. python tests load it via `dlopen`; stale `.so` will pass tests against old code. `release_check.sh` does this for you (and pins `PYO3_PYTHON` to the test interpreter to avoid segfaults); manual workflows must remember.
- **NEON f16 MSRV**: `float16x4_t` and `vcvt_f32_f16` are stable since rustc 1.94, but the workspace MSRV is 1.85 (`rust-version` in the workspace `Cargo.toml` — the single msrv source; clippy reads it too). `crates/urna-runtime/build.rs` probes the compiling rustc and emits `cfg(neon_f16)` at >= 1.94; that cfg gates `simd/neon.rs::dot_f32_f16_neon` and its dispatch arm, and older toolchains fall back to the scalar f16 kernel. the kernel carries `#[clippy::msrv = "1.94"]` to match the cfg guarantee. avoid remove build.rs or the cfg gate without bumping the workspace `rust-version` to >= 1.94.
- **HNSW recall test needs release mode**: debug is 30x slower and hits the 60s default cargo test timeout. always run with `--release`.
- **PT-BR fingerprint corpus**: the model fingerprint is computed against the local sentence-transformers cache. first-time builders must `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"` to populate the cache, otherwise `urna_build_corpus.py` and the fingerprint test fail.
- **squash merge replaces the branch history**: the PR lands on `main` as one commit with a new hash, so a branch that keeps living after its merge conflicts on every file the squash touched. delete merged branches and start each new one from `origin/main`; never rebase old work onto a merged branch.
- **avoid run `cargo clean` casually**: rebuild times are 30-60s for the full workspace. incremental compilation handles most edits.
- **`ask`/`retrieve` embed OFFLINE, routed by the manifest model**: a potion corpus uses `python/forge/embed_query_potion.py` (static table, numpy + tokenizers, no torch, no socket); a corpus whose default text space is a registry model (wemm, clip, jina) routes through `python/forge/embed_query_model.py`, which loads that model locally (network only with `URNA_ALLOW_DOWNLOAD=1`). only `search-text` uses `python/embed_query.py` (sentence-transformers, network on first use). the embedder runs under `python3` unless `URNA_PYTHON` is set; point `URNA_PYTHON` at a venv that carries the forge deps (numpy + tokenizers + the git-lfs potion table) or the embed step fails with `ModuleNotFoundError`. the two flagship e2e tests in `cli_e2e.rs` and `python/forge/test_retrieve.py` need those deps and skip cleanly when absent; they are not run by `release_check.sh`.
- **`cite` is tier-1 only**: it returns the stored canonical text + verifying hashes, NEVER an original-byte reopen. `ask`/`retrieve` print the same tier-1 text. do not let help text or docs claim original-byte reopen (that is net-new tier-2 catalog work, post-gate).

# known gaps

these are documented honest limitations of the current code, not bugs to silently fix. user-visible behavior; flag them in any work that interacts with these areas.

- **`search-text` boot overhead (~300-500ms)**: each invocation forks a python process, imports sentence-transformers, embeds the query, then exits. the latency table in the README and `docs/usage.md` measures the search path AFTER the vector is ready, not end-to-end. python-driven workloads (`urna.UrnaFile.search` in a loop) avoid this.
- **BM25 tokenizer is word-segmented-only**: `crates/urna-runtime/src/bm25/tokenize.rs` splits on non-alphanumeric Unicode boundaries. correct for latin, cyrillic, greek, devanagari. degrades for CJK, thai, lao (each character becomes a token, posting lists explode, recall drops). hybrid search on those languages should disable BM25 (`with_bm25=False`) until a language-aware tokenizer ships.
- **homebrew formula installs the binary only**: the dist-generated formula in the `hoffresearch/homebrew-urna` tap does not lay down the embedder payload, so a brew-installed `urna` reports exit 4 from `urna doctor` until the user also runs the one-liner (or copies `python/forge/` into the data dir by hand). fixing this means a custom formula, deferred until the tap sees real use.
- **st registry models have measured cost cliffs**: wemm-2b runs fp16 on mps with `image_max_side=768` (~0.6 img/s); jina-v5-omni-nano has no `image_max_side` default yet and embeds at native resolution (~0.3 img/s); changing either invalidates that model's cache by design (the knob is recipe-hashed). the siglip2 TEXT tower resolves an hf tokenizer whose optional-file probes can fail in strict offline mode even with the snapshot cached (usage section 12 has the workaround); its image tower is unaffected.
- **the semantic default embedder is english**: `potion-base-8M` is distilled from `bge-base-en-v1.5`, so english synonyms cluster tightly (car ~ automobile +0.78 vs car ~ banana +0.04) but non-english text rides english subword rows and the semantic signal is weak (carro ~ automovel +0.08 vs carro ~ banana -0.05: right direction, small margin). for a primarily non-english corpus, bring a multilingual sentence-transformers model (the ceiling path) or a multilingual potion table. the lexical floor is language-agnostic but captures literal token overlap only.

# things to avoid

- **avoid write markdown that wasn't requested**.
- **avoid bump `URNA_FORMAT_VERSION` for additive changes**. encodings 4-255 and section IDs 0x09+ are reserved within v1. v2 only when an existing field changes meaning.
- **avoid `--no-verify` git hooks** unless explicitly asked.
- **avoid force-push `main` ever** (the ruleset blocks it). force-push a feature branch only after explicit user confirmation, and only with `--force-with-lease`. squash-merge from PR is fine because that goes through GitHub.
- **avoid run `git add -A`** in repos that may carry untracked secrets or LFS payloads. stage explicit paths.
- **avoid bypass `release_check.sh`**. if it fails, fix the underlying issue. suppressing a clippy lint is fine when justified inline (`#[allow(clippy::name)]` + comment); suppressing the whole gate is not.
- **avoid introduce `unsafe` without a `// SAFETY:` comment** that names the invariant the caller is relying on (clippy denies it anyway). prefer `bytemuck` casts over raw-parts casts; keep raw-pointer kernels behind a safe dispatcher that `assert!`s every length in release.
- **avoid a decoder change without running the mutation harness**: `cargo test -p urna-format --test mutation_fuzz -p urna-runtime --test mutation_fuzz` (default counts run in seconds; `URNA_MUTATION_ITERS=25000` for a soak). a new section codec gets an arm in `fuzz/fuzz_targets/section_decoders.rs` and a fixture in the harness. then the coverage-guided soak: `sh scripts/fuzz_soak.sh` (one hour per target, corpus under `fuzz/corpus/`, needs nightly + cargo-fuzz) before the pr; ci repeats it nightly for 30 minutes per target. a finding becomes a `tests/negative_*.rs` before the fix, and its libfuzzer artifact a `fuzz/seeds/regress-*.bin`.
- **avoid add em-dashes or emoji** to project files. consistency check in CI is informal but the maintainer reads diffs.

# documentation

- `README.md`: project overview, install, CLI summary, python surface, benchmarks, hardening, presets, reference index.
- `docs/arc/arc.toml`: the single architecture reference, machine-readable for agents and tooling: the human-readable inventory plus runtime contract summary, and the mermaid sequence diagram of the build and query flows (`diagram.source`). a `schema` table documents its shape.
- `docs/usage.md`: how-to for the twelve engine subcommands (incl `media`, section 15) plus the ask/retrieve/build agent verbs, presets, offline mode, citations, the model registry and multi-model spaces (section 12), declarative builds (section 13), and the compression levers with the dual quality gate (section 14), and the collapsed reference section: every install channel (one-liner, pypi `urna`, brew, binstall, docker, dev build), verification (sha256 + attestations + sbom), offline notes, and the maintainer one-time checklist.
- `docs/CHANGELOG`: 0.1.0 through 0.4.0 and the unreleased deltas, with measured numbers.
- `docs/benchmarks.md`: urna vs usearch / hnswlib / sqlite-vec / lancedb, one table, regenerated by `python/tools/bench_competitors.py`.
- `data/demo/Instructions.md`: what each upstream PT-BR dataset is and how to rebuild the unified corpus.
- `docs/CONTRIBUTING.md`: external contributor flow.
- `docs/CODE_OF_CONDUCT.md`: contributor covenant 2.1, lowercase plain-style.
- `docs/SECURITY.md`: reporting channel, supported versions, security scope, hardening notes, and the data-governance posture (cleartext datastore, erasure and rectification, provenance as a compliance asset, corpus licensing).
- `docs/LICENSE`: mit license text.
- `scripts/release_check.sh`: read it. it documents the gate by being the gate.

# agent instructions

this file is the single instruction source for ai coding agents: use/update/init only .contracts/.agents/AGENTS.md (the core global agent file). codex and most agentic tooling already read .contracts/.agents/AGENTS.md by default; point claude, gemini, cursor rules and similar tools here on init. do not create CLAUDE.md, GEMINI.md, CODEX.md, or any parallel instruction doc.
