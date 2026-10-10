---
project: urna
audience: users and security researchers
status: active
last-updated: 2026-10-10
domain: security
---

# Security

`urna` is maintained by [Hoff Research](https://hoffresearch.com). Author: Brenner Cruvinel.

## Supported versions

Only the latest minor on `main` is supported.

| Version | Status |
|---------|--------|
| 0.5.x   | Supported (current) |
| 0.4.x and earlier | Not supported, please upgrade (the reader still opens 0.4.0 `.nest` files: `LEGACY_MAGIC` in `rust/format/src/layout/mod.rs`) |

## Reporting a vulnerability

Do not open a public GitHub issue for security vulnerabilities.

Use one of:

- Private vulnerability report: <https://github.com/hoffresearch/urna/security/advisories/new>
- Email: brenner@hoffresearch.com

We aim to acknowledge within 72 hours and to publish a fix or mitigation within 14 days for confirmed reports. Coordinated disclosure preferred; we credit reporters who request it.

## Scope

Things we treat as security bugs:

- Malformed `.urna` files that trigger UB / OOB / panic in the Rust runtime
- A citation collision (two distinct chunks producing the same `chunk_id`)
- A `content_hash` collision under the v1 hash domain separation
- A path that bypasses `model_hash` validation in a text query path (`search-text`, `ask`, `retrieve`) without the user passing an explicit skip flag; `search-space` takes a raw vector and validates only when `--expect-model-hash` is given
- A path that executes model-repo code (`trust_remote_code` presets) without the explicit opt-in - the spec's `allow_remote_code` at build time, `URNA_ALLOW_REMOTE_CODE` on the query/bench/bridge side (`presetqry.py`, `modelrank.py`, `uibackend.py`) - or with a code file whose SHA256 is outside the pinned allowlist in `rust/bridge/python/urna/model/presetmap.py`
- Secrets or credentials accidentally committed to the repository

Things we do not treat as security bugs:

- Low recall on a particular corpus
- HNSW recall under user expectation (configuration tuning, see `--ef`)
- BM25 tokenizer degrading on CJK / Thai / Lao (documented limitation)
- Compressed vs raw size differences
- Vulnerabilities in upstream sentence-transformers / HuggingFace stack; report those upstream first
- Weaknesses in the embedding model itself (false positives, biased recall)
- Configuration choices made by the operator (e.g. building a corpus with the placeholder `model_hash` and using `--skip-model-hash-check`)

## What helps a report

- The `.urna` `file_hash` and `content_hash` (`urna stats <file>` prints both)
- The runtime `simd_backend` and platform (`urna stats`)
- The exact CLI or Python invocation
- A minimal reproducer if possible (a synthetic `.urna` is fine, see `rust/format/tests/fixtures/`)
- Whether you have a proposed mitigation

## Hardening notes

- The runtime (Rust) never opens a network socket. Queries are answered from `mmap`. The default query embedders are offline too: `ask`/`retrieve` use the vendored potion table (no network by construction), and the `search-text` sentence-transformers path forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` unless you opt in with `URNA_ALLOW_DOWNLOAD=1` (or pass `--model-path`). A registry query embedder (`urna/embed/presetqry.py`, behind `ask` and `retrieve` on a registry model) fetches only with the same `URNA_ALLOW_DOWNLOAD=1` opt-in; an open_clip preset pinned to a hub revision (SigLIP2) then fetches exactly its listed files at that revision, and without the opt-in a missing file is an error naming it, never a lookup. The other place a model is fetched is the model install (`urna setup --model`, the explorer's install panel): only after an explicit confirmation, only the files the payload's catalog pins at a fixed hub revision, with `URNA_ALLOW_DOWNLOAD=1` set for that fetch child alone, and kept only when they fingerprint to the catalog's `model_hash`; a model that runs its repo's code needs a second, separate consent and its reviewed file hashes. Packages go only into the venv `urna setup` manages.
- `model_hash` is a granular fingerprint over the local model snapshot (config + tokenizer + weights + pooling + dim + normalize); for an open_clip model it covers the loaded weights and the image preprocess instead, and the tokenizer is pinned by loading it from the same snapshot revision. A mismatch fails with a typed error, never silently. The CLI (`search-text`) enforces this; the Python `UrnaFile.retrieve` binding accepts `expected_model_hash` and the flagship `urna/reads/retrieval.py` passes it by default, so the honesty gate holds on the Python surface too.
- `unsafe` lives in the SIMD kernels (`rust/engine/src/simd/`) and the two `mmap` calls (`rust/engine/src/mmap_file.rs`, `mmap_cold.rs`); the former zero-copy casts in `rust/format` (header / footer / section-entry byte views, the int8 row view) are now safe `bytemuck` casts whose layout invariants the compiler checks, and the crate keeps zero `unsafe` by `#![forbid(unsafe_code)]`, and the whole format crate runs under Miri nightly (`gatecheck.yml` job `miri`), so "no undefined behaviour" in the parsers is a run, not a claim. Every remaining `unsafe` block carries a `// SAFETY:` comment naming the invariant, and `clippy::undocumented_unsafe_blocks` is denied workspace-wide so a new undocumented block fails the build. The safe SIMD dispatchers check every slice length with `assert!` (kept in release), so the raw-pointer kernels never run on a mismatched row even if a validation layer upstream regresses.
- No `unwrap()` or `expect()` on a parse path: `clippy::unwrap_used` and `clippy::expect_used` are denied workspace-wide (tests exempt), little-endian field reads go through `urna_format::bytes` and return `UnexpectedEof`, every header-derived size (`n * dim * width`) is overflow-checked, every payload cursor bounds-checks as `need > remaining` (never `pos + need > len`, which wraps), every count read from the file is bounded against the remaining bytes before it sizes an allocation, and every f32 ranking sort is a NaN-last total order.
- Fuzzing: `cargo test` runs a deterministic mutation-fuzz harness on every push (`rust/format/tests/mutation_fuzz.rs`, `rust/engine/tests/mutation_fuzz.rs`: bit flips, byte sets, zero runs, integer specials, truncation, splices, half of them with checksums resealed so the corruption reaches the decoders), and `fuzz/` carries four `cargo-fuzz` targets (reader, section codecs, index codecs, mmap open + every search verb) that `gatecheck.yml` smoke-runs on nightly. The harnesses' first runs found and fixed five classes of bug: an unchecked `n * dim` overflow in the expected-section-size check, a NaN score reaching a `partial_cmp`-based sort (a panic since Rust 1.81) through an unvalidated multimodal band, a wrapping `pos + n` cursor bounds check, BM25 postings whose doc IDs were never checked against `n_docs`, and (from the coverage-guided run) a URI-pool count in the intpack spans repack that reached `Vec::with_capacity` unbounded, so a 90-byte payload asked the allocator for 31 GB. Every count-driven allocation on a decode path is now bounded by what the bytes can hold before it happens. Those are exactly the bug classes this document declares in scope; a `.urna` that panics or aborts the runtime is still a security bug, please report it.
- Untrusted `.urna` files: the header/section/footer checksums are unkeyed SHA-256 (corruption detection, NOT authenticity); an attacker can recompute them, so `validate()` does not prove a file is trustworthy. Safety against a hostile file rests on the parser's memory-safety (bounds-checked indices, capped decompression/allocation); opening an untrusted corpus still executes that parser, so treat unknown `.urna` files with the same care as any untrusted input.
- Release provenance: commits are signed (SSH signing). Checksums: the five binary archives and the embedder payload carry a per-file `.sha256` (the installers and `urna setup` check the payload's before unpacking), and `sha256.sum` lists the archives and the npm package; the SBOM and the Homebrew formula carry none. Attestations: the five archives carry a Sigstore keyless build attestation (`gh attestation verify <archive> --repo hoffresearch/urna`); the payload, the SBOM, the npm package and the formula carry none (`gh release verify-asset` only proves they are the bytes GitHub stored). Release tags are annotated and SSH-signed and verified against `.github/trustkeys` (`.github/workflows/tagverify.yml`) before the host job releases anything or a publish job runs; the archive builds run beside that check. Every release ships one CycloneDX SBOM for the CLI package (`cargo cyclonedx`, not attested), the binaries embed their dependency tree (`cargo auditable`), every action in the workflows is pinned to a commit SHA (`release.yml` through cargo-dist's `github-action-commits` in `Cargo.toml`, which `rehearsal.yml` reads too), and `cargo deny` gates advisories, licenses and sources in CI (`deny.toml`). `Cargo.lock` is committed so the Rust dependency set is pinned and auditable. Since 0.5.4 the release attests the four wheels and attaches them to the GitHub release with a `.sha256` each, and publishes those same files to PyPI with OIDC trusted publishing and PEP 740 attestations (maintainer checklist step 4 in `docs/USAGE.md`). 0.5.0 to 0.5.3 reached PyPI on a token and carry no PEP 740 provenance.
- Model registry remote code: presets that require `trust_remote_code` (wemm, jina) load only with an explicit `allow_remote_code` opt-in in the build spec AND matching pinned SHA256 allowlists for the model-repo code files. A hash identifies a version, it does not make it safe; review the pinned files before trusting a new pin, and prefer building in an isolated environment when the model directory is not fully trusted. Sentence-transformers models run in a per-model worker process.

## Data governance

A shipped `.urna` is an immutable, self-contained, content-addressed file meant to be copied around (phones, edge nodes, air-gapped boxes). When the embedded content is personal data that design has consequences; this section records the posture so it is auditable.

- A `.urna` is a datastore, not a cache. It stores the canonical chunk text and `source_uri` in cleartext plus the embeddings (a derived representation of that text). There is no at-rest encryption in the format; `zstd` is compression, not confidentiality. A file built over personal or sensitive data (a clinical corpus, say) is a copy of that data and needs the same controls as the source, encryption at rest included (FileVault / APFS / LUKS on any volume that holds it). For special-category data that is a required control, not a residual risk.
- Erasure and rectification (LGPD art. 18, GDPR art. 16 and 17) cannot be honored by editing a distributed file in place. A citation is `urna://<content_hash>/<chunk_id>`, changing any chunk changes `content_hash` and invalidates every issued citation, and the runtime never opens a socket, so there is no callback channel to recall copies already shipped. Before shipping a `.urna` that contains personal data: establish a lawful basis and run a DPIA (GDPR art. 35, LGPD art. 38), prefer anonymized or CC0 / permissively licensed corpora (the `demo/corpora/intro` CC0 path exists for this), define the revocation process up front (version the corpus, each build has its own `content_hash` and `file_hash`; publish a revocation list; put an operational obligation on operators to re-pull the current build and destroy superseded copies; treat embeddings as in-scope derived personal data), and record consent and provenance for embedded third-party content. Do not distribute a `.urna` containing special-category data (health, etc.) without counsel confirming the immutable-distribution model is compatible with the applicable data-subject rights.
- Provenance and build integrity are compliance assets: reproducible builds (`reproducible=True` + same chunks + same model fingerprint = byte-identical `file_hash` on any machine), the four SHA-256 checks (an 8-byte header checksum, an 8-byte checksum per section over its physical bytes, `file_hash` over the whole file, and `content_hash` over the decoded canonical sections, stable across encodings), and the `model_hash` fingerprint that fails the honesty gate when a query is embedded by a different model. Declarative builds add the versioned build manifest (`manifest_schema_version`, canonical serialization, provenance redaction modes minimal / standard / full) and `build.lock.json` (package versions, tool binary hashes, model hashes, the materialized spec); a byte-identical rebuild claim is only valid under a matching lock. The on-disk checksums are unkeyed SHA-256: integrity, not authenticity (see the hardening notes above).
