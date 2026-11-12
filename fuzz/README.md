---
project: urna
audience: contributors
status: active
last-updated: 2026-09-22
domain: fuzzing
---

# Fuzzing

Coverage-guided fuzzing of every byte-level entry point in `urna-format` and
`urna-engine` with [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
(libFuzzer + AddressSanitizer). The contract is docs/SECURITY.md: a malformed
`.urna` may be rejected with a typed error, never with a panic, a hang, or an
out-of-bounds read.

The deterministic twins of these targets run on stable under plain `cargo
test` (`rust/format/tests/mutation_fuzz.rs`,
`rust/engine/tests/mutation_fuzz.rs`), so every CI run already
executes a few thousand mutations; this directory is the long soak.

## Targets

| Target | Entry point | What a crash means |
|---|---|---|
| `urna-view` | `UrnaView::from_bytes` + every section decoder + hashes | Reader / decoder bug |
| `section-decoders` | One section codec picked by the first byte, fed the rest | Codec bug reachable behind a valid container |
| `runtime-indexes` | `HnswIndex::from_bytes`, `Bm25Index::from_bytes`, `CsrIndex::from_bytes` | Index codec bug |
| `mmap-open-search` | File on disk, `MmapUrnaFile::open` + every search verb | Runtime open / search bug |
| `zstd-decoder` | urna-format's zstd decoder on raw bytes, with a dictionary, and on libzstd round trips | Decoder panic, or a disagreement with libzstd |

`urna-view` and `mmap-open-search` reseal half of their inputs (recompute
the header checksum, section checksums and footer hash) so the fuzzer gets
past the integrity layer and into the decoders; the other half tests the
integrity layer itself.

## Run

```sh
cargo install cargo-fuzz          # needs a nightly toolchain
cd fuzz
mkdir -p corpus/urna-view && cp seeds/*.bin ../rust/format/tests/fixtures/golden_v1_minimal.urna corpus/urna-view/
cargo +nightly fuzz run urna-view -- -max_total_time=600
cargo +nightly fuzz run section-decoders -- -max_total_time=600
cargo +nightly fuzz run runtime-indexes -- -max_total_time=600
cargo +nightly fuzz run mmap-open-search -- -max_total_time=600 -rss_limit_mb=4096
```

A finding lands in `artifacts/<target>/`; reproduce with `cargo +nightly fuzz
run <target> artifacts/<target>/<file>` and turn it into a negative test
under `rust/*/tests/` before fixing.

## Seeds

`seeds/*.bin` are real `.urna` files in every dtype / text-encoding
combination with every optional section present. Regenerate them from the
stable harness:

```sh
mkdir -p fuzz/seeds
URNA_FUZZ_SEED_DIR=$PWD/fuzz/seeds cargo test -p urna-format --test mutation_fuzz
URNA_FUZZ_SEED_DIR=$PWD/fuzz/seeds cargo test -p urna-engine --test mutation_fuzz
```

`gatecheck.yml` runs every target for a short bounded time on each push (smoke,
not soak) with these seeds as the corpus.
