# contributing

`urna` is maintained by [hoff research](https://hoffresearch.com). author: brenner cruvinel ([brenner@hoffresearch.com](mailto:brenner@hoffresearch.com)). all contributions are welcome.

## how to contribute

1. fork the repo at https://github.com/hoffresearch/urna.
2. branch from `main`: `git checkout -b feature/short-description origin/main`.
3. keep each pr focused on one concern. small is better.
4. add or update tests for the change. new behavior needs a new test. write real tests against real artifacts (built .urna files, golden fixtures, real corpora), no mocks; cover the happy path, the error path, and one edge case.
5. if the change alters architecture, module boundaries, data flow, or doc locations, update the arc pair (`doc/arc/arc.yaml`, `doc/arc/arc.mmd`) in the same pr. keep both concise and pragmatic. do not add a separate human architecture doc; `arc.yaml` is both the machine map and the human reference.
6. run `./scripts/release_check.sh` locally before pushing. `.github/workflows/ci.yml` runs the same gate on the pr (minus the lfs corpus measurement), plus the mutation-fuzz harnesses and a cargo-fuzz smoke.
7. commit with a clear message in plain english. no conventional commits prefix.
8. open a pr against `main`. the maintainer squash merges it; `main` requires verified (ssh-signed) commits and linear history, so sign your commits (`git config commit.gpgsign true` with an ssh or gpg key registered on github).

## setup

requires rust edition 2024 (`rustc >= 1.85`) and python 3.12+.

```
git clone https://github.com/hoffresearch/urna.git
cd urna

cargo build --release --workspace
cp target/release/lib_urna.dylib python/_urna.so   # macOS
cp target/release/lib_urna.so   python/_urna.so    # linux

python3 -m venv .venv && source .venv/bin/activate
pip install ruff sentence-transformers pandas zstandard pyarrow
```

`dat/corpus_next.v1.urna` is tracked via git lfs. demo datasets under `dat/demo/` are local-only and gitignored; fetch them with the commands documented in `dat/demo/Instructions.md`. without those datasets, runtime unit tests still pass.

## conventions and writing style

these conventions are not aesthetic preferences. they exist to keep the repo readable for humans, agents, and vector search at the same time. if you find yourself wanting to break one, open an issue first and explain why; do not silently deviate. the goal is gentle communal pressure to keep the codebase legible.

### naming

- top-level directories and most root tokens use **3-char codes** (`dat`, `doc`, `ref`). this keeps directory glyphs atomic, easy to scan at a glance, and consistent across the hoff research repos that follow the same pattern. rust workspace conventions (`crates/`, `target/`) and language defaults (`python/`, `scripts/`, `tests/`) are kept as-is so the project stays idiomatic to its stack.
- multi-word documentation and asset names use **kebab-case in english** (`code-of-conduct.md`-style filenames, dataset folders, etc.).
- source files follow the conventions of their language (`snake_case.rs`, `snake_case.py`).
- when proposing renames or moves, list exact `mv` commands first, execute the move, fix every touched import, and run the test suite after.

### writing style

- write in **diataxis style**: separate tutorial, how-to, reference, and explanation. mixing them produces noise.
- **all lowercase** in body text. acronyms used as words keep their canonical case (`CLI`, `BM25`, `HNSW`, `SIMD`, `MIT`).
- **no emoji**, anywhere. **no em-dash** (`-`); use `,`, `;`, `.`, or a regular hyphen.
- short paragraphs, direct voice, no marketing copy. commits explain the **why**; the diff already shows the what. no conventional-commits prefix.
- every governance or architecture doc starts with a yaml frontmatter block (`project`, `audience`, `status`, `last-updated`, `domain`) so llm and vector tooling can resolve it semantically.

### agent instruction files

- `.contracts/.agents/AGENTS.md` is the single instruction source for ai coding agents working in this repo: use/update/init only `.contracts/.agents/AGENTS.md` (the core global agent file).
- do not create per-tool instruction files (CLAUDE.md, GEMINI.md, CODEX.md, cursor rules). most agentic tooling already reads .contracts/.agents/AGENTS.md by default; point the rest at it on init.

### file hygiene

human working memory holds four plus or minus one chunks at once (cowan, 2001). neural networks behave better the same way. a file that does not fit the mental window forces internal context switching and raises bug rates. this is the same principle ui designers apply to information density.

- **operational target for new files: 220 lines.** aim here.
- **hard limit: 333 lines.** above this, refactor along single-responsibility lines in the same pr.
- **rust source carve-out: 300 lines** for `crates/**/src/**`. test files and the `crates/urna-format/tests/roundtrip.rs` carve-out are exempt.

## code style

rust:

- edition 2024. `cargo fmt --all` enforced, rules pinned in `rustfmt.toml`.
- `cargo clippy --workspace --all-targets -- -D warnings` is a hard gate. suppress an individual lint with `#[allow(clippy::name)]` and a one-line justification, never globally.
- every `unsafe` block needs a `// SAFETY:` comment naming the invariant the caller is relying on.
- public items get a doc comment that explains the why, not the what. the name already says what.
- file hygiene as above: 300 lines for `crates/**/src/**`.

python:

- target `py312`, line length 100. ruff config in `pyproject.toml`.
- lints: `E F W I B UP SIM`. run `ruff check .` and `ruff format --check .`.
- private helpers in `python/tools/` use the `_` prefix (e.g. `_baseline_decoder.py`).
- file hygiene as above: 220-line target, 333-line hard limit.

format and runtime invariants:

the format is frozen at v1. any byte-level change either fits inside v1 (new section ids and encodings 4-255 are reserved) or bumps `URNA_FORMAT_VERSION` and ships as v2.

## tests

```
cargo test --release --workspace
python tests/test_e2e.py
python tests/test_builder.py
python tests/test_search_text_model_hash.py
./scripts/release_check.sh
```

`release_check.sh` is the source of truth. if it passes locally, ci passes.

two lints are denied workspace-wide and will fail the build: `clippy::unwrap_used` (tests are exempt; parse paths read fields through `urna_format::bytes`) and `clippy::undocumented_unsafe_blocks` (every `unsafe` block states its invariant in a `// SAFETY:` comment). a change to any section decoder or search path should also run the mutation harness, and a new codec gets an arm in `fuzz/fuzz_targets/section_decoders.rs`:

```
cargo test -p urna-format --test mutation_fuzz -p urna-runtime --test mutation_fuzz
URNA_MUTATION_ITERS=25000 cargo test --release -p urna-format --test mutation_fuzz
cargo +nightly fuzz run urna-view -- -max_total_time=600      # needs cargo-fuzz, see fuzz/README.md
```

## reporting issues

- bugs and feature requests: [github issues](https://github.com/hoffresearch/urna/issues).
- security vulns: do not open a public issue. email [brenner@hoffresearch.com](mailto:brenner@hoffresearch.com). target ack within 72 hours.
- questions about the format: open a discussion, or read `doc/arc/arc.yaml` and `doc/arc/arc.mmd`.

bug reports should include the `.urna` `file_hash` and `content_hash` (from `urna stats <file>`), the runtime `simd_backend` (also in `urna stats`), the exact cli or python invocation, and the error output.

## code of conduct

this project follows [code_of_conduct.md](CODE_OF_CONDUCT.md). by participating you agree to it.

## license

contributions are licensed under the [mit license](LICENSE). copyright vests in hoff research as the maintainer. mit keeps your right to use, copy, modify, distribute, or sublicense your own copies of the resulting software intact.
