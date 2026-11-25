#!/usr/bin/env bash
# fullcheck.sh - the local gate, in three tiers.
#
#   tool/tasks/fullcheck.sh --tier quick   the fast loop on the working tree
#   tool/tasks/fullcheck.sh --tier push    what CI runs on a pull request, on
#                                          the commit (the pre-push hook)
#   tool/tasks/fullcheck.sh [--tier full]  push, then the release gate below
#
# quick and push are tool/tasks/tiersteps.sh: quick runs names, fmt,
# clippy, the 639-line guard, ruff, the gate map and the tests that need no
# end-to-end setup; push runs, in a clean worktree of HEAD (URNA_GATE_REV
# names another commit) at target/gate/tree, every check of gatecheck.yml
# and lintworks.yml that can run on macOS arm64, with CI's flags (debug,
# RUSTFLAGS=-D warnings) and URNA_TEST_REDUCED=1 (the three slowest tests
# on fewer cases; CI runs them whole). tool/tasks/gatesteps.tsv lists every
# CI check with the tier that runs it, or why none can.
#
# full runs push, then the release gate on the working tree (what it skips:
# docs/USAGE.md section 10):
#   0. the name check (length table, exceptions, lexicon)
#   1. cargo build (release), then the PyO3 extension (.so) the tests load
#   2. cargo test/clippy/fmt (release profile), the 639-line guard
#   3. the python suites (the `step "python tool/tests/..."` lines below), ruff
#   4. presetrun --json on the benchmark corpus (tool/tasks/benchdata.py)
#   5. benchgate regression gates vs tool/bench/reference.json
#
# Exits non-zero on the first failure. A step that already passed with the
# same inputs is skipped (tool/tasks/stepcache.py): its key hashes the files
# it reads, the toolchain and the step's own text, so a run with nothing
# changed takes seconds. A cold run takes about 5 min on an m-series mac
# with a warm cargo cache: steps 1 to 3 about 3 min, and step 4 builds the
# twelve presets and mrl variants of the 23,335-chunk corpus, which share
# one text-codec choice (the first build makes it, about 40 s). System tools
# (ffmpeg, cjxl, ssimulacra2) and the network are not in the key: after
# changing any of them, run with URNA_FRESH=1.
#
# A test that lacks an external resource names the need it lacks
# (tool/tests/testneeds.tsv). The needs a tier provides itself
# (REQUIRED_NEEDS below) are required: lacking one fails the run. The
# others are optional: each skip goes to target/fullcheck/skips.log, a
# step that skipped is not remembered as passed, and the run ends with the
# table of what it skipped.
#
# Override knobs (env vars):
#   URNA_BASELINE  - baseline JSON to compare against (default: tool/bench/reference.json,
#                    measured on the default corpus; another corpus needs its own)
#   URNA_QUERIES   - presetrun query count (default: 100)
#   URNA_K         - presetrun top-k (default: 10)
#   URNA_PYTHON    - python interpreter (default: ./.venv/bin/python if present, else python3)
#   URNA_OUT       - where to write the post-run JSON (default: /tmp/fullcheck_post.json)
#   URNA_CORPUS    - the corpus presetrun measures (default: the pinned benchmark corpus in
#                    the hugging face cache; tool/tasks/benchdata.py fetches it, only with
#                    URNA_ALLOW_DOWNLOAD=1, and checks its sha-256)
#   URNA_FRESH     - 1 runs every step, ignoring what passed before (default: 0)
#   URNA_CARDS_CORPUS - the cards corpus the tokenizer and av1 parity compare on
#                    (default: unset, the parity step is reported skipped)
#   URNA_GATE_REV  - the commit the push tier checks (default: HEAD)
#   URNA_SEMVER_BASE - the baseline of cargo-semver-checks in the push tier
#                    (default: where HEAD left origin/main)
#
# Before any step: exit 3 when the default corpus is not cached and downloads
# were not confirmed (6 when the cached file is not the pinned one), exit 9
# when URNA_BASELINE names no usable baseline, exit 10 when the baseline was
# measured on another corpus (tool/tasks/benchdata.py lists every code); the
# full tier only. Exit 2 on a usage error.

set -euo pipefail

# a corpus named relative to where the script was called from, resolved
# before the cd below; cargo test runs each crate from its own directory too.
abspath() { if [[ -z "$1" || "$1" == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "$PWD" "$1"; fi; }
[[ -n "${URNA_CORPUS:-}" ]] && URNA_CORPUS="$(abspath "$URNA_CORPUS")"
[[ -n "${URNA_CARDS_CORPUS:-}" ]] && URNA_CARDS_CORPUS="$(abspath "$URNA_CARDS_CORPUS")" && export URNA_CARDS_CORPUS

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

TIER=full
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="${2:-}"; shift 2 || shift ;;
    --tier=*) TIER="${1#--tier=}"; shift ;;
    *) TIER="?$1"; break ;;
  esac
done
case "$TIER" in
  quick | push | full) ;;
  *) printf 'usage: tool/tasks/fullcheck.sh [--tier quick|push|full]\n' >&2; exit 2 ;;
esac
# the push tier runs itself again inside the clean tree, with this set.
INNER="${URNA_GATE_TREE:-0}"
# CI's flags (gatecheck.yml's env), and the slowest tests on fewer cases;
# set before the step keys are computed, which read them.
if [[ "$TIER" == push ]]; then
  export RUSTFLAGS="-D warnings" URNA_TEST_REDUCED=1
fi

# ---- knobs ----
BASELINE="${URNA_BASELINE:-tool/bench/reference.json}"
QUERIES="${URNA_QUERIES:-100}"
K="${URNA_K:-10}"
OUT="${URNA_OUT:-/tmp/fullcheck_post.json}"
CORPUS="${URNA_CORPUS:-}"
FRESH="${URNA_FRESH:-0}"

# hnsw_recall's graph_bytes_are_pinned prints the digests and asserts
# nothing while PIN_HNSW_DIGESTS is set: a mode for re-pinning them, which
# a gate must not pass on.
if [[ -n "${PIN_HNSW_DIGESTS+set}" ]]; then
  printf 'fullcheck: PIN_HNSW_DIGESTS is set (the hnsw digest re-pin mode); unset it to run the gate\n' >&2
  exit 2
fi

if [[ -n "${URNA_PYTHON:-}" ]]; then
  PY="$URNA_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

# the corpus, before any step: a missing one stops the run here, with the
# command that fetches it, not after the tests.
if [[ "$TIER" == full ]]; then
  if [[ -z "$CORPUS" ]]; then
    CORPUS="$("$PY" tool/tasks/benchdata.py fetch)"
  fi
  # the same for the baseline, which benchgate reads only at the last step: it
  # must exist (exit 9) and be measured on this corpus (exit 10), never on
  # another one whose numbers would pass or fail this run for no reason.
  "$PY" tool/tasks/benchdata.py match "$BASELINE" "$CORPUS"
fi

step() {
  printf '\n\033[1;36m== %s ==\033[0m\n' "$*" >&2
}

ok() {
  printf '\033[1;32m  PASS:\033[0m %s\n' "$*" >&2
}

# the steps of the quick and push tiers, and those the full one shares.
# shellcheck source=tool/tasks/tiersteps.sh
. tool/tasks/tiersteps.sh

# ---- step cache ----
# the inputs of each step, as paths git sees. the rust steps read the
# workspace; presetrun reads the crates that build and search a file, the
# python package that calls them, the bench, benchdata.py (which presetrun
# imports) and the corpus (#489); the test steps read the whole tree. the
# parity workspaces build their path dependencies against the root
# Cargo.toml (`workspace = true`) and include the needs helper and table.
RUST_IN="rust Cargo.toml Cargo.lock clippy.toml rustfmt.toml"
BENCH_IN="rust/format rust/engine rust/bridge tool/bench tool/tasks/benchdata.py Cargo.toml Cargo.lock"
ALL_IN="."
NEEDS_IN="rust/clitui/tests/common/needs.rs tool/tests/testneeds.tsv"
GATE_IN=".github tool/tasks/gatesteps.tsv tool/tasks/fullcheck.sh tool/tasks/tiersteps.sh tool/tests/test_gatematch.py"
# the toolchain also lists the packages of the $PY environment, read with
# importlib.metadata, not pip (a uv venv has no pip).
TOOLCHAIN="$(rustc -vV; cargo -V; "$PY" -VV; uname -sm
  "$PY" -c 'import importlib.metadata as m
print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in m.distributions())))'
  printf 'queries=%s k=%s corpus=%s RUSTFLAGS=%s\n' "$QUERIES" "$K" "$CORPUS" "${RUSTFLAGS:-}"
  env | grep '^URNA_' | grep -v -e '^URNA_OUT=' -e '^URNA_FRESH=' -e '^URNA_CLEANHOST=' -e '^URNA_CARDS_CORPUS=' -e '^URNA_SKIP_LOG=' | sort || true)"
STAMPS="$ROOT/target/fullcheck"
mkdir -p "$STAMPS"

# ---- the clean tree (push, full; gate_tree in tool/tasks/tiersteps.sh) ----
if [[ "$INNER" != 1 && "$TIER" != quick ]]; then
  gate_tree
  if [[ "$TIER" == push ]]; then
    printf '\n\033[1;32m== fullcheck --tier push passed ==\033[0m\n'
    exit 0
  fi
fi

# ---- needs (tool/tests/testneeds.tsv) ----
# what a tier provides itself, so a test that lacks it fails. push: the
# forge python ($PY, the repo .venv), the git-lfs files, the cbinds
# library it builds, ruff and the cross-check targets, what CI's jobs
# provide. full adds what the release build leaves in target/release (the
# binary) and the release tools (checklist step 8). quick requires none.
# the cards corpus, the avif dir, the media tools, the network, the
# sentence-transformers and siglip2 stacks and a clean host are optional
# everywhere: listed at the end, never failed.
case "$TIER" in
  quick) REQUIRED_NEEDS="" ;;
  push) REQUIRED_NEEDS="forge lfs native ruff targets" ;;
  full) REQUIRED_NEEDS="forge lfs native binary release" ;;
esac
for need in $REQUIRED_NEEDS; do
  export "URNA_REQUIRE_$(printf '%s' "$need" | tr '[:lower:]' '[:upper:]')=1"
done
# this run's skips, one `need<TAB>test<TAB>reason` per line; a full run
# lists the push tier's too.
export URNA_SKIP_LOG="$STAMPS/skips.log"
: > "$URNA_SKIP_LOG"
if [[ "$TIER" == full ]]; then
  cat "$ROOT/target/gate/tree/target/fullcheck/skips.log" >> "$URNA_SKIP_LOG"
fi
# skip NEED WHAT REASON: a skip of fullcheck's own (a step it cannot run),
# through the same contract; a required need fails here.
skip() { "$PY" tool/tests/testneeds.py skip "$@"; }
skipped_bytes() { wc -c < "$URNA_SKIP_LOG" | tr -d ' '; }

# cached NAME INPUTS [--file PATH]... [--extra TEXT]... [--output PATH]... -- FUNCTION
# runs FUNCTION unless NAME passed before with the same key and outputs.
cached() {
  local name="$1" inputs="$2" keyargs=() outs=()
  shift 2
  while [[ "$1" != -- ]]; do
    case "$1" in
      --file) keyargs+=(--file "$2") ;;
      --extra) keyargs+=(--extra "$2") ;;
      --output) outs+=(--output "$2") ;;
      *) printf 'cached: unknown flag %s\n' "$1" >&2; exit 2 ;;
    esac
    shift 2
  done
  local fn="$2" key from
  # shellcheck disable=SC2086 # INPUTS is a space-separated list of paths
  key="$("$PY" tool/tasks/stepcache.py key ${keyargs[@]+"${keyargs[@]}"} \
    --extra "$TOOLCHAIN" --extra "$(declare -f "$fn")" $inputs)"
  if [[ "$FRESH" != 1 ]] \
    && "$PY" tool/tasks/stepcache.py hit "$name" "$key" ${outs[@]+"${outs[@]}"}; then
    step "$name"
    printf '\033[1;33m  SKIP:\033[0m same inputs as its last pass\n' >&2
    return 0
  fi
  from="$(skipped_bytes)"
  "$fn"
  # a run that skipped a test for a missing need is not stamped.
  "$PY" tool/tasks/stepcache.py mark "$name" "$key" ${outs[@]+"${outs[@]}"} \
    --skip-log "$URNA_SKIP_LOG" --skip-from "$from"
}

# uncached NAME -- FUNCTION: a step that runs every time (the network's
# answer, or a check cheaper than its key).
uncached() {
  [[ "$2" == -- ]] || { printf 'uncached: usage: uncached NAME -- FUNCTION\n' >&2; exit 2; }
  "$3"
}

case "$TIER" in
  quick | push)
    "tier_$TIER"
    step "skipped needs (python tool/tests/testneeds.py report $URNA_SKIP_LOG)"
    "$PY" tool/tests/testneeds.py report "$URNA_SKIP_LOG"
    [[ "$INNER" == 1 ]] || printf '\n\033[1;32m== fullcheck --tier %s passed ==\033[0m\n' "$TIER"
    exit 0
    ;;
esac

# ---- tier full: the release gate on the working tree ----
# every `cached` step from here down is the full tier's own; the push
# tier's ran above, in the clean tree.
cached names "$ALL_IN" -- names
cached layers "$ALL_IN" -- layers

# ---- cargo build (release) ----
cargo_build() {
  step "cargo build --release --workspace"
  cargo build --release --workspace
  ok "release build"
}
cached cargo-build "$RUST_IN" -- cargo_build

# ---- rebuild PyO3 .so ----
# before cargo test: the cli e2e tests (cli_e2e.rs) build their demo corpus
# through the urna package, so a fresh checkout without
# rust/bridge/python/urna/_urna.so failed
# there before reaching this step. the copy is what the tests load; a later
# cargo build of urna-bridge without the feature does not touch it.
bridge_so() {
  step "rebuild rust/bridge/python/urna/_urna.so"
  # build the extension against the SAME interpreter that runs the tests, so a
  # .venv that differs from the default build python can never load a mismatched
  # _urna.so (that mismatch segfaults test_pythonapi). PYO3_PYTHON pins it to $PY.
  # pyo3/extension-module keeps libpython OUT of the dylib (extension modules
  # resolve symbols from the host process): without it the .so hard-links a
  # libpython path and segfaults under statically-embedded interpreters (uv's
  # python-build-standalone) by loading a second runtime. maturin builds the
  # published wheel the same way.
  PYO3_PYTHON="$PY" cargo build --release -p urna-bridge \
    --features pyo3/extension-module >/dev/null
  # a new file, not a copy over the old one: macOS keeps the old code
  # signature for the inode and kills (SIGKILL) whatever loads the new bytes.
  rm -f rust/bridge/python/urna/_urna.so
  case "$(uname)" in
    Darwin) cp target/release/lib_urna.dylib rust/bridge/python/urna/_urna.so ;;
    Linux)  cp target/release/lib_urna.so    rust/bridge/python/urna/_urna.so ;;
    *) printf "unknown OS, copy lib_urna.* manually\n" >&2; exit 1 ;;
  esac
  ok "_urna.so built and copied"
}
cached bridge-so "$RUST_IN" --output rust/bridge/python/urna/_urna.so -- bridge_so

# ---- cargo test (release) ----
cargo_test() {
  step "cargo test --release --workspace"
  cargo test --release --workspace 2>&1 \
    | grep -E "^(test result|running [0-9]+ tests)" \
    | awk '/^test result/ { passed += $4; failed += $6; ignored += $8 } END { printf "  passed=%d failed=%d ignored=%d\n", passed, failed, ignored; if (failed > 0) exit 1 }'
  ok "all tests"
}
cached cargo-test "$ALL_IN" -- cargo_test

# ---- parity of the in-tree tokenizer and av1 decoder ----
# rust/neural/parity and rust/images/parity are workspaces of their own (the
# reference crates stay out of urna's lock and cargo-deny), so the workspace
# test above never runs them. their corpus pass reads the cards corpus named
# by URNA_CARDS_CORPUS; without it the step is a skip of the cards need.
parity() {
  step "cargo test --release in rust/neural/parity and rust/images/parity"
  URNA_PARITY_LIMIT="${URNA_PARITY_LIMIT:-2000}" cargo test --release \
    --manifest-path rust/neural/parity/Cargo.toml 2>&1 \
    | grep -E "^test result|panicked|FAILED"
  URNA_PARITY_LIMIT="${URNA_PARITY_LIMIT:-2000}" cargo test --release \
    --manifest-path rust/images/parity/Cargo.toml 2>&1 \
    | grep -E "^test result|panicked|FAILED"
  ok "tokenizer and av1 parity on $URNA_CARDS_CORPUS"
}
if [[ -n "${URNA_CARDS_CORPUS:-}" ]]; then
  cached parity "rust/neural rust/images rust/av1dec rust/engine rust/format Cargo.toml $NEEDS_IN" \
    --file "$URNA_CARDS_CORPUS" -- parity
else
  step "parity"
  skip cards "fullcheck: tokenizer and av1 parity" "URNA_CARDS_CORPUS unset"
fi

# ---- parity of urna-format's own f16, SHA-256 and zstd decoder ----
# rust/format/parity holds all three to the crates they replaced (`half`,
# `sha2` and libzstd), a workspace of its own so those stay out of the
# readers' build; one run covers them, and it needs no corpus.
format_parity() {
  step "cargo test --release in rust/format/parity"
  cargo test --release --manifest-path rust/format/parity/Cargo.toml 2>&1 \
    | grep -E "^test result|panicked|FAILED"
  ok "f16, sha-256 and zstd decoder parity"
}
cached format-parity "rust/format rust/engine Cargo.toml $NEEDS_IN" -- format_parity

# ---- parity of the cli's color fold ----
# rust/clitui/parity is a workspace of its own (coolor, the crate the fold
# replaced, stays out of urna's lock); it needs no corpus.
tone_parity() {
  step "cargo test in rust/clitui/parity"
  cargo test --locked --manifest-path rust/clitui/parity/Cargo.toml 2>&1 \
    | grep -E "^test result|panicked|FAILED"
  ok "xterm-256 fold matches coolor"
}
cached tone-parity "rust/clitui/src/cmd/tone.rs rust/clitui/parity" -- tone_parity

# ---- cargo clippy, cargo fmt, the 639-line guard (tool/tasks/tiersteps.sh) ----
cached cargo-clippy "$RUST_IN" -- cargo_clippy
cached cargo-fmt "$RUST_IN" -- cargo_fmt
uncached guard639 -- guard639

# ---- python tests ----
python_tests() {
  # the step cache of this script: a key moves with every input, never with
  # an ignored file, and a hit needs the outputs it recorded.
  step "python tool/tests/test_stepcache.py"
  "$PY" tool/tests/test_stepcache.py
  ok "stepcache (10 cases)"

  # the corpus of step 4: fetched only with consent, measured only if pinned.
  step "python tool/tests/test_benchdata.py"
  "$PY" tool/tests/test_benchdata.py
  ok "benchdata (7 cases)"

  # the atlas summary is the current state in at most three sentences; the
  # history is in docs/CHANGELOG.
  step "python tool/tests/test_atlasdocs.py"
  "$PY" tool/tests/test_atlasdocs.py
  ok "atlasdocs (4 cases)"

  # rust/ingest inherits nothing from the root workspace: its lint tables
  # and rust-version are a copy, held to the root's here.
  step "python tool/tests/test_lintmatch.py"
  "$PY" tool/tests/test_lintmatch.py
  ok "lintmatch (4 cases)"

  step "python tool/tests/test_pythonapi.py"
  "$PY" tool/tests/test_pythonapi.py
  ok "pythonapi"

  step "python tool/tests/test_ingestion.py"
  "$PY" tool/tests/test_ingestion.py
  ok "ingestion"

  step "python tool/tests/test_hashguard.py"
  "$PY" tool/tests/test_hashguard.py
  ok "hashguard: search-text model_hash gate (7 cases)"

  # builds its own dataset, so it needs no demo corpus; the compressed cases
  # skip themselves when ffmpeg/libsvtav1 is absent.
  step "python tool/tests/test_imagepipe.py"
  "$PY" tool/tests/test_imagepipe.py
  ok "imagepipe: image corpus pipeline (43 cases)"

  # declarative builds: spec validation, fake-preset e2e, triad cache, dedup,
  # output modes, L3 rebuild. no heavy ML deps; media legs skip without ffmpeg.
  step "python tool/tests/test_specrules.py"
  "$PY" tool/tests/test_specrules.py
  step "python tool/tests/test_chunkcols.py"
  "$PY" tool/tests/test_chunkcols.py
  ok "specrules: build spec + pipeline"

  # dual quality gate + jxl round-trip; skips cleanly without ssimulacra2/cjxl.
  step "python tool/tests/test_mediagate.py"
  "$PY" tool/tests/test_mediagate.py
  ok "mediagate: quality gate + jxl"

  step "python tool/tests/test_clispaces.py"
  "$PY" tool/tests/test_clispaces.py
  ok "clispaces: cli space verbs (8 cases)"

  step "python tool/tests/test_askrouter.py"
  "$PY" tool/tests/test_askrouter.py
  ok "askrouter: query embedder routing (10 cases; 7 to 10 need sentence-transformers, URNA_ST_PYTHON)"

  # the release payload, staged and run from outside the checkout: both query
  # embedders answer, the registry route names its missing deps, a half tree
  # does not stage.
  step "python tool/tests/test_embedpack.py"
  "$PY" tool/tests/test_embedpack.py
  ok "embedpack: embedder payload (3 cases)"

  # the st_multimodal model_hash of a synthetic snapshot is the value the code
  # had before the move to the urna package: the fingerprint's key names are
  # data hashed into model_hash. needs numpy only; --models none keeps the
  # model cases out, so the gate never loads a model (#488).
  step "python tool/tests/test_stbackend.py --models none"
  "$PY" tool/tests/test_stbackend.py --models none
  ok "stbackend: pinned model_hash"

  # importing the search-text embedder, the model registry and the image path
  # forces the hub offline unless URNA_ALLOW_DOWNLOAD=1.
  step "python tool/tests/test_nonetwork.py"
  "$PY" tool/tests/test_nonetwork.py
  ok "nonetwork: offline by default and the opt-in (6 cases)"

  # the model catalog setup offers is the registry's validated presets, with a
  # reason for every preset it leaves out.
  step "python tool/tests/test_catalogue.py"
  "$PY" tool/tests/test_catalogue.py
  ok "catalogue: model catalog (5 cases)"

  # the uv setup downloads and the python range it builds on are read from
  # their upstreams into upstreams.json, never written by hand (stubbed here).
  step "python tool/tests/test_upstreams.py"
  "$PY" tool/tests/test_upstreams.py
  ok "upstreams: uv, the python range and the sample (7 cases)"

  # the preset job: an unknown --only name is a usage error, a measured 0 is
  # the probe failing, and an mps that compiles no kernel (github's mac
  # runner) runs every preset on the cpu (subprocess stubbed, no urna).
  step "python tool/tests/test_presetjob.py"
  "$PY" tool/tests/test_presetjob.py
  ok "presetjob: preset job (4 cases)"

  # the model fetch: confirmed downloads only, the pinned files only, kept
  # only when the fingerprint is the catalog's (an 18 MB hub model; the
  # download cases skip by name when huggingface.co does not answer).
  step "python tool/tests/test_modelpull.py"
  "$PY" tool/tests/test_modelpull.py
  ok "modelpull: model install (5 cases)"

  # the benchmark rebuild builds beside the corpus and renames at the end, so
  # an interrupted gate never leaves target/bench without its corpora.
  step "python tool/tests/test_benchmark.py"
  "$PY" tool/tests/test_benchmark.py
  ok "benchmark: bench runner (5 cases)"

  # the version a release names agrees across the manifests, the lockfile,
  # CITATION.cff and the changelog; the tag checks need a tag and run in ci.
  step "python tool/tests/test_preflight.py"
  "$PY" tool/tests/test_preflight.py
  ok "preflight (7 cases)"

  # the pypi upload takes the release's wheels from the run its index allows,
  # and a rerun uploads only what the index does not have yet.
  step "python tool/tests/test_pypiindex.py"
  "$PY" tool/tests/test_pypiindex.py
  ok "pypiindex (5 cases)"

  # release.yml's edits (distpatch.py) land once on dist's text; the full
  # regenerate check needs the dist binary and runs in gatecheck.yml.
  step "python tool/tests/test_distpatch.py"
  "$PY" tool/tests/test_distpatch.py
  ok "distpatch (3 cases)"

  # the release rehearsal is generated from release.yml, publishes nothing,
  # and its required check fails a needed build that did not pass.
  step "python tool/tests/test_rehearsal.py"
  "$PY" tool/tests/test_rehearsal.py
  ok "rehearsal (7 cases)"

  # setuptest waits for the exact version on each registry, and the release
  # report names what every channel serves, a failed or cancelled run included.
  step "python tool/tests/test_chanprobe.py"
  "$PY" tool/tests/test_chanprobe.py
  ok "chanprobe (8 cases)"

  # each channel's manifest is rendered from the release: its URLs, its
  # hashes, the texts escaped for the file, or nothing written at all.
  step "python tool/tests/test_pkgrender.py"
  "$PY" tool/tests/test_pkgrender.py
  ok "pkgrender (3 cases)"

  # the PPA's source package: debian/ filled from the workspace texts, the
  # vendored crates' checksums, and a wrong tag refused before any write.
  step "python tool/tests/test_ppasource.py"
  "$PY" tool/tests/test_ppasource.py
  ok "ppasource (5 cases)"

  # what serves urna.dev/apt: created once, kept on a second run, and no
  # route of the Worker beyond urna.dev/apt/*.
  step "python tool/tests/test_aptserver.py"
  "$PY" tool/tests/test_aptserver.py
  ok "aptserver (3 cases)"

  # a test that lacks an external resource fails where the run requires
  # the need, and otherwise logs its skip (tool/tests/testneeds.tsv).
  step "python tool/tests/test_testneeds.py"
  "$PY" tool/tests/test_testneeds.py
  ok "testneeds (5 cases)"

  # the embedders' own self-tests: the stdlib lexical floor, the model
  # registry, the potion table through the C ABI against numpy + tokenizers
  # (the release build left the library in target/release), the offline
  # retrieve demo, and the siglip2 query path (the clip need: its packages
  # and pinned snapshot are optional here).
  step "python tool/tests/test_lexifloor.py"
  "$PY" tool/tests/test_lexifloor.py
  ok "lexifloor: the stdlib static embedder"
  step "python tool/tests/test_presetmap.py"
  "$PY" tool/tests/test_presetmap.py
  ok "presetmap: the model registry (12 cases)"
  step "python tool/tests/test_potiontab.py"
  "$PY" tool/tests/test_potiontab.py
  ok "potiontab: the potion embedder, native and reference (8 cases)"
  step "python tool/tests/test_retrieval.py"
  "$PY" tool/tests/test_retrieval.py
  ok "retrieval: the offline retrieve demo (3 cases)"
  step "python tool/tests/test_clipsnaps.py"
  "$PY" tool/tests/test_clipsnaps.py
  ok "clipsnaps: the siglip2 query path (2 cases)"

  # the bridge's media blob pair and multimodal space bands, on files built
  # through urna.build.
  step "python tool/tests/test_mediablob.py"
  "$PY" tool/tests/test_mediablob.py
  ok "mediablob: the blob pair (5 cases)"
  step "python tool/tests/test_spaceband.py"
  "$PY" tool/tests/test_spaceband.py
  ok "spaceband: the space bands (5 cases)"

  # the release pull request is prepared by cargo-release in its own worktree,
  # signed, and touches only the version, the lockfile, the changelog and
  # CITATION.cff; skips without the pinned cargo-release.
  step "python tool/tests/test_releasepr.py"
  "$PY" tool/tests/test_releasepr.py
  ok "releasepr (4 cases)"
}
cached python-tests "$ALL_IN" -- python_tests

# ---- ruff (tool/tasks/tiersteps.sh) ----
# ruff missing from $PY is a skip here, a failure in the push tier and in ci.
cached ruff "$ALL_IN" -- ruff

# ---- cleanhost ----
# the install on a machine with nothing on it, which cleanhost.yml runs in CI
# on the paths of its own filter, read from it here: a tart macOS VM, else a
# docker or podman linux container. without either it says so and runs
# nothing, as URNA_CLEANHOST=0 does; neither marks the step passed.
CLEANHOST_IN="$(sed -n 's/^      - "\(.*\)"$/\1/p' .github/workflows/cleanhost.yml | tr '\n' ' ')"
# an empty list would key no file, and the step would skip forever.
if [[ -z "${CLEANHOST_IN// /}" ]]; then
  printf 'fullcheck: read no path filter from .github/workflows/cleanhost.yml (its format changed?)\n' >&2
  exit 2
fi
CLEANHOST_MODE=""
if command -v "${TART:-tart}" >/dev/null 2>&1 && [[ "$(uname -sm)" == "Darwin arm64" ]]; then
  CLEANHOST_MODE=macos
elif command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1; then
  CLEANHOST_MODE=linux
fi
cleanhost() {
  step "sh tool/tasks/cleanhost.sh $CLEANHOST_MODE"
  sh tool/tasks/cleanhost.sh "$CLEANHOST_MODE"
  ok "cleanhost $CLEANHOST_MODE"
}
CLEANHOST_SKIP=""
[[ -z "$CLEANHOST_MODE" ]] && CLEANHOST_SKIP="no tart (TART=<path>), docker or podman here"
[[ "${URNA_CLEANHOST:-1}" == 0 ]] && CLEANHOST_SKIP="URNA_CLEANHOST=0"
if [[ -n "$CLEANHOST_SKIP" ]]; then
  step "cleanhost"
  skip cleanhost "fullcheck: cleanhost (CI runs it when these change: $CLEANHOST_IN)" \
    "$CLEANHOST_SKIP"
else
  # the mode is in the key: a pass in a linux container is not the macOS VM's.
  cached cleanhost "$CLEANHOST_IN" --extra "cleanhost mode: $CLEANHOST_MODE" -- cleanhost
fi

# ---- presetrun + compare ----
# one cached step: the metrics pass only when benchgate passes them, so a
# noisy run (latency on a busy machine) is measured again on the next run,
# never kept. the metrics are kept beside the stamp, so a skipped step still
# writes the numbers of the run that passed with these inputs to $OUT.
PRESETS="$STAMPS/presetrun.json"
bench() {
  step "python tool/bench/presetrun.py --baseline $CORPUS --n-queries $QUERIES --k $K --json"
  "$PY" tool/bench/presetrun.py --baseline "$CORPUS" --n-queries "$QUERIES" --k "$K" --json \
    > "$PRESETS.tmp"
  mv "$PRESETS.tmp" "$PRESETS"
  ok "metrics written to $PRESETS"

  step "python tool/bench/benchgate.py $BASELINE $PRESETS"
  "$PY" tool/bench/benchgate.py "$BASELINE" "$PRESETS"
  ok "regression gates"
}
cached bench "$BENCH_IN" --file "$CORPUS" --file "$BASELINE" --output "$PRESETS" -- bench
cp "$PRESETS" "$OUT"

# ---- what did not run ----
# every skip of this run, by need; a required one fails the run here too
# (its test failed already, unless something skipped without asking).
step "skipped needs (python tool/tests/testneeds.py report $URNA_SKIP_LOG)"
"$PY" tool/tests/testneeds.py report "$URNA_SKIP_LOG"

# ---- summary ----
printf '\n\033[1;32m== fullcheck passed ==\033[0m\n'
printf '  baseline: %s\n' "$BASELINE"
printf '  post:     %s\n' "$OUT"
printf '  next:     tool/tasks/releasepr.sh X.Y.Z (docs/USAGE.md, maintainer checklist step 8)\n'
