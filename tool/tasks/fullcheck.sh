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
# names another commit) at target/gate/tree, the checks of gatecheck.yml
# and lintworks.yml that tool/tasks/gatesteps.tsv gives it, with CI's
# flags (debug, RUSTFLAGS=-D warnings) and URNA_TEST_REDUCED=1 (the three
# slowest tests on fewer cases; CI runs them whole). the map lists every
# check of those two workflows with the tier that runs it, or why none
# can: the format and av1 parity workspaces are the full tier's.
#
# full runs push, then the release gate on the working tree (what it skips:
# docs/USAGE.md section 10):
#   0. the name check (length table, exceptions, lexicon)
#   1. cargo build (release), then the PyO3 extension (.so) the tests load
#   2. cargo test/clippy/fmt (release profile), the 639-line guard
#   3. the python suites (tool/tasks/fullsuite.sh), ruff
#   4. presetrun --json on the benchmark corpus (tool/tasks/benchdata.py)
#   5. benchgate regression gates vs tool/bench/reference.json
#
# Exits non-zero on the first failure. A step that already passed with the
# same inputs is skipped (tool/tasks/stepcache.py): its key hashes the files
# it reads, the toolchain and the step's own text, so a run with nothing
# changed takes seconds. A run with nothing remembered starts with the push
# tier (its measured times: docs/CONTRIBUTING.md, "Local gate"), and its
# step 4 builds the twelve presets and mrl variants of the 23,335-chunk
# corpus, which share one text-codec choice (the first build makes it,
# about 40 s). System tools
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
# python_tests, the full tier's python suites.
# shellcheck source=tool/tasks/fullsuite.sh
. tool/tasks/fullsuite.sh

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
  env | grep '^URNA_' | grep -v -e '^URNA_OUT=' -e '^URNA_FRESH=' -e '^URNA_CLEANHOST=' -e '^URNA_CARDS_CORPUS=' -e '^URNA_SKIP_LOG=' -e '^URNA_GATE_REV=' -e '^URNA_PUSHGUARD=' | sort || true)"
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
# this run's skips, one `need<TAB>test<TAB>reason` per line. a full run
# starts with the push tier, whose own run already judged its skips by
# push's required needs: they are listed at the end but not judged again
# by full's, which would fail push's debug tree for lacking the release
# binary that only full builds.
export URNA_SKIP_LOG="$STAMPS/skips.log"
: > "$URNA_SKIP_LOG"
PUSH_SKIPS="$ROOT/target/gate/tree/target/fullcheck/skips.log"
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
# with URNA_AVIF_DIR set too, the av1 tests read that directory's .avif
# files in place of the corpus blobs, so each of them is in the key.
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
  PARITY_FILES=(--file "$URNA_CARDS_CORPUS")
  if [[ -n "${URNA_AVIF_DIR:-}" ]]; then
    for avif in "$URNA_AVIF_DIR"/*.avif; do PARITY_FILES+=(--file "$avif"); done
  fi
  cached parity "rust/neural rust/images rust/av1dec rust/engine rust/format Cargo.toml $NEEDS_IN" \
    "${PARITY_FILES[@]}" -- parity
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

# ---- python tests (tool/tasks/fullsuite.sh) ----
cached python-tests "$ALL_IN" -- python_tests

# ---- ruff (tool/tasks/tiersteps.sh) ----
# ruff missing from $PY is a skip here, a failure in the push tier and in ci.
cached ruff "$ALL_IN" -- ruff

# ---- cleanhost ----
# the install on a machine with nothing on it, which cleanhost.yml runs in CI
# on the paths of its own filter, read from it here: a tart macOS VM, else a
# docker or podman linux container. without either it says so and runs
# nothing, as URNA_CLEANHOST=0 does; neither marks the step passed. a
# filter ending in /** is keyed as its directory: left as a glob, the shell
# expanded it to every entry there, an ignored __pycache__ among them,
# which the key refuses as a path git sees no file under.
CLEANHOST_IN="$(sed -n 's/^      - "\(.*\)"$/\1/p' .github/workflows/cleanhost.yml | sed 's|/\*\*$||' | tr '\n' ' ')"
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
if [[ -s "$PUSH_SKIPS" ]]; then
  step "skipped needs of the push tier (judged by its own run, listed here)"
  # without this run's URNA_REQUIRE_*: the list, not a second verdict.
  drop=()
  while IFS= read -r var; do drop+=(-u "$var"); done \
    < <(env | sed -n 's/^\(URNA_REQUIRE_[A-Z_]*\)=.*/\1/p')
  env ${drop[@]+"${drop[@]}"} "$PY" tool/tests/testneeds.py report "$PUSH_SKIPS"
fi
step "skipped needs (python tool/tests/testneeds.py report $URNA_SKIP_LOG)"
"$PY" tool/tests/testneeds.py report "$URNA_SKIP_LOG"

# ---- summary ----
printf '\n\033[1;32m== fullcheck passed ==\033[0m\n'
printf '  baseline: %s\n' "$BASELINE"
printf '  post:     %s\n' "$OUT"
printf '  next:     tool/tasks/releasepr.sh X.Y.Z (docs/USAGE.md, maintainer checklist step 8)\n'
