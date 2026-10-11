#!/usr/bin/env bash
# fullcheck.sh - full release verification pipeline.
#
# Runs the local release gate end-to-end (what it skips: docs/USAGE.md section 10):
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
#
# Before any step: exit 3 when the default corpus is not cached and downloads
# were not confirmed (6 when the cached file is not the pinned one), exit 9
# when URNA_BASELINE names no usable baseline, exit 10 when the baseline was
# measured on another corpus (tool/tasks/benchdata.py lists every code).

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

# ---- knobs ----
BASELINE="${URNA_BASELINE:-tool/bench/reference.json}"
QUERIES="${URNA_QUERIES:-100}"
K="${URNA_K:-10}"
OUT="${URNA_OUT:-/tmp/fullcheck_post.json}"
CORPUS="${URNA_CORPUS:-}"
FRESH="${URNA_FRESH:-0}"

if [[ -n "${URNA_PYTHON:-}" ]]; then
  PY="$URNA_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

# the corpus, before any step: a missing one stops the run here, with the
# command that fetches it, not after the tests.
if [[ -z "$CORPUS" ]]; then
  CORPUS="$("$PY" tool/tasks/benchdata.py fetch)"
fi
# the same for the baseline, which benchgate reads only at the last step: it
# must exist (exit 9) and be measured on this corpus (exit 10), never on
# another one whose numbers would pass or fail this run for no reason.
"$PY" tool/tasks/benchdata.py match "$BASELINE" "$CORPUS"

step() {
  printf '\n\033[1;36m== %s ==\033[0m\n' "$*" >&2
}

ok() {
  printf '\033[1;32m  PASS:\033[0m %s\n' "$*" >&2
}

# ---- step cache ----
# the inputs of each step, as paths git sees. the rust steps read the
# workspace; presetrun reads the crates that build and search a file, the
# python package that calls them, the bench and the corpus (#489); the test
# steps read the whole tree.
RUST_IN="rust Cargo.toml Cargo.lock clippy.toml rustfmt.toml"
BENCH_IN="rust/format rust/engine rust/bridge tool/bench Cargo.toml Cargo.lock"
ALL_IN="."
# the toolchain also lists the packages of the $PY environment, read with
# importlib.metadata, not pip (a uv venv has no pip).
TOOLCHAIN="$(rustc -vV; cargo -V; "$PY" -VV; uname -sm
  "$PY" -c 'import importlib.metadata as m
print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in m.distributions())))'
  printf 'queries=%s k=%s corpus=%s\n' "$QUERIES" "$K" "$CORPUS"
  env | grep '^URNA_' | grep -v -e '^URNA_OUT=' -e '^URNA_FRESH=' | sort || true)"
STAMPS="$ROOT/target/fullcheck"
mkdir -p "$STAMPS"

# cached NAME INPUTS [--file PATH]... [--output PATH]... -- FUNCTION
# runs FUNCTION unless NAME passed before with the same key and outputs.
cached() {
  local name="$1" inputs="$2" keyargs=() outs=()
  shift 2
  while [[ "$1" != -- ]]; do
    case "$1" in
      --file) keyargs+=(--file "$2") ;;
      --output) outs+=(--output "$2") ;;
      *) printf 'cached: unknown flag %s\n' "$1" >&2; exit 2 ;;
    esac
    shift 2
  done
  local fn="$2" key
  # shellcheck disable=SC2086 # INPUTS is a space-separated list of paths
  key="$("$PY" tool/tasks/stepcache.py key ${keyargs[@]+"${keyargs[@]}"} \
    --extra "$TOOLCHAIN" --extra "$(declare -f "$fn")" $inputs)"
  if [[ "$FRESH" != 1 ]] \
    && "$PY" tool/tasks/stepcache.py hit "$name" "$key" ${outs[@]+"${outs[@]}"}; then
    step "$name"
    printf '\033[1;33m  SKIP:\033[0m same inputs as its last pass\n' >&2
    return 0
  fi
  "$fn"
  "$PY" tool/tasks/stepcache.py mark "$name" "$key" ${outs[@]+"${outs[@]}"}
}

# ---- names ----
names() {
  step "python tool/tasks/namecheck.py"
  "$PY" tool/tasks/namecheck.py
  "$PY" tool/tests/test_namecheck.py
  ok "names follow the length table and the lexicon"
}
cached names "$ALL_IN" -- names

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

# ---- cargo clippy ----
cargo_clippy() {
  step "cargo clippy --workspace --all-targets -- -D warnings"
  cargo clippy --workspace --all-targets -- -D warnings >/dev/null 2>&1
  ok "clippy clean"
}
cached cargo-clippy "$RUST_IN" -- cargo_clippy

# ---- cargo fmt ----
cargo_fmt() {
  step "cargo fmt --all --check"
  cargo fmt --all --check
  ok "rustfmt clean"
}
cached cargo-fmt "$RUST_IN" -- cargo_fmt

# ---- 639-line guard ----
step "no Rust file in rust/**/src exceeds 639 lines"
overlong="$(find rust -name '*.rs' -not -path '*/tests/*' \
  | xargs wc -l \
  | awk '$1 > 639 {print}' \
  | grep -v 'total$' || true)"
if [[ -n "$overlong" ]]; then
  printf '\033[1;31m  FAIL:\033[0m\n%s\n' "$overlong" >&2
  exit 1
fi
ok "all source files ≤ 639 lines"

# ---- python tests ----
python_tests() {
  # the step cache of this script: a key moves with every input, never with
  # an ignored file, and a hit needs the outputs it recorded.
  step "python tool/tests/test_stepcache.py"
  "$PY" tool/tests/test_stepcache.py
  ok "stepcache (8 cases)"

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

  # the release pull request is prepared by cargo-release in its own worktree,
  # signed, and touches only the version, the lockfile, the changelog and
  # CITATION.cff; skips without the pinned cargo-release.
  step "python tool/tests/test_releasepr.py"
  "$PY" tool/tests/test_releasepr.py
  ok "releasepr (4 cases)"
}
cached python-tests "$ALL_IN" -- python_tests

# ---- ruff (best-effort) ----
# the file list lives in tool/tasks/ruffcheck.sh so gatecheck.yml and this gate stay
# in lockstep; ruff missing from $PY is a skip here, a failure in ci.
ruff() {
  if "$PY" -c "import ruff" 2>/dev/null || "$PY" -m ruff --version 2>/dev/null | head -1 >/dev/null; then
    step "ruff check / format on the files we own (tool/tasks/ruffcheck.sh)"
    URNA_PYTHON="$PY" sh tool/tasks/ruffcheck.sh
    ok "ruff clean"
  else
    printf '  skip: ruff not importable in %s\n' "$PY" >&2
  fi
}
cached ruff "$ALL_IN" -- ruff

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

# ---- summary ----
printf '\n\033[1;32m== fullcheck passed ==\033[0m\n'
printf '  baseline: %s\n' "$BASELINE"
printf '  post:     %s\n' "$OUT"
printf '  next:     tool/tasks/releasepr.sh X.Y.Z (docs/USAGE.md, maintainer checklist step 8)\n'
