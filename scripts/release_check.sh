#!/usr/bin/env bash
# release_check.sh — full release verification pipeline.
#
# Runs every CI gate end-to-end:
#   1. cargo build/test/clippy/fmt (release profile)
#   2. rebuild PyO3 extension (.so)
#   3. python tests: e2e, builder, search-text model_hash, image corpus
#   4. measure_presets --json on the LFS-tracked corpus
#   5. compare_measure regression gates vs data/measure/baseline.json
#
# Exits non-zero on the first failure. Total runtime ≈ 2–3 min on a
# warm cache (most of it is the measure_presets re-build of the four
# presets).
#
# Override knobs (env vars):
#   URNA_BASELINE  — baseline JSON to compare against (default: data/measure/baseline.json)
#   URNA_QUERIES   — measure_presets query count (default: 100)
#   URNA_K         — measure_presets top-k (default: 10)
#   URNA_PYTHON    — python interpreter (default: ./.venv/bin/python if present, else python3)
#   URNA_OUT       — where to write the post-run JSON (default: /tmp/release_check_post.json)

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ---- knobs ----
BASELINE="${URNA_BASELINE:-data/measure/baseline.json}"
QUERIES="${URNA_QUERIES:-100}"
K="${URNA_K:-10}"
OUT="${URNA_OUT:-/tmp/release_check_post.json}"

if [[ -n "${URNA_PYTHON:-}" ]]; then
  PY="$URNA_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

step() {
  printf '\n\033[1;36m== %s ==\033[0m\n' "$*" >&2
}

ok() {
  printf '\033[1;32m  PASS:\033[0m %s\n' "$*" >&2
}

# ---- cargo build (release) ----
step "cargo build --release --workspace"
cargo build --release --workspace
ok "release build"

# ---- cargo test (release) ----
step "cargo test --release --workspace"
cargo test --release --workspace 2>&1 \
  | grep -E "^(test result|running [0-9]+ tests)" \
  | awk '/^test result/ { passed += $4; failed += $6; ignored += $8 } END { printf "  passed=%d failed=%d ignored=%d\n", passed, failed, ignored; if (failed > 0) exit 1 }'
ok "all tests"

# ---- cargo clippy ----
step "cargo clippy --workspace --all-targets -- -D warnings"
cargo clippy --workspace --all-targets -- -D warnings >/dev/null 2>&1
ok "clippy clean"

# ---- cargo fmt ----
step "cargo fmt --all --check"
cargo fmt --all --check
ok "rustfmt clean"

# ---- 300-line guard ----
step "no Rust file in crates/**/src exceeds 300 lines"
overlong="$(find crates -name '*.rs' -not -path '*/tests/*' \
  | xargs wc -l \
  | awk '$1 > 300 {print}' \
  | grep -v 'total$' || true)"
if [[ -n "$overlong" ]]; then
  printf '\033[1;31m  FAIL:\033[0m\n%s\n' "$overlong" >&2
  exit 1
fi
ok "all source files ≤ 300 lines"

# ---- rebuild PyO3 .so ----
step "rebuild python/_urna.so"
# build the extension against the SAME interpreter that runs the tests, so a
# .venv that differs from the default build python can never load a mismatched
# _urna.so (that mismatch segfaults test_e2e). PYO3_PYTHON pins it to $PY.
# pyo3/extension-module keeps libpython OUT of the dylib (extension modules
# resolve symbols from the host process): without it the .so hard-links a
# libpython path and segfaults under statically-embedded interpreters (uv's
# python-build-standalone) by loading a second runtime. maturin builds the
# published wheel the same way.
PYO3_PYTHON="$PY" cargo build --release -p urna-python \
  --features pyo3/extension-module >/dev/null
case "$(uname)" in
  Darwin) cp target/release/lib_urna.dylib python/_urna.so ;;
  Linux)  cp target/release/lib_urna.so    python/_urna.so ;;
  *) printf "unknown OS, copy lib_urna.* manually\n" >&2; exit 1 ;;
esac
ok "_urna.so built and copied"

# ---- python tests ----
step "python tests/test_e2e.py"
"$PY" tests/test_e2e.py
ok "e2e"

step "python tests/test_builder.py"
"$PY" tests/test_builder.py
ok "builder"

step "python tests/test_search_text_model_hash.py"
"$PY" tests/test_search_text_model_hash.py
ok "search-text model_hash gate (5 cases)"

# builds its own dataset, so it needs no demo corpus; the compressed cases
# skip themselves when ffmpeg/libsvtav1 is absent.
step "python tests/test_image_corpus.py"
"$PY" tests/test_image_corpus.py
ok "image corpus pipeline (37 cases)"

# declarative builds: spec validation, fake-preset e2e, triad cache, dedup,
# output modes, L3 rebuild. no heavy ML deps; media legs skip without ffmpeg.
step "python tests/test_forge_spec.py"
"$PY" tests/test_forge_spec.py
ok "forge spec + pipeline"

# dual quality gate + jxl round-trip; skips cleanly without ssimulacra2/cjxl.
step "python tests/test_quality_gate.py"
"$PY" tests/test_quality_gate.py
ok "quality gate + jxl"

step "python tests/test_cli_space.py"
"$PY" tests/test_cli_space.py
ok "cli space verbs (8 cases)"

step "python tests/test_query_embedder_routing.py"
"$PY" tests/test_query_embedder_routing.py
ok "query embedder routing (4 cases)"

# ---- ruff (best-effort) ----
# the file list lives in scripts/ruff_check.sh so ci.yml and this gate stay
# in lockstep; ruff missing from $PY is a skip here, a failure in ci.
if "$PY" -c "import ruff" 2>/dev/null || "$PY" -m ruff --version 2>/dev/null | head -1 >/dev/null; then
  step "ruff check / format on the files we own (scripts/ruff_check.sh)"
  URNA_PYTHON="$PY" sh scripts/ruff_check.sh
  ok "ruff clean"
else
  printf '  skip: ruff not importable in %s\n' "$PY" >&2
fi

# ---- measure_presets + compare ----
step "python python/tools/measure_presets.py --n-queries $QUERIES --k $K --json"
"$PY" python/tools/measure_presets.py --n-queries "$QUERIES" --k "$K" --json > "$OUT"
ok "metrics written to $OUT"

step "python python/tools/compare_measure.py $BASELINE $OUT"
"$PY" python/tools/compare_measure.py "$BASELINE" "$OUT"
ok "regression gates"

# ---- summary ----
printf '\n\033[1;32m== release check passed ==\033[0m\n'
printf '  baseline: %s\n' "$BASELINE"
printf '  post:     %s\n' "$OUT"
printf '  next:     git tag -a vX.Y.Z -m "..." && git push --tags\n'
