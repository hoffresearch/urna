# tiersteps.sh - the quick and push tiers of tool/tasks/fullcheck.sh.
#
# sourced by fullcheck.sh, which defines step, ok, cached, uncached, skip
# and $PY before it. tool/tasks/gatesteps.tsv maps every check CI runs on
# a pull request to the step here that runs the same command, or says why
# none can (tool/tests/test_gatematch.py holds the three to each other):
#
#   tier_quick - the fast loop on the working tree: names, fmt, clippy,
#                the 639-line guard, ruff, the gate map and the tests that
#                need no end-to-end setup (the nextest `quick` profile).
#   tier_push  - the checks CI runs on a pull request that
#                tool/tasks/gatesteps.tsv gives this tier (the format and
#                av1 parity workspaces are the full tier's; what only CI
#                can run has its reason there), with CI's flags: the
#                debug profile, RUSTFLAGS=-D warnings, every clippy
#                variant, the read path without zstd, rust/ingest, fuzz/,
#                the cross-target checks, cargo deny, cargo-semver-checks,
#                the python job and the workflow linters. fullcheck.sh
#                runs it in a clean worktree of the commit
#                (target/gate/tree), never on the working tree.
#
# a command copied from a workflow is copied verbatim: the gate map checks
# every line of an `exact` entry against the function its step runs here.

# shellcheck shell=bash

# ---- the clean tree (fullcheck.sh runs it for push and full) ----
# the push tier checks the commit, never the working tree: a worktree of
# it at target/gate/tree, cleaned of every file git does not track, so an
# ignored demo output or a stale _urna.so cannot make it pass, as on a CI
# checkout. its target/ dirs stay between runs (git clean -e target/), so a
# push rebuilds what changed, not the world. the git-lfs files are checked
# out and verified. it runs before any need is exported here: the tree
# requires its own.
gate_tree() {
  local tree="$ROOT/target/gate/tree" rev
  # a hook's GIT_DIR would point `git -C "$tree"` at the checkout's index.
  unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_PREFIX
  rev="$(git rev-parse --verify "${URNA_GATE_REV:-HEAD}^{commit}")"
  step "the push tier on $rev, in a clean worktree ($tree)"
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    printf '\033[1;33m  NOTE:\033[0m the working tree has uncommitted changes; the push tier checks %s without them\n' "$rev" >&2
  fi
  git worktree prune
  if git worktree list --porcelain | grep -qxF "worktree $tree"; then
    git -C "$tree" checkout --quiet --detach --force "$rev"
  else
    rm -rf "$tree"
    git worktree add --quiet --detach "$tree" "$rev"
  fi
  git -C "$tree" clean -ffdxq -e target/
  if command -v git-lfs >/dev/null 2>&1; then
    git -C "$tree" lfs fsck >/dev/null
    if git -C "$tree" lfs ls-files | grep -q ' - '; then
      printf 'fullcheck: a git-lfs file of %s is a pointer in the tree (git lfs pull):\n' "$rev" >&2
      git -C "$tree" lfs ls-files | grep ' - ' >&2
      exit 1
    fi
  fi
  URNA_GATE_TREE=1 URNA_PYTHON="$PY" "$tree/tool/tasks/fullcheck.sh" --tier push
}
# ---- shared by the tiers ----

names() {
  step "python tool/tasks/namecheck.py"
  "$PY" tool/tasks/namecheck.py
  "$PY" tool/tests/test_namecheck.py
  ok "names follow the length table and the lexicon"
}

layers() {
  step "python tool/tasks/layerscan.py"
  "$PY" tool/tasks/layerscan.py
  "$PY" tool/tests/test_layerscan.py
  ok "urna keeps its import contracts"
}

cargo_fmt() {
  step "cargo fmt --all --check"
  cargo fmt --all --check
  ok "rustfmt clean"
}

# the file list lives in tool/tasks/ruffcheck.sh, as in gatecheck.yml.
ruff() {
  if "$PY" -m ruff --version >/dev/null 2>&1; then
    step "ruff check / format on the files we own (tool/tasks/ruffcheck.sh)"
    URNA_PYTHON="$PY" sh tool/tasks/ruffcheck.sh
    ok "ruff clean"
  else
    skip ruff "fullcheck: ruff" "ruff not importable in $PY"
  fi
}

# rust/av1dec is the rav1d fork vendored whole; its files keep upstream's shape.
guard639() {
  step "no Rust file in rust/**/src exceeds 639 lines"
  overlong="$(find rust -name '*.rs' -path '*/src/*' -not -path '*/tests/*' -not -path '*/target/*' -not -path 'rust/av1dec/*' \
    -exec wc -l {} + | awk '$1 > 639 {print}' | grep -v 'total$' || true)"
  if [ -n "$overlong" ]; then echo "$overlong"; exit 1; fi
  ok "all source files ≤ 639 lines"
}

# every check CI runs has its local step or the reason it has none.
gatematch() {
  step "python tool/tests/test_gatematch.py"
  "$PY" tool/tests/test_gatematch.py
  ok "gatematch: the CI checks and their local steps"
}

# ---- quick ----

# both clippy runs of the rust job, with the output shown on a failure.
cargo_clippy() {
  step "cargo clippy --workspace --all-targets -- -D warnings"
  cargo clippy --workspace --all-targets -- -D warnings
  cargo clippy -p urna --no-default-features --all-targets -- -D warnings
  ok "clippy clean"
}

quick_test() {
  step "cargo nextest run --workspace --profile quick (URNA_TEST_REDUCED=1)"
  URNA_TEST_REDUCED=1 cargo nextest run --workspace --profile quick
  ok "the quick tests"
}

tier_quick() {
  cached names "$ALL_IN" -- names
  cached layers "$ALL_IN" -- layers
  cached cargo-fmt "$RUST_IN" -- cargo_fmt
  cached ruff "$ALL_IN" -- ruff
  cached gatematch "$GATE_IN" -- gatematch
  uncached guard639 -- guard639
  cached cargo-clippy "$RUST_IN" -- cargo_clippy
  cached quick-test "$ALL_IN" -- quick_test
}

# ---- push: gatecheck.yml's rust job ----

ci_clippy() {
  step "cargo clippy: the workspace, the engine-only cli, the read path"
  cargo clippy --workspace --all-targets -- -D warnings
  cargo clippy -p urna --no-default-features --all-targets -- -D warnings
  cargo clippy -p urna-format --no-default-features -- -D warnings
  cargo clippy -p urna-engine -p urna-cbinds -- -D warnings
  readers="$(cargo tree -p urna-engine -p urna-cbinds -e normal,build)"
  if printf '%s\n' "$readers" | grep zstd; then
    echo "a reader depends on the zstd crate" >&2
    exit 1
  fi
  ok "clippy clean; the read path links no C"
}

# the windows release archive and the linux ones build on runners this mac
# is not: the crates that link no C type-check here for both targets
# (zstd-sys, the writer's libzstd, builds C with the target's compiler,
# which a mac has for neither; gatesteps.tsv says what CI alone checks).
cross_check() {
  local t
  for t in x86_64-unknown-linux-gnu x86_64-pc-windows-msvc; do
    if ! rustup target list --installed | grep -qx "$t"; then
      skip targets "fullcheck: cargo check --target $t" "the $t standard library is not installed"
      continue
    fi
    step "cargo check --target $t (the crates that link no C)"
    cargo check --target "$t" -p urna-format --no-default-features
    cargo check --target "$t" -p urna-engine -p urna-cbinds
  done
  ok "the read path type-checks for linux and windows x86_64"
}

ci_build() {
  step "cargo build --workspace --all-targets"
  cargo build --workspace --all-targets
  v=$(./target/debug/urna --version)
  echo "$v"
  case "$v" in
    *+dev.*) ;;
    *) echo "expected a development build name (<version>+dev.<sha>), got: $v" >&2; exit 1 ;;
  esac
  ok "debug build, a development build name"
}

# the rust job's forge step, against $PY instead of a pip install.
forge_so() {
  step "the potion table and the debug _urna extension against $PY"
  sh tool/tasks/getpotion.sh
  rm -f rust/bridge/python/urna/_urna.so
  PYO3_PYTHON="$PY" cargo build -p urna-bridge --features pyo3/extension-module
  case "$(uname)" in
    Darwin) cp target/debug/lib_urna.dylib rust/bridge/python/urna/_urna.so ;;
    *) cp target/debug/lib_urna.so rust/bridge/python/urna/_urna.so ;;
  esac
  ok "_urna.so built and copied"
}

# the forge's python is the first python3 on PATH, as setup-python's is in CI.
ci_test() {
  step "cargo nextest run --workspace --profile ci (URNA_TEST_REDUCED=1)"
  PATH="$(dirname "$PY"):$PATH" cargo nextest run --workspace --profile ci
  cargo test --workspace --doc
  ok "the workspace tests and the doctests"
}

ci_tone_parity() {
  step "cargo test in rust/clitui/parity"
  cargo test --locked --manifest-path rust/clitui/parity/Cargo.toml
  ok "xterm-256 fold matches coolor"
}

benches() {
  step "cargo bench -p urna-engine --no-run"
  cargo bench -p urna-engine --no-run
  ok "the benches compile"
}

mutation() {
  step "mutation fuzz at CI's counts (release)"
  URNA_MUTATION_ITERS=6000 cargo test --release -p urna-format --test mutation_fuzz
  URNA_MUTATION_ITERS=800 cargo test --release -p urna-engine --test mutation_fuzz
  ok "mutation fuzz"
}

ingest() {
  step "rust/ingest (separate workspace)"
  cargo fmt --manifest-path rust/ingest/Cargo.toml --all --check
  cargo clippy --manifest-path rust/ingest/Cargo.toml --all-targets -- -D warnings
  cargo test --manifest-path rust/ingest/Cargo.toml
  ok "rust/ingest"
}

fuzz_lints() {
  step "cargo clippy --manifest-path fuzz/Cargo.toml --all-targets"
  cargo clippy --manifest-path fuzz/Cargo.toml --all-targets -- -D warnings
  ok "fuzz/ lints"
}

# ---- push: the other gatecheck.yml jobs ----

# potion-native: the library (native) and numpy + tokenizers (forge) are
# both here, so the parity may skip neither.
potion_native() {
  step "potion through the C ABI against numpy + tokenizers"
  cargo build --release -p urna-cbinds
  local lib=target/release/liburna_cbinds.so
  [[ "$(uname)" == Darwin ]] && lib=target/release/liburna_cbinds.dylib
  URNA_REQUIRE_NATIVE=1 URNA_REQUIRE_FORGE=1 URNA_CBINDS="$PWD/$lib" "$PY" tool/tests/test_potiontab.py
  ok "native and reference agree"
}

# the advisory database is fetched every run, as in CI: never cached.
deny() {
  step "cargo deny check (the workspace, rust/ingest, fuzz)"
  if ! command -v cargo-deny >/dev/null 2>&1; then
    printf 'fullcheck: cargo-deny is not installed (cargo install cargo-deny --locked)\n' >&2
    exit 1
  fi
  # the arguments cargo-deny-action passes (its `arguments` default is
  # --all-features), with each step's command and command-arguments.
  cargo deny --log-level warn --manifest-path ./Cargo.toml --all-features \
    check advisories licenses bans sources
  cargo deny --log-level warn --manifest-path rust/ingest/Cargo.toml --all-features \
    check advisories licenses bans sources -A unmatched-skip
  cargo deny --log-level warn --manifest-path fuzz/Cargo.toml --all-features \
    check advisories licenses bans sources -A unmatched-skip
  ok "cargo deny"
}

# CI compares against the pull request's base commit; here, the commit
# where HEAD left origin/main (URNA_SEMVER_BASE names another).
semver() {
  local base="${URNA_SEMVER_BASE:-$(git merge-base HEAD origin/main)}"
  step "cargo semver-checks -p urna-format --baseline-rev $base"
  if ! git grep -q '^name = "urna-format"' "$base" -- '*Cargo.toml'; then
    printf '  base commit has no urna-format package; semver check skipped\n' >&2
    return 0
  fi
  cargo semver-checks -p urna-format --baseline-rev "$base"
  ok "urna-format's public api against $base"
}

# ---- push: gatecheck.yml's python job and lintworks.yml ----

python_job() {
  step "gatecheck.yml's python job"
  "$PY" tool/tests/test_atlasdocs.py
  "$PY" tool/tests/test_testneeds.py
  "$PY" tool/tests/test_lexifloor.py
  "$PY" tool/tests/test_lintmatch.py
  "$PY" rust/bridge/python/urna/model/catalogue.py --check
  "$PY" tool/tests/test_upstreams.py
  "$PY" tool/tests/test_stbackend.py --models none
  "$PY" tool/tests/test_nonetwork.py
  "$PY" tool/tasks/preflight.py
  "$PY" tool/tests/test_preflight.py
  "$PY" tool/tests/test_pypiindex.py
  # the dist CI downloads by its pinned hash; here the one on PATH, whose
  # version distpatch.py checks against Cargo.toml's cargo-dist-version.
  DIST="${DIST:-dist}" "$PY" tool/tasks/distpatch.py generate --check
  "$PY" tool/tests/test_distpatch.py
  "$PY" tool/tasks/rehearsal.py generate --check
  "$PY" tool/tests/test_rehearsal.py
  "$PY" tool/tests/test_chanprobe.py
  "$PY" tool/tests/test_pkgrender.py
  "$PY" tool/tests/test_ppasource.py
  "$PY" tool/tests/test_aptserver.py
  ok "the python job's suites"
}

# lintworks.yml: actionlint, then the bash of the composite actions
# (through shellcheck), then zizmor. zizmor's online audits (known-vulnerable actions)
# need a GitHub token: with GH_TOKEN exported it runs them, as CI does.
workflows() {
  local tool
  for tool in actionlint shellcheck zizmor; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      printf 'fullcheck: %s is not installed (brew install actionlint shellcheck zizmor)\n' "$tool" >&2
      exit 1
    fi
  done
  step "actionlint"
  SHELLCHECK_OPTS="-S warning" actionlint -color \
    -ignore 'property "output" is not defined in object type'
  step "shellcheck the composite actions"
  "$PY" - <<'PY'
import pathlib, subprocess, sys, tempfile, yaml
fail = 0
for action in sorted(pathlib.Path(".github/actions").glob("*/action.yml")):
    steps = yaml.safe_load(action.read_text(encoding="utf-8"))["runs"].get("steps", [])
    for i, step in enumerate(steps):
        if "run" not in step or step.get("shell") != "bash":
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/usr/bin/env bash\n" + step["run"])
        r = subprocess.run(["shellcheck", "-S", "warning", f.name], capture_output=True, text=True)
        if r.returncode:
            fail = 1
            print(f"{action} step {i} '{step.get('name', '')}':")
            print(r.stdout.replace(f.name, str(action)))
sys.exit(fail)
PY
  step "zizmor"
  if [[ -n "${GH_TOKEN:-}" ]]; then zizmor .; else zizmor --offline .; fi
  ok "the workflows lint clean"
}

tier_push() {
  # the static checks first: a misspelled name fails in seconds.
  cached names "$ALL_IN" -- names
  cached layers "$ALL_IN" -- layers
  cached ruff "$ALL_IN" -- ruff
  cached gatematch "$GATE_IN" -- gatematch
  cached python-job "$ALL_IN" -- python_job
  cached workflows ".github" -- workflows
  cached cargo-fmt "$RUST_IN" -- cargo_fmt
  uncached guard639 -- guard639
  cached cargo-clippy "$RUST_IN" -- ci_clippy
  cached ingest "rust/ingest" -- ingest
  cached fuzz-lints "fuzz $RUST_IN" -- fuzz_lints
  cached cross-check "$RUST_IN" -- cross_check
  cached ci-build "$RUST_IN" -- ci_build
  cached forge-so "$RUST_IN" --output rust/bridge/python/urna/_urna.so -- forge_so
  cached ci-test "$ALL_IN" -- ci_test
  cached tone-parity "rust/clitui/src/cmd/tone.rs rust/clitui/parity" -- ci_tone_parity
  cached benches "$RUST_IN" -- benches
  cached mutation "$RUST_IN" -- mutation
  cached potion-native "$ALL_IN" -- potion_native
  uncached deny -- deny
  uncached semver -- semver
}
