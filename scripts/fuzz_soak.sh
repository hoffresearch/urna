#!/bin/sh
# local coverage-guided fuzz soak: every cargo-fuzz
# target for SECONDS each (default 3600), corpus under fuzz/corpus/<target>
# (gitignored, accumulates across runs). run it after any decoder change,
# before the pr. a crash lands in fuzz/artifacts/<target>/; turn it into a
# crates/*/tests/negative_*.rs regression and a fuzz/seeds/regress-*.bin
# seed before fixing.
#
#   sh scripts/fuzz_soak.sh            # 1 hour per target
#   sh scripts/fuzz_soak.sh 600        # 10 minutes per target
#   URNA_FUZZ_TARGETS="section-decoders" sh scripts/fuzz_soak.sh 300
#
# needs the nightly toolchain and cargo-fuzz (`cargo install cargo-fuzz`).

set -eu

SECONDS_PER_TARGET="${1:-3600}"
TARGETS="${URNA_FUZZ_TARGETS:-urna-view section-decoders runtime-indexes mmap-open-search}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/fuzz"

for t in $TARGETS; do
    mkdir -p "corpus/$t"
    if [ -z "$(ls -A "corpus/$t")" ]; then
        cp seeds/*.bin ../crates/urna-format/tests/fixtures/golden_v1_minimal.urna "corpus/$t/"
    fi
    echo "fuzz-soak: $t for ${SECONDS_PER_TARGET}s"
    cargo +nightly fuzz run "$t" -- \
        -max_total_time="$SECONDS_PER_TARGET" -max_len=65536 -rss_limit_mb=4096
    cargo +nightly fuzz cmin "$t" >/dev/null 2>&1 || true
done
echo "fuzz-soak: clean ($TARGETS)"
