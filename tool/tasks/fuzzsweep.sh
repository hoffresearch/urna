#!/bin/sh
# local coverage-guided fuzz soak: every cargo-fuzz
# target for SECONDS each (default 3600), corpus under fuzz/corpus/<target>
# (gitignored, accumulates across runs). run it after any decoder change,
# before the pr. a crash lands in fuzz/artifacts/<target>/; turn it into a
# rust/*/tests/negative_*.rs regression and a fuzz/seeds/regress-*.bin
# seed before fixing.
#
#   sh tool/tasks/fuzzsweep.sh            # 1 hour per target
#   sh tool/tasks/fuzzsweep.sh 600        # 10 minutes per target
#   URNA_FUZZ_TARGETS="section-decoders" sh tool/tasks/fuzzsweep.sh 300
#
# needs cargo-fuzz (`cargo install cargo-fuzz`) and the nightly CI pins
# (NIGHTLY in .github/workflows/gatecheck.yml, read from there;
# `rustup toolchain install <it>`); NIGHTLY=<toolchain> overrides it.

set -eu

SECONDS_PER_TARGET="${1:-3600}"
TARGETS="${URNA_FUZZ_TARGETS:-urna-view section-decoders runtime-indexes mmap-open-search zstd-decoder}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NIGHTLY="${NIGHTLY:-$(sed -n 's/^  NIGHTLY: //p' "$ROOT/.github/workflows/gatecheck.yml")}"
if [ -z "$NIGHTLY" ]; then
    echo "fuzzsweep: no NIGHTLY in .github/workflows/gatecheck.yml" >&2
    exit 2
fi
cd "$ROOT/fuzz"

for t in $TARGETS; do
    mkdir -p "corpus/$t"
    if [ -z "$(ls -A "corpus/$t")" ]; then
        cp seeds/*.bin ../rust/format/tests/fixtures/golden_v1_minimal.urna "corpus/$t/"
    fi
    echo "fuzzsweep: $t for ${SECONDS_PER_TARGET}s"
    cargo +"$NIGHTLY" fuzz run "$t" -- \
        -max_total_time="$SECONDS_PER_TARGET" -max_len=65536 -rss_limit_mb=4096
    cargo +"$NIGHTLY" fuzz cmin "$t" >/dev/null 2>&1 || true
done
echo "fuzzsweep: clean ($TARGETS)"
