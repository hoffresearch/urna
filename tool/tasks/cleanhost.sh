#!/bin/sh
# tool/tasks/cleanhost.sh -- install urna the way a first-time user meets it,
# on a machine with nothing on it (no python, no uv, no checkout), and run
# `urna setup` end to end: the check epic #535 asks of every pull request.
#
#   sh tool/tasks/cleanhost.sh linux [--model NAME]
#       docker (or podman): debian trixie-slim with curl, ca-certificates
#       and xz-utils only, the binary built for this arch's linux-musl.
#   sh tool/tasks/cleanhost.sh macos [--model NAME] [--control]
#       tart: a disposable vanilla macOS VM. it has no Command Line Tools,
#       so /usr/bin/python3 is Apple's stub, the one that opens the
#       installer dialog when anything runs it. --control runs the stub at
#       the end, to prove the check below would see the dialog.
#
# both build a release of the current tree into a scratch dir (the binary
# archive and the embedder payload, each with its .sha256, the layout the
# release publishes; the binary is built with URNA_RELEASE_TAG=v<version>,
# as the release workflow builds it, since a development build looks for
# the checkout it came from, which a clean machine does not have), hand it to tool/tasks/installer.sh as
# URNA_RELEASE_BASE, then run `urna setup --yes --model NAME` (default
# minilm-multilingual) and `urna doctor`, and check:
#   - setup and doctor exit 0;
#   - the env's python is inside the range the payload's catalogue names;
#   - setup downloaded the uv the catalogue names (none was there before);
#   - setup saved the sample corpus the catalogue names, it validates, and
#     `urna ask` on it answers with a citation, outside any checkout;
#   - macos: Apple's Command Line Tools installer never opened, and
#     `xcode-select -p` still fails (nothing installed them on the way).
# the machine reaches the network for what a user's would: uv, a python
# (python-build-standalone, through uv), pypi and the hugging face hub (the
# model and the sample).
#
# needs: python3 3.11+, curl, and docker/podman (linux) or tart (macos);
# the potion table comes through tool/tasks/getpotion.sh. tart from
# `brew install openai/tools/tart` (after `brew trust --formula` on
# openai/tools/tart and openai/tools/softnet), or the release's tart.app with
# TART=<dir>/tart.app/Contents/MacOS/tart. ssh logs into the VM through
# SSH_ASKPASS (OpenSSH 8.4+), so no sshpass.
# a macOS VM takes ~25 GB of disk and 8 GB of RAM while it runs; it is
# deleted at the end, pass or fail.

set -eu
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

# pinned by tag and digest, like the Dockerfile; bump both together.
DEBIAN="debian:trixie-slim@sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f"
RUST="rust:1.98.1-bookworm@sha256:93ce27a88655056a51dbdd8f5f2d7ddc071c7b0070fb288a37b5a285fc83971e"
MACOS="ghcr.io/cirruslabs/macos-tahoe-vanilla:latest"

say() { printf 'cleanhost: %s\n' "$*"; }
die() { printf 'cleanhost: error: %s\n' "$*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: cleanhost.sh linux|macos [--model NAME] [--control]"
MODE="$1"
shift
MODEL="minilm-multilingual"
CONTROL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --control) CONTROL=1; shift ;;
        *) die "unknown flag: $1" ;;
    esac
done

VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("Cargo.toml","rb"))["workspace"]["package"]["version"])')"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/urna-cleanhost.XXXXXX")"
REL="$WORK/release"
mkdir -p "$REL"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# the release layout installer.sh reads: urna-<target>.tar.xz holding
# urna-<target>/urna, the payload tarball, a .sha256 beside each.
publish() {
    target="$1"
    bin="$2"
    mkdir -p "$WORK/urna-$target"
    cp "$bin" "$WORK/urna-$target/urna"
    tar -C "$WORK" -cJf "$REL/urna-$target.tar.xz" "urna-$target"
    sh tool/tasks/getpotion.sh >/dev/null
    python3 tool/tasks/embedpack.py "$WORK/stage" --tar "$REL/urna-embedder-payload.tar.gz" >/dev/null
    for f in "urna-$target.tar.xz" urna-embedder-payload.tar.gz; do
        printf '%s  %s\n' "$(sha256 "$REL/$f")" "$f" > "$REL/$f.sha256"
    done
    cp tool/tasks/installer.sh "$REL/installer.sh"
    say "release v$VERSION for $target staged in $REL"
}

# what runs on the clean machine, after the release is there at $1.
guest_script() {
    cat <<EOF
set -eu
fail() { printf 'cleanhost(guest): FAIL: %s\n' "\$*" >&2; exit 1; }
command -v uv >/dev/null 2>&1 && fail "uv is already installed: not a clean machine"
if [ "\$(uname -s)" = Darwin ]; then
    xcode-select -p >/dev/null 2>&1 && fail "the Command Line Tools are installed: not a clean machine"
else
    command -v python3 >/dev/null 2>&1 && fail "python3 is already installed: not a clean machine"
fi
URNA_RELEASE_BASE="file://$1" sh "$1/installer.sh"
export PATH="\$HOME/.local/bin:\$PATH"
urna setup --yes --model "$MODEL" || fail "urna setup exited \$?"
urna doctor || fail "urna doctor exited \$?"
data="\${XDG_DATA_HOME:-\$HOME/.local/share}/urna"
[ -x "\$data/bin/uv" ] || fail "setup did not install the catalogue's uv at \$data/bin/uv"
"\$data/venv/bin/python" - "\$data/python/urna/model/catalogue.json" <<'PY' || fail "the env is outside the catalogue's range"
import json, sys
spec = json.load(open(sys.argv[1]))["python"]["requires"]
lo, hi = (int(p.split("3.")[1]) for p in spec.split(","))
have = sys.version_info[:2]
print(f"cleanhost(guest): env python {have[0]}.{have[1]}, catalogue range {spec}")
sys.exit(0 if have[0] == 3 and lo <= have[1] < hi else 1)
PY
sample="\$(ls "\$data"/sample/*.urna 2>/dev/null | head -n 1)"
[ -n "\$sample" ] || fail "setup saved no sample corpus under \$data/sample"
urna validate "\$sample" >/dev/null || fail "urna validate rejects the sample"
(cd / && urna ask "\$sample" "imposto sobre pix" -k 1) | grep -q "urna://" \
    || fail "urna ask on the sample gave no cited hit"
echo "cleanhost(guest): sample \$sample validates and answers"
if [ "\$(uname -s)" = Darwin ]; then
    sleep 3
    pgrep -fl "Install Command Line Developer Tools" && fail "Apple's Command Line Tools installer opened"
    xcode-select -p >/dev/null 2>&1 && fail "the Command Line Tools got installed on the way"
    if [ "$CONTROL" = 1 ]; then
        /usr/bin/python3 --version >/dev/null 2>&1 || true
        sleep 5
        pgrep -fl "Install Command Line Developer Tools" >/dev/null \
            || fail "control: running the stub opened no dialog this check can see"
        echo "cleanhost(guest): control ok: the stub opens the installer, and setup never ran it"
    fi
fi
echo "cleanhost(guest): PASS"
EOF
}

linux() {
    if command -v docker >/dev/null 2>&1; then CT=docker
    elif command -v podman >/dev/null 2>&1; then CT=podman
    else die "linux needs docker or podman"; fi
    case "$(uname -m)" in
        arm64 | aarch64) arch=aarch64 ;;
        x86_64 | amd64) arch=x86_64 ;;
        *) die "unsupported arch $(uname -m)" ;;
    esac
    target="$arch-unknown-linux-musl"
    say "building urna for $target in $RUST"
    "$CT" run --rm -v "$ROOT:/src" -w /src -e "CC_${arch}_unknown_linux_musl=musl-gcc" \
        -e "URNA_RELEASE_TAG=v$VERSION" "$RUST" sh -c "
        apt-get update -qq && apt-get install -y -qq --no-install-recommends musl-tools >/dev/null &&
        rustup target add $target >/dev/null &&
        cargo build --locked --profile dist -p urna --target $target --target-dir /src/target/cleanhost"
    publish "$target" "target/cleanhost/$target/dist/urna"
    guest_script /release > "$REL/guest.sh"
    say "running the install in a clean $DEBIAN"
    "$CT" run --rm -v "$REL:/release:ro" "$DEBIAN" sh -c "
        apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates xz-utils >/dev/null &&
        sh /release/guest.sh"
}

# the vanilla images' only account is admin/admin; the VM is thrown away.
# ssh reads the password from an askpass helper, never from a terminal.
vm_auth() {
    printf '#!/bin/sh\necho admin\n' > "$WORK/askpass"
    chmod 700 "$WORK/askpass"
}
VM_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no"
vm_ssh() {
    # VM_OPTS splits into its words; "$@" is the host and the command, as given.
    # shellcheck disable=SC2086,SC2029
    SSH_ASKPASS="$WORK/askpass" SSH_ASKPASS_REQUIRE=force ssh $VM_OPTS "$@"
}
vm_scp() {
    # shellcheck disable=SC2086
    SSH_ASKPASS="$WORK/askpass" SSH_ASKPASS_REQUIRE=force scp $VM_OPTS "$@"
}
TART="${TART:-tart}"

macos() {
    command -v "$TART" >/dev/null 2>&1 || die "macos needs tart (brew install openai/tools/tart, or TART=<path>)"
    vm_auth
    [ "$(uname -m)" = arm64 ] || die "tart runs macOS VMs on apple silicon only"
    target="aarch64-apple-darwin"
    say "building urna for $target"
    URNA_RELEASE_TAG="v$VERSION" cargo build --locked --profile dist -p urna --target "$target"
    publish "$target" "target/$target/dist/urna"
    vm="urna-cleanhost-$$"
    trap '"$TART" stop "$vm" >/dev/null 2>&1 || true; "$TART" delete "$vm" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT
    say "cloning $MACOS into $vm (the first pull is ~25 GB)"
    "$TART" clone "$MACOS" "$vm"
    "$TART" run --no-graphics "$vm" >/dev/null 2>&1 &
    ip="$("$TART" ip --wait 180 "$vm")" || die "the VM got no ip"
    i=0
    until vm_ssh "admin@$ip" true 2>/dev/null; do
        i=$((i + 1))
        [ "$i" -lt 60 ] || die "no ssh on $ip"
        sleep 2
    done
    vm_scp -q -r "$REL" "admin@$ip:release"
    guest_script /Users/admin/release > "$WORK/guest.sh"
    say "running the install in $vm ($ip)"
    vm_ssh "admin@$ip" sh -s < "$WORK/guest.sh"
}

case "$MODE" in
    linux) trap 'rm -rf "$WORK"' EXIT; linux ;;
    macos) macos ;;
    *) die "unknown mode: $MODE (linux or macos)" ;;
esac
say "PASS: $MODE, urna setup --model $MODEL on a clean machine"
