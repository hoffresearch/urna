#!/usr/bin/env bash
# aptrender.sh - the signed apt index of every release's .deb, for urna.dev/apt.
#
#   APT_SIGNING_FPR=<fingerprint> tool/tasks/aptrender.sh OUT [REPO]
#
# downloads the .deb packages of every vX.Y.Z release of REPO (default: the
# repository gh points at) into OUT/pool/<tag>/, then writes OUT/dists/stable:
# Packages and Packages.gz per architecture (dpkg-scanpackages, so Filename
# is pool/<tag>/<file>, the path the urna.dev Worker redirects to the
# release asset), Release (apt-ftparchive, with the SHA-256 of each index),
# and its two signatures, InRelease and Release.gpg, by the key
# APT_SIGNING_FPR names, already in the gpg keyring. OUT/urna.gpg is that
# key's public half, the keyring users point signed-by at.
#
# the pool is only for the scan: aptsigner.yml uploads dists/ and urna.gpg,
# never a .deb.

set -euo pipefail

die() { echo "aptrender: $*" >&2; exit 1; }

out="${1:?usage: aptrender.sh OUT [REPO]}"
repo="${2:-}"
fpr="${APT_SIGNING_FPR:?APT_SIGNING_FPR is not set}"
for tool in gh gpg dpkg-scanpackages apt-ftparchive; do
  command -v "$tool" >/dev/null || die "needs $tool"
done
gpg --batch --list-secret-keys "$fpr" >/dev/null 2>&1 || die "no secret key $fpr in the keyring"

rm -rf "$out"
mkdir -p "$out/pool"
cd "$out"
repo_args=()
[[ -n "$repo" ]] && repo_args=(--repo "$repo")

tags="$(gh release list "${repo_args[@]}" --limit 200 --exclude-drafts --exclude-pre-releases \
  --json tagName --jq '.[].tagName' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true)"
[[ -n "$tags" ]] || die "no release"
for tag in $tags; do
  if gh release download "$tag" "${repo_args[@]}" --pattern 'urna_*.deb' --dir "pool/$tag" 2>/dev/null; then
    echo "aptrender: $tag" "pool/$tag"/*.deb
  fi
done
compgen -G 'pool/*/urna_*.deb' >/dev/null || die "no release has a .deb"

for arch in amd64 arm64; do
  dir="dists/stable/main/binary-$arch"
  mkdir -p "$dir"
  dpkg-scanpackages --arch "$arch" --multiversion pool /dev/null 2>/dev/null > "$dir/Packages"
  [[ -s "$dir/Packages" ]] || die "no $arch package"
  gzip -9nk "$dir/Packages"
done

apt-ftparchive \
  -o APT::FTPArchive::Release::Origin=urna \
  -o APT::FTPArchive::Release::Label=urna \
  -o APT::FTPArchive::Release::Suite=stable \
  -o APT::FTPArchive::Release::Codename=stable \
  -o APT::FTPArchive::Release::Architectures="amd64 arm64" \
  -o APT::FTPArchive::Release::Components=main \
  -o APT::FTPArchive::Release::Description="urna, the single-file vector database" \
  release dists/stable > dists/stable/Release.tmp
mv dists/stable/Release.tmp dists/stable/Release

gpg --batch --yes --local-user "$fpr" --clearsign -o dists/stable/InRelease dists/stable/Release
gpg --batch --yes --local-user "$fpr" --armor --detach-sign -o dists/stable/Release.gpg dists/stable/Release
gpg --batch --export "$fpr" > urna.gpg
echo "aptrender: $(grep -c '^Package:' dists/stable/main/binary-amd64/Packages) amd64 and $(grep -c '^Package:' dists/stable/main/binary-arm64/Packages) arm64 versions, signed by $fpr"
