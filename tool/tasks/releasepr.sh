#!/usr/bin/env bash
# releasepr.sh - open the pull request that prepares a release.
#
#   tool/tasks/releasepr.sh X.Y.Z [--base REF] [--worktree DIR] [--no-push]
#
# works in a new worktree of origin/main (never the current checkout) on the
# branch release-X.Y.Z:
#   1. cargo-release, at the version pinned below, sets the workspace version,
#      the urna-format and urna-engine pins and the lockfile (`cargo release
#      version`); towncrier, at the version pinned below, turns the news/
#      fragments into the release's dated changelog section and removes
#      them (`towncrier build`); cargo-release sets the versioned fields of
#      CITATION.cff (`cargo release replace`, the list in
#      rust/clitui/Cargo.toml); the rest of CITATION.cff does not move;
#   2. tool/tasks/preflight.py checks the result, and nothing outside
#      those four files and the removed fragments may have changed;
#   3. a signed commit, an explicit push of that one branch, the pull request.
# the tag and the publication are separate steps after the merge (usage,
# maintainer checklist step 8): nothing here tags, publishes or touches main.
#
# --base and --worktree pick another start and place; --no-push stops after
# the signed commit (tool/tests/test_releasepr.py runs it that way).

set -euo pipefail

CARGO_RELEASE_VERSION=1.1.6
TOWNCRIER_VERSION=26.9.0
FILES=(CITATION.cff Cargo.lock Cargo.toml docs/CHANGELOG)

die() { echo "releasepr: $*" >&2; exit 1; }

version="" base=origin/main worktree="" push=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) base="${2:?--base needs a ref}"; shift 2 ;;
    --worktree) worktree="${2:?--worktree needs a directory}"; shift 2 ;;
    --no-push) push=0; shift ;;
    -*) die "unknown option $1" ;;
    *) [[ -z "$version" ]] || die "one version only"; version="$1"; shift ;;
  esac
done
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || die "usage: releasepr.sh X.Y.Z [--base REF] [--worktree DIR] [--no-push]"

have="$(cargo release --version 2>/dev/null | awk '{print $2}' || true)"
[[ "$have" == "$CARGO_RELEASE_VERSION" ]] \
  || die "needs cargo-release $CARGO_RELEASE_VERSION, found ${have:-none}: cargo install cargo-release --version $CARGO_RELEASE_VERSION --locked"
have="$(towncrier --version 2>/dev/null | awk '{print $3}' || true)"
[[ "$have" == "$TOWNCRIER_VERSION" ]] \
  || die "needs towncrier $TOWNCRIER_VERSION, found ${have:-none}: uv tool install towncrier==$TOWNCRIER_VERSION"

repo="$(git rev-parse --show-toplevel)"
branch="release-$version"
worktree="${worktree:-${TMPDIR:-/tmp}/urna-release-$version}"
if [[ "$push" == 1 ]]; then
  git -C "$repo" fetch --quiet --tags origin main
fi
git -C "$repo" rev-parse --verify --quiet "$base^{commit}" >/dev/null || die "no commit $base"
! git -C "$repo" rev-parse --verify --quiet "refs/heads/$branch" >/dev/null || die "branch $branch already exists"
! git -C "$repo" rev-parse --verify --quiet "refs/tags/v$version" >/dev/null || die "tag v$version already exists"
[[ ! -e "$worktree" ]] || die "$worktree already exists"

# the version moves forward and news/ has something to release.
current="$(git -C "$repo" show "$base:Cargo.toml" | sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"$/\1/p')"
[[ -n "$current" ]] || die "no [workspace.package] version in $base:Cargo.toml"
newest="$(printf '%s\n%s\n' "$current" "$version" | LC_ALL=C sort -t. -k1,1n -k2,2n -k3,3n | tail -1)"
[[ "$version" != "$current" && "$newest" == "$version" ]] || die "$version is not after $current"
[[ -n "$(git -C "$repo" ls-tree --name-only "$base" -- news/)" ]] \
  || die "news/ has nothing to release"

GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" worktree add --quiet -b "$branch" "$worktree" "$base"
cd "$worktree"
cargo release version "$version" --execute --no-confirm
# cargo-release's {{date}} is the UTC day too, and preflight compares them.
towncrier build --version "$version" --date "$(date -u +%F)" --yes
cargo release replace --execute --no-confirm
python3 tool/tasks/preflight.py

# untracked files count too: anything a tool created is a change. towncrier
# removed every fragment (git rm), which is the one other change allowed.
changed="$(git status --porcelain | grep -v '^D  news/' | awk '{print $2}' | LC_ALL=C sort | tr '\n' ' ' || true)"
[[ "$changed" == "${FILES[*]} " ]] || die "expected changes in exactly ${FILES[*]}, got: $changed"
[[ -z "$(git ls-files -- news/)" ]] || die "towncrier left fragments in news/"
git add -- "${FILES[@]}"
date="$(sed -n 's/^date-released: "\(.*\)"$/\1/p' CITATION.cff)"
git commit --quiet -S -m "Release $version" -m "The workspace version, the urna-format and urna-engine pins, the lockfile,
the changelog section dated $date from the news/ fragments and the versioned
fields of CITATION.cff, prepared by tool/tasks/releasepr.sh with cargo-release
$CARGO_RELEASE_VERSION and towncrier $TOWNCRIER_VERSION."
echo "releasepr: $branch at $(git rev-parse --short HEAD) in $worktree, release date $date (UTC)"

if [[ "$push" == 0 ]]; then
  exit 0
fi
git push --quiet origin "refs/heads/$branch:refs/heads/$branch"
gh pr create --base main --head "$branch" --title "Release $version" --body "Prepares $version: the workspace version, the urna-format and urna-engine pins, the lockfile, the changelog section dated $date and the versioned fields of \`CITATION.cff\`, by \`tool/tasks/releasepr.sh\` (cargo-release $CARGO_RELEASE_VERSION). The preflight passed on this tree.

After CI and the required \`rehearsal\` check pass and this merges, the release is the signed tag on that merge commit, a separate step (\`docs/USAGE.md\`, maintainer checklist step 8). The tag's UTC day must not be before $date."
