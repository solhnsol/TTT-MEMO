#!/usr/bin/env bash
# Hourly code snapshot to GitHub branch `autosnap/a100` (run from cron on the A100).
#
# Commits every tracked change plus untracked, non-ignored files < 5 MB under
# scripts/, env/, docker/, upstream/ and top-level *.md (the repo is PUBLIC), using a
# throwaway index. It never touches HEAD, the real index, or the working tree, so
# it is safe while experiments or other git work run in this checkout.
# Also refreshes env/ (pip freezes) so a fresh machine can rebuild the venvs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BRANCH="autosnap/a100"
MAX_BYTES=$((5 * 1024 * 1024))
cd "$REPO"
exec 9>"$REPO/.git/autosnap.lock"
flock -n 9 || exit 0

bash "$REPO/scripts/backup/freeze_env.sh" >/dev/null 2>&1 || true

export GIT_INDEX_FILE="$(mktemp)"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree HEAD
git add -u
git ls-files -z -o --exclude-standard -- scripts env docker upstream ':(glob)*.md' | while IFS= read -r -d '' f; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -lt "$MAX_BYTES" ] && printf '%s\0' "$f"
done | git add --pathspec-from-file=- --pathspec-file-nul 2>/dev/null || true
tree="$(git write-tree)"

prev="$(git rev-parse -q --verify "refs/heads/$BRANCH" || true)"
base="${prev:-$(git rev-parse HEAD)}"
[ "$(git rev-parse "$base^{tree}")" = "$tree" ] && exit 0

parents=(-p "$base")
if [ -n "$prev" ] && ! git merge-base --is-ancestor HEAD "$prev"; then
  parents+=(-p HEAD)
fi
commit="$(git commit-tree "$tree" "${parents[@]}" -m "autosnap $(date -u +%FT%TZ) (HEAD $(git rev-parse --short HEAD))")"
git update-ref "refs/heads/$BRANCH" "$commit"
git push -q origin "refs/heads/$BRANCH:refs/heads/$BRANCH"
