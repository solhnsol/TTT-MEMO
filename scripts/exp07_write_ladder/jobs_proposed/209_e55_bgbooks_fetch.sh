#!/bin/bash
# Fetch of 20 extra public-domain background books (Project Gutenberg) for the 512-chunk E55 world v3_64_4 (E57 section 12 recommendation D). No GPU, no model. Writes data/narrative_gutenberg/pg<ID>.txt (header / licence cut at the START / END markers, title checked) and the manifest $OUT/e55_bgbooks_manifest.json (id, title key, url, bytes, sha256). The 4 existing books and the canary books are not touched. The list and the reason are in scripts/exp07_write_ladder/e55_bgbooks.py (public-domain English novels; none of pg11 / pg1342 (canary) or the 4 v3 pool books).
# After it ran: the sha256 of every file are copied into scripts/exp07_write_ladder/e55_bgbooks_sha.json (pin file, committed by the code thread); job 210 verifies them. A failed book (wrong id, no network, no markers) = exit 6 with the list of failures; nothing is overwritten (existing files are kept).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. 1-5 minutes (20 downloads of 0.2-1.5 MB each, 3 URL patterns tried with 3 attempts each); needs outbound HTTPS to gutenberg.org from the lab server (untested: the code sandbox has no access). No GPU. CPU_ONLY=1 runs the offline selftest only.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_bk FETCH_HEAD && cd /tmp/dry_bk && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/209_e55_bgbooks_fetch.sh   (then git worktree remove --force /tmp/dry_bk)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH of code (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=904556b2aa276945a720dbec83860d2701b05c07
FILES="$D/e55_bgbooks.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
G=$(sha256sum $D/e55_bgbooks.py | cut -c1-12); [ "$G" = "3d8657814bbb" ] || { echo "VERSION MISMATCH e55_bgbooks.py: have $G (exit 5)"; exit 5; }
$PY $D/e55_bgbooks.py --selftest
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: selftest done, no download"; exit 0; }
MAN=$OUT/e55_bgbooks_manifest.json
if [ -f $MAN ] && [ -z "$RERUN" ]; then echo "manifest $MAN exists: not redone (exit 8; RERUN=1 to redo; existing book files are kept either way)"; exit 8; fi
mkdir -p data/narrative_gutenberg
$PY $D/e55_bgbooks.py fetch --dest data/narrative_gutenberg --manifest $MAN || { echo "some books failed (exit 6): see the list above"; exit 6; }
$PY $D/e55_bgbooks.py verify --dest data/narrative_gutenberg
$PY - <<PYEOF
import json;m=json.load(open("$MAN"))
print("book  bytes  sha256 (to be pinned in e55_bgbooks_sha.json):")
for r in m["books"]:print(f'pg{r["id"]}  {r["bytes"]}  {r["sha256"]}  ({r["key"]})')
print("total bytes", sum(r["bytes"] for r in m["books"]))
PYEOF
