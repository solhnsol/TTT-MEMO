#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 40 min GPU [design thread estimate from the old runs of the same cell: K3-all measured eval 5.5 min + save, SENT 50-55 min; worlds A and B, 4 passes, --keep]
# E48 frame-disjoint world (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread). World seed 101. E48_MODE=pair (default) = W1: every pair has its own R = 16 sentence frames, no frame shared between pairs (the two persons of a pair share the pair's frames); E48_MODE=sentence = W2: every exposure sentence has its own frame. Only the exposure sentence templates change (names, places, occupations, schedule, background order are the same rng stream; the stream positions shift a little because the sentence lengths differ); gen_world.py reads E48_FRAMES (this job exports it) and defaults to the old shared frames, bit-identical (e48_world_test.py --old). Result tags carry the prefix w1_ / w2_ so nothing of the old worlds is overwritten.
# PINNED INSTALL, NO FETCH: step 0 checks that the lab ladder.py is the lab version b9fd20049420 or the patched file and that the lab gen_world.py is the version the E48 patch was made on (c1ae126eadda is the patched file, base 1514a13e67c7), then checks the code out of the local repository at commit $COMMIT (exit 4 if absent: fetch by hand; exit 6 if a lab file is neither base nor patched); step 1 verifies sha256 (first 12 hex) against the manifest (exit 5). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py is VERIFIED ONLY (lab copy used).
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e48 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e48/data && cd /tmp/dry_e48 && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/132_e48_k6p.sh   (afterwards: git worktree remove --force /tmp/dry_e48)
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=7dc1b841bee081e90cc4118ba9edd2379d4669c7
FILES="scripts/exp07_write_ladder/s3.py scripts/exp07_write_ladder/s3_sets.py scripts/exp07_write_ladder/e43_sets_test.py scripts/exp07_write_ladder/e44_sets_test.py scripts/exp07_write_ladder/e46_sets_test.py scripts/exp07_write_ladder/e47_sets_test.py scripts/exp07_write_ladder/e42_k2_tokens.py scripts/exp07_write_ladder/k3_counts.py scripts/exp07_write_ladder/suppress.py scripts/exp07_write_ladder/ladder.py scripts/exp07_write_ladder/margins_split.py scripts/exp07_write_ladder/margins_split_test.py scripts/exp07_write_ladder/e48_world_test.py scripts/exp07_write_ladder/e48_world_stats.py scripts/exp04_what_ttt_retains/gen_world.py scripts/exp04_what_ttt_retains/frame_bank.py "
LADDER_OK="b9fd20049420 714b08ea3b90"
GW_OK="1514a13e67c7 c1ae126eadda"
E48_MODE="${E48_MODE:-pair}"
case "$E48_MODE" in pair) TP=w1;; sentence) TP=w2;; *) echo "E48_MODE must be pair or sentence"; exit 2;; esac
if [ -z "$NO_INSTALL" ]; then
  for X in "$D/ladder.py:$LADDER_OK" "$D4/gen_world.py:$GW_OK"; do
    GL=$(sha256sum ${X%%:*} | cut -c1-12); case " ${X#*:} " in *" $GL "*) ;; *) echo "lab file ${X%%:*} ($GL) is neither the lab version the patch was made on nor the patched file: not overwritten (exit 6), the code thread must rebase"; exit 6;; esac
  done
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $(for F in $FILES; do [ "$F" = "$D/suppress.py" ] || echo $F; done)
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $F | cut -c1-12)
    if [ "$F" = "$D/suppress.py" ]; then [ "$G" = "$HH" ] || echo "WARNING suppress.py differs from the repo copy: have $G, repo $HH (the lab copy is used)"; continue; fi
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH"; exit 5; }
  done <<'MANIFEST'
5442628d45b2 scripts/exp07_write_ladder/s3.py
b69477a8f778 scripts/exp07_write_ladder/s3_sets.py
a256dbbcb9ec scripts/exp07_write_ladder/e43_sets_test.py
3dc2e49e33fc scripts/exp07_write_ladder/e44_sets_test.py
0b7f4b351d23 scripts/exp07_write_ladder/e46_sets_test.py
99da9b569218 scripts/exp07_write_ladder/e47_sets_test.py
9153e937b8c1 scripts/exp07_write_ladder/e42_k2_tokens.py
fe0ba0b228a6 scripts/exp07_write_ladder/k3_counts.py
9dac9fdda99e scripts/exp07_write_ladder/suppress.py
714b08ea3b90 scripts/exp07_write_ladder/ladder.py
2638a505ef0c scripts/exp07_write_ladder/margins_split.py
1daebd776335 scripts/exp07_write_ladder/margins_split_test.py
70a70838d527 scripts/exp07_write_ladder/e48_world_test.py
364c5a73bb47 scripts/exp07_write_ladder/e48_world_stats.py
c1ae126eadda scripts/exp04_what_ttt_retains/gen_world.py
1927a9ee330b scripts/exp04_what_ttt_retains/frame_bank.py

MANIFEST
  echo "installed code verified against the manifest"
fi
export E48_FRAMES="$E48_MODE"
# CPU (the same lines in the dry run): unit tests of the old sets and of the new world (frame disjointness, slots, one exposure per triple and round, frame field of the sentence table; the old-code identity test needs --old and is run by the code thread)
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/e47_sets_test.py
$PY $S/margins_split_test.py
$PY $S/e48_world_test.py
[ -n "$CPU_ONLY" ] && exit 0
for M in save eval; do
  $PY $S/s3.py --cell k6p --world-seed 101 --passes 4 --worlds AB --keep --tag ${TP}_k6p --mode $M
done
