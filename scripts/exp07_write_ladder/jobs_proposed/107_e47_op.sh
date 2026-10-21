#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 35-40 min GPU [estimate from the design thread: the masked set O + P is smaller than K3, worlds A+B, 4 passes, --keep]. CPU steps ~3 min.
# E47 SENT-gain split, cell 'op' (O + P: the oracle tokens plus the sentences of the same pair, another triple (no K3 part); reference cell oracle) on seed 101 (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread; queue order: after the E46 jobs 97 / 98). Masked set, chunks WITH an oracle set only: O (name + place tokens of the earlier same-triple sentences, as the oracle cell) plus the extra set stated for the cell, WITHOUT the other tokens of the earlier same-triple sentences (no K3 part). P = tokens of sentences of the same pair, another triple (E44 type P); X = tokens of sentences of other pairs (E44 type X); the two do not overlap K3 (asserted per chunk, with SENT == K3 + P + X + later same-triple window tokens).
# PINNED INSTALL, NO FETCH: step 0 checks the code out of the local repository at commit $COMMIT (exit 4 if the commit is not in the local clone: fetch it first by hand); step 1 verifies sha256 (first 12 hex) of every file against the manifest and stops (exit 5) on a mismatch. NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py is VERIFIED ONLY (the lab copy is used, a difference prints a WARNING).
# Output: $OUT/S3_op_s101_{A,B}.json (.saved.json), meta_S3_op_s101.json, states under $OUT/states/S3_op_s101/{A,B}/ . The DCLM first-chunk damage read of the pass-4 states is a separate proposal (one run for several states).
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e47op FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e47op/data && cd /tmp/dry_e47op && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 E47_COUNTS_DIR=/tmp/dry_e47op_counts bash scripts/exp07_write_ladder/jobs_proposed/107_e47_op.sh   (afterwards: git worktree remove --force /tmp/dry_e47op)
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=1e24fae259f9da8240aa40dc4a6a991a273d7339
FILES="$D/s3.py $D/s3_sets.py $D/e43_sets_test.py $D/e44_sets_test.py $D/e46_sets_test.py $D/e47_sets_test.py $D/e46_counts.py $D/e47_counts.py $D/e47_p_split.py $D/k3_counts.py $D/e42_k2_tokens.py $D/suppress.py "
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $(for F in $FILES; do [ "$F" = "$D/suppress.py" ] || echo $F; done)
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    if [ "$F" = suppress.py ]; then [ "$G" = "$HH" ] || echo "WARNING suppress.py differs from the repo copy: have $G, repo $HH (the lab copy is used)"; continue; fi
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH"; exit 5; }
  done <<'MANIFEST'
189b8990da41 s3.py
be6bf37f43dd s3_sets.py
a256dbbcb9ec e43_sets_test.py
3dc2e49e33fc e44_sets_test.py
0b7f4b351d23 e46_sets_test.py
99da9b569218 e47_sets_test.py
b8e94c1cfb4d e46_counts.py
f3973540244b e47_counts.py
07e32342b1dd e47_p_split.py
fe0ba0b228a6 k3_counts.py
9153e937b8c1 e42_k2_tokens.py
9dac9fdda99e suppress.py

MANIFEST
  echo "installed code verified against the manifest"
fi
# CPU (the same lines in the dry run): unit tests and the per-chunk masked counts by kind (P, X, same-frame share) on the real sentence table, seed 101, worlds A and B, passes 1-4
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/e47_sets_test.py
$PY $S/e47_counts.py --selftest
$PY $S/e47_p_split.py --selftest >/dev/null
$PY $S/e47_counts.py --seeds 101 --passes 4 --worlds AB --out-dir ${E47_COUNTS_DIR:-/tmp/e47_counts}
[ -n "$CPU_ONLY" ] && exit 0
for M in save eval; do
  $PY $S/s3.py --cell op --world-seed 101 --passes 4 --worlds AB --keep --tag op --mode $M
done
