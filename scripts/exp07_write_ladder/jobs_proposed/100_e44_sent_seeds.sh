#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 50-55 min GPU per seed [estimate: SENT save + eval ~45 min per seed + the DCLM first-chunk read of the pass-4 states of A and B and W0 ~5 min incl. model load]; seeds 102 and 103 = about 1.7-1.8 h in total. CPU steps ~3 min.
# E44 SENT replication on seeds 102 and 103 (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread; queue order: after job 97). Per seed: cell sent (O + every token of every exposure sentence in the window, chunks with O only) save then eval (worlds A+B, passes 1-4, --keep), then the first-chunk NLL on the DCLM validation subset (E37c method, e37_firstchunk_nll.py) for the saved PASS-4 states of A and B (W0 included).
# PINNED INSTALL, NO FETCH: step 0 checks the code out of the local repository at commit 7bdcfa078d1dafca5a9c9b22a8563b46d96c5058 (the commit must already be in the local clone, otherwise exit 4: fetch it first by hand); step 1 verifies sha256 (first 12 hex) of every file against the manifest below and stops (exit 5) on a mismatch. NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py (imported by s3.py and the counts tools) is VERIFIED ONLY, never overwritten: the lab's own copy is used; a difference prints a WARNING (the repo copy is the lab's autosnap copy of 2026-10-21 04:07Z).
# Output: $OUT/S3_sent_s{102,103}_{A,B}.json (.saved.json), meta_S3_sent_s{102,103}.json, states under $OUT/states/S3_sent_s{102,103}/{A,B}/, $OUT/e37_firstchunk_e44sents{102,103}.{json,csv}, $K3_COUNTS_DIR/k3_counts.{md,json} (K3 / K3b / SENT / BG / NOO per chunk for seeds 102, 103; CPU).
# CPU-only dry run (separate directory, nothing under the lab tree is touched):
#   git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e44sent FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e44sent/data && cd /tmp/dry_e44sent && PY=python3 S=scripts/exp07_write_ladder NO_INSTALL=1 CPU_ONLY=1 K3_COUNTS_DIR=/tmp/dry_e44sent_counts bash scripts/exp07_write_ladder/jobs_proposed/100_e44_sent_seeds.sh
#   (run it from the lab repository root; the worktree is the branch tip, which carries this job file and the pinned code files unchanged; the symlink gives gen_world the tokenizer at data/tokenizer/llama3/tokenizer.json; remove afterwards with: git worktree remove --force /tmp/dry_e44sent)
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=7bdcfa078d1dafca5a9c9b22a8563b46d96c5058
FILES="$D/s3.py $D/s3_sets.py $D/e43_sets_test.py $D/e44_sets_test.py $D/k3_counts.py $D/e42_k2_tokens.py $D/e37_firstchunk_nll.py $D/suppress.py "
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $(for F in $FILES; do [ "$F" = "$D/suppress.py" ] || echo $F; done)
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read H F; do
    [ -n "$H" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    if [ "$F" = suppress.py ]; then [ "$G" = "$H" ] || echo "WARNING suppress.py differs from the repo copy: have $G, repo $H (the lab copy is used)"; continue; fi
    [ "$G" = "$H" ] || { echo "VERSION MISMATCH $F: have $G, expected $H"; exit 5; }
  done <<'MANIFEST'
77ce2c79464e s3.py
5072cfae0d81 s3_sets.py
a256dbbcb9ec e43_sets_test.py
3dc2e49e33fc e44_sets_test.py
fe0ba0b228a6 k3_counts.py
9153e937b8c1 e42_k2_tokens.py
e167e94bc935 e37_firstchunk_nll.py
9dac9fdda99e suppress.py

MANIFEST
  echo "installed code verified against the manifest"
fi
# CPU (the same lines in the dry run): unit tests and the per-chunk masked counts of seeds 102 and 103 on the real sentence table (n_SENT, n_BG, n_NOO, K3, K3b)
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/k3_counts.py --selftest
$PY $S/k3_counts.py --seeds 102,103 --passes 4 --worlds AB --out-dir ${K3_COUNTS_DIR:-/tmp/k3_counts_sent_seeds}
[ -n "$CPU_ONLY" ] && exit 0
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
for SEED in 102 103; do
  for M in save eval; do
    $PY $S/s3.py --cell sent --world-seed $SEED --passes 4 --worlds AB --keep --tag sent --mode $M
  done
  if [ -f "$IDS" ]; then
    ARGS=""
    for W in A B; do ARGS="$ARGS --state SENT_s${SEED}_${W}_p4=$OUT/states/S3_sent_s${SEED}/$W/state_004.npz"; done
    $PY $S/e37_firstchunk_nll.py --ids $IDS --seed $SEED --tag e44sents$SEED $ARGS || echo "first-chunk NLL read failed for seed $SEED (the D results stand)"
  else
    echo "first-chunk NLL skipped: $IDS missing"
  fi
done
