#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 45 min GPU per seed [estimate: K3 save + eval ~40 min per seed as in the approval request, + the DCLM first-chunk read of the pass-4 states of A and B and W0 ~5 min incl. model load]; seeds 102 and 103 = about 1.5 h in total. CPU steps ~3 min.
# E43 K3 replication on seeds 102 and 103 (PROPOSED, NOT QUEUED; waits for Hans's approval; goes after job 87). Per seed: K3 (sentence-cover oracle: O + earlier-round sentences of the same triple, chunks with O only) save then eval (worlds A+B, passes 1-4, --keep), then the first-chunk NLL on the DCLM validation subset (E37c method, e37_firstchunk_nll.py) for the SAVED PASS-4 STATES of A and B (W0 is always included in that read).
# This job installs everything it needs itself (the earlier job 87 did not install k3_counts.py and an old copy overwrote /tmp/k3_counts): step 0 fetches the branch and checks out the files below, step 1 verifies their sha256 (first 12 hex) against the manifest of this job and stops (exit 5) if one differs. NO_INSTALL=1 skips the checkout (the files must already be the right version), SKIP_VERIFY=1 skips the check.
# Output: $OUT/S3_k3_s{102,103}_{A,B}.json (.saved.json), meta_S3_k3_s{102,103}.json, states under $OUT/states/S3_k3_s{102,103}/{A,B}/, $OUT/e37_firstchunk_e43k3s{102,103}.{json,csv}, /tmp/k3_counts_seeds/k3_counts.{md,json} (K3 / K3b / SENT / BG / NOO per chunk for seeds 102, 103; CPU).
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
FILES="$D/s3.py $D/s3_sets.py $D/e43_sets_test.py $D/e44_sets_test.py $D/k3_counts.py $D/e42_k2_tokens.py $D/e37_firstchunk_nll.py"
if [ -z "$NO_INSTALL" ]; then git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- $FILES; fi
if [ -z "$SKIP_VERIFY" ]; then
  while read H F; do
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$H" ] || { echo "VERSION MISMATCH $F: have $G, expected $H"; exit 5; }
  done <<'MANIFEST'
827afb358a78 s3.py
ed261a37e596 s3_sets.py
a256dbbcb9ec e43_sets_test.py
3dc2e49e33fc e44_sets_test.py
fe0ba0b228a6 k3_counts.py
9153e937b8c1 e42_k2_tokens.py
e167e94bc935 e37_firstchunk_nll.py
MANIFEST
  echo "installed code verified against the manifest"
fi
# CPU (run BEFORE queueing, the same lines): unit tests and the per-chunk masked counts of seeds 102 and 103 on the real sentence table
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/k3_counts.py --selftest
$PY $S/k3_counts.py --seeds 102,103 --passes 4 --worlds AB --out-dir ${K3_COUNTS_DIR:-/tmp/k3_counts_seeds}
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
for SEED in 102 103; do
  for M in save eval; do
    $PY $S/s3.py --cell k3 --world-seed $SEED --passes 4 --worlds AB --keep --tag k3 --mode $M
  done
  if [ -f "$IDS" ]; then
    ARGS=""
    for W in A B; do ARGS="$ARGS --state K3_s${SEED}_${W}_p4=$OUT/states/S3_k3_s${SEED}/$W/state_004.npz"; done
    $PY $S/e37_firstchunk_nll.py --ids $IDS --seed $SEED --tag e43k3s$SEED $ARGS || echo "first-chunk NLL read failed for seed $SEED (the D results stand)"
  else
    echo "first-chunk NLL skipped: $IDS missing"
  fi
done
