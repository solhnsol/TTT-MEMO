#!/bin/bash
# JOB 1345x = ADD A NARROWER BAND ARM TO THE KEPT-WINDOW BLOCK STREAM (prepared 11/23 on Hans' request: 'later 256 or narrower band widths can be added without rerunning NAT or 1024'): the SAME wrapper e59_bs8_window.py (same cell blk16x8_c16, same seeds 82..85, same shuffled block assignment, same reads, save-apps and nat-loss-every as job 1345w), ONLY the band arm differs: BAND=256 (default; 128, 64, 32 the same way) -> --arms NATMASK --mask-arm ALLQ$BAND --tag bs8w$BAND into the SAME output directory as job 1345w (files E55_bs8w<BAND>_NATMASK_s<seed>.*). NATS / NATMASK(1024) are NOT run again. Needs job 1345w finished (its sidecars E55_bs8w_ctx_s<seed>.json and W0 are the reference): the analysis asserts that the context sidecars (sha256 of every context tail, items, lags) are identical and prints the max |W0 difference| across tags. Pinned runner hashes of 1296 are checked. Run: BAND=256 bash 1345x_bs8_window_band.sh (queue copy only after Hans' own sentence). Exploratory.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${SEEDS:=82 83 84 85}"; : "${SAVE:=16,32,48,64,80,96,112,128}"; : "${BAND:=256}"; : "${KEEP_STATES:=0}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e59_bs8_window; OD=$OUT/e57_blockstream8; TAG=bs8w$BAND
sha256sum -c $OD/pinned_sha256.txt || { echo "REUSE REFUSED: a runner file of job 1296 differs from the pinned hash"; exit 12; }
for SEED in $SEEDS; do test -f $O/E55_bs8w_NATS_s$SEED.saved.json -a -f $O/E55_bs8w_ctx_s$SEED.json || { echo "job 1345w output missing for seed $SEED: exit 13"; exit 13; }; done
sha256sum $D/e59_bs8_window.py $D/e59_bs8_window_analyze.py
for SEED in $SEEDS; do
  $PY $D/e59_bs8_window.py --cell blk16x8_c16 --seed $SEED --out $O --masks none --mask-arm ALLQ$BAND --arms NATMASK --save-apps $SAVE --nat-loss-every 8 --tag $TAG --shuffle-blocks
  echo "RUN bs8_window_band $BAND seed $SEED rc=$?"
done
$PY $D/e59_bs8_window_analyze.py --new $O --orig $OD --runs NATS=bs8w:NATS,NATMASK1024=bs8w:NATMASK,NATMASK$BAND=$TAG:NATMASK | tee $O/bs8_window_band${BAND}_summary.txt
if [ "$KEEP_STATES" = "0" ]; then rm -rf $O/states/E55_${TAG}_*; fi
