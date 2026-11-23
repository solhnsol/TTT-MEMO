#!/bin/bash
# JOB 1345w = BLOCK-STREAM RE-READ WITH THE KEPT WINDOW (Hans 11/23: 'W0 group 0 question; write group 0, window holds group 0, ask group 0; write group 1, window holds group 1, ask groups 0 and 1 ...'; replaces job 1345 (other-block context); code e59_bs8_window.py + e59_bs8_window_analyze.py (new) around the UNCHANGED job-1296 runner chain e57_blockstream8_run.py -> e57_oracle_run.py, pinned by the sha256 file of job 1296): identical writes to 1296 (blk16x8_c16, shuffled block assignment, seeds 82..85, arms NATS and NATMASK = band ALLQ1024); reads at the end of block k: ONE context w<k> = the real window (last 8 chunks) asks the facts of the blocks 0..k (lag 0 in-window; lag >= 1 window kept, facts not in the window); W0 of the same contexts; neutral out read at the final point only. Later band arms (ALLQ256, ALLQ128, ALLQ64) = job 1345x with another --tag; NATS / NATMASK are not run again. Pinned hashes of 1296 are checked; final_hash of the re-run is compared with the stored one by the analyzer. Exploratory.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${SEEDS:=82 83 84 85}"; : "${SAVE:=16,32,48,64,80,96,112,128}"; : "${KEEP_STATES:=1}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e59_bs8_window; OD=$OUT/e57_blockstream8; mkdir -p $O/smoke
sha256sum -c $OD/pinned_sha256.txt || { echo "REUSE REFUSED: a runner file of job 1296 differs from the pinned hash"; exit 12; }
sha256sum $D/e59_bs8_window.py $D/e59_bs8_window_analyze.py
$PY $D/e59_bs8_window.py --selftest | tail -1 | cut -c1-200
$PY $D/e57_oracle_run.py --selftest | tail -1; $PY $D/e57_blockstream8_run.py --selftest | tail -1
$PY $D/e59_bs8_window.py --cell blk16x8_c16 --seed 82 --out $O/smoke --masks none --mask-arm ALLQ1024 --arms NATS --save-apps 128 --nat-loss-every 8 --tag bs8w --shuffle-blocks --smoke 32 --rerun
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/E55_bs8w_NATS_s82.read.json'))
assert set(r['read'])=={'32'} and set(r['read']['32'])=={'w1'},(list(r['read']),{k:list(v) for k,v in r['read'].items()})
for nm in ('w1',):
    x=np.asarray(r['read']['32'][nm]);w=np.asarray(r['W0'][nm]);assert x.shape==(36,16)==w.shape and np.isfinite(x).all() and np.isfinite(w).all() and np.abs(x-w).max()>1e-3,(nm,'the read must differ from W0')
    assert len(r['meta']['item_idx']['32'][nm])==36
print('BS8 WINDOW SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for SEED in $SEEDS; do
  $PY $D/e59_bs8_window.py --cell blk16x8_c16 --seed $SEED --out $O --masks none --mask-arm ALLQ1024 --arms NATS,NATMASK --save-apps $SAVE --nat-loss-every 8 --tag bs8w --shuffle-blocks
  echo "RUN bs8_window seed $SEED rc=$?"
done
$PY $D/e59_bs8_window_analyze.py --new $O --orig $OD | tee $O/bs8_window_summary.txt
if [ "$KEEP_STATES" = "0" ]; then rm -rf $O/states; fi
