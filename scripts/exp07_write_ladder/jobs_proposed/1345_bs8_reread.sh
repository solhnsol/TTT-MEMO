#!/bin/bash
# JOB 1345 = BLOCK-STREAM RE-READ WITH A REAL STREAM CONTEXT (Hans via the coordinator 11/23: is the gain of the masked write an artefact of the neutral read context?; code e59_bs8_reread.py + e59_bs8_reread_analyze.py (new) around the UNCHANGED job-1296 runner chain e57_blockstream8_run.py -> e57_oracle_run.py, pinned by the sha256 file of job 1296): the 1296 experiment (blk16x8_c16, 8 blocks of 16 facts streamed one after the other, shuffled block assignment, seeds 82..85, arms NATS and NATMASK with band ALLQ1024) is re-run with identical writes (the reads only score frozen weights; final_hash compared with the stored one) and the reads changed: at every block end the facts of every block j already written are read with the context = BOS + 8191 tokens of the first 8 stream chunks of ANOTHER block (j + 4) % 8 (none of block j's facts in it, asserted) instead of the neutral SQuAD tail; the neutral 'out' read of all items is kept at the final read point only (the original neutral reads of every read point are the pairing when the hash matches). States of every block end are SAVED (--save-apps 16,32,...,128; ~5.5 GB each, ~44 GB per run, kept unless KEEP_STATES=0) so that further reads can be made without new writes. Margin = gold - best wrong over 16 candidates, same definition as the original. Main result: NATMASK - NATS per lag in the stream context vs the neutral context (seed-paired, n = 4), dilution slopes. EXPLORATORY; NOT queued until Hans' sentence; est. ~3.4 h (smoke ~8 min; per seed ~50 min: 2 runs x (writes ~7 min + 576 block-context item-reads ~13 min + final neutral read ~3 min + state saves ~4 min) + W0 block contexts ~3 min; the item-read time (~1.4 s) is inferred from the sec_total of the original runs).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${SEEDS:=82 83 84 85}"; : "${SAVE:=16,32,48,64,80,96,112,128}"; : "${KEEP_STATES:=1}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e59_bs8_reread; OD=$OUT/e57_blockstream8; mkdir -p $O/smoke
sha256sum -c $OD/pinned_sha256.txt || { echo "REUSE REFUSED: a runner file of job 1296 differs from the pinned hash"; exit 12; }
sha256sum $D/e59_bs8_reread.py $D/e59_bs8_reread_analyze.py
$PY $D/e59_bs8_reread.py --selftest | tail -1 | cut -c1-200
$PY $D/e57_oracle_run.py --selftest | tail -1; $PY $D/e57_blockstream8_run.py --selftest | tail -1
$PY $D/e59_bs8_reread.py --cell blk16x8_c16 --seed 82 --out $O/smoke --masks none --mask-arm ALLQ1024 --arms NATS --save-apps 128 --nat-loss-every 8 --tag bs8c --shuffle-blocks --smoke 32 --rerun
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/E55_bs8c_NATS_s82.read.json'))
assert set(r['read'])=={'32'} and set(r['read']['32'])=={'bc0','bc1'},(list(r['read']),{k:list(v) for k,v in r['read'].items()})
for nm in ('bc0','bc1'):
    x=np.asarray(r['read']['32'][nm]);w=np.asarray(r['W0'][nm]);assert x.shape==(16,16)==w.shape and np.isfinite(x).all() and np.isfinite(w).all() and np.abs(x-w).max()>1e-3,(nm,'the read must differ from W0')
    assert len(r['meta']['item_idx']['32'][nm])==16
print('BS8 REREAD SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for SEED in $SEEDS; do
  $PY $D/e59_bs8_reread.py --cell blk16x8_c16 --seed $SEED --out $O --masks none --mask-arm ALLQ1024 --arms NATS,NATMASK --save-apps $SAVE --nat-loss-every 8 --tag bs8c --shuffle-blocks
  echo "RUN bs8_reread seed $SEED rc=$?"
done
$PY $D/e59_bs8_reread_analyze.py --new $O --orig $OD | tee $O/bs8_reread_summary.txt
if [ "$KEEP_STATES" = "0" ]; then rm -rf $O/states; fi
