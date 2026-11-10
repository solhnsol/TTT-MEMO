#!/bin/bash
# JOB 1273 = E57 ORIGINAL-NAT (clipped SGD) WRITE ARMS, SYNTHETIC 64-FACT WORLD (cell v3_64_4), NEW 5 SEEDS 62 63 64 65 66 (Hans 11/10 12:56 KST: new pre-registration with fresh seeds; pilot s61 = job 1271 reported separately as exploratory). Arms NATS, NATMASK (ALLQ256), NATMASKC, DIRmaskNORMfull, DIRfullNORMmask. Output $OUT/e57_natmask_d64/ (tag synngd64). ESTIMATE ONLY: job 1271 took 2 h 14 min for 5 arms of ONE seed => ~11 h for 5 seeds. HELD until the pre-registration text is fixed and Hans approves the GPU slot.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_natmask_run.py --selftest | tail -1
for SEED in ${SEEDS:-62 63 64 65 66}; do
  $PY $D/e57_natmask_run.py --seed $SEED --cell v3_64_4 --save-apps 16,32,64,96,128 --out $OUT/e57_natmask_d64 --masks none --mask-arm ALLQ256 --arms NATS,NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask --tag synngd64
done
echo "outputs: $OUT/e57_natmask_d64/E55_synngd64_*_s*.read.json / .saved.json"
