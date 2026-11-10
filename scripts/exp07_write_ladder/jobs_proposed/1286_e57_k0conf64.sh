#!/bin/bash
# JOB 1286 = E57 K0=0 CONFIRMATION, synthetic 64-fact worlds (v3_64_4, 512 chunks), NEW seeds 71..75 (Hans 11/10 '추천대로'; GPU start needs Hans' own sentence; Council check first): NATS, NATMASK, NATMASKC, DIRmaskNORMfull, DIRfullNORMmask at K0 = 0, save points 16/32/64/96/128. Output $OUT/e57_k0conf64/ tag k0c64. ESTIMATE 5 arms x 5 seeds x 28 min ~ 11.7 h.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
for SEED in 71 72 73 74 75; do
  $PY $D/e57_oracle_run.py --seed $SEED --cell v3_64_4 --save-apps 16,32,64,96,128 --out $OUT/e57_k0conf64 --masks none --mask-arm ALLQ256 --arms NATS,NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask --nat-loss-every 8 --tag k0c64
done
echo "outputs: $OUT/e57_k0conf64/E55_k0c64_<arm>_s<seed>.{saved,read}.json"
