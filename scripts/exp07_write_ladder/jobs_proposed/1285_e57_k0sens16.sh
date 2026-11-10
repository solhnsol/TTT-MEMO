#!/bin/bash
# JOB 1285 = E57 K0 SENSITIVITY, synthetic 16-fact worlds, the SAME 8 seeds as jobs 1270/1272 (64 61 62 63 65 66 67 68) with the mask from chunk 0 (Hans 11/10 '추천대로'; GPU start needs Hans' own sentence): arms NATMASK, NATMASKC, DIRmaskNORMfull, DIRfullNORMmask at K0 = 0; NATS of the K0 = 8 jobs is K0-independent and reused (analysis --ref-dir). Run type k0_sensitivity. Output $OUT/e57_k0sens16/ tag k0s16. ESTIMATE 4 arms x 8 seeds x 8.5 min ~ 4.5 h.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
for SEED in 64 61 62 63 65 66 67 68; do
  $PY $D/e57_oracle_run.py --seed $SEED --out $OUT/e57_k0sens16 --masks none --mask-arm ALLQ256 --arms NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask --nat-loss-every 8 --tag k0s16
done
echo "outputs: $OUT/e57_k0sens16/E55_k0s16_<arm>_s<seed>.{saved,read}.json"
