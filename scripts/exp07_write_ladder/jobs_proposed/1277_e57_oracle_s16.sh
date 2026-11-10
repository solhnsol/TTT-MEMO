#!/bin/bash
# JOB 1277 = E57 ORACLE / HALF-ORACLE UPDATE-PATH MASKS, synthetic 16-fact worlds (v3_16_1), seeds 64 61 62 63 65 (core-claims thread 11/10; HELD until 1276 has finished and Hans' sentence): arms NATORACLE (a sentence row does not see the earlier sentences of its own fact), NATHALF (rows see the previous 256 tokens + the earlier sentences of their own fact). NATS / NATMASK of the same seeds are the files of job 1270 (e57_natmask/, tag synng). A 10-chunk smoke (seed 64) runs first. ESTIMATE ~8.5 min per arm-seed => ~85 min + smoke ~10 min.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
$PY $D/e57_oracle_run.py --seed 64 --out $OUT/e57_oracle_smoke --masks none --mask-arm ALLQ256 --arms NATORACLE,NATHALF --smoke 10 --tag orcsm
for SEED in 64 61 62 63 65; do
  $PY $D/e57_oracle_run.py --seed $SEED --out $OUT/e57_oracle --masks none --mask-arm ALLQ256 --arms NATORACLE,NATHALF --tag orc
done
echo "outputs: $OUT/e57_oracle/E55_orc_<arm>_s<seed>.{saved,read}.json"
