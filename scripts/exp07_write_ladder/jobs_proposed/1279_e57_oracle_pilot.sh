#!/bin/bash
# JOB 1279 = E57 DIRECT-ACCESS ORACLE / HALF-ORACLE PILOT (core-claims thread 11/10; HELD until 1276 has finished and Hans' sentence; EXPLORATORY pilot, the confirmation run uses NEW seeds with a pre-registered rule): 32 facts, 16 exposures, cells v3_32_4_a16 (p=4, 64 chunks, 8 sentences/chunk) and v3_32_16_a16 (p=16, 256 chunks, 2 sentences/chunk), read at exposure 16, seeds 62 63 64. Arms p=4: NATS, NATMASK (ALLQ256), NATORACLE, NATHALF, NATORACLE_RAND (token-matched random control); p=16: NATS, NATMASK, NATORACLE_RAND; the bit-identity gate (p=16: NATORACLE == NATS and NATHALF == NATMASK in every logged number) runs on seed 62 only. Logs: per-token NLL (fact first / later exposure / filler; natural and update path), fact-gradient share every 8 chunks, clip stats, read-hash before/after reads. A 10-chunk smoke (p=4, seed 62, 5 arms) runs first. ESTIMATE ~90 min per seed + ~45 min gate => ~5.3 h for 3 seeds.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
$PY $D/e57_oracle_run.py --seed 62 --cell v3_32_4_a16 --out $OUT/e57_orcmain_smoke --masks none --mask-arm ALLQ256 --arms NATS,NATMASK,NATORACLE,NATHALF,NATORACLE_RAND --smoke 10 --tag orcsm
for SEED in 62 63 64; do
  $PY $D/e57_oracle_run.py --seed $SEED --cell v3_32_4_a16 --out $OUT/e57_orcmain --masks none --mask-arm ALLQ256 --arms NATS,NATMASK,NATORACLE,NATHALF,NATORACLE_RAND --tag orcp4
  $PY $D/e57_oracle_run.py --seed $SEED --cell v3_32_16_a16 --out $OUT/e57_orcmain --masks none --mask-arm ALLQ256 --arms NATS,NATMASK,NATORACLE_RAND --tag orcp16
  [ "$SEED" = 62 ] && $PY $D/e57_oracle_run.py --seed $SEED --cell v3_32_16_a16 --out $OUT/e57_orcmain --masks none --mask-arm ALLQ256 --arms NATORACLE,NATHALF --tag orcp16
done
echo "outputs: $OUT/e57_orcmain/E55_orcp{4,16}_<arm>_s<seed>.{saved,read}.json"
