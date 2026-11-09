#!/bin/bash
# JOB 1270 = E57 ORIGINAL-NAT (clipped SGD) WRITE ARMS, SYNTHETIC 16-FACT WORLDS (v3_16_1), 5 seeds 64 61 62 63 65 (Hans 11/09 22:54 KST '통과하면 Council 반영 설계 ... 본 실행까지 밤새 이어서'; smoke 1260 passed the gates: NATS==stored NAT chunks 0-14 bitwise, NATBINF==NATS bitwise, mask path active): arms NATS, NATMASK (ALLQ256), NATMASKC (same number of hidden keys, random keys), DIRmaskNORMfull, DIRfullNORMmask; reads at appearances 16/32/64/128 (out) + 128 (in window), canary at the last chunk, predict-path loss every 8 chunks, cos/clip logs. Output $OUT/e57_natmask/. ESTIMATE ~10 min per arm x 5 arms x 5 seeds = ~4-4.5 h.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_natmask_run.py --selftest | tail -1
for SEED in 64 61 62 63 65; do
  M=$OUT/e57_synq/syn_s${SEED}_qmasks.npz
  $PY $D/e57_natmask_run.py --seed $SEED --out $OUT/e57_natmask --masks $M --mask-arm ALLQ256 --arms NATS,NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask --tag synng
done
echo "outputs: $OUT/e57_natmask/E55_synng_*_s*.read.json / .saved.json"
