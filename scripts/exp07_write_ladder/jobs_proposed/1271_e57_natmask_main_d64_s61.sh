#!/bin/bash
# JOB 1271 = E57 ORIGINAL-NAT (clipped SGD) WRITE ARMS, SYNTHETIC 64-FACT WORLD (cell v3_64_4, 512 chunks), seed 61 (Hans 11/09 22:54 KST 's64 -> s61 -> QA'): arms NATS, NATMASK (ALLQ256), NATMASKC, DIRmaskNORMfull, DIRfullNORMmask; save points appearances 16/32/64/96/128, inline reads at 16/32/64/128, canary, predict-path loss every 8 chunks. Output $OUT/e57_natmask_d64/. ESTIMATE ONLY (from job 970: 58 min per arm of 512 chunks; swap arms run 2 backward passes per chunk): NATS ~60 min, NATMASK ~70, NATMASKC ~70, each DIR arm ~110 => ~7 h for the 5 arms of ONE seed. More seeds (s62, s63 have NAT stored) would add ~7 h each. NOTE (Council 11/09): the 64-fact world is EXPLORATORY and DESCRIPTIVE ONLY: one seed (s61), report per-arm values and ranges only, no significance test or inference claim; it does not change the main verdict (16-fact, 5 seeds).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_natmask_run.py --selftest | tail -1
for SEED in ${SEEDS:-61}; do
  $PY $D/e57_natmask_run.py --seed $SEED --cell v3_64_4 --save-apps 16,32,64,96,128 --out $OUT/e57_natmask_d64 --masks none --mask-arm ALLQ256 --arms NATS,NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask --tag synngd64
done
echo "outputs: $OUT/e57_natmask_d64/E55_synngd64_*_s*.read.json / .saved.json"
