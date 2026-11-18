#!/bin/bash
# JOB 1312 = FictionalQA W0 pre-screen (GPU forward only, no weight update; Hans 11/18 'fictionalQA 사전 점검 승인'; one process, 8 worlds = run seed labels 211..218 = builder seeds 201..208; ~5-8 min est.): fq_screen_run.py reads all 24 facts (16 world + 8 reserve) of every world at W0 (E2E 3B v3_16_1 weights, same reader as the NATS runs). Output experiments/exp07_write_ladder/fq/w0screen/fq_w0screen_<label>.json. The replacement of margin>0 facts (fq_w0_select.py + fq_w0_apply.py) is CPU and runs afterwards.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/fq_world.py --selftest | tail -1; $PY $D/fq_screen_run.py --selftest | tail -1
sha256sum $D/fq_world.py $D/fq_screen_run.py $OUT/fq/worlds/*.json
$PY $D/fq_screen_run.py --worlds $OUT/fq/worlds --out $OUT/fq/w0screen
