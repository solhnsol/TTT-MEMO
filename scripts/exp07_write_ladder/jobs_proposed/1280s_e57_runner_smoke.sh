#!/bin/bash
# JOB 1280s = E57 1280 RUNNER SMOKE (first GPU run of e57_1280_runner.py; HELD until the principal investigator's own sentence in the server-queue thread; instrumentation seed 91 on the cell v3_32_4_a16, 20 chunks, branch chunks 10 and 12; not part of any registered analysis, the outputs are code checks and timing). It measures: whether every branch computation runs, the time per branch chunk (NLL arms, the four one-step margin reads, the variant window, the t3 pair), c - nc, the isolation check against the trajectory log, t2, t3 pair differences. ESTIMATE about 20 min (model load, natural K,V record about 10 s for 20 chunks, 2 branch chunks of an unmeasured length, one final read about 5 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
for T in e57_1280_margin e57_1280_validity e57_1280_gainnp e57_1280_recomp e57_1280_runner_records e57_1280_t3_sensitivity; do $PY $D/$T.py --selftest | tail -1; done
$PY $D/e57_oracle_run.py --selftest | tail -1
$PY $D/e57_1280_runner.py --seed 91 --out $OUT/e57_1280_smoke --smoke 20 --branches 10,12 --tag r1280sm
echo "outputs: $OUT/e57_1280_smoke/{validity_s91.json,results_s91.json,E55_r1280sm_NATS_s91.saved.json}"
