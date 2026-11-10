#!/bin/bash
# JOB 1282 = E57 DCLM preservation gate with the mask from chunk 0 (Hans 11/10: K0 = 0): NATMASK (ALLQ256 from chunk 0) and NATMASKC on the same 4 DCLM streams as job 1274 (W0 and NATS of 1274 are reused, K0 does not change them), same protocol, output $OUT/e57_dclm_ttt_k0/ and $OUT/e57_dclm_ttt_k0_gate.json (HELD until Hans' sentence). ESTIMATE ~40 min (NATMASK ~9 min/stream x 4 + load).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_dclm_ttt_run.py --selftest | tail -1
$PY $D/e57_dclm_ttt_run.py --stage run --out $OUT/e57_dclm_ttt_k0 --k0 0 --streams 0,1,2,3 --arms NATMASK
$PY $D/e57_dclm_ttt_run.py --stage analyze --out $OUT/e57_dclm_ttt_k0 --ref-dir $OUT/e57_dclm_ttt --json $OUT/e57_dclm_ttt_k0_gate.json
echo "outputs: $OUT/e57_dclm_ttt_k0_gate.json"
