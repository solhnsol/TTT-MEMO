#!/bin/bash
# JOB 1278 = E57 DISTANCE x VISIBILITY (gap) CELLS (core-claims thread 11/10; HELD until 1276/1277 have finished and Hans' sentence; EXPLORATORY, not comparable with the 16-fact results): 32 facts, 16 exposures, period p = gap G in {4, 16, 32} chunks (64 / 256 / 512 chunks), read at exposure 16 (out-of-window and in-window group 0), arms NATS and NATMASK (ALLQ256 from chunk 8), seeds 62 63 64 65 66 (5). Cells v3_32_<p>_a16 are registered at run time (gen_facts.py untouched). A 10-chunk smoke on cell v3_32_4_a16 runs first. ESTIMATE: 832 chunks per (arm, seed) at ~4 s/chunk ~ 55 min => 10 arm-seeds ~ 9.2 h.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
$PY $D/e57_oracle_run.py --seed 62 --cell v3_32_4_a16 --out $OUT/e57_gap_smoke --masks none --mask-arm ALLQ256 --arms NATS,NATMASK --smoke 10 --tag gapsm
for SEED in 62 63 64 65 66; do
  for P in 4 16 32; do
    $PY $D/e57_oracle_run.py --seed $SEED --cell v3_32_${P}_a16 --out $OUT/e57_gap --masks none --mask-arm ALLQ256 --arms NATS,NATMASK --tag gap${P}
  done
done
echo "outputs: $OUT/e57_gap/E55_gap<P>_<arm>_s<seed>.{saved,read}.json"
