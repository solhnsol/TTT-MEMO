#!/bin/bash
# JOB 1281 = E57 NUMERICAL NOISE FLOOR (core-claims thread 11/10; HELD until Hans' sentence; independent of the pilot): the SAME NATS run (cell v3_32_4_a16, seed 62, 64 chunks) repeated under controlled nondeterminism: A) deterministic ops, 2 repeats (must be bit identical); B) nondeterministic XLA kernels (no --xla_gpu_deterministic_ops), 3 repeats; C) matmul precision 'highest' (TF32 off) and 'default' (TF32 on), 1 run each under deterministic ops; D) W0 fast weights perturbed by a relative 1e-7 (perturb seeds 0..4) under deterministic ops. Every run logs the same files as the pilot (update_norm, ce, gnorm, coef, token loss, read margins). ESTIMATE ~12 runs x ~9 min (model load + W0 read + 64 chunks) ~ 1.8 h.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
run(){ # tag, extra args...
  local tag=$1; shift
  $PY $D/e57_oracle_run.py --seed 62 --cell v3_32_4_a16 --out $OUT/e57_noise --masks none --mask-arm ALLQ256 --arms NATS --tag "$tag" "$@"
}
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
run noiseA1; run noiseA2
export XLA_FLAGS=""
run noiseB1; run noiseB2; run noiseB3
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
JAX_DEFAULT_MATMUL_PRECISION=highest run noiseChigh
JAX_DEFAULT_MATMUL_PRECISION=default run noiseCdef
for S in 0 1 2 3 4; do run noiseD$S --perturb 1e-7 --perturb-seed $S; done
echo "outputs: $OUT/e57_noise/E55_noise*_NATS_s62.{saved,read}.json"
