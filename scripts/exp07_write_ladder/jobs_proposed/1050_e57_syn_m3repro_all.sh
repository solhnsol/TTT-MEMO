#!/bin/bash
# JOB 1050 (wrapper, ~2.7 h on an idle server; Hans '진행해봐' 11/05 23:28 KST via coordinator): W0 window-gain passes (e57_syn_w0pass.py, bands 256) and e57_syn_mmasks.py for the fresh 16-fact seeds 62 and 65 where missing, then 1050_e57_syn_m3repro.sh (6 arms each) for SEED=62 and SEED=65.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for SEED in 62 65; do
  mkdir -p "$OUT/e57_syn_s$SEED"
  [ -f "$OUT/e57_syn_s$SEED/syn_s${SEED}_nll.npz" ] || $PY $S/e57_syn_w0pass.py --seed $SEED --out "$OUT/e57_syn_s$SEED" --bands 256
  [ -f "$OUT/e57_synq/syn_s${SEED}_m30_masks.npz" ] || $PY $S/e57_syn_mmasks.py --seed $SEED --dir "$OUT/e57_syn_s$SEED" --out "$OUT/e57_synq/syn_s${SEED}_m30_masks.npz"
done
for SEED in 62 65; do SEED=$SEED bash $S/jobs_proposed/1050_e57_syn_m3repro.sh; done
