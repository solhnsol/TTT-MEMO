#!/bin/bash
# JOB 1210 = E57 SUFFIX-ONLY BAND g' on the six synthetic worlds (GPM thread, Hans 11/07 '고고'): prefix 24 layers = natural window, only the 8 suffix layers get the band 128; g' = nll_suffixband128 - nll_win. Forward only, W0, no update. Outputs <OUT>/e57_suffixband/s<seed>/syn_s<seed>_nllsuf.npz (+ .json with the timings). nll_win is checked against the stored W0 pass.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_suffixband_pass.py e57_longcell.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_suffixband_pass.py --selftest | tail -1
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
for W in 62:v3_16_1 71:v3_4_1 72:v3_32_1 65:v3_16_1 73:v3_4_1 74:v3_32_1; do
  SEED=${W%%:*}; CELL=${W##*:}; O=$OUT/e57_suffixband/s$SEED
  [ -f "$O/syn_s${SEED}_nllsuf.npz" ] || $PY $S/e57_suffixband_pass.py --seed $SEED --cell $CELL --out "$O" --bands 128
  $PY -I - <<PYE
import numpy as np
a=np.load("$O/syn_s${SEED}_nllsuf.npz");b=np.load("$OUT/e57_syn_s$SEED/syn_s${SEED}_nll128.npz")
assert (a["lm"]==b["lm"]).all();d=np.abs(a["nll_win"]-b["nll_win"])[a["lm"]].max();print("s$SEED nll_win max|diff| vs stored W0 pass:",d);assert d<1e-2,d
PYE
done
echo "outputs: $OUT/e57_suffixband/s{62,71,72,65,73,74}/syn_s*_nllsuf.npz"
