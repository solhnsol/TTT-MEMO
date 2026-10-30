#!/bin/bash
# JOB 540 = E57 NAT SGD AT 2x AND 4x THE NAT UPDATE NORM (prepared; Hans 01:25Z via the Rethink-momentum thread: queue after 530): no mask, no Adam, one cycle, 1 seed: arm NATM (the NAT clipped-gradient direction at the current weights) with the per-chunk update norm = 2x (NATX2) / 4x (NATX4) the per-chunk norms of the saved NAT run (E57_main_NAT.arms.json), OUT reads of the 42 questions at chunks 99 / 179 / 279 + canary. Output $OUT/e57_natsgd/. ESTIMATE ONLY: load + record ~13 min + 2 arms x ~7 min + reads ~12 min = ~40 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 31d3f6683be1f91b967a828cde05e7d52398cb41^{commit} && git cat-file -e bae5f22ebfa72badf78fc296aa4767c7b5cf3fd3^{commit} || { echo "commit 31d3f66 or bae5f22ebfa72badf78fc296aa4767c7b5cf3fd3 missing (exit 4)"; exit 4; }; git checkout 31d3f6683be1f91b967a828cde05e7d52398cb41 -- $D/e57_adamdecomp_run.py; git checkout bae5f22ebfa72badf78fc296aa4767c7b5cf3fd3 -- $D/e57_natsgd_run.py; fi
sha256sum $D/e57_adamdecomp_run.py $D/e57_natsgd_run.py $D/e57_qcurve_run.py | cut -c1-16,65-
$PY $S/e57_natsgd_run.py --selftest | tail -1
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_natsgd}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] || { echo "input missing: $NATARMS (exit 7)"; exit 7; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_NATX2.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_NATX2.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARMS=${ARMS:-NATX2,NATX4}
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms $ARMS --ntg-from $NATARMS"
$PY $S/e57_natsgd_run.py $ARGS --stage arms
$PY $S/e57_natsgd_run.py $ARGS --stage reads
for A in $(echo "$ARMS" | tr ',' ' '); do rm -f "$MOUT"/states/E57_main_$A/*_099.npz "$MOUT"/states/E57_main_$A/*_179.npz; done; ls "$MOUT"/states/*/ | head -20
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy to the shared root as e57_natsgd_scores.json / e57_natsgd_<ARM>.arms.json); states of the chunk 279 kept in $MOUT/states/"
