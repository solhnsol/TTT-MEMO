#!/bin/bash
# JOB 430 = E57 ADAM DECOMPOSITION, 4 arms + NATADAMH rerun (post-hoc; Hans 10:02Z decision card "4팔+재실행, 42문항만", thread "Adam이 바인딩을 올리는 이유 규명", E57 section 125): one process, one cycle, no mask, every chunk update of norm 0.5 x the NAT (SGD) norm (the NATADAMH step size), arms NATADAMH (rerun, control), R1RMS (beta1 = 0), R2MOM (momentum only), R3SGN (sign of the clipped gradient), R4VFIX (second moment frozen after the first 8 chunks, m kept; e55_core adam_dir_vfix): e57_adamdecomp_run.py. States saved at the chunks 99, 179, 279, canary there, OUT reads of the 42 questions at W0 and every state (E57_qcurve_scores.json format); afterwards the states of the chunks 99 / 179 are deleted, only 279 is kept (5 x 1.4 GB). Output $OUT/e57_adamdecomp/. ESTIMATE ONLY: load 15 min + record 5 min + 5 arms x ~7.5 min + reads 16 x ~0.5 min = ~65 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 31d3f6683be1f91b967a828cde05e7d52398cb41^{commit} || { echo "commit 31d3f6683be1f91b967a828cde05e7d52398cb41 missing (exit 4)"; exit 4; }; git checkout 31d3f6683be1f91b967a828cde05e7d52398cb41 -- $D/e57_adamdecomp_run.py; fi
sha256sum $D/e57_adamdecomp_run.py $D/e57_qcurve_run.py | cut -c1-16,65-
$PY $S/e57_adamdecomp_run.py --selftest | tail -1
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_adamdecomp}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] || { echo "input missing: $NATARMS (exit 7)"; exit 7; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_NATADAMH.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_NATADAMH.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARMS=${ARMS:-NATADAMH,R1RMS,R2MOM,R3SGN,R4VFIX}
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms $ARMS --ntg-from $NATARMS"
$PY $S/e57_adamdecomp_run.py $ARGS --stage arms
$PY $S/e57_adamdecomp_run.py $ARGS --stage reads
for A in $(echo "$ARMS" | tr ',' ' '); do rm -f "$MOUT"/states/E57_main_$A/*_099.npz "$MOUT"/states/E57_main_$A/*_179.npz; done; ls "$MOUT"/states/*/ | head -20
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy flat to the shared root as e57_adamdecomp_scores.json); states of the chunk 279 kept in $MOUT/states/"
