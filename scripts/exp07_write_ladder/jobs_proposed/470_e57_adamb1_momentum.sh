#!/bin/bash
# JOB 470 = E57 MOMENTUM-ONLY WRITE DIRECTION, beta1 0.5 / 0.9 / 0.99 + NATADAMH (post-hoc; prepared, NOT APPROVED to run, needs Hans's approval line; brainstorm thread option (a), Hans 13:26Z 'a 부터 해보자'): the method of job 430 (e57_adamdecomp_run.py, commit 31d3f66; one process, one cycle, no mask, 1 seed, every chunk update of norm 0.5 x the NAT norm, 42 questions OUT, states at chunks 99 / 179 / 279 with canary) with the R2MOM direction d = m / c1 (momentum only, no sqrt(v)) at beta1 = 0.5 (R2B05), 0.9 (R2B09 = 430 R2MOM, in-job control) and 0.99 (R2B099), plus the NATADAMH control. Wrapper e57_adamb1_run.py. Output $OUT/e57_adamb1/ (same file formats as 430). No pass criterion. ESTIMATE ONLY (430 took 52 min for 5 arms): load + record ~15 min + 4 arms x ~7 min + reads ~13 x 0.5 min = ~50 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 31d3f6683be1f91b967a828cde05e7d52398cb41^{commit} && git cat-file -e d6617f150f4f2c0f7f03f51b8c530bfaa09bc3a7^{commit} || { echo "commit 31d3f66 or d6617f150f4f2c0f7f03f51b8c530bfaa09bc3a7 missing (exit 4)"; exit 4; }; git checkout 31d3f6683be1f91b967a828cde05e7d52398cb41 -- $D/e57_adamdecomp_run.py; git checkout d6617f150f4f2c0f7f03f51b8c530bfaa09bc3a7 -- $D/e57_adamb1_run.py; fi
sha256sum $D/e57_adamdecomp_run.py $D/e57_adamb1_run.py $D/e57_qcurve_run.py | cut -c1-16,65-
$PY $S/e57_adamb1_run.py --selftest | tail -1
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_adamb1}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] || { echo "input missing: $NATARMS (exit 7)"; exit 7; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_NATADAMH.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_NATADAMH.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARMS=${ARMS:-NATADAMH,R2B05,R2B09,R2B099}
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms $ARMS --ntg-from $NATARMS"
$PY $S/e57_adamb1_run.py $ARGS --stage arms
$PY $S/e57_adamb1_run.py $ARGS --stage reads
for A in $(echo "$ARMS" | tr ',' ' '); do rm -f "$MOUT"/states/E57_main_$A/*_099.npz "$MOUT"/states/E57_main_$A/*_179.npz; done; ls "$MOUT"/states/*/ | head -20
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy to the shared root as e57_adamb1_scores.json / e57_adamb1_<ARM>.arms.json); states of the chunk 279 kept in $MOUT/states/"
