#!/bin/bash
# JOB 510 = E57 SLOT-MOMENTUM WRITE DIRECTION (prepared; Hans 17:19Z via the idea thread: build, queue and run without asking again): the R2 arm of job 470 (momentum only, no sqrt(v), beta1 .9, clipped gradient, update norm 0.5 x NAT, no mask, one cycle, 1 seed, OUT reads of the 42 questions at the chunks 99 / 179 / 279, canary) with the momentum buffer split into 10 SLOTS: chunk k updates only the slot r[k] of a routing table (e57_slot_routing_tables.json, sha256 21007a08...), own bias correction per slot; arms S_ONE (one slot: must reproduce R2B09 of job 470, hash 8c61a9924782), S_ORACLE (dominant event), S_KMEANS (offline k-means on the C sketch of job 490, exploratory), S_RANDOM; controls R2B09 / NATADAMH are the results of job 470 (not rerun). Output $OUT/e57_slotmom/. No pass criterion. ESTIMATE ONLY: load + record ~13 min + 4 arms x ~7 min + reads 12 x 0.5 min = ~50 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 31d3f6683be1f91b967a828cde05e7d52398cb41^{commit} && git cat-file -e d3ddf964c504c12eb8c39a4705148104dcda1da6^{commit} || { echo "commit 31d3f66 or d3ddf964c504c12eb8c39a4705148104dcda1da6 missing (exit 4)"; exit 4; }; git checkout 31d3f6683be1f91b967a828cde05e7d52398cb41 -- $D/e57_adamdecomp_run.py; git checkout d3ddf964c504c12eb8c39a4705148104dcda1da6 -- $D/e57_slotmom_run.py $D/e57_adamb1_run.py; fi
sha256sum $D/e57_adamdecomp_run.py $D/e57_adamb1_run.py $D/e57_slotmom_run.py $D/e57_qcurve_run.py | cut -c1-16,65-
$PY $S/e57_slotmom_run.py --selftest | tail -1
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_slotmom}; RT=${RT:-$OUT/e57_slotmom_in/e57_slot_routing_tables.json}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] && [ -f "$RT" ] || { echo "input missing: $NATARMS or $RT (exit 7)"; exit 7; }
[ "$(sha256sum "$RT" | cut -c1-64)" = "21007a08a2cc3a83803e17bfbc85fb8050f434d608ddc3d8d9c471f2e74a6c57" ] || { echo "routing tables differ from the frozen copy (exit 9)"; exit 9; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_S_ONE.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_S_ONE.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARMS=${ARMS:-S_ONE,S_ORACLE,S_KMEANS,S_RANDOM}
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms $ARMS --ntg-from $NATARMS"
$PY $S/e57_slotmom_run.py --routing $RT $ARGS --stage arms
$PY $S/e57_slotmom_run.py --routing $RT $ARGS --stage reads
for A in $(echo "$ARMS" | tr ',' ' '); do rm -f "$MOUT"/states/E57_main_$A/*_099.npz "$MOUT"/states/E57_main_$A/*_179.npz; done; ls "$MOUT"/states/*/ | head -20
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy to the shared root as e57_slotmom_scores.json / e57_slotmom_<ARM>.arms.json); states of the chunk 279 kept in $MOUT/states/"
