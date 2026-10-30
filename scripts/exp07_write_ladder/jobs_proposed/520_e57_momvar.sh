#!/bin/bash
# JOB 520 = E57 MOMENTUM-ONLY DIRECTION VARIANTS (prepared; Hans 00:43Z via the idea thread: build, queue and run without asking again): the R2 arm of job 470 (R2B09: momentum only, no sqrt(v), beta1 .9, update norm 0.5 x NAT, no mask, one cycle, 1 seed, OUT reads of the 42 questions at chunks 99 / 179 / 279, canary) with the DIRECTION changed: W_MULTI (.8 fast beta .9 + .2 slow beta .99), W_CAUTIOUS (keep sign(m)==sign(g)), W_MUON (Newton-Schulz orthogonalised w2 momentum, norm kept), W_WHITEN_A / W_WHITEN_B (key-side (C + lambda I)^-1 on the 5632-dim input of w2, lambda = alpha tr(C)/5632, alpha .1 / 1.0, C = bias-corrected EMA .99 of the per-chunk key second moment from an extra forward). Controls R2B09 / NATADAMH = job 470. Output $OUT/e57_momvar/. No thresholds. ESTIMATE ONLY: load + record ~13 min + 3 arms x 7 min + 2 whiten arms x ~20-25 min (extra forward with 1 GB of callbacks and 8 solves per chunk) + reads ~15 min = ~1.6-1.8 h.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 31d3f6683be1f91b967a828cde05e7d52398cb41^{commit} && git cat-file -e a25d2ecad5c5495b84fed558fdb30c14190ee121^{commit} || { echo "commit 31d3f66 or a25d2ecad5c5495b84fed558fdb30c14190ee121 missing (exit 4)"; exit 4; }; git checkout 31d3f6683be1f91b967a828cde05e7d52398cb41 -- $D/e57_adamdecomp_run.py; git checkout a25d2ecad5c5495b84fed558fdb30c14190ee121 -- $D/e57_momvar_run.py $D/e57_adamb1_run.py; fi
sha256sum $D/e57_adamdecomp_run.py $D/e57_adamb1_run.py $D/e57_momvar_run.py $D/e57_qcurve_run.py | cut -c1-16,65-
$PY $S/e57_momvar_run.py --selftest | tail -1
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_momvar}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] || { echo "input missing: $NATARMS (exit 7)"; exit 7; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_W_MULTI.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_W_MULTI.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --ntg-from $NATARMS"
$PY $S/e57_momvar_run.py $ARGS --arms W_MULTI,W_CAUTIOUS,W_MUON --stage arms
set +e; $PY $S/e57_momvar_run.py $ARGS --arms W_WHITEN_A,W_WHITEN_B --stage arms; WRC=$?; set -e
echo "whiten arms stage rc=$WRC"
ARMS=W_MULTI,W_CAUTIOUS,W_MUON; for A in W_WHITEN_A W_WHITEN_B; do [ -f "$MOUT/E57_main_$A.arms.json" ] && ARMS="$ARMS,$A"; done; echo "arms to read: $ARMS"
$PY $S/e57_momvar_run.py $ARGS --arms $ARMS --stage reads
for A in $(echo "$ARMS" | tr ',' ' '); do rm -f "$MOUT"/states/E57_main_$A/*_099.npz "$MOUT"/states/E57_main_$A/*_179.npz; done; ls "$MOUT"/states/*/ | head -20
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy to the shared root as e57_momvar_scores.json / e57_momvar_<ARM>.arms.json); states of the chunk 279 kept in $MOUT/states/"
