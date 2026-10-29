#!/bin/bash
# JOB 401 = E57 NEWS REWRITE OF THREE ARMS WITH STATES AT CHUNKS 99 / 179 / 279 (post-hoc; Hans via the discussion thread 05:05Z "추천대로", step A): NATADAMH (no mask, Adam direction x 0.5 NAT norm), GAINQ512 p=.26 and RANDQ512 p=.10 are written again for one cycle (engine and masks as in the jobs 291 / 292 / 296; masks merged by e57_qcurve_masks.py, content hash checked) with the weights saved at the chunks 99, 179, 279, the canary there, and the 42 questions of bank v1 read OUT only (8192-token neutral tail) from W0 and the 9 states (e57_qcurve_run.py). Output $OUT/e57_qcurve/E57_main_<ARM>.arms.json, E57_qcurve_scores.json, 9 states of 1.4 GB. The new chunk-279 states are NOT assumed to equal the old ones (hashes of repeated runs differ); job 402 reads all 9 states on the stream. NOT APPROVED yet. ESTIMATE ONLY, unmeasured, lab A100: load 15 min + record 5 min + 3 arms x (7 min write + 1 min saves) + OUT scorer 3.5 min + 3 x 3 reads x 73 s = about 1.0 h.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
sha256sum $D/e57_qcurve_run.py $D/e57_qcurve_masks.py | cut -c1-16,65-
$PY $S/e57_qcurve_masks.py --selftest
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_qcurve}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
MG=${MG:-$OUT/e57_gainq_p0.26/e57_qband_masks.npz}; MR=${MR:-$OUT/e57_randq_p0.10/e57_qband_masks.npz}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
for F in "$NATARMS" "$MG" "$MR"; do [ -f "$F" ] || { echo "input missing: $F (exit 7)"; exit 7; }; done
mkdir -p "$MOUT"; $PY $S/e57_qcurve_masks.py --gain "$MG" --rand "$MR" --out "$MOUT/e57_qcurve_masks.npz" --expect-det dbde4b0663ab29b4b25e0be06ca64f160a290cf9fa075b5f533e29cdf33d4260 || { echo "mask content hash differs (exit 9)"; exit 9; }
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_GAINQ512.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_GAINQ512.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARMS=${ARMS:-NATADAMH,GAINQ512,RANDQ512}
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms $ARMS --qband-masks $MOUT/e57_qcurve_masks.npz --ntg-from $NATARMS"
$PY $S/e57_qcurve_run.py $ARGS --stage arms
$PY $S/e57_qcurve_run.py $ARGS --stage reads
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy flat to the shared root as e57_qcurve_scores.json)"
