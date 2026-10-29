#!/bin/bash
# JOB 495 = E57 GRADIENT-SKETCH DUMP A AGAIN AT 2048 DIMS (prepared; Hans 16:41Z via the idea thread 'good'): the stage A (and C) of job 490 only, NAT 280 chunks, native write_step unchanged, count-sketch of the raw down_proj write gradient at 2048 dims (same method and seed as 490, table for 2048 buckets), plus the exact-vs-sketch cosine check of the first 20 chunks (layer 0) and the comparison with the saved NAT run (update norms, CE). No fragments (stage B), no pass criterion. Output $OUT/e57_gradsketch2048/E57_gradsketch_2048.npz + E57_gradsketch_2048_meta.json (job 490 files untouched). ESTIMATE ONLY: load + record ~8 min + 280 chunks x 2 s ~10 min + check ~1 min = ~20 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e c731db3d69ba47b5adedb56dd005f0dc4cf07561^{commit} || { echo "commit c731db3d69ba47b5adedb56dd005f0dc4cf07561 missing (exit 4)"; exit 4; }; git checkout c731db3d69ba47b5adedb56dd005f0dc4cf07561 -- $D/e57_gradsketch.py; fi
sha256sum $D/e57_gradsketch.py | cut -c1-16,65-
$PY $S/e57_gradsketch.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_gradsketch2048}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] || { echo "input missing: $NATARMS (exit 7)"; exit 7; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_gradsketch_2048.npz" ] && [ -z "$RERUN" ] && { echo "E57_gradsketch_2048.npz exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_gradsketch.py --nb 2048 --no-frags --out-suffix _2048 --nat-ref "$NATARMS" --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --read-at 279 --canary-npos 1 --arms NAT --stage arms
echo "outputs: $MOUT/E57_gradsketch_2048.npz $MOUT/E57_gradsketch_2048_meta.json (copy to the shared root as e57_gradsketch2048.npz / e57_gradsketch2048_meta.json)"
