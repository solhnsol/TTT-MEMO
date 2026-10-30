#!/bin/bash
# JOB 561 = E57 PREFIX-ONLY BAND-512 SCORE (prepared; NOT QUEUED until Hans confirms job name + commit; Hans 'prefix 점수 계산부터' via the coordinator): forward-only pass on the W0 weights, chunks 8..279, in which the band-512 keep (every query sees only the previous 512 tokens, window AND chunk keys) is given to the PREFIX layers only and the suffix layers keep the natural window keep. Same stream walk as job 290 / 301 (e57_closure_run.py --bands) in a NEW file (e57_closure_prefixband_run.py); no existing e57_*.py is edited. The natural nll_win is recomputed ONLY for chunks 8-15 as a sanity control (Hans: no full natural pass) and compared with the stored e57_band/e57_band_W0.npz (nll_win max abs diff over chunks 8-15 in the json, and the both-layers band-512 mean over all chunks for comparison); chunks 16..279 get the prefix-only band-512 forward only. Output $MOUT/E57_prefixband_W0.npz (nll_win for chunks 8-15 only, nll_prefixband [1,280,1024], lm, tgt) + .json. No write, no state saved. ESTIMATE ONLY: load 15 min + record 5 min + 272 chunks x 1 band forward = about 6 min -> about 25-30 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_closure_prefixband_run.py e57_closure_run.py e57_closure_masks.py e57_main_run.py e55_core.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_closure_prefixband_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_prefixband512}; REF=${REF:-$OUT/e57_band/e57_band_W0.npz}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] && [ -f "$REF" ] || { echo "bundle store, bank or reference missing (exit 7)"; exit 7; }
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_prefixband_W0.npz" ] && [ -z "$RERUN" ] && { echo "E57_prefixband_W0.npz exists (exit 8; RERUN=1 to redo)"; exit 8; }
mkdir -p "$MOUT"
$PY $S/e57_closure_prefixband_run.py $ARGS --bands 512 --ref "$REF" --stage arms
echo "outputs: $MOUT/E57_prefixband_W0.npz $MOUT/E57_prefixband_W0.json"
