#!/bin/bash
# JOB 313 = E57 NEWS SUFFIX-ONLY BAND GAIN (W0, forward only, no write; post-hoc, exploratory; Hans 2026-10-29 19:2x KST "RunPod에서 병렬로"). Same input, bundle, bank and engine as jobs 287a / 289 / 290 (e57_closure_run.py version b29b5ef21a24 is the base of e57_suffixband_run.py). For chunks 8..279 it computes nll_sufband[B] = per-token NLL when every query sees only the previous B tokens in the 8 SUFFIX layers, with the NATURAL prefix output (no prefix recompute). Analysis (CPU, separate): rank agreement of nll_sufband[512] - nll_win with nll_band[512] - nll_win of e57_band_W0.npz. ESTIMATE ONLY, unmeasured: model load 13-17 min + natural K,V record about 5 min + per chunk window and chunk-only forwards (as 287a, about 1.4 s) + 3 suffix-only band forwards (about 0.15 s each) = about 10 min of compute. Needs host RAM >= 128 GB (K,V record about 66 GiB), a writable volume (df first), A100.
# Code must already be in the tree: NO_INSTALL=1 is the default here (the pod tree of jobs 289/290 holds the pinned e57_closure_masks.py d520326 and e57_main_run.py 7b74b02); e57_suffixband_run.py is a NEW file copied in.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
for x in "091506dbe7b8 e57_closure_masks.py" "d13a0ac9ee4a e57_main_run.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (have $G, exit 5)"; exit 5; }; done
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_suffixband}; REF=${REF:-$OUT/e57_windowless/E57_windowless_W0.npz}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
mkdir -p "$MOUT"; df -h "$MOUT" | tail -1; echo probe > "$MOUT/.write_probe" && rm "$MOUT/.write_probe"
$PY $D/e57_suffixband_run.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $D/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles ${CYCLES:-1} --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_suffixband_W0.npz" ] && [ -z "$RERUN" ] && { echo "E57_suffixband_W0.npz exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARGS="--bundle $BUNDLE --out $MOUT --cycles ${CYCLES:-1} --npos 1 --canary-npos 1 --arms NAT --smoke 0"
$PY $D/e57_suffixband_run.py $ARGS --bands 128,512,1024 --ref "$REF" --stage arms
