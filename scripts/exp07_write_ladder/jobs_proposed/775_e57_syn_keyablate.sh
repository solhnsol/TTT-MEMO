#!/bin/bash
# JOB 775 = E57 SYNTHETIC KEY-LEVEL MASKING DIAGNOSTIC (method thread card 'M4 + 키 단위 가림' part (b), Hans '진단 먼저'; seed 64, W0, forward only, no write): on the deciding rows (top 30% by window gain) of the chunks 16..127 the fraction of the window gain g removed when only a set of far keys is masked: ALLV top m, ALL top m (W0 key-mass dump), m random far keys, all same-fact far keys (oracle), m = 16 and 64. New file e57_syn_keyablate.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_keyablate.py e57_syn_pass.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_keyablate.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_keylevel}; mkdir -p "$TOUT"
for f in $OUT/e57_synq/syn_s64_m30_masks.npz $OUT/e57_syn_s64/syn_s64_nll.npz $OUT/e57_syn_s64/syn_s64_dump_j07.npz; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_syn_keyablate.json" ] && [ -z "$RERUN" ] && { echo "e57_syn_keyablate.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_keyablate.py --out "$TOUT"
echo "outputs: $TOUT/e57_syn_keyablate.json $TOUT/e57_syn_keyablate.npz"
