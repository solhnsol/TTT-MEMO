#!/bin/bash
# JOB 881 (restart of job 880, whose one-call K,V record ran at ~250 s per layer; segmented record / prefix as in the 6-cycle news run, K,V shared between the arms) = E57 SYNTHETIC LONG STREAM, 1024 CHUNKS (method thread; Hans '합성 스트림을 더 길게 ... 격차가 계속 벌어지는지', estimate <= 3 h approved in advance): world v3_16_1_L1024 (e57_longcell.py: seed 64, 16 facts, one sentence per fact in every chunk, 1024 exposures per fact = 1024 chunks, bank of 300 frames cycled about 3.4 times per fact, 24-book background pool of the cell v3_64_4; the first 128 chunks are NOT the stream of v3_16_1, nothing of the old runs is reused); (1) W0 window pass over chunks 8..1023 (band 256), (2) deciding rows (top 30% by g), (3) in one process the plain NAT write (norm source) and the arms M3W1N05 (= ALLQ256) and M3W8N05 with states at a = 128 / 256 / 512 / 768 / 1024 exposures and the canary (no inline reads), (4) per-token reads of the 10 states and the curve of the branch-token rank and d(A-B). No existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_longcell.py e57_pqa2_run.py e57_main_run.py e57_syn_w0pass.py e57_syn_mmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py rope_fly.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_w0pass.py --selftest | tail -1; $PY $S/e57_syn_mmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
CELL=v3_16_1_L1024; PL=$OUT/e57_syn_long; SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_long_curve}; mkdir -p "$PL" "$TT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_s64.json" ] && [ -z "$RERUN" ] && { echo "curve json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$PL/syn_s64_nll.npz" ] || $PY $S/e57_syn_w0pass.py --seed 64 --cell $CELL --out "$PL" --bands 256 --segmented
[ -f "$SQ/syn_s64_L1024_m30_masks.npz" ] || $PY $S/e57_syn_mmasks.py --seed 64 --cell $CELL --dir "$PL" --out "$SQ/syn_s64_L1024_m30_masks.npz"
$PY $S/e57_syn_mrun.py --seed 64 --cell $CELL --out $SQ --tag synqL --m30 $SQ/syn_s64_L1024_m30_masks.npz --ntg-nat auto --arms NATSGD,M3W1N05,M3W8N05 --save-apps 128,256,512,768,1024 --no-inline --segmented
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --cell $CELL --out "$TT" --tag synqL --arms M3W1N05,M3W8N05
$PY $S/e57_curve_report.py --kind syn --seed 64 --tag synqL --tokread $TT/e57_syn_tokread_s64.json --out $TT/e57_curve_s64.json
echo "outputs: $TT/e57_curve_s64.json $TT/e57_syn_tokread_s64.json $SQ/E55_synqL_{NATSGD,M3W1N05,M3W8N05}_s64.saved.json"
