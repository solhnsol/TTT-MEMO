#!/bin/bash
# JOB 905 = E57 SYNTHETIC P1 CONTRAST WRITE: PER-TOKEN READ OF THE STATES SAVED BEFORE THE STOP (investigation thread, Hans card '읽기 먼저, 이후 상한 대조'; read only): M3PB05W8N05 and M3PB10W8N05 at chunk 15 and 31 (the arms were stopped at chunk 45 by the nll' rule of job 900), merged with the reads of M3W1N05 / M3W8N05 of job 800 at the same chunks; curve report. New file e57_syn_p1read.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_p1read.py e57_syn_tokread.py e57_curve_report.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
TT=${TT:-$OUT/e57_syn_p1read}; mkdir -p "$TT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_p1read_s64.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_p1read.py --stage read --out "$TT"
$PY $S/e57_syn_p1read.py --stage report --out "$TT"
$PY $S/e57_curve_report.py --kind syn --seed 64 --tag synqc --tokread $TT/e57_syn_tokread_s64_all.json --out $TT/e57_curve_p1read_s64.json
echo "outputs: $TT/e57_curve_p1read_s64.json $TT/e57_syn_tokread_s64.json"
