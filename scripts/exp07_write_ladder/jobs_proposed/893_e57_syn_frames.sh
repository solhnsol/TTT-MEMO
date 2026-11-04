#!/bin/bash
# JOB 893 = E57 SYNTHETIC LONG WORLD: INTERVENING-CLAUSE AND REVERSE READ FRAMES (investigation thread, Hans 11/04 16:41 KST card '문구 3종씩, NAT 포함 실행'; read only on the saved states of job 881): base read, 3 intervening-clause phrasings ('{X}, who may be a friend of {Y}, now lives in the village of' ...; Y = person of another fact; 16 place candidates) and 3 reverse phrasings ('The person who lives in {P} is' ...; 16 name candidates) for W0, ALLQ256 (M3W1N05), M3 w=8 (M3W8N05) and NAT (NATSGD) at a = 256 and 768 (chunks 255 / 767); per-token reads; the base frames are packed like job 881 and compared with its stored reads. New file e57_syn_frames.py (uses e57_longcell.py, e57_syn_tokread.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_frames.py e57_longcell.py e57_syn_tokread.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_frames.py --selftest | tail -1
TT=${TT:-$OUT/e57_syn_frames}; mkdir -p "$TT"
for l in NATSGD M3W1N05 M3W8N05; do [ -f "$OUT/e57_synq/E55_synqL_${l}_s64.saved.json" ] || { echo "input missing: $l state (exit 7)"; exit 7; }; done
[ -f "$TT/e57_syn_frames_check.json" ] || CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_syn_frames.py --stage check --out "$TT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_syn_frames_report.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TT/e57_syn_frames_read.json" ] || $PY $S/e57_syn_frames.py --stage read --out "$TT"
$PY $S/e57_syn_frames.py --stage analyze --out "$TT"
echo "outputs: $TT/e57_syn_frames_report.json $TT/e57_syn_frames_read.json $TT/e57_syn_frames_check.json"
