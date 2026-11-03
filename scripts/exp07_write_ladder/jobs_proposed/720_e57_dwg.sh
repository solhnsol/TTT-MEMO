#!/bin/bash
# JOB 720 = E57 DECISION-TOKEN WINDOW GAIN DIAGNOSIS (thread 'QA에서 가림이 안 되는 이유 규명', Hans approved one run; W0 model, read only, NO weight update): at every position where the answer string occurs in the write stream, the log-prob of the first content token (space-only skipped), the second token and the mean of the later tokens under the natural window and under the band-erased window (b = 128 for QA / news, 256 for synthetic = the ALLQ mask range); window gain = lp(window) - lp(band). QA: GPU W0 pass over the 33 group streams of job 640 (bands 128 and 512, every chunk); synthetic s64 and news stream A: the stored W0 pass files (e57_syn_s64/syn_s64_nll.npz, e57_band/e57_band_W0.npz, chunks >= 8). New files e57_dwg_qa_pass.py, e57_dwg_analyze.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_dwg_qa_pass.py e57_dwg_analyze.py e57_pqa7_run.py e57_pqa6_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_dwg_qa_pass.py --selftest | tail -1
$PY $S/e57_dwg_analyze.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_dwg}; mkdir -p "$TOUT"
[ -f "$TOUT/e57_decision_window_gain.json" ] && [ -z "$RERUN" ] && { echo "e57_decision_window_gain.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_dwg_positions.json" ] || $PY $S/e57_dwg_analyze.py --stage locate --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_dwg_qa_nll.npz" ] || $PY $S/e57_dwg_qa_pass.py --src640 $OUT/e57_public_qa_stage6 --out "$TOUT" --bands 128,512
$PY $S/e57_dwg_analyze.py --stage aggregate --out "$TOUT" --qa-nll "$TOUT/e57_dwg_qa_nll.npz"
echo "outputs: $TOUT/e57_decision_window_gain.json $TOUT/e57_dwg_raw.json $TOUT/e57_dwg_positions.json"
