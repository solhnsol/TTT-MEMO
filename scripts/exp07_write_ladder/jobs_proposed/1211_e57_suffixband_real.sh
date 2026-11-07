#!/bin/bash
# JOB 1211 = E57 SUFFIX-ONLY BAND g' on the QA bundles 5/8/9/10/12 and the news stream A (GPM thread, Hans 11/07 '고고', 'QA 번들과 뉴스 A도 가능하면'): prefix = natural window, suffix 8 layers get the band 128; forward only, W0, no update. Outputs <OUT>/e57_suffixband/qa/e57_dwg_qa_nllsuf.npz and <OUT>/e57_suffixband/news/E57_suffixband_W0.npz.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_suffixband_qa_pass.py e57_closure_suffixband_run.py e57_closure_masks.py e57_main_run.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_suffixband_qa_pass.py --selftest | tail -1; $PY $S/e57_closure_suffixband_run.py --selftest | tail -1
BUNDLE=$OUT/e57_news; REF=$OUT/e57_band/e57_band_W0.npz
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] && [ -f "$REF" ] && [ -f "$OUT/e57_public_qa_n65/e57_dwg_qa_nll.npz" ] || { echo "inputs missing (exit 7)"; exit 7; }
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$OUT/e57_suffixband/news" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
mkdir -p "$OUT/e57_suffixband/qa" "$OUT/e57_suffixband/news"
[ -f "$OUT/e57_suffixband/qa/e57_dwg_qa_nllsuf.npz" ] || $PY $S/e57_suffixband_qa_pass.py --src640 "$OUT/e57_public_qa_n65" --out "$OUT/e57_suffixband/qa" --bands 128
[ -f "$OUT/e57_suffixband/news/E57_suffixband_W0.npz" ] || $PY $S/e57_closure_suffixband_run.py --bundle "$BUNDLE" --out "$OUT/e57_suffixband/news" --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --bands 128 --ref "$REF" --stage arms
echo "outputs: $OUT/e57_suffixband/qa/e57_dwg_qa_nllsuf.npz $OUT/e57_suffixband/news/E57_suffixband_W0.npz"
