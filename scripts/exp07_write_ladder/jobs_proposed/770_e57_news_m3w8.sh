#!/bin/bash
# JOB 770 = E57 NEWS T1/T2 M3 w=8 AGAINST ALLQ128 (method thread, Hans '실행' 11/03 23:33 KST on the design: stream A, one cycle, 280 chunks): write arm M3W8Q128 = the ALLQ128 write of job 570 (engine SELC1, Adam direction, update norm 0.5 x the plain NAT norm of E57_main_NAT, ALLQ128 from chunk 8) with the loss of the deciding rows (top 30% of the eligible rows of every chunk by g = nll_band128 - nll_win of the stored W0 pass) weighted 8; canary stored at chunk 279; the end state is read like job 677 (OUT reader, T1/T2 v1 frames, 150 frames incl. the 100 overlap-free ones; per-token log-probs; divergence-token metric) and compared per frame with the stored reads of job 677 (W0, NAT, NATQ, ALLQ128, ...). New file e57_news_m3.py, two small additions to e57_news_tokread.py (--sets, --extra, more paired references); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_news_m3.py e57_news_tokread.py e57_syn_mrun.py e57_syn_tokread.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_news_m3.py --selftest | tail -1; $PY $S/e57_news_tokread.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_news_m3}; mkdir -p "$TOUT"
for f in $OUT/e57_band/e57_band_W0.npz $OUT/e57_main/E57_main_NAT.arms.json $OUT/e57_t1t2v2_in/e57_stream_sentences.jsonl $OUT/e57_news_tokread/e57_news_tokread.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$TOUT/news_m30_masks.npz" ] || $PY $S/e57_news_m3.py --stage masks --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/merged/e57_news_tokread_keys.json" ] && [ -z "$RERUN" ] && { echo "e57_news_tokread_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/E57_news_M3W8Q128.arms.json" ] || $PY $S/e57_news_m3.py --stage write --out "$TOUT" --weight 8
EXTRA=$($PY -c "import json,sys;d=json.load(open('$TOUT/E57_news_M3W8Q128.arms.json'));s=d['states']['279'];print('M3W8Q128,'+s['file']+','+s['hash'])")
[ -f "$TOUT/read/e57_news_tokread.json" ] || $PY $S/e57_news_tokread.py --stage read --out "$TOUT/read" --sets v1 --extra "$EXTRA"
$PY $S/e57_news_m3.py --stage merge --out "$TOUT"
$PY $S/e57_news_tokread.py --stage analyze --out "$TOUT/merged"
echo "outputs: $TOUT/merged/e57_news_tokread_keys.json $TOUT/merged/e57_news_tokread.json $TOUT/E57_news_M3W8Q128.arms.json"
