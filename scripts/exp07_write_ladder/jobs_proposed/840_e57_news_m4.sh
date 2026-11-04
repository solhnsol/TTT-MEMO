#!/bin/bash
# JOB 840 = E57 NEWS STREAM A, M4 ARM WITH THE CURVE (thread 'ALLQ보다 나은 가림 방법 찾기', Hans card 's61 + 뉴스 실행'): M4W8Q128 = M3 w=8 on the news stream (ALLQ128 mask from chunk 8, loss weight 8 on the deciding rows = top 30% by window gain of the stored W0 pass, update norm 0.5 x NAT) with the deciding rows seeing only the previous 32 tokens (the synthetic ratio 64 / 256 = 1 / 4 applied to the news band 128); states at chunks 55 / 111 / 167 / 223 / 279 (canary at every point), T1/T2 v1 per-token reads like job 825, curve and the paired differences to ALLQ128 and M3 w=8 of job 825. No existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_news_m3.py e57_news_tokread.py e57_curve_report.py e57_syn_mrun.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_news_m3.py --selftest | tail -1; $PY $S/e57_news_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
C0=$OUT/e57_news_curve; TOUT=${TOUT:-$OUT/e57_news_curve_m4}; mkdir -p "$TOUT"
for f in $C0/news_m30_masks.npz $C0/merged/e57_news_tokread.json $OUT/e57_main/E57_main_NAT.arms.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
cp -n $C0/news_m30_masks.npz "$TOUT/news_m30_masks.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_curve_news_m4.json" ] && [ -z "$RERUN" ] && { echo "curve json exists (exit 8; RERUN=1 to redo)"; exit 8; }
SAVES=55,111,167,223,279
[ -f "$TOUT/E57_news_M4W8Q128C.arms.json" ] || $PY $S/e57_news_m3.py --stage write --out "$TOUT" --weight 8 --arm M4W8Q128 --saves $SAVES
EXTRA=$($PY - "$TOUT" <<'PYE'
import json,sys
d=json.load(open(f'{sys.argv[1]}/E57_news_M4W8Q128C.arms.json'));print(';'.join(f"M4W8Q128@c{ch},{s['file']},{s['hash']}" for ch,s in sorted(d['states'].items(),key=lambda x:int(x[0]))))
PYE
)
[ -f "$TOUT/read/e57_news_tokread.json" ] || $PY $S/e57_news_tokread.py --stage read --out "$TOUT/read" --sets v1 --extra "$EXTRA"
$PY $S/e57_news_m3.py --stage merge --out "$TOUT" --old $C0/merged/e57_news_tokread.json
$PY $S/e57_curve_report.py --kind news --tokread $TOUT/merged/e57_news_tokread.json --out $TOUT/e57_curve_news_m4.json
echo "outputs: $TOUT/e57_curve_news_m4.json $TOUT/E57_news_M4W8Q128C.arms.json"
