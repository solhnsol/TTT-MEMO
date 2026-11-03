#!/bin/bash
# JOB 820 = E57 NEWS STREAM A CURVE OF THE BRANCH-TOKEN RANK OVER THE CYCLE (thread 'ALLQ보다 나은 가림 방법 찾기', Hans '사이클 안에서의 w=8 곡선' / '합성 + 뉴스'): the writes of M3 w=8 (ALLQ128 mask, deciding rows = top 30% by window gain, update norm 0.5 x NAT; job 770) and ALLQ128 (the original write of job 570, SelMaskQ) are repeated with the same code and the state is saved at 5 points (chunks 55 / 111 / 167 / 223 / 279; canary at every point); the end-state hashes are compared with the original states; the 10 states are read like job 677 (T1/T2 v1, 150 frames incl. the 100 overlap-free ones; per-token log-probs) and the branch-token rank curve is made. New files e57_news_m3.py (arm / saves arguments), e57_curve_report.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_news_m3.py e57_news_tokread.py e57_curve_report.py e57_syn_mrun.py e57_syn_tokread.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_news_m3.py --selftest | tail -1; $PY $S/e57_news_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
M3=$OUT/e57_news_m3; TOUT=${TOUT:-$OUT/e57_news_curve}; mkdir -p "$TOUT"
for f in $M3/news_m30_masks.npz $M3/E57_news_M3W8Q128.arms.json $OUT/e57_main/E57_main_NAT.arms.json $OUT/e57_news_tokread/e57_news_tokread.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
cp -n $M3/news_m30_masks.npz "$TOUT/news_m30_masks.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_curve_news.json" ] && [ -z "$RERUN" ] && { echo "e57_curve_news.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
SAVES=55,111,167,223,279
[ -f "$TOUT/E57_news_M3W8Q128C.arms.json" ] || $PY $S/e57_news_m3.py --stage write --out "$TOUT" --weight 8 --arm M3W8Q128 --saves $SAVES
[ -f "$TOUT/E57_news_ALLQ128C.arms.json" ] || $PY $S/e57_news_m3.py --stage write --out "$TOUT" --weight 1 --arm ALLQ128 --saves $SAVES
EXTRA=$($PY - "$TOUT" <<'PYE'
import json,sys
o=[]
for a in ('M3W8Q128','ALLQ128'):
    d=json.load(open(f'{sys.argv[1]}/E57_news_{a}C.arms.json'))
    for ch,s in sorted(d['states'].items(),key=lambda x:int(x[0])):o.append(f"{a}@c{ch},{s['file']},{s['hash']}")
print(';'.join(o))
PYE
)
[ -f "$TOUT/read/e57_news_tokread.json" ] || $PY $S/e57_news_tokread.py --stage read --out "$TOUT/read" --sets v1 --extra "$EXTRA"
$PY $S/e57_news_m3.py --stage merge --out "$TOUT"
$PY $S/e57_curve_report.py --kind news --tokread $TOUT/merged/e57_news_tokread.json --out $TOUT/e57_curve_news.json
echo "outputs: $TOUT/e57_curve_news.json $TOUT/E57_news_M3W8Q128C.arms.json $TOUT/E57_news_ALLQ128C.arms.json"
