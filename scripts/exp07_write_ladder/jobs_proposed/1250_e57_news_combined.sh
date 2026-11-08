#!/bin/bash
# JOB 1250 = E57 NEWS STREAM A COMBINED-SCORE ARMS (GPM thread, Hans 11/08 '응 돌려보자'): rows = top 5% (w=16) / top 10% (w=8) by the logistic combined score (g128, g', nll_win, token type; e57_combined_masks_cpu.py) of the W0 pass, variant A trained on the news tune events, variant B trained on QA; ALLQ128 on every row from chunk 8, update norms of E57_main_NAT x 0.5 (engine SELC1): NEWSqcmA05W16, NEWSqcmA10W8, NEWSqcmB05W16, NEWSqcmB10W8. Reused: ALLQ128, M3 w=8, TOP5, W0-ABS2, TOP10.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_news_m3.py e57_news_tokread.py e57_rule_masks_real.py e57_combined_real_report.py e57_syn_mrun.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_news_m3.py --selftest | tail -1; $PY $S/e57_combined_real_report.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_news_combined}; MASKS=$OUT/e57_news_m3/news_combined_masks.npz; mkdir -p "$TOUT"
for f in $MASKS $OUT/e57_main/E57_main_NAT.arms.json $OUT/e57_news_rules/read/e57_news_tokread.json $OUT/e57_news_massgrid/read/e57_news_tokread.json $OUT/e57_news_tokread/e57_news_tokread.json $OUT/e57_news_split.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
for X in "NEWSqcmA05W16 16 qcmA05" "NEWSqcmA10W8 8 qcmA10" "NEWSqcmB05W16 16 qcmB05" "NEWSqcmB10W8 8 qcmB10"; do
  set -- $X
  [ -f "$TOUT/E57_news_${1}C.arms.json" ] || $PY $S/e57_news_m3.py --stage write --out "$TOUT" --weight $2 --arm $1 --saves 279 --mask-npz $MASKS --mask-key $3
done
EXTRA=$($PY - "$TOUT" <<'PYE'
import json,sys
t=sys.argv[1];parts=[]
for a in ('NEWSqcmA05W16','NEWSqcmA10W8','NEWSqcmB05W16','NEWSqcmB10W8'):
    d=json.load(open(f'{t}/E57_news_{a}C.arms.json'));s=d['states']['279'];parts.append(f"{a}@c279,{s['file']},{s['hash']}")
print(';'.join(parts))
PYE
)
[ -f "$TOUT/read/e57_news_tokread.json" ] || $PY $S/e57_news_tokread.py --stage read --out "$TOUT/read" --sets v1,v2A --extra "$EXTRA"
$PY $S/e57_combined_real_report.py --source news --out "$OUT/e57_combined_news_report.json"
echo "outputs: $OUT/e57_combined_news_report.json $TOUT/read/e57_news_tokread.json"
