#!/bin/bash
# JOB 1090 = E57 NEWS STREAM A WITH RULE ARMS (GPM thread, Hans 11/05 '진행시켜'): ABS-NAT rows (g128 > 2 / 4 / 6 nats) and the fixed top 5% rows, loss weight 16, ALLQ128 on every row from chunk 8, update norms of E57_main_NAT (as the M3 w=8 write of job 770); masks e57_news_m3/news_rules_masks.npz (e57_rule_masks_real.py). Four arms NEWSqtA2W16 / NEWSqtA4W16 / NEWSqtA6W16 / NEWSqt05W16, end state (chunk 279) read per token on the question sets v1 (150) and v2A (74) together with the stored M3 w=8 end state of job 770 (E57_news_M3W8Q128.arms.json); ALLQ128 / NATQ / W0 values come from the stored tokread of job 570+. Scoring by the tune / hold-out event split (e57_news_split.json): e57_split_score.py.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_news_m3.py e57_news_tokread.py e57_rule_masks_real.py e57_rules.py e57_syn_mrun.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_news_m3.py --selftest | tail -1; $PY $S/e57_news_tokread.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_news_rules}; MASKS=$OUT/e57_news_m3/news_rules_masks.npz; mkdir -p "$TOUT"
for f in $MASKS $OUT/e57_main/E57_main_NAT.arms.json $OUT/e57_news_m3/E57_news_M3W8Q128.arms.json $OUT/e57_news_tokread/e57_news_tokread.json $OUT/e57_news_split.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/read/e57_news_tokread.json" ] && [ -z "$RERUN" ] && { echo "tokread exists (exit 8; RERUN=1 to redo)"; exit 8; }
for R in qtA2 qtA4 qtA6 qt05; do
  [ -f "$TOUT/E57_news_NEWS${R}W16C.arms.json" ] || $PY $S/e57_news_m3.py --stage write --out "$TOUT" --weight 16 --arm NEWS${R}W16 --saves 279 --mask-npz $MASKS --mask-key $R
done
EXTRA=$($PY - "$TOUT" "$OUT" <<'PYE'
import json,sys
t,o=sys.argv[1:3];parts=[]
for r in ('qtA2','qtA4','qtA6','qt05'):
    d=json.load(open(f'{t}/E57_news_NEWS{r}W16C.arms.json'));s=d['states']['279'];parts.append(f"NEWS{r}W16@c279,{s['file']},{s['hash']}")
d=json.load(open(f'{o}/e57_news_m3/E57_news_M3W8Q128.arms.json'));s=d['states']['279'];parts.append(f"M3W8Q128@c279,{s['file']},{s['hash']}")
print(';'.join(parts))
PYE
)
$PY $S/e57_news_tokread.py --stage read --out "$TOUT/read" --sets v1,v2A --extra "$EXTRA"
echo "outputs: $TOUT/read/e57_news_tokread.json $TOUT/E57_news_NEWS*W16C.arms.json (scoring: e57_split_score.py)"
