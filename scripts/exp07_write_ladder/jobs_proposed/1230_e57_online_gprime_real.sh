#!/bin/bash
# JOB 1230 = E57 ONLINE g' ABS2 RULE ON THE QA 65-VERSION STREAM AND NEWS STREAM A, plus the news ONLINE g ABS2 baseline (GPM thread, Hans 11/08 "g′ + 뉴스 온라인 g 팔" via coordinator): g' = nll_suffixband128 - nll_win from the CURRENT fast weights (natural prefix outputs, the band only in the 8 suffix layers), rows with g' > 2 nats among the eligible rows get loss weight 16, ALLQ128 on every row (from chunk 8 for news). Arms: QA M3OS2W16Q128 (bundles 5/8/9/10/12, e57_pqa25_run.py), news NEWSOS2W16 and NEWSOA2W16 (online g ABS2, the baseline of the never-run job 1170) with the news M3 write of e57_news_m3.py (saves at chunk 279). ALLQ128, M3 w=8, TOP5, W0-ABS2 and the QA online g (job 1160) are reused.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa25_run.py e57_pqa19_run.py e57_online_uv.py e57_news_m3.py e57_news_tokread.py e57_online_gprime_real_report.py e57_syn_mrun.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa25_run.py --selftest | tail -1; $PY $S/e57_news_m3.py --selftest | tail -1; $PY $S/e57_online_uv.py --selftest | tail -1; $PY $S/e57_online_gprime_real_report.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; N65=$OUT/e57_public_qa_n65; RD=${RD:-$OUT/e57_public_qa_n65_gprime}; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$OUT/e57_public_qa_stage4/E57_pqa4_assignB.json
TOUT=${TOUT:-$OUT/e57_news_online_gprime}; MASKS=$OUT/e57_news_m3/news_rules_masks.npz; mkdir -p "$TOUT"
for f in $N65/e57_pqa6_groups.json $N65/e57_pqa6_group_ids.npz $N65/e57_dwg_qa_nll.npz $N17/e57_pqa6_groups.json $ITEMS $BANK $ST1 $ASG $OUT/e57_public_qa_n65_online/E57_pqa15_rows_A_M3OA2W16Q128_n65.json $MASKS $OUT/e57_main/E57_main_NAT.arms.json $OUT/e57_news_rules/read/e57_news_tokread.json $OUT/e57_news_tokread/e57_news_tokread.json $OUT/e57_news_split.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
for f in e57_pqa6_groups.json e57_pqa6_group_ids.npz e57_dwg_qa_nll.npz; do [ -f "$RD/$f" ] || cp "$N65/$f" "$RD/$f"; done
if [ ! -f "$RD/E57_pqa15_rows_A_M3OS2W16Q128_n65.json" ] || [ -n "$RERUN" ]; then
  $PY $S/e57_pqa25_run.py --stage write --out "$RD"
  $PY $S/e57_pqa25_run.py --stage read --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
fi
$PY - "$RD" <<'PYE'
import json,sys
W=json.load(open(sys.argv[1]+'/e57_pqa6_writes.json'));can={}
for gid,arms in W.items():
    for arm,v in arms.items():
        for ch,s in v.get('states',{}).items():
            c=s.get('canary')
            if c:can.setdefault(arm,{}).setdefault(gid,{})[ch]=dict(gen_nll_rise=c['_gen_nll']-v['canary_w0']['_gen_nll'],tail_nll_rise=c['_tail_nll']-v['canary_w0']['_tail_nll'])
json.dump(can,open(sys.argv[1]+'/e57_qa_online_canary.json','w'),indent=1);print('canary written',list(can))
PYE
for A in NEWSOS2W16 NEWSOA2W16; do
  [ -f "$TOUT/E57_news_${A}C.arms.json" ] || $PY $S/e57_news_m3.py --stage write --out "$TOUT" --weight 16 --arm $A --saves 279 --mask-npz $MASKS
done
EXTRA=$($PY - "$TOUT" <<'PYE'
import json,sys
t=sys.argv[1];parts=[]
for a in ('NEWSOS2W16','NEWSOA2W16'):
    d=json.load(open(f'{t}/E57_news_{a}C.arms.json'));s=d['states']['279'];parts.append(f"{a}@c279,{s['file']},{s['hash']}")
print(';'.join(parts))
PYE
)
[ -f "$TOUT/read/e57_news_tokread.json" ] || $PY $S/e57_news_tokread.py --stage read --out "$TOUT/read" --sets v1,v2A --extra "$EXTRA"
$PY $S/e57_online_gprime_real_report.py --out "$OUT/e57_online_gprime_real_report.json"
echo "outputs: $OUT/e57_online_gprime_real_report.json $RD/e57_qa_online_canary.json $TOUT/read/e57_news_tokread.json"
