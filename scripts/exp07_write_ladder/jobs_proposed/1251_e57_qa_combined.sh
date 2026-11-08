#!/bin/bash
# JOB 1251 = E57 QA 65-VERSION STREAM COMBINED-SCORE ARMS (GPM thread, Hans 11/08 '응 돌려보자'): bundles 5 / 8 / 9 / 10 / 12; rows = top 5% (w=16) / top 10% (w=8) by the logistic combined score (g128, g', nll_win, token type) of the W0 pass, variant A trained on the QA tune bundles, variant B trained on news; ALLQ128 on every row, update norm = the plain NAT write of this run: M3qcmA05W16Q128, M3qcmA10W8Q128, M3qcmB05W16Q128, M3qcmB10W8Q128. Reused: ALLQ128, M3 w=8, TOP5, W0-ABS2, TOP10, gain-mass. Masks e57_public_qa_n65/qa_combined_masks.npz.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa27_run.py e57_pqa19_run.py e57_pqa16_run.py e57_pqa15_run.py e57_combined_real_report.py e57_syn_mrun.py e57_pqa6_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa27_run.py --selftest | tail -1; $PY $S/e57_combined_real_report.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; N65=$OUT/e57_public_qa_n65; RD=${RD:-$OUT/e57_public_qa_n65_combined}; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$OUT/e57_public_qa_stage4/E57_pqa4_assignB.json
for f in $N65/e57_pqa6_groups.json $N65/e57_pqa6_group_ids.npz $N65/e57_dwg_qa_nll.npz $N65/qa_combined_masks.npz $N17/e57_pqa6_groups.json $ITEMS $BANK $ST1 $ASG; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
for f in e57_pqa6_groups.json e57_pqa6_group_ids.npz e57_dwg_qa_nll.npz; do [ -f "$RD/$f" ] || cp "$N65/$f" "$RD/$f"; done
if [ ! -f "$RD/E57_pqa15_rows_A_M3qcmB10W8Q128_n65.json" ] || [ -n "$RERUN" ]; then
  $PY $S/e57_pqa27_run.py --stage write --out "$RD"
  $PY $S/e57_pqa27_run.py --stage read --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
fi
$PY - "$RD" <<'PYE'
import json,sys
W=json.load(open(sys.argv[1]+'/e57_pqa6_writes.json'));can={}
for gid,arms in W.items():
    for arm,v in arms.items():
        for ch,s in v.get('states',{}).items():
            c=s.get('canary')
            if c:can.setdefault(arm,{}).setdefault(gid,{})[ch]=dict(gen_nll_rise=c['_gen_nll']-v['canary_w0']['_gen_nll'],tail_nll_rise=c['_tail_nll']-v['canary_w0']['_tail_nll'])
json.dump(can,open(sys.argv[1]+'/e57_qa_combined_canary.json','w'),indent=1);print('canary written',list(can))
PYE
$PY $S/e57_combined_real_report.py --source qa --out "$OUT/e57_combined_qa_report.json"
echo "outputs: $OUT/e57_combined_qa_report.json $RD/e57_qa_combined_canary.json"
