#!/bin/bash
# JOB 1110 = E57 QA 65-VERSION STREAM WITH GAIN-MASS ARMS (GPM thread, Hans 11/05 '확인했어. 진행시켜'): bundles 5 / 8 / 9 / 10 / 12 of job 930 (same stream, groups, band-128 W0 file copied into e57_public_qa_n65_gm), plain NAT write for the update norms, then ABS-NAT rows (g128 > 2 / 4 / 6 nats) and the fixed top 5% rows, loss weight 16, ALLQ128 on every row (arms M3qtA2W16Q128 / M3qtA4W16Q128 / M3qtA6W16Q128 / M3qt05W16Q128; masks e57_rule_masks_real.py -> e57_public_qa_n65/qa_rules_masks_gm.npz); read points n = 9 / 33 / 65 (A, B, BUN rows as in jobs 930 / 940). ALLQ128 and M3 w=8 are not rerun (job 930 values). Scoring by tune bundles 8 / 10 / 12 and hold-out bundles 5 / 9: e57_split_score.py.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa22_run.py e57_pqa19_run.py e57_pqa16_run.py e57_pqa15_run.py e57_rule_masks_real.py e57_rules.py e57_syn_mrun.py e57_pqa12_run.py e57_pqa6_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa22_run.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; N65=$OUT/e57_public_qa_n65; RD=${RD:-$OUT/e57_public_qa_n65_gm}; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$OUT/e57_public_qa_stage4/E57_pqa4_assignB.json
for f in $N65/e57_pqa6_groups.json $N65/e57_pqa6_group_ids.npz $N65/e57_dwg_qa_nll.npz $N65/qa_rules_masks_gm.npz $N17/e57_pqa6_groups.json $ITEMS $BANK $ST1 $ASG; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$RD/e57_qa_gm_canary.json" ] && [ -z "$RERUN" ] && { echo "e57_qa_gm_canary.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
for f in e57_pqa6_groups.json e57_pqa6_group_ids.npz e57_dwg_qa_nll.npz; do [ -f "$RD/$f" ] || cp "$N65/$f" "$RD/$f"; done
$PY $S/e57_pqa22_run.py --stage write --out "$RD"
$PY $S/e57_pqa22_run.py --stage read --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY - "$RD" <<'PYE'
import json,sys
W=json.load(open(sys.argv[1]+'/e57_pqa6_writes.json'));can={}
for gid,arms in W.items():
    for arm,v in arms.items():
        for ch,s in v.get('states',{}).items():
            c=s.get('canary')
            if c:can.setdefault(arm,{}).setdefault(gid,{})[ch]=dict(gen_nll_rise=c['_gen_nll']-v['canary_w0']['_gen_nll'],tail_nll_rise=c['_tail_nll']-v['canary_w0']['_tail_nll'])
json.dump(can,open(sys.argv[1]+'/e57_qa_gm_canary.json','w'),indent=1);print('canary written',list(can))
PYE
echo "outputs: $RD/e57_qa_gm_canary.json, rows $RD/E57_pqa15_rows_*.json (scoring: e57_split_score.py)"
