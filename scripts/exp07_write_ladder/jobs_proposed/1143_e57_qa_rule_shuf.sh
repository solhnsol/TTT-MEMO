#!/bin/bash
# JOB 1143 = E57 PAIR-SHUFFLE CONTROL OF THE RULE ARMS, QA (GPM thread, Hans 11/06 '1번 2번 먼저 하자'): the pair-shuffled 65-version stream of job 940 (bundles 8 / 10 / 12 only, n = 65), plain NAT write for the update norms, then ABS2 w=16 (g128 > 2 nats) and the fixed top 5% w=16 with ALLQ128 on every row; the masks from the g128 of the SHUFFLED stream (e57_public_qa_n65_pairshuffle/e57_dwg_qa_nll.npz -> qa_rules_masks_shuf.npz). ALLQ128 and M3 w=8 on the shuffled stream are the rows of job 940. Scored with the ORIGINAL pairs: e57_rule_shuf_report_qa.py.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa23_run.py e57_pqa19_run.py e57_pqa16_run.py e57_pqa15_run.py e57_rule_masks_real.py e57_rule_shuf_report_qa.py e57_rules.py e57_syn_mrun.py e57_pqa6_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa23_run.py --selftest | tail -1; $PY $S/e57_rule_shuf_report_qa.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; PS=$OUT/e57_public_qa_n65_pairshuffle; RD=${RD:-$OUT/e57_public_qa_n65_ruleshuf}; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$OUT/e57_public_qa_stage4/E57_pqa4_assignB.json
for f in $PS/e57_pqa6_groups.json $PS/e57_pqa6_group_ids.npz $PS/e57_dwg_qa_nll.npz $PS/E57_pqa15_rows_A_M3W8Q128_n65.json $N17/e57_pqa6_groups.json $ITEMS $BANK $ST1 $ASG $OUT/e57_public_qa_n65_rules/E57_pqa15_rows_A_M3qtA2W16Q128_n65.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$PS/qa_rules_masks_shuf.npz" ] || $PY $S/e57_rule_masks_real.py --qa-out "$PS/qa_rules_masks_shuf.npz" --qa-nll "$PS/e57_dwg_qa_nll.npz" --taus 2
$PY - "$PS" "$RD" <<'PYE'
import json,sys,shutil,pathlib
ps,rd=map(pathlib.Path,sys.argv[1:3]);g=json.load(open(ps/'e57_pqa6_groups.json'));keep=[x for x in g['groups'] if x['gid'] in (8,10,12)];assert len(keep)==3
g2=dict(g);g2['groups']=keep;g2['n_groups']=3;(rd/'e57_pqa6_groups.json').write_text(json.dumps(g2))
for f in ('e57_pqa6_group_ids.npz','e57_dwg_qa_nll.npz'):shutil.copy(ps/f,rd/f)
print('tune bundles',[x['gid'] for x in keep])
PYE
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$RD/E57_pqa15_rows_A_M3qtA2W16Q128_n65.json" ] && [ -z "$RERUN" ] && { echo "rows exist (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_pqa23_run.py --stage write --out "$RD"
$PY $S/e57_pqa23_run.py --stage read --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_rule_shuf_report_qa.py --out "$RD/e57_rule_shuf_report_qa.json"
echo "outputs: $RD/e57_rule_shuf_report_qa.json; masks $PS/qa_rules_masks_shuf.json"
