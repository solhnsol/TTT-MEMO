#!/bin/bash
# JOB 890 = E57 QA BUNDLE EPISODES, 17-VERSION STREAM REPEATED R = 3 TIMES WITHOUT RE-DESCRIPTION (method thread, Hans 'R=3 실행' 11/04 12:43 KST): bundles 5 / 8 / 9 / 10 / 12 (73 items), stream = the 17 sections of job 790 x 3 (51 sections) + filler; W0 pass (band 128), arms NATQ, ALLQ128, M3 w=8 (NAT norm source; canary at every read chunk), read points n = 9 / 17 / 26 / 34 / 43 / 51; reads and metrics as job 870 (e57_pqa15_run.py functions). New file e57_pqa17_run.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa17_run.py e57_pqa16_run.py e57_pqa15_run.py e57_dwg_qa_pass.py e57_syn_mrun.py e57_pqa12_run.py e57_pqa6_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa17_run.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; RD=${RD:-$OUT/e57_public_qa_n51}; D620=$OUT/e57_public_qa_stage4; D610=$OUT/e57_public_qa_stage3; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
for f in $N17/e57_pqa6_groups.json $N17/e57_pqa6_group_ids.npz $D610/data/paragraphs.json data/squad/train-v1.1.json data/squad/dev-v1.1.json $ITEMS $BANK $ST1 $ASG; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$RD/e57_pqa6_groups.json" ] || $PY $S/e57_pqa17_run.py --stage build --out "$RD" --src790 "$N17" --para $D610/data/paragraphs.json --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$RD/e57_qa_repeat.json" ] && [ -z "$RERUN" ] && { echo "e57_qa_repeat.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$RD/e57_dwg_qa_nll.npz" ] || $PY $S/e57_dwg_qa_pass.py --src640 "$RD" --out "$RD" --bands 128
$PY $S/e57_pqa17_run.py --stage write --out "$RD"
[ -f "$RD/e57_pqa15_read_meta.json" ] || $PY $S/e57_pqa17_run.py --stage read --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa17_run.py --stage analyze --out "$RD" --items-file $ITEMS --json "$RD/e57_qa_repeat.json"
$PY - "$RD" <<'PYE'
import json,sys,numpy as np
W=json.load(open(sys.argv[1]+'/e57_pqa6_writes.json'));can={}
for gid,arms in W.items():
    for arm,v in arms.items():
        for ch,s in v.get('states',{}).items():
            c=s.get('canary')
            if c:can.setdefault(arm,{}).setdefault(gid,{})[ch]=dict(gen_nll_rise=c['_gen_nll']-v['canary_w0']['_gen_nll'],tail_nll_rise=c['_tail_nll']-v['canary_w0']['_tail_nll'])
json.dump(can,open(sys.argv[1]+'/e57_qa_repeat_canary.json','w'),indent=1);print('canary written',list(can))
PYE
echo "outputs: $RD/e57_qa_repeat.json $RD/e57_qa_repeat_canary.json $RD/E57_pqa15_rows_*.json"
