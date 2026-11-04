#!/bin/bash
# JOB 930 = E57 QA EXTENSION TO 65 VERSIONS (QA cause thread, Hans '수리부터 하고 한번에 돌려보자' 11/05 00:39 KST): bundles 5 / 8 / 9 / 10 / 12 (73 items), stream = original + 64 re-descriptions per paragraph (rewrites_64_subset.json, sha256 5f99556baa5d251d...; up to 65 versions, 90-111 chunks per bundle; the first 17 sections equal the streams of jobs 790 / 870), arms NATQ, ALLQ128, M3 w=8, M4 (deciding band 32) with the NAT norm source, canary at every read chunk, read points n = 1 / 3 / 5 / 9 / 17 / 25 / 33 / 49 / 65; reads of n <= 33 first (intermediate analysis file), then n = 49 / 65 and the final analysis; metrics as job 870. New file e57_pqa19_run.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa19_run.py e57_pqa16_run.py e57_pqa15_run.py e57_dwg_qa_pass.py e57_syn_mrun.py e57_pqa12_run.py e57_pqa6_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa19_run.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; RD=${RD:-$OUT/e57_public_qa_n65}; D620=$OUT/e57_public_qa_stage4; D610=$OUT/e57_public_qa_stage3; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
[ "$(sha256sum $OUT/e57_pqa15_in/rewrites_64_subset.json | cut -c1-12)" = 5f99556baa5d ] || { echo "rewrites_64_subset.json sha256 differs (exit 9)"; exit 9; }
for f in $N17/e57_pqa6_groups.json $N17/e57_pqa6_group_ids.npz $D610/data/paragraphs.json data/squad/train-v1.1.json data/squad/dev-v1.1.json $ITEMS $BANK $ST1 $ASG; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$RD/e57_pqa6_groups.json" ] || $PY $S/e57_pqa19_run.py --stage build --out "$RD" --src790 "$N17" --para $D610/data/paragraphs.json --rew64 $OUT/e57_pqa15_in/rewrites_64_subset.json --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$RD/e57_qa_ext.json" ] && [ -z "$RERUN" ] && { echo "e57_qa_ext.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$RD/e57_dwg_qa_nll.npz" ] || $PY $S/e57_dwg_qa_pass.py --src640 "$RD" --out "$RD" --bands 128
$PY $S/e57_pqa19_run.py --stage write --out "$RD"
if [ ! -f "$RD/e57_qa_ext_n33.json" ]; then
  $PY $S/e57_pqa19_run.py --stage read --n-set A --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
  cp "$RD/e57_pqa15_read_meta.json" "$RD/e57_pqa15_read_meta_A.json"
  $PY $S/e57_pqa19_run.py --stage analyze --n-set A --out "$RD" --items-file $ITEMS --json "$RD/e57_qa_ext_n33.json"
  echo "INTERMEDIATE (n <= 33): $RD/e57_qa_ext_n33.json"
fi
if [ ! -f "$RD/e57_pqa15_read_meta_B.json" ]; then
  $PY $S/e57_pqa19_run.py --stage read --n-set B --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
  cp "$RD/e57_pqa15_read_meta.json" "$RD/e57_pqa15_read_meta_B.json"
fi
$PY - "$RD" <<'PYE'
import json,sys,pathlib
R=pathlib.Path(sys.argv[1]);a=json.load(open(R/'e57_pqa15_read_meta_A.json'));b=json.load(open(R/'e57_pqa15_read_meta_B.json'))
assert a['item_ids']==b['item_ids']
m=dict(a);m['labels']=a['labels']+[l for l in b['labels'] if l!='W0'];(R/'e57_pqa15_read_meta.json').write_text(json.dumps(m));print('labels',len(m['labels']))
PYE
$PY $S/e57_pqa19_run.py --stage analyze --n-set all --out "$RD" --items-file $ITEMS --json "$RD/e57_qa_ext.json"
$PY - "$RD" <<'PYE'
import json,sys
W=json.load(open(sys.argv[1]+'/e57_pqa6_writes.json'));can={}
for gid,arms in W.items():
    for arm,v in arms.items():
        for ch,s in v.get('states',{}).items():
            c=s.get('canary')
            if c:can.setdefault(arm,{}).setdefault(gid,{})[ch]=dict(gen_nll_rise=c['_gen_nll']-v['canary_w0']['_gen_nll'],tail_nll_rise=c['_tail_nll']-v['canary_w0']['_tail_nll'])
json.dump(can,open(sys.argv[1]+'/e57_qa_ext_canary.json','w'),indent=1);print('canary written',list(can))
PYE
echo "outputs: $RD/e57_qa_ext.json $RD/e57_qa_ext_n33.json $RD/e57_qa_ext_canary.json $RD/E57_pqa15_rows_*.json"
