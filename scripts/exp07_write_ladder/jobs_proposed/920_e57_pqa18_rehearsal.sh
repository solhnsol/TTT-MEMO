#!/bin/bash
# JOB 920 = E57 QA SPACED REHEARSAL OF THE DECIDING ROWS, P3 (investigation thread 'binding learned fast at few exposures', Hans card 'P3 결정 행 간격 복습, QA에서 두 팔 실행' 11/04 20:41 KST; design ref/few-exposure-binding-methods-2026-11-04.md section 7; QA bundle episodes of jobs 790 / 870): base ALLQ128 (every row, from chunk 0), update norm 0.5 x the NAT norm of job 790; rehearsal fragments (63 tokens before a deciding row + the next token) of the chunks k-1, k-2, k-4, k-8 rewritten at chunk k, at most 16 per step in one 1024-token sequence with a block-diagonal mask in a second suffix forward, loss = sum of the NLL of both forwards / total loss tokens; arms REP (deciding rows) and REPRAND (random eligible rows of the same source chunks, same number); n = 1 / 3 / 5 / 9 / 13 / 17 with the canary; reads and analysis as job 870, comparison with the ALLQ128 state of job 890 at n = 34. New file e57_pqa18_run.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa18_run.py e57_pqa16_run.py e57_pqa15_run.py e57_pqa12_run.py e57_pqa6_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa18_run.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; N51=$OUT/e57_public_qa_n51; M34=$OUT/e57_public_qa_n17_m3m4; RD=${RD:-$OUT/e57_public_qa_n17_rehearsal}; D620=$OUT/e57_public_qa_stage4; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
for f in $N17/e57_pqa6_groups.json $N17/e57_pqa6_group_ids.npz $N17/e57_pqa6_writes.json $N17/e57_pqa15_read_meta.json $M34/e57_dwg_qa_nll.npz $N51/E57_pqa15_rows_A_ALLQ128_n34.json $ITEMS $BANK $ST1 $ASG; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
cp -n $N17/e57_pqa6_groups.json $N17/e57_pqa6_group_ids.npz $N17/e57_pqa6_writes.json "$RD/"
[ -f "$RD/e57_pqa18_plan.json" ] || $PY $S/e57_pqa18_run.py --stage plan --out "$RD" --dwg $M34/e57_dwg_qa_nll.npz
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$RD/e57_qa_rehearsal.json" ] && [ -z "$RERUN" ] && { echo "e57_qa_rehearsal.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_pqa18_run.py --stage write --out "$RD"
[ -f "$RD/new_rows/e57_pqa15_read_meta.json" ] || $PY $S/e57_pqa18_run.py --stage read --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY - "$N17" "$RD" <<'PYE'
import json,sys,os,pathlib,numpy as np
N,M=map(pathlib.Path,sys.argv[1:3]);mg=M/'merged';mg.mkdir(exist_ok=True)
old=json.load(open(N/'e57_pqa15_read_meta.json'));new=json.load(open(M/'new_rows/e57_pqa15_read_meta.json'))
for f in ('e57_pqa6_groups.json','e57_pqa6_group_ids.npz','e57_pqa15_tokens.json','e57_pqa15_layout.json'):
    if not (mg/f).exists():os.symlink(os.path.abspath(N/f),mg/f)
chk={}
for c in ('A','B','BUN'):
    for l in old['labels']:
        dst=mg/f'E57_pqa15_rows_{c}_{l.replace("@","_")}.json'
        if not dst.exists():os.symlink(os.path.abspath(N/f'E57_pqa15_rows_{c}_{l.replace("@","_")}.json'),dst)
    for l in new['labels']:
        if l=='W0':continue
        dst=mg/f'E57_pqa15_rows_{c}_{l.replace("@","_")}.json'
        if not dst.exists():os.symlink(os.path.abspath(M/f'new_rows/E57_pqa15_rows_{c}_{l.replace("@","_")}.json'),dst)
    a=json.load(open(N/f'E57_pqa15_rows_{c}_W0.json'));b=json.load(open(M/f'new_rows/E57_pqa15_rows_{c}_W0.json'));chk[c]=float(max(np.abs(np.asarray(x['cand_lp'])-np.asarray(y['cand_lp'])).max() for x,y in zip(a,b)))
print('W0 rows of this read vs job 790 (packing check), max|diff|:',chk)
meta=dict(old);meta['labels']=old['labels']+[l for l in new['labels'] if l!='W0'];meta['w0_packing_check_vs_job790']=chk;(mg/'e57_pqa15_read_meta.json').write_text(json.dumps(meta))
W=json.load(open(M/'e57_pqa18_writes.json'));can={};rl={}
for gid,arms in W.items():
    for arm,v in arms.items():
        rl.setdefault(arm,{})[gid]=v.get('rep_log')
        for ch,s in v['states'].items():
            c=s.get('canary')
            if c:can.setdefault(arm,{}).setdefault(gid,{})[ch]=dict(gen_nll_rise=c['_gen_nll']-v['canary_w0']['_gen_nll'],tail_nll_rise=c['_tail_nll']-v['canary_w0']['_tail_nll'])
(M/'e57_qa_rehearsal_canary.json').write_text(json.dumps(can,indent=1));(M/'e57_qa_rehearsal_replog.json').write_text(json.dumps(rl));print('canary and rehearsal log written')
PYE
$PY $S/e57_pqa18_run.py --stage analyze --out "$RD/merged" --items-file $ITEMS --json "$RD/e57_qa_rehearsal.json"
$PY $S/e57_pqa18_run.py --stage compare --out "$RD" --merged "$RD/merged" --r890 "$N51" --json "$RD/e57_qa_rehearsal_compare890.json"
echo "outputs: $RD/e57_qa_rehearsal.json $RD/e57_qa_rehearsal_compare890.json $RD/e57_qa_rehearsal_canary.json $RD/e57_qa_rehearsal_replog.json"
