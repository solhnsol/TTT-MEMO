#!/bin/bash
# JOB 870 = E57 QA BUNDLE EPISODES WITH 17 EXPOSURES: M3 w=8 AND M4 AGAINST NATQ AND ALLQ128 (method thread, Hans '같은 규칙 그대로 실행' 11/04 11:33 KST): the episodes of job 790 (bundles 5 / 8 / 9 / 10 / 12, 73 items, rewrites_16_subset.json); W0 pass over the streams (band 128; e57_dwg_qa_pass.py) -> deciding rows = top 30% of the eligible rows of every chunk by g = nll_band128 - nll_win; arms M3W8Q128 (ALLQ128 mask on every row from chunk 0, loss weight 8 on the deciding rows, update norm 0.5 x NAT of job 790) and M4W8Q128 (the deciding rows see only the previous 32 tokens); states at n = 1 / 3 / 5 / 9 / 13 / 17 with the canary; reads like job 790 (question read A / B, closed same-bundle read); NATQ and ALLQ128 rows of job 790 reused; first content token (space-only skipped) dA, d(A-B), masking extras, rank ratios, second token, rest. New file e57_pqa16_run.py (uses e57_pqa15_run.py, e57_syn_mrun.py, e57_dwg_qa_pass.py), small additions to e57_pqa15_run.py (--skip_repro, more extra pairs); no other existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa16_run.py e57_pqa15_run.py e57_dwg_qa_pass.py e57_syn_mrun.py e57_pqa12_run.py e57_pqa6_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa16_run.py --selftest | tail -1; $PY $S/e57_dwg_qa_pass.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; MD=${MD:-$OUT/e57_public_qa_n17_m3m4}; D620=$OUT/e57_public_qa_stage4; mkdir -p "$MD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
for f in $N17/e57_pqa6_groups.json $N17/e57_pqa6_group_ids.npz $N17/e57_pqa6_writes.json $N17/e57_pqa15_read_meta.json $N17/E57_pqa15_rows_A_W0.json $N17/E57_pqa15_rows_A_ALLQ128_n17.json $ITEMS $BANK $ST1 $ASG; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
cp -n $N17/e57_pqa6_groups.json $N17/e57_pqa6_group_ids.npz $N17/e57_pqa6_writes.json "$MD/"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MD/e57_qa_m3m4.json" ] && [ -z "$RERUN" ] && { echo "e57_qa_m3m4.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$MD/e57_dwg_qa_nll.npz" ] || $PY $S/e57_dwg_qa_pass.py --src640 "$MD" --out "$MD" --bands 128
[ -f "$MD/e57_pqa16_writes.json" ] && $PY - "$MD" <<'PYE' || true
import json,sys;W=json.load(open(sys.argv[1]+'/e57_pqa16_writes.json'));print('writes present for bundles',sorted(W))
PYE
$PY $S/e57_pqa16_run.py --stage write --out "$MD"
[ -f "$MD/new_rows/e57_pqa15_read_meta.json" ] || $PY $S/e57_pqa16_run.py --stage read --out "$MD" --src640 "$N17" --src790 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY - "$N17" "$MD" <<'PYE'
import json,sys,os,pathlib,numpy as np
N,M=map(pathlib.Path,sys.argv[1:3]);mg=M/'merged';mg.mkdir(exist_ok=True)
old=json.load(open(N/'e57_pqa15_read_meta.json'));new=json.load(open(M/'new_rows/e57_pqa15_read_meta.json'))
for f in ('e57_pqa6_groups.json','e57_pqa6_group_ids.npz'):
    if not (mg/f).exists():os.symlink(os.path.abspath(N/f),mg/f)
for f in ('e57_pqa15_tokens.json','e57_pqa15_layout.json'):
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
W=json.load(open(M/'e57_pqa16_writes.json'));can={}
for gid,arms in W.items():
    for arm,v in arms.items():
        for ch,s in v['states'].items():
            c=s.get('canary')
            if c:can.setdefault(arm,{}).setdefault(gid,{})[ch]=dict(gen_nll_rise=c['_gen_nll']-v['canary_w0']['_gen_nll'],tail_nll_rise=c['_tail_nll']-v['canary_w0']['_tail_nll'])
(M/'e57_qa_m3m4_canary.json').write_text(json.dumps(can,indent=1));print('canary written')
PYE
$PY $S/e57_pqa16_run.py --stage analyze --out "$MD/merged" --items-file $ITEMS --json "$MD/e57_qa_m3m4.json"
echo "outputs: $MD/e57_qa_m3m4.json $MD/e57_qa_m3m4_canary.json $MD/new_rows/ $MD/e57_pqa16_writes.json"
