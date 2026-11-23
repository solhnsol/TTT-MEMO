#!/bin/bash
# JOB 1343 = REVERSAL CONTROL on Qwen3-4B-Base FictionalQA worlds (labels 211..218; review follow-up 11/23; code e59_qwen_hide.py arms ALLHID / SHOWSAME / SHOWOTHER with --fq-worlds, e59_hidearms.py; the update-path mask acts in ALL 36 decoder layers (asserted), fast weights = the last 9 layers' MLP, fixed per-chunk update norm T = 0.30426): the same three arms as job 1334 on the synthetic world; reference NAT / CHK / ISO / BANDALL1024 = the 1320 / 1324 runs (experiments/exp09_qwen_ttt/fq_qwen). Main metric = out-of-window margin at exposure 128, (arm - NAT) and vs W0, 8 seeds; secondary rank / top-1. Qwen window = 7 earlier chunks (E2E 8). EXPLORATORY; NOT queued until Hans' sentence; est. ~2.7 h (1331 ran 4 arms in 3.65 h).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; W=experiments/exp07_write_ladder/fq/qwen_prep/qwen_c;O=experiments/exp09_qwen_ttt/show_fq
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 20"; exit 20; }
mkdir -p $O/smoke
sha256sum $E/e59_qwen_hide.py $E/e59_qwen_mask.py $E/e59_qwen_run.py scripts/exp07_write_ladder/e59_hidearms.py $E/bandmask.py $E/cell_band_ladder.py scripts/exp07_write_ladder/e59_srv_core.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_hide.py --selftest | tail -1
$PYQ $E/e59_qwen_hide.py --fq-worlds $W --world-seed 211 --step-norm 0.30426 --arms SHOWSAME,SHOWOTHER,ALLHID --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun
$PYQ - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];A=('SHOWSAME', 'SHOWOTHER', 'ALLHID');R={a:json.load(open(f'{d}/e59q_{a}_s211.json')) for a in A};N=json.load(open('experiments/exp09_qwen_ttt/fq_qwen/e59q_NAT_s211.json'))
for a,r in R.items():
    s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3))
    assert all(x['hook_layers']==36 for x in s),(a,'the mask must act in all 36 decoder layers')
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],N['steps'][:16]))
assert all(abs(R[a]['steps'][0]['ce']-N['steps'][0]['ce'])<5e-3 for a in A) and all(R[a]['steps'][5]['ce']!=N['steps'][5]['ce'] for a in A) and len({R[a]['steps'][5]['ce'] for a in A})==len(A)
assert all(R[a]['steps'][5]['masked_pairs']>0 for a in A)
print('SHOW_QWEN_FQ SMOKE PASSED')
PYE
for s in 211 212 213 214 215 216 217 218; do
  set +e
  $PYQ $E/e59_qwen_hide.py --fq-worlds $W --world-seed $s --step-norm 0.30426 --arms SHOWSAME,SHOWOTHER,ALLHID --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --out $O
  echo "RUN 1343 qwen seed $s rc=$?"
  set -e
done
