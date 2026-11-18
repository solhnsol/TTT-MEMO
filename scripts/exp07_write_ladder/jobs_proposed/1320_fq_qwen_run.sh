#!/bin/bash
# JOB 1320 = FictionalQA on Qwen3-4B-Base: CPU prep (replacement from the Qwen screen of job 1319, Qwen re-tokenisation, controls to 4, adapter checks: scripts/exp07_write_ladder/fq_qwen_prep.sh; if it fails the job stops BEFORE any GPU work) -> smoke (seed 211, 16 chunks, NAT + BANDALL1024) -> NAT / BAND1024 / BANDALL1024 for seeds 211..218 (fullmlp last 9 layers fp32, fixed per-chunk update norm T = 0.30426 = 6%, reads exposures 16 / 32 / 64 / 128; BAND1024 = band in the 9 fast layers as 1311, BANDALL1024 = band in all 36 layers = the E2E mask: both are run so that the layer choice is not decided here). EXPLORATORY; NOT queued until Hans' sentence; est. 3-3.5 h. Needs job 1319 finished (queue order 1319 < 1320).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; FQ=experiments/exp07_write_ladder/fq; O=experiments/exp09_qwen_ttt/fq_qwen
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 18"; exit 18; }
ls $FQ/w0screen_qwen/fq_w0screen_21[1-8].json > /dev/null
bash scripts/exp07_write_ladder/fq_qwen_prep.sh $FQ/w0screen_qwen $FQ/qwen_prep
W=$FQ/qwen_prep/qwen_c; mkdir -p $O/smoke
sha256sum $E/fq_qwen_run.py $E/e59_qwen_run.py $E/e59_qwen_mask.py scripts/exp07_write_ladder/fq_world.py scripts/exp07_write_ladder/e59_srv_core.py $W/*.json
$PYQ $E/fq_qwen_run.py --world-seed 211 --fq-worlds $W --step-norm 0.30426 --arms NAT,BANDALL1024 --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun
$PYQ - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];R={a:json.load(open(f'{d}/e59q_{a}_s211.json')) for a in ('NAT','BANDALL1024')}
for a,r in R.items():
    s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3),'fact_ce',round(s[1]['fact_ce'],3))
assert [x['fact_hash'] for x in R['NAT']['steps']]==[x['fact_hash'] for x in R['BANDALL1024']['steps']] and R['NAT']['steps'][1]['ce']!=R['BANDALL1024']['steps'][1]['ce']
print('FQ QWEN SMOKE PASSED')
PYE
for s in 211 212 213 214 215 216 217 218; do
  set +e
  $PYQ $E/fq_qwen_run.py --world-seed $s --fq-worlds $W --step-norm 0.30426 --arms NAT,BAND1024,BANDALL1024 --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --out $O
  echo "RUN fq_qwen seed $s rc=$?"
  set -e
done
