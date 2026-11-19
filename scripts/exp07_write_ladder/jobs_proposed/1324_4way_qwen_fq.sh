#!/bin/bash
# JOB 1324 = FictionalQA Qwen 4-way completion (Hans 11/19): the arms CHK and ISO on the Qwen FictionalQA worlds of job 1320 (labels 211..218; NAT and BANDALL1024 are the 1320 runs; the 9-layer BAND1024 is discarded); fixed per-chunk norm T = 0.30426, same reads / records; code fq_qwen_run.py (arms CHK, ISO of e59_qwen_run). EXPLORATORY; NOT queued until Hans' sentence; est. ~1.3 h (smoke ~4 min; per seed ~9 min; W0 reads reused from the 1320 output). Output experiments/exp09_qwen_ttt/fq_qwen (same directory as 1320).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; FQ=experiments/exp07_write_ladder/fq; W=$FQ/qwen_prep/qwen_c; O=experiments/exp09_qwen_ttt/fq_qwen
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 22"; exit 22; }
mkdir -p $O/smoke4
sha256sum $E/fq_qwen_run.py $E/e59_qwen_run.py $E/e59_qwen_mask.py $E/bandmask.py $E/cell_band_ladder.py scripts/exp07_write_ladder/fq_world.py scripts/exp07_write_ladder/e59_srv_core.py $W/*.json
$PYQ $E/fq_qwen_run.py --world-seed 211 --fq-worlds $W --step-norm 0.30426 --arms CHK,ISO --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke4 --rerun
$PYQ - $O/smoke4 <<'PYE'
import json,math,sys
d=sys.argv[1]
for a in ('CHK','ISO'):
    s=json.load(open(f'{d}/e59q_{a}_s211.json'))['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3))
print('4WAY QWEN FQ SMOKE PASSED')
PYE
for s in 211 212 213 214 215 216 217 218; do
  set +e
  $PYQ $E/fq_qwen_run.py --world-seed $s --fq-worlds $W --step-norm 0.30426 --arms CHK,ISO --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --out $O
  echo "RUN 4way qwen fq seed $s rc=$?"
  set -e
done
