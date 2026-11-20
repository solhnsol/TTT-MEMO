#!/bin/bash
# JOB 1336 = FEW-REPETITION read-out on Qwen3-4B-Base (Hans 11/20): synthetic worlds seeds 221..228 (the 1322 worlds), arms NAT / MASK1024 (band-1024 mask in ALL 36 layers) / CHK / ISO trained on the FIRST 8 chunks only (--max-chunks 8) and read out-of-window (and in-window) after 2 / 4 / 8 exposures (--reads 2,4,8); fixed per-chunk norm T = 0.30426, fullmlp last 9 layers, same code as 1322 (e59_qwen_mask.py -> e59_qwen_run.py; nothing new in the arms). Main metric = out-of-window margin gain vs NAT (and vs W0) at exposures 2 / 4 / 8. EXPLORATORY; NOT queued until Hans' sentence; est. ~3.2 h (smoke ~6 min; per seed ~24 min; unmeasured).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/fewrep
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 20"; exit 20; }
mkdir -p $O/smoke
sha256sum $E/e59_qwen_mask.py $E/e59_qwen_run.py $E/bandmask.py $E/cell_band_ladder.py $E/inner.py $E/qwen_runtime.py scripts/exp07_write_ladder/e59_srv_core.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_mask.py --selftest | tail -1
$PYQ $E/e59_qwen_mask.py --world-seed 221 --step-norm 0.30426 --arms NAT,MASK1024,CHK,ISO --max-chunks 8 --reads 2,4,8 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun
$PYQ - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];A=('NAT','MASK1024','CHK','ISO');R={a:json.load(open(f'{d}/e59q_{a}_s221.json')) for a in A};N=json.load(open('experiments/exp09_qwen_ttt/q4way/e59q_NAT_s221.json'))
for a,r in R.items():
    s=r['steps'];assert len(s)==8 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3))
    assert set(r['reads'])=={'2','4','8'} and len(r['reads']['2']['out'])==20,a
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],N['steps'][:8]))
assert abs(R['NAT']['steps'][0]['ce']-N['steps'][0]['ce'])<5e-3 and len({R[a]['steps'][5]['ce'] for a in A})==4
print('FEWREP QWEN SMOKE PASSED')
PYE
for s in 221 222 223 224 225 226 227 228; do
  set +e
  $PYQ $E/e59_qwen_mask.py --world-seed $s --step-norm 0.30426 --arms NAT,MASK1024,CHK,ISO --max-chunks 8 --reads 2,4,8 --fast fullmlp --lr 1.0 --clip 1.0 --out $O
  echo "RUN fewrep qwen seed $s rc=$?"
  set -e
done
