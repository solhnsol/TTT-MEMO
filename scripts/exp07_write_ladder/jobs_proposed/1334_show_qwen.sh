#!/bin/bash
# JOB 1334 = DECISIVE CONTROL 'show only the same fact' on Qwen3-4B-Base synthetic worlds (Hans 11/20): arms ALLHID (every row loses ALL earlier-chunk keys = the chunk alone, in all 36 layers; stream-engine counterpart of CHK), SHOWSAME (sentence rows keep the earlier same-fact sentences), SHOWOTHER (sentence rows keep the earlier other-fact sentences; placebo), filler rows see only their own chunk; fixed norm T = 0.30426, K0 = 0, seeds 221..228; reference NAT / CHK / ISO / MASK1024 = the 1322 runs. Main metric = out-of-window margin gain at exposure 128 vs NAT (and vs W0); SHOWSAME back to the NAT level => the same fact is the cause. EXPLORATORY; NOT queued until Hans' sentence; est. ~3.8 h (smoke ~6 min; per seed ~28 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/show
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 20"; exit 20; }
mkdir -p $O/smoke
sha256sum $E/e59_qwen_hide.py $E/e59_qwen_mask.py $E/e59_qwen_run.py scripts/exp07_write_ladder/e59_hidearms.py $E/bandmask.py $E/cell_band_ladder.py scripts/exp07_write_ladder/e59_srv_core.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_hide.py --selftest | tail -1
$PYQ $E/e59_qwen_hide.py --world-seed 221 --step-norm 0.30426 --arms SHOWSAME,SHOWOTHER,ALLHID --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun
$PYQ - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];A=('SHOWSAME', 'SHOWOTHER', 'ALLHID');R={a:json.load(open(f'{d}/e59q_{a}_s221.json')) for a in A};N=json.load(open('experiments/exp09_qwen_ttt/q4way/e59q_NAT_s221.json'))
for a,r in R.items():
    s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3))
    assert all(x['hook_layers']==36 for x in s),(a,'the mask must act in all 36 decoder layers')
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],N['steps'][:16]))
assert all(abs(R[a]['steps'][0]['ce']-N['steps'][0]['ce'])<5e-3 for a in A) and all(R[a]['steps'][5]['ce']!=N['steps'][5]['ce'] for a in A) and len({R[a]['steps'][5]['ce'] for a in A})==len(A)

print('SHOW_QWEN SMOKE PASSED')
PYE
for s in 221 222 223 224 225 226 227 228; do
  set +e
  $PYQ $E/e59_qwen_hide.py --world-seed $s --step-norm 0.30426 --arms SHOWSAME,SHOWOTHER,ALLHID --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --out $O
  echo "RUN 1334 qwen seed $s rc=$?"
  set -e
done
