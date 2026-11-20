#!/bin/bash
# JOB 1330 = SAME-FACT SPECIFICITY on Qwen3-4B-Base synthetic worlds (Hans 11/20): arms SAME (earlier same-fact sentences hidden) / OTHERONLY (earlier other-fact sentences hidden, same fact visible) / BOTH (all earlier fact sentences hidden) / RANDF (same number of earlier keys per chunk as BOTH, random FILLER keys, scattered), definitions of job 1308z, update-path mask in ALL 36 layers, sentence rows only, K0 = 0, fixed per-chunk norm T = 0.30426, world seeds 221..228 (the 1322 worlds); reference NAT / CHK / ISO / MASK1024 = the 1322 runs (experiments/exp09_qwen_ttt/q4way); code e59_qwen_hide.py + e59_hidearms.py (new); Qwen cache window = 7 earlier chunks (E2E 8). Main metric = out-of-window margin gain at exposure 128 vs NAT (and vs W0); rank / top-1 secondary. EXPLORATORY; NOT queued until Hans' sentence; est. ~5 h (smoke ~8 min; per seed ~38 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/same_spec
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 20"; exit 20; }
mkdir -p $O/smoke
sha256sum $E/e59_qwen_hide.py $E/e59_qwen_mask.py $E/e59_qwen_run.py scripts/exp07_write_ladder/e59_hidearms.py $E/bandmask.py $E/cell_band_ladder.py scripts/exp07_write_ladder/e59_srv_core.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_hide.py --selftest | tail -1
$PYQ $E/e59_qwen_hide.py --world-seed 221 --step-norm 0.30426 --arms SAME,OTHERONLY,BOTH,RANDF --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun
$PYQ - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];A=('SAME', 'OTHERONLY', 'BOTH', 'RANDF');R={a:json.load(open(f'{d}/e59q_{a}_s221.json')) for a in A};N=json.load(open('experiments/exp09_qwen_ttt/q4way/e59q_NAT_s221.json'))
for a,r in R.items():
    s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3))
    assert all(x['hook_layers']==36 for x in s),(a,'the mask must act in all 36 decoder layers')
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],N['steps'][:16]))
assert all(abs(R[a]['steps'][0]['ce']-N['steps'][0]['ce'])<5e-3 for a in A) and all(R[a]['steps'][5]['ce']!=N['steps'][5]['ce'] for a in A) and len({R[a]['steps'][5]['ce'] for a in A})==len(A)
assert all(R[a]['steps'][5]['masked_pairs']>0 for a in A if a not in ('ALLHID',))
print('SAME_SPEC_QWEN_SYN SMOKE PASSED')
PYE
for s in 221 222 223 224 225 226 227 228; do
  set +e
  $PYQ $E/e59_qwen_hide.py --world-seed $s --step-norm 0.30426 --arms SAME,OTHERONLY,BOTH,RANDF --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --out $O
  echo "RUN 1330 qwen seed $s rc=$?"
  set -e
done
