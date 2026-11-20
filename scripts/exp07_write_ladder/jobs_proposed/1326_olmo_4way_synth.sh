#!/bin/bash
# JOB 1326 = OLMo-3-7B (allenai/Olmo-3-1025-7B) 4-way arms on the synthetic world (Hans 11/19: OLMo as the third model; fast weights = last 8 layers' MLP (gate/up/down, fp32, ||W_fast|| = 721.32), mask = ALL 32 layers via olmo_port.LayerPlan (asserted); NAT, MASK1024 (band 1024), CHK, ISO; fixed per-chunk update norm T = 0.33812 = 6% of ||W_fast|| over 128 chunks (the same relative size as Qwen 0.30426 and E2E 12% path -> 6% like the Qwen runs); world seeds 251..258 (proposed; Hans may change); reads exposures 16/32/64/128 out + in-window + reverse readout; code scripts/exp11_olmo3/olmo_port.py + olmo_run.py (new files), e59_qwen_run.py / e59_qwen_mask.py (patched in the process). EXPLORATORY; NOT queued until Hans' sentence; first GPU use of the OLMo port: the smoke also records peak GPU memory and step time; est. ~9-11 h (smoke ~15 min; per seed ~70 min).
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp11_olmo3; O=experiments/exp11_olmo3/q4way
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--allenai--Olmo-3-1025-7B/snapshots/* | head -1)
T=0.33812
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 24"; exit 24; }
mkdir -p $O/smoke
sha256sum $E/olmo_port.py $E/olmo_run.py scripts/exp09_qwen_ttt/e59_qwen_run.py scripts/exp09_qwen_ttt/e59_qwen_mask.py scripts/exp07_write_ladder/e59_srv_core.py scripts/exp07_write_ladder/e59_rev.py $SNAP/model.safetensors.index.json $SNAP/config.json
CUDA_VISIBLE_DEVICES= $PYQ $E/olmo_port.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= $PYQ $E/olmo_run.py --selftest | tail -1
$PYQ $E/olmo_run.py --world-seed 251 --model $SNAP --fast fullmlp --olmo-fast-n 8 --lr 1.0 --clip 1.0 --step-norm $T --arms NAT,MASK1024,CHK,ISO --max-chunks 16 --reads 16 --reverse --readout-batch 4 --out $O/smoke --rerun
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
$PYQ - $O/smoke $T <<'PYE'
import json,math,sys
d=sys.argv[1];T=float(sys.argv[2]);A=('NAT','MASK1024','CHK','ISO');R={a:json.load(open(f'{d}/e59q_{a}_s251.json')) for a in A}
for a,r in R.items():
    s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-T)<=1e-4*T and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3),'sec',round(r['sec']))
    assert set(r['reads_rev'])=={'16'} and len(r['reads_rev']['16']['out'])==16,a
assert all(x['fact_hash']==y['fact_hash'] for a in A[1:] for x,y in zip(R[a]['steps'],R['NAT']['steps'])) and R['NAT']['steps'][1]['ce']!=R['MASK1024']['steps'][1]['ce'] and abs(R['NAT']['steps'][0]['ce']-R['MASK1024']['steps'][0]['ce'])<5e-2
print('OLMO SMOKE PASSED')
PYE
for s in 251 252 253 254 255 256 257 258; do
  set +e
  $PYQ $E/olmo_run.py --world-seed $s --model $SNAP --fast fullmlp --olmo-fast-n 8 --lr 1.0 --clip 1.0 --step-norm $T --arms NAT,MASK1024,CHK,ISO --reads 16,32,64,128 --reverse --readout-batch 4 --out $O
  echo "RUN olmo seed $s rc=$?"
  set -e
done
