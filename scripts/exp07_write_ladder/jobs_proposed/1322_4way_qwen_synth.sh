#!/bin/bash
# JOB 1322 = 4-way arms on Qwen3-4B-Base synthetic worlds + REVERSE-question readout (--reverse; secondary, Hans 11/19) (Hans 11/19: NAT, all-layer band, CHK, ISO; world seeds 221..228; EXPLORATORY, no pass/fail gating; code e59_qwen_mask.py -> e59_qwen_run.py): fullmlp (last 9 layers, fp32), fixed per-chunk update norm T = 0.30426 (6%), arms NAT, MASK1024 (band-1024 mask in ALL 36 layers = the E2E mask), CHK (chunk alone, all layers), ISO (16 isolated sentences per round); reads exposures 16 / 32 / 64 / 128 (out + in-window), W0 reads computed per seed. NOT queued until Hans' sentence; est. ~3.5 h (smoke ~6 min; per seed ~25 min). Output experiments/exp09_qwen_ttt/q4way.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/q4way
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 20"; exit 20; }
mkdir -p $O/smoke
sha256sum $E/e59_qwen_mask.py $E/e59_qwen_run.py $E/bandmask.py $E/cell_band_ladder.py $E/inner.py $E/qwen_runtime.py $E/world_text.py scripts/exp07_write_ladder/e59_srv_core.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_mask.py --selftest | tail -1
$PYQ $E/e59_qwen_mask.py --world-seed 221 --step-norm 0.30426 --arms NAT,MASK1024,CHK,ISO --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun --reverse
$PYQ - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];A=('NAT','MASK1024','CHK','ISO');R={a:json.load(open(f'{d}/e59q_{a}_s221.json')) for a in A}
for a,r in R.items():
    s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ce step1',round(s[1]['ce'],3))
assert all(x['fact_hash']==y['fact_hash'] for a in A[1:] for x,y in zip(R[a]['steps'],R['NAT']['steps'])) and R['NAT']['steps'][1]['ce']!=R['MASK1024']['steps'][1]['ce']
for a,r in R.items():assert set(r['reads_rev'])=={'16'} and len(r['reads_rev']['16']['out'])==16 and len(r['reads_rev']['16']['in'])==16 and r['meta_rev']['gold_idx']==[0]*16,a
print('4WAY QWEN SMOKE PASSED (forward + reverse readout)')
PYE
for s in 221 222 223 224 225 226 227 228; do
  set +e
  $PYQ $E/e59_qwen_mask.py --world-seed $s --step-norm 0.30426 --arms NAT,MASK1024,CHK,ISO --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --out $O --reverse
  echo "RUN 4way qwen seed $s rc=$?"
  set -e
done
