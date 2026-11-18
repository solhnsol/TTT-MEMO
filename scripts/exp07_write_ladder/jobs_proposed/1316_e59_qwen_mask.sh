#!/bin/bash
# JOB 1316 = E59 follow-up (Hans 11/18 'why does Qwen band1024 bind less than ISO'): Qwen3-4B-Base fullmlp, world seeds 161 / 162, fixed per-chunk update norm T = 0.30426 (6%), arms BAND1024 (as 1311 band 1024 on the 9 fast layers) and CHKMASK9 (own-chunk-only mask on the same 9 fast layers, carried KV / earlier layers / reads natural) with the E59 per-step records; references = E59 NAT / NATF / CHK / CHKF / FACT / ISO of the same seeds (experiments/exp09_qwen_ttt/e59q/p6). EXPLORATORY; NOT queued until Hans' sentence; est. ~45 min (smoke ~5 min). If the smoke fails nothing else starts.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/e59q/mask6; REF=experiments/exp09_qwen_ttt/cband_main
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 14"; exit 14; }
mkdir -p $O/smoke
sha256sum $E/e59_qwen_mask.py $E/e59_qwen_run.py scripts/exp07_write_ladder/e59_srv_core.py $E/bandmask.py $E/inner.py $E/cell_band_ladder.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_mask.py --selftest | tail -1
$PYQ $E/e59_qwen_mask.py --world-seed 161 --step-norm 0.30426 --arms BAND1024,CHKMASK9 --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun
$PYQ - $O/smoke <<'PYE'
import json,math
d='experiments/exp09_qwen_ttt/e59q/mask6/smoke';R={a:json.load(open(f'{d}/e59q_{a}_s161.json')) for a in ('BAND1024','CHKMASK9')};N=json.load(open('experiments/exp09_qwen_ttt/e59q/smoke/e59q_NAT_s161.json'))
for a,r in R.items():
    s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: fact_ce step1',round(s[1]['fact_ce'],3),'ce',round(s[1]['ce'],3))
assert all(x['fact_hash']==y['fact_hash'] for a in R for x,y in zip(R[a]['steps'],N['steps'])),'fact hash differs from the E59 NAT run'
assert abs(R['BAND1024']['steps'][0]['ce']-N['steps'][0]['ce'])<1e-5 and R['CHKMASK9']['steps'][1]['ce']!=N['steps'][1]['ce']
print('E59 QWEN MASK SMOKE PASSED')
PYE
for s in 161 162; do
  set +e
  $PYQ $E/e59_qwen_mask.py --world-seed $s --step-norm 0.30426 --arms BAND1024,CHKMASK9 --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --ref-dir $REF --out $O
  echo "RUN e59q mask seed $s rc=$?"
  set -e
done
