#!/bin/bash
# JOB 1315 = E59 write-signal decomposition on Qwen3-4B-Base fullmlp (last 9 layers, fp32), world seeds 161 / 162, fixed per-chunk update norm (EXPLORATORY; code scripts/exp09_qwen_ttt/e59_qwen_run.py, new file; NOT queued until Hans' sentence; GPU one job at a time; est. 1.5-2 h for 6% + ~1 h for the 2% sensitivity): smoke (seed 161, 16 chunks, all six arms) -> 6% (T = 0.30426) seeds 161, 162 -> 2% (T = 0.10142, sensitivity) seeds 161, 162. W0 reads reused from experiments/exp09_qwen_ttt/cband_main (same worlds, contexts, stream sha asserted). If the smoke fails nothing else starts.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/e59q; REF=experiments/exp09_qwen_ttt/cband_main
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 13"; exit 13; }
mkdir -p $O/smoke $O/p6 $O/p2
sha256sum $E/e59_qwen_run.py scripts/exp07_write_ladder/e59_srv_core.py $E/cell_band_ladder.py $E/inner.py $E/qwen_runtime.py $E/world_text.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_run.py --selftest | tail -1
COMMON="--fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --ref-dir $REF"
$PYQ $E/e59_qwen_run.py --world-seed 161 --step-norm 0.30426 --arms NAT,NATF,CHK,CHKF,FACT,ISO --max-chunks 16 --reads 16 --fast fullmlp --lr 1.0 --clip 1.0 --out $O/smoke --rerun
$PYQ - $O/smoke <<'PYE'
import json,sys,math
d=sys.argv[1];A=('NAT','NATF','CHK','CHKF','FACT','ISO');R={a:json.load(open(f'{d}/e59q_{a}_s161.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(abs(x['applied_norm']-0.30426)<=1e-4*0.30426 and math.isfinite(x['ce']) for x in s),a;print('smoke',a,'ok: ratio step0',round(s[0]['ratio'],3),'fact_ce',round(s[0]['fact_ce'],3),'ce',round(s[0]['ce'],3))
assert all(x['fact_hash']==y['fact_hash'] for a in A[1:] for x,y in zip(R[a]['steps'],R['NAT']['steps'])),'fact token hash differs between arms'
assert abs(R['NAT']['steps'][0]['raw_norm']-R['CHK']['steps'][0]['raw_norm'])<1e-6*R['NAT']['steps'][0]['raw_norm']+1e-9 and R['NAT']['steps'][1]['ce']!=R['CHK']['steps'][1]['ce']
print('E59 QWEN SMOKE PASSED')
PYE
for s in 161 162; do
  set +e
  $PYQ $E/e59_qwen_run.py --world-seed $s --step-norm 0.30426 --arms NAT,NATF,CHK,CHKF,FACT,ISO $COMMON --out $O/p6
  echo "RUN e59q 6pct seed $s rc=$?"
  set -e
done
for s in 161 162; do
  set +e
  $PYQ $E/e59_qwen_run.py --world-seed $s --step-norm 0.10142 --arms NAT,NATF,CHK,CHKF,FACT,ISO $COMMON --out $O/p2
  echo "RUN e59q 2pct seed $s rc=$?"
  set -e
done
