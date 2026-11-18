#!/bin/bash
# JOB 1314 = E59 write-signal decomposition, E2E 3B v3_16_1 seeds 71 / 72 (EXPLORATORY; code e59_srv_core.py / e59_srv_run.py, new files; NOT queued until Hans' sentence; GPU one job at a time; est. 2.5-3 h): smoke (seed 71, 16 rounds, all arms) + self-checks, then per seed NAT (reference norms) -> NATREF -> NATF -> CHK -> CHKF -> FACT -> ISO -> ISOORIG, then the noise-floor rerun of NAT (seed 71, tag e59rerun). If the smoke or its checks fail nothing else is started.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 12"; exit 12; }
D=scripts/exp07_write_ladder;O=$OUT/e59
mkdir -p $O/smoke
sha256sum $D/e59_srv_core.py $D/e59_srv_run.py $D/e55_core.py $D/e55_iso.py $D/e55_run.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/ladder.py $D/s3.py
$PY $D/e59_srv_run.py --selftest | tail -2
$PY $D/e59_srv_run.py --seed 71 --out $O/smoke --arms NAT,NATREF,NATF,CHK,CHKF,FACT,ISO,ISOORIG --tag e59smoke --smoke 16 --rerun
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
d=sys.argv[1];A=('NAT','NATREF','NATF','CHK','CHKF','FACT','ISO','ISOORIG');R={a:json.load(open(f'{d}/E59_e59smoke_{a}_s71.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a;print('smoke',a,'ok: applied norm step0',round(s[0]['applied_norm'],4),'ratio',s[0]['ratio'],'fact_ce',round(s[0]['fact_ce'],3))
ref=[x['applied_norm'] for x in R['NAT']['steps']]
for a in ('NATREF','NATF','CHK','CHKF','FACT','ISO'):
    for x,r in zip(R[a]['steps'],ref):assert abs(x['applied_norm']-r)<=1e-6*r+1e-9,(a,x['applied_norm'],r)
assert all(abs(x['ratio']-1)<1e-4 for x in R['NATREF']['steps']),'NATREF ratio must be 1 (reproduces NAT)'
assert all(x['fact_hash']==y['fact_hash'] for a in A[1:] for x,y in zip(R[a]['steps'],R['NAT']['steps'])),'fact token hash differs between arms'
assert R['NAT']['steps'][0]['raw_norm']==R['CHK']['steps'][0]['raw_norm'],'r=1: NAT and CHK gradients must be identical (no window yet)'
assert R['NAT']['steps'][1]['raw_norm']!=R['CHK']['steps'][1]['raw_norm'],'CHK must differ from NAT once a window exists'
assert abs(R['NATREF']['steps'][5]['net_displacement']-R['NAT']['steps'][5]['net_displacement'])<1e-4*R['NAT']['steps'][5]['net_displacement']+1e-9,'NATREF does not reproduce NAT'
print('E59 SMOKE PASSED')
PYE
for s in 71 72; do
  $PY $D/e59_srv_run.py --seed $s --out $O --arms NAT,NATREF,NATF,CHK,CHKF,FACT,ISO,ISOORIG --tag e59
  echo "RUN e59 seed $s rc=$?"
done
$PY $D/e59_srv_run.py --seed 71 --out $O/rerun --arms NAT --tag e59 --rerun
echo "RUN e59 rerun rc=$?"
