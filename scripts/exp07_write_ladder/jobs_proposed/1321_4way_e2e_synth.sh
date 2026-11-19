#!/bin/bash
# JOB 1321 = 4-way arms on E2E synthetic worlds + REVERSE-question readout (--reverse: place -> person, 16 name candidates, out + in-window, secondary readout; Hans 11/19) (Hans 11/19: NAT, all-layer band, CHK, ISO; world seeds 231..238; EXPLORATORY, no pass/fail gating; code e59_srv_run.py): NAT (original rule: SGD lr 1 clip 1, its applied per-chunk norms are the reference norms), MASK1024 (band-1024 mask on every row in ALL 32 layers, original rule, K0 = 0 = the NATMASK1024 arm of jobs 1297c / 1308 / 1313), CHK (chunk alone, direction rescaled to the NAT norms), ISO (16 isolated sentences / round, one step, direction rescaled to the NAT norms). NOT queued until Hans' sentence; est. ~5 h (smoke ~10 min; per seed ~35 min). Output experiments/exp07_write_ladder/e59_4way. Order per seed: NAT first (reference norms).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 19"; exit 19; }
D=scripts/exp07_write_ladder;O=$OUT/e59_4way
mkdir -p $O/smoke $O/repro;df -h /home/solhnsol | tail -1
sha256sum $D/e59_srv_core.py $D/e59_srv_run.py $D/e55_core.py $D/e55_iso.py $D/e55_run.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_srv_run.py --selftest | tail -2
# smoke: seed 231, 16 rounds, the four arms
$PY $D/e59_srv_run.py --seed 231 --out $O/smoke --arms NAT,MASK1024,CHK,ISO --tag smoke --smoke 16 --rerun --reverse
$PY - $O/smoke <<'PYE'
import json,sys,math
d=sys.argv[1];A=('NAT','MASK1024','CHK','ISO');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a;print('smoke',a,'ok: applied norm step0',round(s[0]['applied_norm'],4),'ce',round(s[1]['ce'] if 'ce' in s[1] else 0,3))
ref=[x['applied_norm'] for x in R['NAT']['steps']]
for a in ('CHK','ISO'):assert all(abs(x['applied_norm']-r)<=1e-4*r+1e-9 for x,r in zip(R[a]['steps'],ref)),a
assert all(x['fact_hash']==y['fact_hash'] for a in A[1:] for x,y in zip(R[a]['steps'],R['NAT']['steps']))
assert R['NAT']['log'][1]['ce']!=R['MASK1024']['log'][1]['ce'] and abs(R['NAT']['log'][0]['ce']-R['MASK1024']['log'][0]['ce'])<1e-2
for a in A:
    rv=json.load(open(f'{d}/E59_smoke_{a}_s231.rev.read.json'));assert rv['meta']['gi']==[0]*16 and len(rv['read']['16']['out'])==16 and len(rv['read']['16']['out'][0])==16 and len(rv['W0']['out'])==16,a
print('4WAY E2E SMOKE PASSED (forward + reverse readout files present)')
PYE
rm -rf $O/smoke/states
# reproduction check (information, not a gate): BAND1024 seed 71 vs the stored band-1024 run of job 1293 (e57_band_dist2/w1024)
$PY $D/e59_srv_run.py --seed 71 --out $O/repro --arms MASK1024 --tag repro
$PY - <<'PYE' || true
import json,glob,numpy as np
def mar(S,meta):
    x=np.asarray(S,float);g=np.asarray(meta['gi']);f=np.asarray(meta['kind'])=='fact';x=x[f];g=g[f];gold=x[np.arange(len(g)),g];w=x.copy();w[np.arange(len(g)),g]=-np.inf;return float((gold-w.max(1)).mean())
a=json.load(open('experiments/exp07_write_ladder/e59_4way/repro/E59_repro_MASK1024_s71.read.json'));ms=glob.glob('experiments/exp07_write_ladder/e57_band_dist2/w1024/*NATMASK_s71.read.json')
print('REPRO new MASK1024 s71 margin128 %.6f'%mar(a['read']['128']['out'],a['meta']));[print('REPRO stored',f.split('/')[-1],'%.6f'%mar(json.load(open(f))['read']['128']['out'],json.load(open(f))['meta'])) for f in ms]
PYE
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms NAT,MASK1024,CHK,ISO --tag a4 --reverse
  echo "RUN 4way e2e seed $s rc=$?"
  rm -rf $O/states/E59_a4_*_s$s      # the post-write weight files (4 x 1.4 GB per arm) are not kept: the disk of /home has ~57 GB free
  set -e
done
