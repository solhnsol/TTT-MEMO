#!/bin/bash
# JOB 1283 = E57 K0=0 CONFIRMATION, synthetic 16-fact worlds (v3_16_1), NEW seeds 71..78 (Hans 11/10 '추천대로'; GPU start needs Hans' own sentence; Council check of the registration first): original-NAT arms NATS, NATBINF (gate: == NATS bit for bit on every seed), NATMASK, NATMASKC, DIRmaskNORMfull, DIRfullNORMmask, plus the UNCONDITIONAL clip-sensitivity arms NATS_clip10, NATMASK_clip10, NATMASKC_clip10 (sensitivity only), mask from chunk 0 (K0 = 0, tagged in every saved.json); reads at exposures 16/32/64/128, token diag every 8 chunks (paper cost, not the per-chunk mechanism logging). Output $OUT/e57_k0conf16/ tag k0c16. ESTIMATE 9 arms x 8 seeds x 8.5 min ~ 10.2 h. versions.json (library versions, flags, driver, git head, file hashes) is written first and frozen.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
mkdir -p $OUT/e57_k0conf16
$PY - $OUT/e57_k0conf16 <<'PYE'
import sys,json,os,subprocess,hashlib,platform
o=sys.argv[1];v={'python':platform.python_version(),'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'JAX_DEFAULT_MATMUL_PRECISION':os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')}
for m in ('jax','jaxlib','numpy','scipy'):
    try:v[m]=__import__(m).__version__
    except Exception as e:v[m]=str(e)
v['driver']=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
v['git_head']=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
v['file_sha256']={f:hashlib.sha256(open(f'scripts/exp07_write_ladder/{f}','rb').read()).hexdigest()[:16] for f in ('e57_oracle_run.py','e57_natmask_run.py','e57_oracle_masks.py','e57_k0conf16_analyze.py','e55_core.py','ladder.py','s3.py')}
json.dump(v,open(f'{o}/versions.json','w'),indent=1);print('versions.json written',v['jax'],v['git_head'][:8])
PYE
# 10-CHUNK SMOKE (Hans/core-claims 11/11, before any real seed): all 9 arms on seed 71 for 10 chunks; aborts the job on a non-finite value, on NATBINF != NATS, or if NATMASK is not active from chunk 0
$PY $D/e57_oracle_run.py --seed 71 --out $OUT/e57_k0conf16_smoke --masks none --mask-arm ALLQ256 --arms NATS,NATBINF,NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask,NATS_clip10,NATMASK_clip10,NATMASKC_clip10 --nat-loss-every 8 --smoke 10 --tag k0c16sm
$PY - $OUT/e57_k0conf16_smoke <<'PYE'
import json,sys,math
o=sys.argv[1];A=['NATS','NATBINF','NATMASK','NATMASKC','DIRmaskNORMfull','DIRfullNORMmask','NATS_clip10','NATMASK_clip10','NATMASKC_clip10']
R={a:json.load(open(f'{o}/E55_k0c16sm_{a}_s71.saved.json')) for a in A}
for a,r in R.items():
    assert all(math.isfinite(e[k]) for e in r['log'] for k in ('ce','gnorm','update_norm','coef')),('non-finite',a)
n,b=R['NATS']['log'],R['NATBINF']['log']
for k in range(9):
    for f in ('update_norm','ce','gnorm','coef'):assert n[k][f]==b[k][f],('NATBINF != NATS',k,f)
assert R['NATMASK']['K0']==0 and R['NATMASK']['log'][0]['ce']!=n[0]['ce'],'NATMASK not active from chunk 0 (K0 = 0)'
print('smoke ok: finite, NATBINF == NATS (chunks 0-8), NATMASK active from chunk 0, K0 tags',{a:r.get('K0') for a,r in R.items()})
PYE
for SEED in 71 72 73 74 75 76 77 78; do
  $PY $D/e57_oracle_run.py --seed $SEED --out $OUT/e57_k0conf16 --masks none --mask-arm ALLQ256 --arms NATS,NATBINF,NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask,NATS_clip10,NATMASK_clip10,NATMASKC_clip10 --nat-loss-every 8 --tag k0c16
done
echo "outputs: $OUT/e57_k0conf16/E55_k0c16_<arm>_s<seed>.{saved,read}.json"
