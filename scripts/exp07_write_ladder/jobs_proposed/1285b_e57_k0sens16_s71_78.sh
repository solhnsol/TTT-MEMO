#!/bin/bash
# JOB 1285b = E57 K0 SENSITIVITY, paired with job 1283 on the SAME seeds 71..78 (pre-specification: reports/E57-1285-prespec.md; exploratory secondary evidence; GPU start needs the principal investigator's own sentence in the server-queue thread). Supersedes the proposed job 1285 (seeds 61..68, 4 arms, about 4.5 h), which is NOT run. Runs ONLY the K0 = 8 arm NATMASK_K08 (mask from chunk 8) on seeds 71..78 of the synthetic 16-fact worlds (v3_16_1, same config as job 1283: --masks none --mask-arm ALLQ256 --save-apps 16,32,64,128 --nat-loss-every 8); the K0 = 0 arms NATS and NATMASK of job 1283 (tag k0c16) are REUSED only if (1) the library versions and the file hashes of the runner code equal those recorded in the 1283 versions.json (abort 12 otherwise) and (2) a rerun of NATS on seed 71 is bit identical to the 1283 NATS on seed 71 in update_norm, ce, gnorm, coef of every chunk and the final state hash (abort 13 otherwise). Output $OUT/e57_k0sens16_s71/ (tag k0s16). ESTIMATE about 1.7 h: 8 seeds x NATMASK_K08 x about 9.9 min (job 1283: 9 arms x 8 seeds in 11.85 h) + one NATS gate run about 10 min + selftest and versions about 5 min.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
O=$OUT/e57_k0sens16_s71
mkdir -p $O
$PY - $O $OUT/e57_k0conf16/versions.json <<'PYE'
import sys,json,os,subprocess,hashlib,platform
o,ref=sys.argv[1],json.load(open(sys.argv[2]))
v={'python':platform.python_version(),'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'JAX_DEFAULT_MATMUL_PRECISION':os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')}
for m in ('jax','jaxlib','numpy','scipy'):
    try:v[m]=__import__(m).__version__
    except Exception as e:v[m]=str(e)
v['driver']=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
v['git_head']=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
v['file_sha256']={f:hashlib.sha256(open(f'scripts/exp07_write_ladder/{f}','rb').read()).hexdigest()[:16] for f in ref['file_sha256'] if f!='e57_k0conf16_analyze.py'}
json.dump(v,open(f'{o}/versions.json','w'),indent=1)
bad=[k for k in ('python','XLA_FLAGS','jax','jaxlib','numpy','scipy','driver') if v[k]!=ref[k]]+[f for f in v['file_sha256'] if v['file_sha256'][f]!=ref['file_sha256'][f]]
if bad:print('REUSE REFUSED: differs from the 1283 versions.json:',bad);sys.exit(12)
print('versions equal to the 1283 versions.json (library versions, flags, driver, runner code hashes); git head',v['git_head'][:8],'(1283:',ref['git_head'][:8]+')')
PYE
# GATE: NATS on seed 71 must be bit identical to the 1283 NATS on seed 71
$PY $D/e57_oracle_run.py --seed 71 --out $O/gate --masks none --mask-arm ALLQ256 --arms NATS --save-apps 16,32,64,128 --nat-loss-every 8 --tag k0s16gate
$PY - $O $OUT/e57_k0conf16 <<'PYE'
import json,sys
o,r=sys.argv[1],sys.argv[2]
a=json.load(open(f'{o}/gate/E55_k0s16gate_NATS_s71.saved.json'));b=json.load(open(f'{r}/E55_k0c16_NATS_s71.saved.json'))
ok=len(a['log'])==len(b['log']) and all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in ('update_norm','ce','gnorm','coef')) and a.get('final_hash')==b.get('final_hash')
print('gate NATS s71 bit identical to 1283:',ok)
sys.exit(0 if ok else 13)
PYE
for SEED in 71 72 73 74 75 76 77 78; do
  $PY $D/e57_oracle_run.py --seed $SEED --out $O --masks none --mask-arm ALLQ256 --arms NATMASK_K08 --save-apps 16,32,64,128 --nat-loss-every 8 --tag k0s16
done
echo "outputs: $O/E55_k0s16_NATMASK_K08_s<seed>.{saved,read}.json (K0 = 0 arms: $OUT/e57_k0conf16/E55_k0c16_{NATS,NATMASK}_s<seed>)"
