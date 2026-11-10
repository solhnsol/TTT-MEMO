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
for SEED in 71 72 73 74 75 76 77 78; do
  $PY $D/e57_oracle_run.py --seed $SEED --out $OUT/e57_k0conf16 --masks none --mask-arm ALLQ256 --arms NATS,NATBINF,NATMASK,NATMASKC,DIRmaskNORMfull,DIRfullNORMmask,NATS_clip10,NATMASK_clip10,NATMASKC_clip10 --nat-loss-every 8 --tag k0c16
done
echo "outputs: $OUT/e57_k0conf16/E55_k0c16_<arm>_s<seed>.{saved,read}.json"
