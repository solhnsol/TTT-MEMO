#!/bin/bash
# JOB 1302o = E57 ORACLE (CACHE PATH) ON THE 1283 SEEDS (EXPLORATORY; K0 = 0; not queued until Hans' sentence; GPU one job at a time; Hans 11/16 option A, Oracle cache-based): cell v3_16_1, seeds 71..78 (the seeds of job 1283 / 1293), arms NATORACLE (OracleUV: a sentence row of fact f does not see the earlier sentences of fact f in the window, update-path mask over the CACHED window K,V, i.e. the mask of the NLL condition A of job 1280c; e57_oracle_run.py lines 109-110) and NATPLACEBO (PlaceboOracleUV: the cache placebo, condition A_P), same options as job 1283 (--masks none --save-apps 16,32,64,128 --nat-loss-every 8), tag orc16. GATE (abort 12 / 13, as job 1297): pinned runner files and library versions equal to the 1283 versions.json; NATS and NATMASK (band 256) on seed 71 rerun with this job's code path must be bit identical to the 1283 records. Output $OUT/e57_orc16/ ; analysis e57_align71_analyze.py (table of the exposure curve on seeds 71..78: NATS, NATMASK256, NATMASK1024, NATORACLE, NATPLACEBO and, when job 1302i has run, ISO-STD; summed span margin, descriptive, no pass / fail rule) -> $OUT/e57_align71_result.json. ESTIMATE ~3.0 h: gate ~22 min + 16 runs (8 seeds x 2 arms) x ~10 min (job 1283: 72 runs in 11.85 h) ; the two arms of a seed share one model load and the W0 read.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
O=$OUT/e57_orc16
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
PIN={'e57_oracle_run.py':'85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea','e57_natmask_run.py':'24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da','e57_oracle_masks.py':'a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371','e55_core.py':'f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48','ladder.py':'714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4','s3.py':'a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d'}   # FULL SHA-256 of the six runner files, pinned before any K0 = 8 data are generated (their 16-hex prefixes equal those of the 1283 versions.json)
full={f:hashlib.sha256(open(f'scripts/exp07_write_ladder/{f}','rb').read()).hexdigest() for f in PIN};v['file_sha256_full']=full;json.dump(v,open(f'{o}/versions.json','w'),indent=1)
bad=[k for k in ('python','XLA_FLAGS','jax','jaxlib','numpy','scipy','driver') if v[k]!=ref[k]]+[f for f in v['file_sha256'] if v['file_sha256'][f]!=ref['file_sha256'][f]]+[f+' (full hash)' for f in PIN if full[f]!=PIN[f]]
if bad:print('REUSE REFUSED: differs from the 1283 versions.json:',bad);sys.exit(12)
print('versions equal to the 1283 versions.json (library versions, flags, driver, runner code hashes); git head',v['git_head'][:8],'(1283:',ref['git_head'][:8]+')')
PYE
# GATES (about 20 min): NATS and NATMASK (K0 = 0) on seed 71 are rerun with the code path of THIS job and must be bit identical (exact equality, not allclose) to the 1283 NATS and NATMASK on seed 71 in update_norm, ce, gnorm, coef of every chunk, final state hash, and in the run configuration (cell, mask arm, read schedule, save apps, stream hash, chunk count, k, n, bank, K0 tag); the read item meta (kind, gold index) must also be identical. Any failure aborts (13): the K0 = 0 reuse is then refused; a rerun of the K0 = 0 arms (about 4.3 h instead of 1.7 h) needs a NEW principal-investigator sentence.
$PY $D/e57_oracle_run.py --seed 71 --out $O/gate --masks none --mask-arm ALLQ256 --arms NATS,NATMASK --save-apps 16,32,64,128 --nat-loss-every 8 --tag k0s16gate
$PY - $O $OUT/e57_k0conf16 <<'PYE'
import json,sys,hashlib
o,r=sys.argv[1],sys.argv[2];ok=True
EXCL={'log','reads','arm','tag','canary_w0','diag','inline','final_hash','clip_bind_frac','cum_update_norm','wlog','read_hash','summary','sec_total'}
cfgh=lambda d:hashlib.sha256(json.dumps({k:v for k,v in d.items() if k not in EXCL},sort_keys=True).encode()).hexdigest()
t0=json.load(open(f'{r}/E55_k0c16_NATS_s71.saved.json'));t1=dict(t0);t1['n_chunks']=t0['n_chunks']+1
assert cfgh(t0)!=cfgh(t1),'negative control: a changed non-excluded configuration key must change the configuration hash'
for arm in ('NATS','NATMASK'):
    a=json.load(open(f'{o}/gate/E55_k0s16gate_{arm}_s71.saved.json'));b=json.load(open(f'{r}/E55_k0c16_{arm}_s71.saved.json'))
    logs=len(a['log'])==len(b['log']) and all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in ('update_norm','ce','gnorm','coef')) and a.get('final_hash')==b.get('final_hash')
    cfg=cfgh(a)==cfgh(b)
    ra=json.load(open(f'{o}/gate/E55_k0s16gate_{arm}_s71.read.json'));rb=json.load(open(f'{r}/E55_k0c16_{arm}_s71.read.json'))
    meta=({k:v for k,v in ra['meta'].items() if k!='arm'}=={k:v for k,v in rb['meta'].items() if k!='arm'})
    reads=(ra['read']==rb['read'] and ra['W0']==rb['W0'])      # exact equality of every read value (exposures 16/32/64/128, out and in_g0, per item candidate scores) and of the W0 reads
    print(f'gate {arm} s71 vs 1283: logs+final hash {logs}, configuration hash {cfg}, read meta {meta}, read values {reads}');ok=ok and logs and cfg and meta and reads
sys.exit(0 if ok else 13)
PYE
for SEED in 71 72 73 74 75 76 77 78; do
  $PY $D/e57_oracle_run.py --seed $SEED --out $O --masks none --mask-arm ALLQ256 --arms NATORACLE,NATPLACEBO --save-apps 16,32,64,128 --nat-loss-every 8 --tag orc16
done
$PY $D/e57_align71_analyze.py --dir $OUT --seeds 71,72,73,74,75,76,77,78 --json $OUT/e57_align71_result.json > /dev/null
echo "outputs: $OUT/e57_orc16/E55_orc16_<arm>_s<seed>.{saved,read}.json, $OUT/e57_align71_result.json"

