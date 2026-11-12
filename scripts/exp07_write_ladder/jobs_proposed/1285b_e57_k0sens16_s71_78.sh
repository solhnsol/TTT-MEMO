#!/bin/bash
# JOB 1285b = E57 K0 SENSITIVITY, paired with job 1283 on the SAME seeds 71..78 (pre-specification: reports/E57-1285-prespec.md; exploratory secondary evidence; GPU start needs the principal investigator's own sentence in the server-queue thread). Supersedes the proposed job 1285 (seeds 61..68, 4 arms, about 4.5 h), which is NOT run. Runs ONLY the K0 = 8 arm NATMASK_K08 (mask from chunk 8) on seeds 71..78 of the synthetic 16-fact worlds (v3_16_1, same config as job 1283: --masks none --mask-arm ALLQ256 --save-apps 16,32,64,128 --nat-loss-every 8); the K0 = 0 arms NATS and NATMASK of job 1283 (tag k0c16) are REUSED only if (1) the library versions and the file hashes of the runner code equal those recorded in the 1283 versions.json (abort 12 otherwise) and (2) reruns of NATS and NATMASK (K0 = 0) on seed 71 with this job's code path are bit identical to the 1283 arms of seed 71 (logs, final hash, run configuration, read item meta; abort 13 otherwise). Output $OUT/e57_k0sens16_s71/ (tag k0s16). ESTIMATE about 1.7 h (about 104 min): 8 seeds x NATMASK_K08 x about 9.9 min (job 1283: 9 arms x 8 seeds in 11.85 h) + two gate runs (NATS and NATMASK K0 = 0 on seed 71) about 20 min + selftest, versions and per-seed assertions about 5 min.
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
PIN={'e57_oracle_run.py':'85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea','e57_natmask_run.py':'24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da','e57_oracle_masks.py':'a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371','e55_core.py':'f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48','ladder.py':'714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4','s3.py':'a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d'}   # FULL SHA-256 of the six runner files, pinned before any K0 = 8 data are generated (their 16-hex prefixes equal those of the 1283 versions.json)
full={f:hashlib.sha256(open(f'scripts/exp07_write_ladder/{f}','rb').read()).hexdigest() for f in PIN};v['file_sha256_full']=full;json.dump(v,open(f'{o}/versions.json','w'),indent=1)
bad=[k for k in ('python','XLA_FLAGS','jax','jaxlib','numpy','scipy','driver') if v[k]!=ref[k]]+[f for f in v['file_sha256'] if v['file_sha256'][f]!=ref['file_sha256'][f]]+[f+' (full hash)' for f in PIN if full[f]!=PIN[f]]
if bad:print('REUSE REFUSED: differs from the 1283 versions.json:',bad);sys.exit(12)
print('versions equal to the 1283 versions.json (library versions, flags, driver, runner code hashes); git head',v['git_head'][:8],'(1283:',ref['git_head'][:8]+')')
PYE
# CPU PRE-CHECK of the read items and candidate token spans (all seeds 71..78, 16 candidates of every item with token ids, spans, order and gold index): the stored manifest file must have the pinned SHA-256 and every regenerated per-seed manifest hash must equal the stored one (abort 14)
echo "d21f596255c4abf3f547d83845ec7fea5c215ac3efcdbe3c2b874dc0949580b4  $O/token_manifest.json" | sha256sum -c - || exit 14
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $D/e57_1285_token_manifest.py --check $O/token_manifest.json || exit 14
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
  $PY $D/e57_oracle_run.py --seed $SEED --out $O --masks none --mask-arm ALLQ256 --arms NATMASK_K08 --save-apps 16,32,64,128 --nat-loss-every 8 --tag k0s16
  # per seed assertions: world / stream realization, the complete read item meta (items, facts, groups, gold index, all 16 candidate fact ids and the spans' item structure), the W0 reads, and the in_g0 item count n_in are the same as in the 1283 run of this seed
  $PY - $O $OUT/e57_k0conf16 $SEED <<'PYE'
import json,sys
o,r,s=sys.argv[1],sys.argv[2],sys.argv[3]
a=json.load(open(f'{o}/E55_k0s16_NATMASK_K08_s{s}.read.json'));b=json.load(open(f'{r}/E55_k0c16_NATMASK_s{s}.read.json'))
ma={k:v for k,v in a['meta'].items() if k!='arm'};mb={k:v for k,v in b['meta'].items() if k!='arm'}
assert ma==mb and a['W0']==b['W0'] and a['place_len']==b['place_len'],('read item meta, W0 reads or place length differ between K0 = 8 and K0 = 0',s)
sa=json.load(open(f'{o}/E55_k0s16_NATMASK_K08_s{s}.saved.json'));sb=json.load(open(f'{r}/E55_k0c16_NATMASK_s{s}.saved.json'))
import hashlib
EXCL={'log','reads','arm','tag','canary_w0','diag','inline','final_hash','clip_bind_frac','cum_update_norm','wlog','read_hash','summary','sec_total'}
cfgh=lambda d:hashlib.sha256(json.dumps({k:v for k,v in d.items() if k not in EXCL},sort_keys=True).encode()).hexdigest()
assert sa['K0']==8 and sb['K0']==0 and sa['arm']=='NATMASK_K08' and sb['arm']=='NATMASK'
sa0=dict(sa);sa0['K0']=0
assert cfgh(sa0)==cfgh(sb),('configuration differs between K0 = 8 and K0 = 0 beyond the arm name and the K0 value',s)
rn=json.load(open(f'{r}/E55_k0c16_NATS_s{s}.read.json'));assert a['W0']==rn['W0'],('W0 reads of the K0 = 8 run differ from the 1283 NATS record',s)
ga=len(a['read']['128']['in_g0']);gb=len(b['read']['128']['in_g0']);assert ga==gb,('in_g0 item count differs',s,ga,gb)
nin=sum(1 for k in a['meta']['kind'][:ga] if k=='fact');print('seed',s,'in_g0 items',ga,'n_in (fact items)',nin,'(same items, meta, W0 and stream as 1283)')
PYE
done
echo "outputs: $O/E55_k0s16_NATMASK_K08_s<seed>.{saved,read}.json (K0 = 0 arms: $OUT/e57_k0conf16/E55_k0c16_{NATS,NATMASK}_s<seed>)"
