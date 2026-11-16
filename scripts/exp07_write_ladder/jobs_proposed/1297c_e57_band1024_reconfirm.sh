#!/bin/bash
# JOB 1297c = E57 BAND-1024 RE-CONFIRMATION WITH THE CORRECTED GATE (K0 = 0; GPU one job at a time; Hans 11/16 '정정 규칙으로 재확증'; registration = docstring of e57_band1024_confirm_analyze_c.py, its sha256 is frozen and a signed tag freeze-1297c-v1 is pushed to the private registry BEFORE the first seed is run). Same design as job 1297 (synthetic 16-fact worlds v3_16_1, 128 exposures, arms NATS, NATMASK ALLQ256, NATMASK ALLQ1024, reads at exposures 16/32/64/128, gates as 1283: abort 12 / 13) with NEW seeds 151..158; the ONLY change to the registered analysis is the activity gate of NATMASK1024 (chunk 1 instead of chunk 0; reason in the docstring). P3 (DCLM): no unused DCLM validation documents remain on this server (all 200 documents of val_c0 were used by 1283d / 1294 / 1297), so P3 is evaluated on the DCLM preservation result of job 1297 (9 streams 16..24; the runs are deterministic per stream, a repetition would reproduce it bit for bit): NOT an independent confirmation of P3 (disclosed). ESTIMATE: gates ~20 min + 24 runs x ~9.9 min = ~4.4 h in total. Analysis: e57_band1024_confirm_analyze_c.py --dir $O --seeds 151,...,158 --dclm-json $OUT/e57_band1024_conf_preservation_1024.json. Run through the job runner (XLA allocator and nvrtc LD_LIBRARY_PATH are set there).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
O=$OUT/e57_band1024_confc
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
for SEED in 151 152 153 154 155 156 157 158; do
  $PY $D/e57_oracle_run.py --seed $SEED --out $O --masks none --mask-arm ALLQ256 --arms NATS,NATMASK --save-apps 16,32,64,128 --nat-loss-every 8 --tag bc
  $PY $D/e57_oracle_run.py --seed $SEED --out $O --masks none --mask-arm ALLQ1024 --arms NATMASK --save-apps 16,32,64,128 --nat-loss-every 8 --tag bc1024
done
$PY $D/e57_band1024_confirm_analyze_c.py --dir $O --seeds 151,152,153,154,155,156,157,158 --dclm-json $OUT/e57_band1024_conf_preservation_1024.json --json $OUT/e57_band1024_confc_result.json > /dev/null
echo "outputs: $OUT/e57_band1024_confc_result.json"
