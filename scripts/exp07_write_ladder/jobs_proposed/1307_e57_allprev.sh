#!/bin/bash
# JOB 1307 = E57 NATALLPREV (hide ALL earlier fact sentences; EXPLORATORY; K0 = 0; not queued until Hans' sentence; GPU one job at a time; Hans 11/17 request): cell v3_16_1, seeds 71..78 (the seeds of jobs 1283 / 1293 / 1302o / 1302i), arm NATALLPREV = e57_oracle_run.py with the update-path mask e57_allprev_masks.AllPrevFactUV (a sentence row does not see ANY earlier fact sentence of the window, own fact or other fact; filler rows see everything; the hidden key set contains the OracleUV set), same engine / reads / options as job 1302o (SGDSEL clipped SGD, --masks none --save-apps 16,32,64,128 --nat-loss-every 8), tag allprev, run through the wrapper e57_allprev_run.py (the registered e57_oracle_run.py is imported, not edited). CHECKS: pinned files and library versions equal to the 1283 versions.json (exit 12 on a difference); CPU selftests of the mask, wrapper and analyzer; CPU real-world mask check on seed 71 (every chunk of the real stream equals a brute-force union of the sentence spans, superset of the oracle set; exit on failure). The bit-identity rerun of NATS / NATMASK of job 1302o is NOT repeated: the code path e57_oracle_run.py / e57_natmask_run.py / e57_oracle_masks.py is pinned with the same sha256 that passed that gate in job 1302o. Output $OUT/e57_allprev/ ; analysis e57_allprev_analyze.py (exposure curve out/in, top-1 share, paired differences vs NATS / NATORACLE / NATPLACEBO, gap recovery vs ISO-STD at exposure 128; descriptive, no pass / fail rule) -> $OUT/e57_allprev_result.json. ESTIMATE ~1.4 h: 8 seeds x ~10 min (jobs 1302o: NATORACLE 514-645 s per seed, model load and W0 read included) plus ~5 min CPU checks.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
sha256sum -c <<GATEPIN
82173f7f7795df77ed7d872bca98b42fe5d0ac046df48612933fdce2a9bfca14  scripts/exp07_write_ladder/e57_allprev_run.py
93a75e759ee9c37854ff74c49f52106f3398024d410399a0128b6a817f20db6b  scripts/exp07_write_ladder/e57_allprev_masks.py
84132298e9bf7f986e9ba48fbce44d76f82131282a981cf1bc5fba3d335f7986  scripts/exp07_write_ladder/e57_allprev_analyze.py
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
c48a57e093ed8c4d652e74e4009f43a8fc5ac0e6a02e75c64feafd830a5e651e  scripts/exp07_write_ladder/e57_qband_run.py
091506dbe7b8a565ccd53594f2a0f9651d80ae07d86448047766cb811cfd5847  scripts/exp07_write_ladder/e57_closure_masks.py
GATEPIN
$PY $D/e57_allprev_masks.py --selftest | tail -1; $PY $D/e57_allprev_run.py --selftest | tail -1; $PY $D/e57_allprev_analyze.py --selftest | tail -1
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY $D/e57_allprev_masks.py --real-check --seed 71 > $OUT/e57_allprev_real_check_s71.json
O=$OUT/e57_allprev
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
for SEED in 71 72 73 74 75 76 77 78; do
  $PY $D/e57_allprev_run.py --seed $SEED --out $O --masks none --mask-arm ALLQ256 --arms NATALLPREV --save-apps 16,32,64,128 --nat-loss-every 8 --tag allprev
done
$PY $D/e57_allprev_analyze.py --dir $OUT --json $OUT/e57_allprev_result.json > /dev/null
echo "outputs: $O/E55_allprev_NATALLPREV_s<seed>.{saved,read}.json, $OUT/e57_allprev_result.json, $OUT/e57_allprev_real_check_s71.json"
