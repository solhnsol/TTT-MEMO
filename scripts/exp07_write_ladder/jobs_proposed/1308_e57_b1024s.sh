#!/bin/bash
# JOB 1308 (v3, Hans 11/17: four weight arms) = E57 BAND 1024 + LOSS WEIGHT ON g ROWS, CLIPPED-SGD ENGINE, K0 = 0 (EXPLORATORY; not queued until Hans' sentence; GPU one job at a time; Hans 11/17 'sgd랑 k0=0': same engine / mask / reads as the band-1024 arm NATMASK1024 of jobs 1283 / 1293; values of NATS and NATMASK1024 are REUSED from the stored runs of seeds 71..78 when the bit-identity gate passes): cell v3_16_1, seeds 71..78. Per seed: (1) W0 window pass e57_syn_w0pass.py --bands 256,1024 (nll_win, nll_band256, nll_band1024; GPU forward only); (2) CPU deciding rows e57_b1024s_masks.py (top 15% and top 30% rows by g1024 = nll_band1024 - nll_win among the eligible rows; for each the same number of random eligible rows); (3) four write arms through e57_b1024s_run.py (wrapper of the unchanged e57_oracle_run.py): NATB1024G15W8 / NATB1024G30W8 (band 1024 on every row from chunk 0, loss weight 8 on the top-15% / top-30% rows by g1024 from chunk 8) and NATB1024R15W8 / NATB1024R30W8 (weight 8 on the same number of random eligible rows; the auxiliary norm-matched arms are not part of this job). The weights enlarge the gradient: gnorm, coef, update_norm, clip activity are logged per chunk (summary.clip_frac_coef_lt1, mean_update_norm) and compared in the analysis. GATES on seed 71 (exit 13 on failure of G2): G1 the band-1024 arm (NATMASK, --mask-arm ALLQ1024) rerun with the code path of this job must be bit identical to the stored e57_band_dist2/w1024 run (log update_norm / ce / gnorm / coef of every chunk, final hash, configuration hash, read values); if it is NOT identical the job does not abort: BAND_STORED=0 and the band-1024 arm is rerun for every seed in e57_b1024s/ (the analysis then uses it, flagged in the result); G2 the weighted code path with weights all 1 (arm NATB1024W1T) must reproduce the band-1024 arm within float rounding: max relative difference of update_norm and ce over the 128 chunks <= 1e-2 (informational numbers are printed; a larger difference = implementation error, abort). CHECKS: pinned files and library versions equal to the 1283 versions.json (exit 12), CPU selftests. Output $OUT/e57_b1024s_s<seed>/ (W0 pass and masks), $OUT/e57_b1024s/ (arms), analysis e57_b1024s_analyze.py -> $OUT/e57_b1024s_result.json. ESTIMATE (jobs 760 / 1302o timings): per seed W0 pass ~6 min + masks ~1 min + 4 arms x ~10 min = ~47 min; 8 seeds ~6.3 h + gates on seed 71 ~20 min + CPU checks ~5 min = ~6.7 h; if G1 is not identical +8 x ~10 min.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
sha256sum -c <<GATEPIN
fd116aa817038697bc28bb7e4eecc7ed6b15a31f5978a2ce3b0676e2bffff7a9  scripts/exp07_write_ladder/e57_b1024s_run.py
f378843442c5ad995da9f58bd477e940e7856b985f9a1daa02e4be4865d07f81  scripts/exp07_write_ladder/e57_b1024s_masks.py
22155054f77844b167e49090dd2386e32cae2be252cd04ef9e7cdc4f107e6061  scripts/exp07_write_ladder/e57_b1024s_analyze.py
84132298e9bf7f986e9ba48fbce44d76f82131282a981cf1bc5fba3d335f7986  scripts/exp07_write_ladder/e57_allprev_analyze.py
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
c48a57e093ed8c4d652e74e4009f43a8fc5ac0e6a02e75c64feafd830a5e651e  scripts/exp07_write_ladder/e57_qband_run.py
091506dbe7b8a565ccd53594f2a0f9651d80ae07d86448047766cb811cfd5847  scripts/exp07_write_ladder/e57_closure_masks.py
e3c7ebfaec6292b5a95b6709369911f138f66f8f33ea59507696fdca232886b4  scripts/exp07_write_ladder/e57_syn_w0pass.py
149e3ece41d9e00d4b84c2c0bd5ec347009a774cdd386eb15cb6557aaf1ee358  scripts/exp07_write_ladder/e57_syn_mmasks.py
5ce5bd6bb36686b6c7d56a588421340369b0c0c013d1c77290cfbcd9d342a515  scripts/exp07_write_ladder/e57_syn_pass.py
71f0fc416f7446c5adcf371ba97244da770c0f4427a3a1f506db6f6ee60b0b2d  scripts/exp07_write_ladder/e55_run.py
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
GATEPIN
$PY $D/e57_b1024s_masks.py --selftest | tail -1; $PY $D/e57_b1024s_run.py --selftest | tail -1; $PY $D/e57_b1024s_analyze.py --selftest | tail -1; $PY $D/e57_syn_w0pass.py --selftest | tail -1
O=$OUT/e57_b1024s
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
mkdir -p $O/gate
ARMS=NATB1024G15W8,NATB1024R15W8,NATB1024G30W8,NATB1024R30W8
STORED=$OUT/e57_band_dist2/w1024
# seed 71 W0 pass and masks first (needed by gate G2), then the gates
P=$OUT/e57_b1024s_s71; mkdir -p $P
[ -f $P/syn_s71_nll.npz ] || $PY $D/e57_syn_w0pass.py --seed 71 --cell v3_16_1 --out $P --bands 256,1024
[ -f $P/b1024s_s71_masks.npz ] || $PY $D/e57_b1024s_masks.py --seed 71 --cell v3_16_1 --dir $P --out $P/b1024s_s71_masks.npz
$PY $D/e57_b1024s_run.py --seed 71 --cell v3_16_1 --out $O/gate --masks none --mask-arm ALLQ1024 --arms NATMASK --save-apps 16,32,64,128 --nat-loss-every 8 --tag b1024sgate
$PY $D/e57_b1024s_run.py --seed 71 --cell v3_16_1 --out $O/gate --masks $P/b1024s_s71_masks.npz --mask-arm ALLQ1024 --arms NATB1024W1T --save-apps 16,32,64,128 --nat-loss-every 8 --tag b1024sgate
set +e
$PY - $O $STORED $OUT/e57_b1024s_gate_verdict.json <<'PYE'
import json,sys,hashlib
o,r,vj=sys.argv[1],sys.argv[2],sys.argv[3]
EXCL={'log','reads','arm','tag','canary_w0','diag','inline','final_hash','clip_bind_frac','cum_update_norm','wlog','read_hash','summary','sec_total'}
cfgh=lambda d:hashlib.sha256(json.dumps({k:v for k,v in d.items() if k not in EXCL},sort_keys=True).encode()).hexdigest()
a=json.load(open(f'{o}/gate/E55_b1024sgate_NATMASK_s71.saved.json'));b=json.load(open(f'{r}/E55_bd1024_NATMASK_s71.saved.json'))
logs=len(a['log'])==len(b['log']) and all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in ('update_norm','ce','gnorm','coef')) and a.get('final_hash')==b.get('final_hash')
cfg=cfgh(a)==cfgh(b)
ra=json.load(open(f'{o}/gate/E55_b1024sgate_NATMASK_s71.read.json'));rb=json.load(open(f'{r}/E55_bd1024_NATMASK_s71.read.json'))
meta=({k:v for k,v in ra['meta'].items() if k!='arm'}=={k:v for k,v in rb['meta'].items() if k!='arm'});reads=(ra['read']==rb['read'] and ra['W0']==rb['W0'])
g1=bool(logs and cfg and meta and reads);print(f'G1 band-1024 rerun vs stored s71: logs+final hash {logs}, configuration hash {cfg}, read meta {meta}, read values {reads} -> identical {g1}')
w=json.load(open(f'{o}/gate/E55_b1024sgate_NATB1024W1T_s71.saved.json'));ref=a if True else b
rel=lambda x,y:abs(x-y)/max(abs(y),1e-12)
du=max(rel(x['update_norm'],y['update_norm']) for x,y in zip(w['log'],ref['log']));dc=max(rel(x['ce'],y['ce']) for x,y in zip(w['log'],ref['log']));dg=max(rel(x['gnorm'],y['gnorm']) for x,y in zip(w['log'],ref['log']))
print(f'G2 weighted path with weights 1 vs band-1024 rerun s71: max relative difference update_norm {du:.3e}, ce {dc:.3e}, gnorm {dg:.3e} (limit 1e-2)')
ok2=bool(du<=1e-2 and dc<=1e-2 and dg<=1e-2)
json.dump(dict(G1_bit_identical=g1,G2_ok=ok2,G2_max_rel_update_norm=du,G2_max_rel_ce=dc,G2_max_rel_gnorm=dg),open(vj,'w'),indent=1)
open(f'{o}/BAND_STORED','w').write('1' if g1 else '0');sys.exit(0 if ok2 else 13)
PYE
GRC=$?; set -e
[ $GRC -eq 0 ] || { echo "gate G2 failed (exit $GRC): the weighted code path does not reproduce the band-1024 arm; abort"; exit 13; }
BAND_STORED=$(cat $O/BAND_STORED); echo "BAND_STORED=$BAND_STORED"
for SEED in 71 72 73 74 75 76 77 78; do
  P=$OUT/e57_b1024s_s$SEED; mkdir -p $P
  [ -f $P/syn_s${SEED}_nll.npz ] || $PY $D/e57_syn_w0pass.py --seed $SEED --cell v3_16_1 --out $P --bands 256,1024
  [ -f $P/b1024s_s${SEED}_masks.npz ] || $PY $D/e57_b1024s_masks.py --seed $SEED --cell v3_16_1 --dir $P --out $P/b1024s_s${SEED}_masks.npz
  REFDIR=$STORED
  if [ "$BAND_STORED" = "0" ]; then
    $PY $D/e57_b1024s_run.py --seed $SEED --cell v3_16_1 --out $O --masks none --mask-arm ALLQ1024 --arms NATMASK --save-apps 16,32,64,128 --nat-loss-every 8 --tag b1024s
    REFDIR=$O/ref; mkdir -p $REFDIR; ln -sf "$(realpath $O/E55_b1024s_NATMASK_s$SEED.saved.json)" $REFDIR/E55_bd1024_NATMASK_s$SEED.saved.json
  fi
  $PY $D/e57_b1024s_run.py --seed $SEED --cell v3_16_1 --out $O --masks $P/b1024s_s${SEED}_masks.npz --mask-arm ALLQ1024 --arms $ARMS --ref-dir $REFDIR --save-apps 16,32,64,128 --nat-loss-every 8 --tag b1024s
done
SRC=stored; [ "$BAND_STORED" = "0" ] && SRC=gate
$PY $D/e57_b1024s_analyze.py --dir $OUT --band-src $SRC --json $OUT/e57_b1024s_result.json > /dev/null
echo "outputs: $O/E55_b1024s_<arm>_s<seed>.{saved,read}.json, $OUT/e57_b1024s_result.json, $OUT/e57_b1024s_gate_verdict.json"
