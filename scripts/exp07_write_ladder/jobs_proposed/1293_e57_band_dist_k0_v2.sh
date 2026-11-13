#!/bin/bash
# JOB 1293 = E57 BAND-DISTANCE EXPERIMENT (RECONFIGURED), CORE (exploratory secondary evidence; pre-specification reports/E57-distB-prespec.md; HELD until the principal investigator's own sentence): NATMASK with the band mask of W tokens (--mask-arm ALLQ<W>) at K0 = 0 on the 16-fact worlds (cell v3_16_1, same options as job 1283: --masks none --save-apps 16,32,64,128 --nat-loss-every 8), seeds 71..78, widths W = 64, 512, 768, 1024, 2048 (NEW; 128 dropped, the CPU check showed bands 64/128/256 hide the same-fact prior exposures 99.7% identically; supersedes job 1292, stopped); W = 256 (NATMASK of job 1283) and the natural window (NATS of job 1283) are REUSED only if (1) the full SHA-256 of the six runner files equals the pinned values (abort 12) and (2) reruns of NATS and NATMASK (band 256) on seed 71 with this job's code path are exactly equal to the 1283 records (logs, final hash, configuration hash, read meta, every read value; abort 13); per-seed assertions after every run (world, stream, meta, W0 reads, configuration identical to the 1283 record except the mask arm, the band and K0 = 0 check). ESTIMATE about 5.7 h: 40 runs x about 8 min + gate about 20 min + setup about 5 min. Output $OUT/e57_band_dist2/w<W>/ (tag bd<W>).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; R=$OUT/e57_k0conf16; O=$OUT/e57_band_dist2; mkdir -p $O
$PY $D/e57_oracle_run.py --selftest | tail -1
$PY $D/e57_band_dist_analyze.py --selftest | tail -1
cat > $O/pinned_sha256.txt <<'PIN'
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
PIN
sha256sum -c $O/pinned_sha256.txt || { echo "REUSE REFUSED: a runner file differs from the pinned hash"; exit 12; }
# GATE: NATS and NATMASK (band 256) on seed 71 equal the 1283 records exactly
$PY $D/e57_oracle_run.py --seed 71 --out $O/gate --masks none --mask-arm ALLQ256 --arms NATS,NATMASK --save-apps 16,32,64,128 --nat-loss-every 8 --tag bdgate
$PY - $O $R <<'PYE'
import json,sys,hashlib
o,r=sys.argv[1],sys.argv[2];ok=True
EXCL={'log','reads','arm','tag','canary_w0','diag','inline','final_hash','clip_bind_frac','cum_update_norm','wlog','read_hash','summary','sec_total'}
cfgh=lambda d:hashlib.sha256(json.dumps({k:v for k,v in d.items() if k not in EXCL},sort_keys=True).encode()).hexdigest()
for arm in ('NATS','NATMASK'):
    a=json.load(open(f'{o}/gate/E55_bdgate_{arm}_s71.saved.json'));b=json.load(open(f'{r}/E55_k0c16_{arm}_s71.saved.json'))
    logs=len(a['log'])==len(b['log']) and all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in ('update_norm','ce','gnorm','coef')) and a.get('final_hash')==b.get('final_hash')
    ra=json.load(open(f'{o}/gate/E55_bdgate_{arm}_s71.read.json'));rb=json.load(open(f'{r}/E55_k0c16_{arm}_s71.read.json'))
    meta=({k:v for k,v in ra['meta'].items() if k!='arm'}=={k:v for k,v in rb['meta'].items() if k!='arm'});reads=(ra['read']==rb['read'] and ra['W0']==rb['W0'])
    print(f'gate {arm} s71 vs 1283: logs+hash {logs}, configuration {cfgh(a)==cfgh(b)}, meta {meta}, reads {reads}');ok=ok and logs and cfgh(a)==cfgh(b) and meta and reads
sys.exit(0 if ok else 13)
PYE
for W in 64 512 768 1024 2048; do
  mkdir -p $O/w$W
  for SEED in 71 72 73 74 75 76 77 78; do
    $PY $D/e57_oracle_run.py --seed $SEED --out $O/w$W --masks none --mask-arm ALLQ$W --arms NATMASK --save-apps 16,32,64,128 --nat-loss-every 8 --tag bd$W
    $PY - $O/w$W $R $SEED $W <<'PYE'
import json,sys,hashlib
o,r,s,W=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
a=json.load(open(f'{o}/E55_bd{W}_NATMASK_s{s}.read.json'));b=json.load(open(f'{r}/E55_k0c16_NATMASK_s{s}.read.json'));rn=json.load(open(f'{r}/E55_k0c16_NATS_s{s}.read.json'))
assert {k:v for k,v in a['meta'].items() if k!='arm'}=={k:v for k,v in b['meta'].items() if k!='arm'} and a['W0']==b['W0']==rn['W0'] and a['place_len']==b['place_len'],('read meta / W0 / place length differ from 1283',s,W)
sa=json.load(open(f'{o}/E55_bd{W}_NATMASK_s{s}.saved.json'));sb=json.load(open(f'{r}/E55_k0c16_NATMASK_s{s}.saved.json'))
EXCL={'log','reads','arm','tag','canary_w0','diag','inline','final_hash','clip_bind_frac','cum_update_norm','wlog','read_hash','summary','sec_total','mask_arm','band'}
cfgh=lambda d:hashlib.sha256(json.dumps({k:v for k,v in d.items() if k not in EXCL},sort_keys=True).encode()).hexdigest()
assert sa['K0']==0 and sb['K0']==0 and sa['band']==W and sa['mask_arm']==f'ALLQ{W}' and cfgh(sa)==cfgh(sb),('configuration differs from 1283 beyond the mask arm and the band',s,W)
print('seed',s,'band',W,'assertions ok')
PYE
  done
done
echo "outputs: $O/w{64,512,768,1024,2048}/E55_bd<W>_NATMASK_s<seed>.{saved,read}.json (analysis: e57_band_dist_analyze.py --k0-dir $R --band-dir $O --widths 64,256,512,768,1024,2048)"
