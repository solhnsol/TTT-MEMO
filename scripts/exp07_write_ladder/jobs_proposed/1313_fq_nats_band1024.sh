#!/bin/bash
# JOB 1313 = FictionalQA second world: smoke (seed 211, 16 chunks, both arms) then NATS + band-1024 (NATMASK, --mask-arm ALLQ1024), seeds 211..218 (EXPLORATORY; Hans 11/18; E2E 3B v3_16_1, clipped SGD, K0 = 0, reads at exposures 16/32/64/128; ~2.7-3 h; GPU one job at a time; NOT queued until Hans' sentence). Worlds: experiments/exp07_write_ladder/fq/worlds_w0c/fq_20x_cap3_VAR_w0.json (W0-replaced, controls completed to 4; 201..208 = labels 211..218). If the smoke fails the main run is NOT started.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder;W=$OUT/fq/worlds_w0c;O=$OUT/fq/run
mkdir -p $O/smoke
sha256sum $D/fq_world.py $D/fq_run.py $D/e57_oracle_run.py $D/e57_natmask_run.py $D/e55_run.py $D/e55_core.py $D/ladder.py $D/s3.py $W/*.json
$PY $D/fq_world.py --selftest | tail -1; $PY $D/fq_run.py --selftest | tail -1
# smoke: seed 211, 16 chunks (read at exposure 16), both arms; checks: finite ce / gnorm, W0 out margin finite, NATMASK differs from NATS after chunk 0
$PY $D/fq_run.py --seed 211 --fq-worlds $W --out $O/smoke --arms NATS,NATMASK --mask-arm ALLQ1024 --save-apps 16,32,64,128 --nat-loss-every 8 --tag fqsmoke --smoke 16 --rerun
$PY - $O/smoke <<'PYE'
import json,sys,math,glob
d=sys.argv[1]
for ar in ('NATS','NATMASK'):
    r=json.load(open(f'{d}/E55_fqsmoke_{ar}_s211.saved.json'));lg=r['log'];assert len(lg)==16,len(lg)
    assert all(math.isfinite(e['ce']) and math.isfinite(e['gnorm']) and e['update_norm']>0 for e in lg),ar
    rd=json.load(open(f'{d}/E55_fqsmoke_{ar}_s211.read.json'));assert '16' in rd['read'] and 'out' in rd['read']['16'] and 'out' in rd['W0']
    print('smoke',ar,'ok: mean ce',sum(e['ce'] for e in lg)/16,'update norm',lg[-1]['update_norm'],'chunk-1 ce',lg[1]['ce'])
a=json.load(open(f'{d}/E55_fqsmoke_NATS_s211.saved.json'))['log'];b=json.load(open(f'{d}/E55_fqsmoke_NATMASK_s211.saved.json'))['log']
assert a[0]['ce']==b[0]['ce'] and any(x['ce']!=y['ce'] for x,y in zip(a[1:],b[1:])),'NATMASK must equal NATS at chunk 0 and differ later'
print('FQ SMOKE PASSED')
PYE
for s in 211 212 213 214 215 216 217 218; do
  $PY $D/fq_run.py --seed $s --fq-worlds $W --out $O --arms NATS,NATMASK --mask-arm ALLQ1024 --save-apps 16,32,64,128 --nat-loss-every 8 --tag fq
  echo "RUN fq seed $s rc=$?"
done
