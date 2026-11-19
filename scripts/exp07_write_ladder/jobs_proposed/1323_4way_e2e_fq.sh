#!/bin/bash
# JOB 1323 = FictionalQA E2E 4-way completion (Hans 11/19): the arms CHK and ISO on the FictionalQA worlds of job 1313 (labels 211..218; NAT = NATS and BAND1024 = NATMASK ALLQ1024 are the 1313 runs); direction rescaled to the applied per-chunk norms of the 1313 NATS run of the same seed (E55_fq_NATS_s21x.saved.json, log[k].update_norm); code e59_srv_run.py with --fq-worlds / --ref-nats (new options). EXPLORATORY; NOT queued until Hans' sentence; est. ~2 h (smoke ~6 min; per seed ~14 min). Output experiments/exp07_write_ladder/e59_fq.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 21"; exit 21; }
D=scripts/exp07_write_ladder;O=$OUT/e59_fq;W=$OUT/fq/worlds_w0c;NATS=$OUT/fq/run/E55_fq_NATS_s{seed}.saved.json
mkdir -p $O/smoke;df -h /home/solhnsol | tail -1
sha256sum $D/e59_srv_core.py $D/e59_srv_run.py $D/fq_run.py $D/fq_world.py $W/*.json
$PY $D/e59_srv_run.py --selftest | tail -2
$PY $D/e59_srv_run.py --seed 211 --out $O/smoke --arms CHK,ISO --tag smoke --smoke 16 --fq-worlds $W --ref-nats $NATS --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];ref=json.load(open('experiments/exp07_write_ladder/fq/run/E55_fq_NATS_s211.saved.json'))['log']
for a in ('CHK','ISO'):
    r=json.load(open(f'{d}/E59_smoke_{a}_s211.json'));s=r['steps'];assert len(s)==16 and all(abs(x['applied_norm']-ref[k]['update_norm'])<=1e-4*ref[k]['update_norm']+1e-9 for k,x in enumerate(s)),a;print('smoke',a,'ok: applied norm step0',round(s[0]['applied_norm'],4),'fact_ce',round(s[0]['fact_ce'],3))
print('4WAY E2E FQ SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 211 212 213 214 215 216 217 218; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms CHK,ISO --tag fq --fq-worlds $W --ref-nats $NATS
  echo "RUN 4way e2e fq seed $s rc=$?"
  rm -rf $O/states/E59_fq_*_s$s      # post-write weights not kept (disk of /home nearly full)
  set -e
done
