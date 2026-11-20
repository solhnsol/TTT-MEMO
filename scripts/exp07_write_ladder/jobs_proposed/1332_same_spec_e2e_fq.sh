#!/bin/bash
# JOB 1332 = SAME-FACT SPECIFICITY on the E2E 3B FictionalQA worlds (labels 211..218; Hans 11/20): arms SAME / OTHERONLY / BOTH / RANDC (random keys, same number per earlier chunk as BOTH, drawn from ALL visible keys of the chunk because the FictionalQA filler (~43%) is smaller than the fact tokens (~57%); fact hit rate logged), definitions of job 1308z, update-path mask of the sentence rows in all 32 layers, window 8 chunks, K0 = 0, original rule; reference NAT = the 1313 NATS runs (experiments/exp07_write_ladder/fq/run), CHK / ISO = 1323; code e59_hidearms.py + e59_srv_run.py. Main metric = out-of-window margin gain at exposure 128 vs NAT (and vs W0). EXPLORATORY; NOT queued until Hans' sentence; est. ~5 h (smoke ~12 min; per seed ~38 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;W=$OUT/fq/worlds_w0c;O=$OUT/same_spec_fq
mkdir -p $O/smoke
sha256sum $D/e59_hidearms.py $D/e59_srv_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e57_allprev_masks.py
$PY $D/e59_hidearms.py --selftest | tail -1; $PY $D/e59_srv_run.py --selftest | tail -1
$PY $D/e59_srv_run.py --seed 211 --out $O/smoke --arms SAME,OTHERONLY,BOTH,RANDC --tag smoke --smoke 16 --fq-worlds $W --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1];A=('SAME', 'OTHERONLY', 'BOTH', 'RANDC');R={a:json.load(open(f'{d}/E59_smoke_{a}_s211.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    rd=json.load(open(f'{d}/E59_smoke_{a}_s211.read.json'));assert len(rd['read']['16']['out'][0])==16,a
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],R['SAME']['steps']))
assert all(abs(R[a]['log'][0]['ce']-R['SAME']['log'][0]['ce'])<1e-2 for a in A) and len({R[a]['log'][5]['ce'] for a in A})==len(A)
print('SAME-SPEC FQ SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 211 212 213 214 215 216 217 218; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms SAME,OTHERONLY,BOTH,RANDC --fq-worlds $W --tag ss
  echo "RUN same_spec_fq seed $s rc=$?"
  rm -rf $O/states/E59_ss_*_s$s
  set -e
done
