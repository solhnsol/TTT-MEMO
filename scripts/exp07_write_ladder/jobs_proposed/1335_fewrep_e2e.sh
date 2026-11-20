#!/bin/bash
# JOB 1335 = FEW-REPETITION read-out (Hans 11/20): E2E 3B, synthetic world v3_16_1 (the 1321 world, seeds 231..238), arms NAT / MASK1024 (band-1024 mask in all 32 layers) / CHK / ISO (direction rescaled to the NAT applied norms) trained on the FIRST 8 chunks only and read out-of-window after 2 / 4 / 8 exposures (--smoke 8 --reads 2,4,8; the stream of the 128-chunk world is identical, checked); same reads / records as 1321. Main metric = out-of-window margin gain vs NAT (and vs W0) at exposures 2 / 4 / 8. EXPLORATORY; NOT queued until Hans' sentence; est. ~3.1 h (smoke ~10 min; per seed ~23 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/fewrep
mkdir -p $O/smoke
sha256sum $D/e59_hidearms.py $D/e59_srv_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e57_allprev_masks.py
$PY $D/e59_hidearms.py --selftest | tail -1; $PY $D/e59_srv_run.py --selftest | tail -1
$PY $D/e59_srv_run.py --seed 231 --out $O/smoke --arms NAT,MASK1024,CHK,ISO --tag smoke --smoke 8 --reads 2,4,8 --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys
d=sys.argv[1];A=('NAT','MASK1024','CHK','ISO');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==8 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    rd=json.load(open(f'{d}/E59_smoke_{a}_s231.read.json'));assert set(rd['read'])=={'2','4','8'} and len(rd['read']['2']['out'])==20 and len(rd['read']['2']['out'][0])==16,a
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],R['NAT']['steps']))
e1=json.load(open('experiments/exp07_write_ladder/e59_4way/E59_a4_NAT_s231.json'))['steps'][:8]
assert all(x['fact_hash']==y['fact_hash'] for x,y in zip(R['NAT']['steps'],e1)),'the first 8 chunks must be the 1321 world'
assert abs(R['NAT']['log'][0]['ce']-R['CHK']['log'][0]['ce'])<1e-2 and len({R[a]['log'][5]['ce'] for a in A})==4
print('FEWREP E2E SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms NAT,MASK1024,CHK,ISO --tag fr --smoke 8 --reads 2,4,8
  echo "RUN fewrep seed $s rc=$?"
  rm -rf $O/states/E59_fr_*_s$s
  set -e
done
