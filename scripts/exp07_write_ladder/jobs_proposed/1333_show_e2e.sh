#!/bin/bash
# JOB 1333 = DECISIVE CONTROL 'show only the same fact' (Hans 11/20 '코드 만들기 시작해'; code e59_hidearms.py + e59_srv_run.py arms SHOWSAME / SHOWOTHER / ALLHID, new/extended files): E2E 3B, synthetic world v3_16_1 (the 1321 world, seeds 231..238). Original rule, K0 = 0, window 8 chunks. ALLHID: every row loses ALL earlier-chunk keys (the chunk alone = stream-engine counterpart of CHK; NOTE the 1321 CHK arm was direction-rescaled to the NAT norms, ALLHID uses the original clipped-SGD rule like NAT). SHOWSAME: as ALLHID except that the sentence rows keep the earlier sentences of the SAME fact visible (filler rows see only their own chunk). SHOWOTHER: sentence rows keep the earlier sentences of the OTHER facts (placebo). If the CHK-level gain disappears with SHOWSAME the same fact is the cause. Reference NAT / CHK / ISO / MASK1024 = the 1321 runs of the same seeds (experiments/exp07_write_ladder/e59_4way). Main metric = out-of-window margin at exposure 128 vs NAT (and vs W0); rank / top-1 secondary. EXPLORATORY; NOT queued until Hans' sentence; est. ~4.8 h (smoke ~12 min; per seed ~36 min: 3 mask arms x ~10.5 min + load).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/show
mkdir -p $O/smoke
sha256sum $D/e59_hidearms.py $D/e59_srv_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e57_allprev_masks.py
$PY $D/e59_hidearms.py --selftest | tail -1; $PY $D/e59_srv_run.py --selftest | tail -1
$PY $D/e59_srv_run.py --seed 231 --out $O/smoke --arms SHOWSAME,SHOWOTHER,ALLHID --tag smoke --smoke 16 --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1];A=('SHOWSAME','SHOWOTHER','ALLHID');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A};R['NAT']=json.load(open('experiments/exp07_write_ladder/e59_4way/E59_a4_NAT_s231.json'))
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    rd=json.load(open(f'{d}/E59_smoke_{a}_s231.read.json'));assert len(rd['read']['16']['out'])==20 and len(rd['read']['16']['out'][0])==16,a
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],R['NAT']['steps'][:16]))
assert abs(R['NAT']['log'][0]['ce']-R['ALLHID']['log'][0]['ce'])<1e-2 and all(R['NAT']['log'][5]['ce']!=R[a]['log'][5]['ce'] for a in A) and len({R[a]['log'][5]['ce'] for a in A})==3
print('SHOW SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms SHOWSAME,SHOWOTHER,ALLHID --tag show
  echo "RUN show seed $s rc=$?"
  rm -rf $O/states/E59_show_*_s$s
  set -e
done
