#!/bin/bash
# JOB 1328 = U1 / U2 hiding experiment (Hans 11/19 '진행시켜'; code e59_u12.py + e59_srv_run.py --u12, new/extended files): E2E 3B, synthetic world with 32 facts (16 read facts + 16 extra facts whose places are NOT read candidates; 32 sentences per chunk, 128 exposures, chunks 1024 tokens), read items = the 16 read facts with the 16 read-fact places as candidates (no controls), arms NAT, SAME (earlier sentences of the same fact hidden), SAME_U1 (+ the other read facts hidden), SAME_U2 (+ the extra facts hidden), SAME_U1U2 (everything of earlier chunks hidden); original clipped-SGD rule, K0 = 0; main metric = out-of-window margin at exposure 128 (gain over W0 and over NAT); world seeds 231..238. EXPLORATORY; NOT queued until Hans' sentence; est. ~7.2 h (smoke ~15 min; per seed ~54 min: NAT ~7.5 min + 4 mask arms ~10.5 min + load).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/u12
mkdir -p $O/smoke
sha256sum $D/e59_u12.py $D/e59_srv_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e57_allprev_masks.py
$PY $D/e59_u12.py --selftest | tail -1; $PY $D/e59_srv_run.py --selftest | tail -1
$PY $D/e59_srv_run.py --seed 231 --u12 --out $O/smoke --arms NAT,SAME,SAME_U1,SAME_U2,SAME_U1U2 --tag smoke --smoke 16 --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1];A=('NAT','SAME','SAME_U1','SAME_U2','SAME_U1U2');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    rd=json.load(open(f'{d}/E59_smoke_{a}_s231.read.json'));assert len(rd['read']['16']['out'])==16 and len(rd['read']['16']['out'][0])==16 and rd['meta']['gi']==[0]*16,a
assert all(x['fact_hash']==y['fact_hash'] for a in A[1:] for x,y in zip(R[a]['steps'],R['NAT']['steps']))
assert abs(R['NAT']['log'][0]['ce']-R['SAME']['log'][0]['ce'])<1e-2 and all(R['NAT']['log'][5]['ce']!=R[a]['log'][5]['ce'] for a in A[1:])
print('U12 SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --u12 --out $O --arms NAT,SAME,SAME_U1,SAME_U2,SAME_U1U2 --tag u12
  echo "RUN u12 seed $s rc=$?"
  rm -rf $O/states/E59_u12_*_s$s
  set -e
done
