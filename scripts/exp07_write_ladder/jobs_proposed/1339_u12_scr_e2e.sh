#!/bin/bash
# JOB 1339 = SCRAMBLED-U2 CHECK, 'why does hiding the OTHER facts help in the synthetic world' (Hans 11/22; code e59_scr.py + e59_srv_run.py --u12 --scramble-u2 (new / extended files), e59_u12.py masks): the U12 world of job 1328 (32 facts: 16 read facts U1 + 16 extra facts U2, seeds 231..238 fixed before the result) with the tokens of every U2 sentence PERMUTED inside its span (bag of words, names and places kept, structure destroyed; U1 and filler untouched). Arms as 1328: NAT, SAME, SAME_U1, SAME_U2, SAME_U1U2 (original rule, K0 = 0), all RE-RUN on the scrambled stream. Main contrast (fixed before the result): the gain that hiding U2 adds on top of U1 hidden, D = SAME_U1U2 - SAME_U1 (margin, exposure 128 out-of-window), compared per seed with the same quantity of job 1328 (intact U2: +1.16 on average, E59_u12 files): D_scr - D_intact; secondary SAME_U2 - SAME, SAME_U1 - SAME, the interaction. Structure reading: D_scr << D_intact; lexical-content reading: D_scr ~ D_intact. EXPLORATORY; NOT queued until Hans' sentence; est. ~5.4 h (as 1328; smoke ~15 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/u12_scr
mkdir -p $O/smoke
sha256sum $D/e59_scr.py $D/e59_scr_analyze.py $D/e59_u12.py $D/e59_srv_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e57_allprev_masks.py
$PY $D/e59_u12.py --selftest | tail -1; $PY $D/e59_scr.py --selftest | tail -1; $PY $D/e59_srv_run.py --selftest | tail -1
$PY $D/e59_srv_run.py --seed 231 --u12 --scramble-u2 --out $O/smoke --arms NAT,SAME,SAME_U1,SAME_U2,SAME_U1U2 --tag smoke --smoke 16 --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1];A=('NAT','SAME','SAME_U1','SAME_U2','SAME_U1U2');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    rd=json.load(open(f'{d}/E59_smoke_{a}_s231.read.json'));assert len(rd['read']['16']['out'])==16 and len(rd['read']['16']['out'][0])==16 and rd['meta']['gi']==[0]*16,a
assert all(x['fact_hash']==y['fact_hash'] for a in A[1:] for x,y in zip(R[a]['steps'],R['NAT']['steps']))
assert abs(R['NAT']['log'][0]['ce']-R['SAME']['log'][0]['ce'])<1e-2 and all(R['NAT']['log'][5]['ce']!=R[a]['log'][5]['ce'] for a in A[1:])
e0=json.load(open('experiments/exp07_write_ladder/u12/E59_u12_NAT_s231.json'))['log'][0]['ce'];assert abs(R['NAT']['log'][0]['ce']-e0)>0.02,('chunk 0 must differ from the intact U12 world (scrambled U2 sentences)',R['NAT']['log'][0]['ce'],e0)
print('U12 SCRAMBLED SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --u12 --scramble-u2 --out $O --arms NAT,SAME,SAME_U1,SAME_U2,SAME_U1U2 --tag us
  echo "RUN u12 seed $s rc=$?"
  rm -rf $O/states/E59_us_*_s$s
  set -e
done
$PY $D/e59_scr_analyze.py | tee $O/scr_summary.txt
