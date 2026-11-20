#!/bin/bash
# JOB 1329 = NUMBER-MATCHED hiding controls for the 'other facts hidden' effect (Hans 11/20 '그래보자고'; code e59_match.py + e59_srv_run.py arms RANDTOK / BGSENT, new/extended files): E2E 3B, synthetic world v3_16_1 (16 facts, 16 sentences/chunk, 128 exposures, the 1321 world, seeds 231..238, same worlds / reads as 1321). Every arm hides the earlier same-fact sentences (SAME, original rule, K0 = 0, window 8 chunks); SAME_U1 additionally hides the sentences of all other facts (~350 tokens per earlier chunk, sentence runs); RANDTOK hides the same NUMBER of keys per earlier chunk as scattered single filler tokens; BGSENT hides the same number as contiguous filler runs with the lengths of the other facts' sentences. Hidden (row, key) pairs identical across SAME_U1 / RANDTOK / BGSENT (asserted in the CPU selftest). Reference NAT = the 1321 NAT runs of the same seeds (experiments/exp07_write_ladder/e59_4way/E59_a4_NAT_s23x). Main metric = out-of-window margin gain at exposure 128 vs NAT; rank / top-1 secondary. EXPLORATORY; NOT queued until Hans' sentence; est. ~6.4 h (smoke ~15 min; per seed ~46 min: 4 mask arms x ~10.5 min + load).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/match
mkdir -p $O/smoke
sha256sum $D/e59_match.py $D/e59_srv_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e57_allprev_masks.py
$PY $D/e59_match.py --selftest | tail -1; $PY $D/e59_srv_run.py --selftest | tail -1
$PY $D/e59_srv_run.py --seed 231 --out $O/smoke --arms SAME,SAME_U1,RANDTOK,BGSENT --tag smoke --smoke 16 --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1];A=('SAME','SAME_U1','RANDTOK','BGSENT');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A};R['NAT']=json.load(open('experiments/exp07_write_ladder/e59_4way/E59_a4_NAT_s231.json'))
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    rd=json.load(open(f'{d}/E59_smoke_{a}_s231.read.json'));assert len(rd['read']['16']['out'])==20 and len(rd['read']['16']['out'][0])==16,a
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],R['NAT']['steps'][:16]))
assert abs(R['NAT']['log'][0]['ce']-R['SAME']['log'][0]['ce'])<1e-2 and all(R['NAT']['log'][5]['ce']!=R[a]['log'][5]['ce'] for a in A) and len({R[a]['log'][5]['ce'] for a in A})==4
print('MATCH SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms SAME,SAME_U1,RANDTOK,BGSENT --tag match
  echo "RUN match seed $s rc=$?"
  rm -rf $O/states/E59_match_*_s$s
  set -e
done
