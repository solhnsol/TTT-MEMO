#!/bin/bash
# JOB 1337 = PARTNER-CUE CHECK (Hans 11/21 '그것도 넣어서 코드 짜 보자'; code e59_partner.py + e59_srv_run.py (--cue, arm PARTNER) + e59_cue_analyze.py, new / extended files): E2E 3B, synthetic world v3_16_1 (the 1333 world, seeds 231..238 fixed before the result), original rule, K0 = 0, 128 steps. Write arms: ALLHID (every row sees only its own chunk), PARTNER (sentence rows of fact A additionally see the earlier sentences of ONE fixed partner fact B(A) only; pairing = order = rng(seed + 7777).permutation(16), partner = next, controls C1 / C2 = 2nd / 3rd next in that order), SHOWOTHER (all other facts visible). After the 128 writes the question of every fact A is read out-of-window at the final weights in four ways: no context (the standard read), and with a context sentence in front of the question: the partner B(A), the control C1(A), the control C2(A) (sentences taken from the last stream chunk). Main metric: cue effect = margin(B) - mean(margin(C1), margin(C2)); main contrast = cue effect(PARTNER) - cue effect(ALLHID), paired over seeds; PARTNER / SHOWOTHER no-context margins vs ALLHID show whether the connection costs binding (dilution). Reference runs ALLHID / SHOWOTHER of job 1333 are NOT reused (new runs with the extra reads, same code path). EXPLORATORY; NOT queued until Hans' sentence; est. ~4 h (smoke ~10 min; per seed ~30 min: 3 arms x (write ~4 min + 4 reads ~6 min); the read times are unmeasured).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/partner_cue
mkdir -p $O/smoke
sha256sum $D/e59_partner.py $D/e59_cue_analyze.py $D/e59_hidearms.py $D/e59_srv_run.py $D/e59_rev.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_partner.py --selftest | tail -1; $PY $D/e59_hidearms.py --selftest | tail -1; $PY $D/e59_srv_run.py --selftest | tail -1
$PY $D/e59_srv_run.py --seed 231 --out $O/smoke --arms ALLHID,PARTNER,SHOWOTHER --tag smoke --smoke 16 --reads 16 --cue --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1];A=('ALLHID','PARTNER','SHOWOTHER');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    rd=json.load(open(f'{d}/E59_smoke_{a}_s231.read.json'));assert len(rd['read']['16']['out'])==20 and len(rd['read']['16']['out'][0])==16,a
    cs={nm:json.load(open(f'{d}/E59_smoke_{a}_s231.cue_{nm}.read.json')) for nm in ('B','C1','C2')}
    for nm,c in cs.items():assert set(c['read'])=={'16'} and len(c['read']['16']['out'])==16 and len(c['read']['16']['out'][0])==16,(a,nm)
    b=np.asarray(cs['B']['read']['16']['out']);c1=np.asarray(cs['C1']['read']['16']['out']);assert np.abs(b-c1).max()>1e-3,'the cue context must change the read'
assert all(x['fact_hash']==y['fact_hash'] for a in A for x,y in zip(R[a]['steps'],R['ALLHID']['steps']))
assert abs(R['ALLHID']['log'][0]['ce']-R['PARTNER']['log'][0]['ce'])<1e-2 and len({R[a]['log'][5]['ce'] for a in A})==3
print('PARTNER-CUE SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms ALLHID,PARTNER,SHOWOTHER --tag pc --reads 128 --cue
  echo "RUN partner_cue seed $s rc=$?"
  rm -rf $O/states/E59_pc_*_s$s
  set -e
done
$PY $D/e59_cue_analyze.py --dir $O | tee $O/partner_cue_summary.txt
