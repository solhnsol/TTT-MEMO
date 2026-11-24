#!/bin/bash
# JOB 1344p = REAL ONE-STEP EFFECT BY POSITION CLASS (Hans 11/24 'B': 8 seeds, 2 measured chunks 16 and 80; code e59_kvpos.py + e59_kvpos_analyze.py (new, derived from e59_kvstep.py); NOT QUEUED: Hans decides the position): conditions C0 and SAMEVIS (U12 world, seeds 231..238); the step built from the positions of one class (ans / name / frame of the 16 read facts, extra = the 16 extra facts' sentences, filler) applied to W0 with the clip coefficient of the full step; fact level: answer tokens / name tokens of facts 0..3 only, margins of all 16 read facts stored (diagonal vs off-diagonal). Needs the read keys of job 1344b (kv1344k/kv_s<seed>_keys.npy). GPU est. ~4.7 h (8 seeds x ~35 min: 2 streams ~12 min + setup ~3 min + 2 chunks x 26 reads x ~20 s; UNMEASURED for this variant). Smoke gate: seed 231, chunk 16. EXPLORATORY.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 33"; exit 33; }
D=scripts/exp07_write_ladder;O=$OUT/kvpos;K=$OUT/kv1344k
mkdir -p $O/smoke
for s in 231 232 233 234 235 236 237 238; do test -f $K/kv_s${s}_keys.npy || { echo "keys of job 1344b missing for seed $s: exit 31"; exit 31; }; done
sha256sum $D/e59_kvpos.py $D/e59_kvpos_analyze.py $D/e59_kvprobe.py $D/e59_shadow.py $D/e59_u12.py $D/e55_core.py $D/e57_natmask_run.py $D/e59_hidearms.py
$PY $D/e59_kvpos.py --selftest | tail -1
$PY $D/e59_kvprobe.py --selftest | tail -1
$PY $D/e59_kvpos.py --seed 231 --out $O/smoke --nch 17 --chunks 16 --keys $K/kv_s231_keys.npy 2>&1 | grep -v "Missing metrics\|metrics not found"
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
r=json.load(open(sys.argv[1]+'/kvpos_s231.json'));row=r['rows'][0];v=r['validation'][0];print('validation',v);assert max(v['w1'],v['w2'],v['w3'])<0.1,v
c=row['combos'];assert all(math.isfinite(x['dmargin']) for x in c.values()) and len(c)==30,len(c)
for k in ('C0','SAMEVIS'):
    cls=sum(c[f'G[{k},{k}]@{p}']['dmargin'] for p in ('ans','name','frame','extra','filler'));print(k,'real step',round(c[f'ACT[{k}]']['dmargin'],3),'full G',round(c[f'G[{k},{k}]']['dmargin'],3),'sum of the 5 classes',round(cls,3))
assert abs(c['ACT[C0]']['dmargin']-c['G[C0,C0]']['dmargin'])<0.05
print('class token counts',row['posn']);print('KVPOS SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_kvpos.py --seed $s --out $O --keys $K/kv_s${s}_keys.npy --chunks 16,80
  echo "RUN kvpos seed $s rc=$?"
  rm -rf $O/states/s${s}_*
  set -e
done
$PY $D/e59_kvpos_analyze.py --dir $O | tee $O/kvpos_summary.txt
