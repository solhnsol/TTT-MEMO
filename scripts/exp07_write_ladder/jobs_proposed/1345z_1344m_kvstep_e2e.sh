#!/bin/bash
# JOB 1344m = ONE-STEP KEY / ERROR SWAP (G-ii) AND MATCHED READS (H) of the key-value decomposition (spec v2; code e59_kvstep.py + e59_kvstep_analyze.py (new); NOT QUEUED: Hans decides after the final 1344k result). U12 world seeds 231..238, chunks 16..127, conditions C0 / SAMEVIS / NATVIS streamed in turn; at every measured chunk 28 combinations (3 real autodiff steps, G[A,B] swaps incl. NATVIS, SIZE-RESTORED, COMMON-KEY-restored, POSITION SPLIT own / other / rest) are applied as ONE step to W0 with the original rule and read (out-of-window margin, neutral tail, 16 facts), and H: weights W0 / W_C0 / W_SAMEVIS / W_NATVIS x read contexts neutral / same-fact prefix (answer hidden by attention) / other-fact prefix. Needs the read keys of job 1344b (kv1344k/kv_s<seed>_keys.npy). GPU est. ~6 h with the 4 measured chunks 16, 48, 80, 112 of the full-run command (8 seeds x ~45 min: 3 streams + 4 chunks x ~8 min of reads, 28 combinations + H; read time measured in the E62 smoke: ~10-12 s per 320 rows; the first estimate of ~4 h was wrong, 8 chunks would be ~11.6 h); host RAM ~20 GB per seed. Smoke gate: seed 231, chunk 16 only. EXPLORATORY.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 30"; exit 30; }
D=scripts/exp07_write_ladder;O=$OUT/kvstep;K=$OUT/kv1344k
mkdir -p $O/smoke
for s in 231 232 233 234 235 236 237 238; do test -f $K/kv_s${s}_keys.npy || { echo "keys of job 1344b missing for seed $s: exit 31"; exit 31; }; done
sha256sum $D/e59_kvstep.py $D/e59_kvstep_analyze.py $D/e59_kvprobe.py $D/e59_shadow.py $D/e59_u12.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e59_hidearms.py
$PY $D/e59_kvstep.py --selftest | tail -1
$PY $D/e59_kvprobe.py --selftest | tail -1
$PY $D/e59_kvstep.py --seed 231 --out $O/smoke --nch 17 --chunks 16 --keys $K/kv_s231_keys.npy 2>&1 | grep -v "Missing metrics\|metrics not found"
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/kvstep_s231.json'));row=r['rows'][0];v=r['validation'][0];print('validation',v)
assert max(v['w1'],v['w2'],v['w3'])<0.1,v
c=row['combos'];assert all(math.isfinite(x['dmargin']) for x in c.values())
print('ACT[C0] dmargin',round(c['ACT[C0]']['dmargin'],3),'G[C0,C0]',round(c['G[C0,C0]']['dmargin'],3),'| ACT[SAMEVIS]',round(c['ACT[SAMEVIS]']['dmargin'],3),'G[S,S]',round(c['G[SAMEVIS,SAMEVIS]']['dmargin'],3),'| ACT[NATVIS]',round(c['ACT[NATVIS]']['dmargin'],3),'G[N,N]',round(c['G[NATVIS,NATVIS]']['dmargin'],3))
for ctx in ('neutral','same','other'):assert all(len(x)==16 and all(math.isfinite(y) for y in x) for x in row['H'][ctx].values()),ctx
print('H W0 margins',{k:round(float(np.mean(v_['W0'])),2) for k,v_ in row['H'].items()});print('KVSTEP SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_kvstep.py --seed $s --out $O --keys $K/kv_s${s}_keys.npy --chunks 16,48,80,112
  echo "RUN kvstep seed $s rc=$?"
  rm -rf $O/states/s${s}_*
  set -e
done
$PY $D/e59_kvstep_analyze.py --dir $O | tee $O/kvstep_summary.txt
