#!/bin/bash
# JOB 1344k = KEY-VALUE DECOMPOSITION OF THE PRE-CLIP FAST-WEIGHT GRADIENT, spec v2 items A-F and G(i) (Hans 11/23 'use the key-value decomposition; the d0 projection is too naive'; spec reports/server-request-1344k-v2-and-E61.md; code e59_kvprobe.py + e59_kv1344k.py + e59_kv_analyze.py (new files); same world / chunks / masks as jobs 1340 and 1344, frozen W0, no weight update): for the conditions C0, SAMEVIS, NATVIS, U1vis, U2vis (U12 world, seeds 231..238, chunks 16, 32, ..., 127) the layer inputs x, x2 and the error signals delta_z1, delta_z3, delta_z2 of the 8 prime SwiGLU layers are taken at every position of the chunk (update objective = mean NLL of all valid tokens, before clipping), sum_p x (x) delta is checked against the autodiff gradient, and the key match with the neutral-read keys, the candidate-logit projection, the three-way decomposition and the key/error swap are saved. GPU est. 3-4 h (8 seeds, ~25-30 min each: 5 conditions x 128 chunk stream + 8 measured chunks with 3 passes; UNMEASURED). Smoke gate: seed 231, 2 conditions, chunk 16, validation error of w2 must be < 0.1 (bf16 compute; the error is reported either way). EXPLORATORY. Items G(ii) / H are job 1344m.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 27"; exit 27; }
D=scripts/exp07_write_ladder;O=$OUT/kv1344k
mkdir -p $O/smoke
sha256sum $D/e59_kvprobe.py $D/e59_kv1344k.py $D/e59_kv_analyze.py $D/e59_shadow.py $D/e59_u12.py $D/e59_leak_probe.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e59_hidearms.py
$PY $D/e59_kvprobe.py --selftest | tail -1
$PY $D/e59_shadow.py --selftest | tail -1 | cut -c1-120
$PY $D/e59_kv1344k.py --seed 231 --out $O/smoke --nch 17 --chunks 16 --conds C0,SAMEVIS
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
m=json.load(open(sys.argv[1]+'/kv_s231.json'));z=np.load(sys.argv[1]+'/kv_s231.npz')
assert len(m['rows'])==2,len(m['rows'])
for r in m['rows']:
    v=np.asarray(r['val_relerr']);print('smoke',r['cond'],'k',r['k'],'|g|',round(r['gnorm'],3),'validation rel.err max over layers w1/w2/w3',v.max(0))
    assert math.isfinite(r['gnorm']) and r['gnorm']>0 and v[:,1].max()<0.1,('Sum_p x2 (x) delta_z2 does not match the autodiff gradient',v[:,1])
for nm in ('X2K','X2KJ','D16','DN','XN','XIN','NLL'):
    for c in ('C0','SAMEVIS'):assert np.isfinite(z[f'{c}|16|{nm}']).all(),(c,nm)
print('KV SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_kv1344k.py --seed $s --out $O
  echo "RUN kv1344k seed $s rc=$?"
  rm -rf $O/states/s${s}_*
  set -e
done
$PY $D/e59_kv_analyze.py --dir $O | tee $O/kv_summary.txt
