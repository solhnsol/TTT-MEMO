#!/bin/bash
# JOB 1344 = SAME-FACT-VISIBLE CONDITIONS added to the shadowing / learning-path check of job 1340 (Hans 11/23 '실험을 못 하는 상황도 아니니 돌리는 게 나을듯'; code e59_shadow.py --set new, e59_shadow_analyze.py, e59_hidearms.py OTHERONLY): the same U12 world (intact), seeds 231..238, the same 8 chunks (16, 32, 48, 64, 80, 96, 112, 127), the same measurements (NLL of the answer / first-entity / frame / all tokens of the U1 sentences; gradient sketches of g_ans, g_frame, g_all, exact norms; frozen W0, no write) for: C0 (all other facts hidden, the same fact hidden: the reference of job 1340, re-run here), SAMEVIS (the earlier sentences of the SAME fact visible, every other fact hidden = e59_hidearms OTHERONLY: C0 with only the same-fact hiding removed) and NATVIS (no mask: the same fact and all other facts visible). Expectation: the answer NLL falls strongly when the same fact is visible (copy); the question is how the gradient (cosine / projection / residual w.r.t. C0, norm) looks in that case compared with the other-facts-visible conditions of job 1340. Main metric as in 1340: cos(g_ans(X), g_ans(C0)) with the NLL_ans difference as co-primary. EXPLORATORY; NOT queued until Hans' sentence (order: after 1341 / 1341n); est. ~1.5 h (smoke ~6 min; per seed ~11 min: 1 record + 3 condition passes + 8 x 3 gradients per condition; unmeasured).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/shadow_same
mkdir -p $O/smoke
sha256sum $D/e59_shadow.py $D/e59_shadow_analyze.py $D/e59_u12.py $D/e59_scr.py $D/e59_leak_probe.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_shadow.py --selftest | tail -1 | cut -c1-120; $PY $D/e59_u12.py --selftest | tail -1 | cut -c1-120; $PY $D/e59_scr.py --selftest | tail -1 | cut -c1-120
$PY $D/e59_shadow.py --seed 231 --out $O/smoke --nch 33 --chunks 16,32 --set new
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/shadow_s231.json'));rows={(x['world'],x['cond'],x['k']):x for x in r['rows']};z=np.load(sys.argv[1]+'/shadow_s231_sketch.npz')
assert len(rows)==6,len(rows)
assert all(math.isfinite(v) for x in rows.values() for kk,v in x.items() if kk.startswith(('nll_','gnorm_'))) and all(x['gnorm_all']>0 and x['gnorm_ans']>0 for x in rows.values())
SK={k:z['sk'][i] for i,k in enumerate(z['keys'])};c=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
d=[rows[('intact','SAMEVIS',k)]['nll_ans']-rows[('intact','C0',k)]['nll_ans'] for k in (16,32)];assert np.mean(d)<-1.0,('the visible same-fact sentences must lower the answer NLL (copy)',d)
cs=[c(SK[f'intact|SAMEVIS|{k}'][0],SK[f'intact|C0|{k}'][0]) for k in (16,32)]
print('SAME-VIS SMOKE PASSED dNLL_ans(SAMEVIS-C0)',[round(x,2) for x in d],'cos_ans',[round(x,3) for x in cs])
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_shadow.py --seed $s --out $O --set new
  echo "RUN shadow seed $s rc=$?"
  rm -rf $O/states/s${s}_*
  set -e
done
$PY $D/e59_shadow_analyze.py --dir $O | tee $O/shadow_summary.txt
