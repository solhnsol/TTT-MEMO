#!/bin/bash
# JOB 1340 = SHADOWING vs LEARNING-PATH CHECK, E2E 3B (Hans 11/23 '섀도잉 점검 만들어' with the point-A idea merged; code e59_shadow.py + e59_shadow_analyze.py (new), e59_u12.py masks, e59_scr.py, e59_leak_probe.py token positions): frozen W0 (forward + backward, NO write), the U12 world of jobs 1328 / 1339 (intact and with scrambled U2 sentences; seeds 231..238 fixed before the result), 8 chunks (16, 32, 48, 64, 80, 96, 112, 127). Conditions (all hide the earlier sentences of the SAME fact): C0 all other facts hidden (reference), U1vis (U2 hidden), U2vis (U1 hidden), ALLVIS (all others visible) in the intact world; C0 and U2vis in the scrambled world. Per condition and chunk: NLL of the answer tokens (second-mentioned entity of the U1 sentences), the first-mentioned entity, the frame tokens, all tokens; gradient (fast weights, 262144-coordinate sketch + exact norm) of three objectives: g_ans, g_frame, g_all. Main metric: cos(g_ans(X), g_ans(C0)) with the co-primary NLL_ans(X) - NLL_ans(C0) (shadowing = lower NLL; learning-path distortion = same NLL, lower cosine / large residual, and the residual aligned with the frame gradient); seed-paired 95% t over 8 seeds. EXPLORATORY; NOT queued until Hans' sentence; est. ~3 h (smoke ~8 min; per seed ~22 min: 2 records + 6 condition passes + 8 x 3 gradients per condition; unmeasured).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/shadow
mkdir -p $O/smoke
sha256sum $D/e59_shadow.py $D/e59_shadow_analyze.py $D/e59_u12.py $D/e59_scr.py $D/e59_leak_probe.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_shadow.py --selftest | tail -1 | cut -c1-120; $PY $D/e59_u12.py --selftest | tail -1 | cut -c1-120; $PY $D/e59_scr.py --selftest | tail -1 | cut -c1-120
$PY $D/e59_shadow.py --seed 231 --out $O/smoke --nch 33 --chunks 16,32
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/shadow_s231.json'));rows={(x['world'],x['cond'],x['k']):x for x in r['rows']};z=np.load(sys.argv[1]+'/shadow_s231_sketch.npz')
assert len(rows)==12,len(rows)
assert all(math.isfinite(v) for x in rows.values() for kk,v in x.items() if kk.startswith(('nll_','gnorm_')) ) and all(x['gnorm_all']>0 and x['gnorm_ans']>0 for x in rows.values())
SK={k:z['sk'][i] for i,k in enumerate(z['keys'])}
c=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
cs=[c(SK[f'intact|ALLVIS|{k}'][0],SK[f'intact|C0|{k}'][0]) for k in (16,32)];assert max(cs)<0.9999,('the visible other facts must change the gradient',cs)
nf=[rows[('intact','ALLVIS',k)]['nll_frame']-rows[('intact','C0',k)]['nll_frame'] for k in (16,32)];assert np.mean(nf)<=0.02,('frame tokens must not get harder when similar sentences are visible',nf)
print('SHADOW SMOKE PASSED cos_ans(ALLVIS,C0)',[round(x,3) for x in cs],'dNLL_frame',[round(x,3) for x in nf],'dNLL_ans',[round(rows[('intact','ALLVIS',k)]['nll_ans']-rows[('intact','C0',k)]['nll_ans'],3) for k in (16,32)])
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_shadow.py --seed $s --out $O
  echo "RUN shadow seed $s rc=$?"
  rm -rf $O/states/s${s}_*
  set -e
done
$PY $D/e59_shadow_analyze.py --dir $O | tee $O/shadow_summary.txt
