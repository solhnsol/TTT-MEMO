#!/bin/bash
# JOB 1341 = CLIP-BUDGET CAUSAL CHECK (Hans 11/23 '서버에 맡겨봐'; code e59_srv_run.py --lr --clip override (extended), e59_clip_analyze.py (new)): in job 1340 the update-objective gradient norm is > 1 in every condition (C0 2.15, ALLVIS 3.05, U1vis 3.22, U2vis 3.21), the clip (1.0) always binds and the applied step is g / ||g||; the extra gradient component of the visible other facts therefore takes a share of the unit step and the component along the C0 direction falls to ~0.65 (projection / norm ratio). Test: the same U12 world and seeds (231..238) with the inner rule changed to a CONSTANT step (lr = $LR, default 0.46 = 1 / the mean C0 gradient norm 2.15 of job 1340; clip = $CLIP, default 8 only as a safety against spikes; the share of clipped steps is recorded) for SAME (all other facts visible) and SAME_U1U2 (all other facts hidden), E2E 3B, original window / masks / reads of job 1328. If the clip budget causes the lost gain, G = SAME_U1U2 - SAME (the gap, +1.49 in job 1328) shrinks without the clip. Main metric (fixed before the result): G_orig - G_const paired by seed (exposure 128 out-of-window margin), and the ratio G_const / G_orig. Stability plan: smoke checks finite norms and that the clip is not binding any more; the update norm of every arm is reported (a constant step has no unit norm: the SAME_U1U2 / C0 arm applies ~1 on average, the visible-other-facts arm ~1.4); a diverged seed is flagged by the driver (last-8 loss > 4 or canary change), not hidden. Alternative if the constant step is unstable: normalise by the C0 gradient norm computed in the same step (second backward pass, ~2x GPU). EXPLORATORY; NOT queued until Hans' sentence; est. ~3.5 h (smoke ~10 min; per seed ~25 min = 2 arms x ~10.5 min + load; as job 1328).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/clip_off;LR=${LR:-0.46};CLIP=${CLIP:-8};ARMS=${ARMS:-SAME,SAME_U1U2}
mkdir -p $O/smoke
sha256sum $D/e59_srv_run.py $D/e59_clip_analyze.py $D/e59_u12.py $D/e55_core.py $D/e55_iso.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_srv_run.py --selftest | tail -1 | cut -c1-80; $PY $D/e59_u12.py --selftest | tail -1 | cut -c1-120
$PY $D/e59_srv_run.py --seed 231 --u12 --out $O/smoke --arms $ARMS --tag smoke --smoke 16 --reads 16 --lr $LR --clip $CLIP --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1];A=('SAME','SAME_U1U2');R={a:json.load(open(f'{d}/E59_smoke_{a}_s231.json')) for a in A}
for a in A:
    s=R[a]['steps'];assert len(s)==16 and all(math.isfinite(x['applied_norm']) and x['applied_norm']>0 for x in s),a
    cf=np.mean([x['coef']<0.9999 for x in R[a]['log']]);assert cf<0.5,('the clip must no longer bind at every step',a,cf)
    print(a,'mean update norm',round(float(np.mean([x['applied_norm'] for x in s])),3),'max',round(max(x['applied_norm'] for x in s),3),'clipped share',round(float(cf),3))
print('CLIP-OFF SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --u12 --out $O --arms $ARMS --tag cl --lr $LR --clip $CLIP
  echo "RUN clip_off seed $s rc=$?"
  rm -rf $O/states/E59_cl_*_s$s
  set -e
done
$PY $D/e59_clip_analyze.py | tee $O/clip_summary.txt
