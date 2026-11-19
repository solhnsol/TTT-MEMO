#!/bin/bash
# JOB 1327 = exposure-DENSITY experiment on the E2E synthetic world (Hans 11/19; code e59_dens_run.py, new file): 16 exposures per fact fixed, repeat gap ~500 / 1000 / 2000 / 4000 / 8000 / 16000 tokens (6 conditions), arms NAT vs MASK1024 (band 1024 in all 32 layers), main metric = out-of-window margin after the last exposure vs W0; UNMATCHED design (steps = chunks of the condition: 8 / 16 / 32 / 64 / 128 / 256) AND the STEP-MATCHED CONTROL (every condition padded with background-only chunks BEFORE its chunks to 256 steps; the p16 condition is the same run in both). World seeds 241..244 (Hans). EXPLORATORY; NOT queued until Hans' sentence; est. ~13 h (smoke ~12 min; per seed ~3.3 h: 22 arm-runs, 3,568 write steps x 2.6 s + 22 x ~100 s for K,V record / W0 / read).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 25"; exit 25; }
D=scripts/exp07_write_ladder;O=$OUT/dens
mkdir -p $O/smoke
sha256sum $D/e59_dens_run.py $D/e55_core.py $D/e55_run.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_dens_run.py --selftest | tail -1
# smoke: seed 241, the two extreme-gap short conditions n2p1 (8 chunks, matched = 8 + 248 background) and one NAT run each mode
$PY $D/e59_dens_run.py --seed 241 --out $O/smoke --conditions n2p1 --matched both --arms NAT,MASK1024 --rerun
$PY - $O/smoke <<'PYE'
import json,math,sys,numpy as np
d=sys.argv[1]
for tag in ('n2p1_u','n2p1_m'):
    R={a:json.load(open(f'{d}/DENS_{tag}_{a}_s241.json')) for a in ('NAT','MASK1024')}
    for a,r in R.items():
        S=np.asarray(r['read']['out']);W0=np.asarray(r['W0']['out']);assert S.shape==(20,16) and np.isfinite(S).all() and np.isfinite(W0).all(),a
        assert all(math.isfinite(e['ce']) for e in r['log']),a
    assert R['NAT']['n_steps']==(8 if tag.endswith('u') else 256) and R['NAT']['n_bg_first']==(0 if tag.endswith('u') else 248),(tag,R['NAT']['n_steps'])
    print('smoke',tag,'ok: steps',R['NAT']['n_steps'])
print('DENSITY SMOKE PASSED')
PYE
rm -rf $O/smoke/states
for s in 241 242 243 244; do
  set +e
  $PY $D/e59_dens_run.py --seed $s --out $O --matched both --arms NAT,MASK1024
  echo "RUN dens seed $s rc=$?"
  set -e
done
