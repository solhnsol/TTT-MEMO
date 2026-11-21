#!/bin/bash
# JOB 1335b = LEAK PROBE 1 for the SHOWOTHER gain (Hans 11/21 '점검 1 만들어'; code e59_leak_probe.py + e59_leak_analyze.py (new), e59_hidearms.py masks): E2E 3B at the FROZEN W0 weights (forward only, NO write), synthetic world v3_16_1 (the 1333 world), seeds 231..238 (fixed before the result), 128 chunks. Four attention conditions of the same stream: NAT, ALLHID, SHOWSAME, SHOWOTHER (masks in all 32 layers, as jobs 1333 / 1334); metric = NLL of the SECOND-mentioned entity of every fact sentence (the binding-dependent answer: place after the person name or person name after the place, both templates occur), mean over sentences of chunks >= 16, with the FIRST-mentioned entity as the vocabulary / format control. Main contrast SHOWOTHER - ALLHID (paired over seeds); SHOWSAME - ALLHID = copy bound. No-leak reading: SHOWOTHER ~ ALLHID on the second entity; leak reading: SHOWOTHER ~ SHOWSAME. EXPLORATORY; NOT queued until Hans' sentence; est. ~1.1 h (smoke ~6 min; per seed ~8 min: natural K,V record ~1.5 min + 4 forward passes of 128 chunks; unmeasured).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/leak_probe
mkdir -p $O/smoke
sha256sum $D/e59_leak_probe.py $D/e59_leak_analyze.py $D/e59_hidearms.py $D/e59_stage0_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_leak_probe.py --selftest | tail -1; $PY $D/e59_hidearms.py --selftest | tail -1
$PY $D/e59_leak_probe.py --seed 231 --out $O/smoke --chunks 24 --first 8
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/leak_probe_s231.json'));rows=[q for q in r['rows'] if q['k']>=8];C=('NAT','ALLHID','SHOWSAME','SHOWOTHER')
m={c:float(np.mean([q['second_'+c] for q in rows])) for c in C};f={c:float(np.mean([q['first_'+c] for q in rows])) for c in C}
assert all(math.isfinite(v) for v in list(m.values())+list(f.values())),(m,f)
assert m['NAT']<m['ALLHID']-0.5 and m['SHOWSAME']<m['ALLHID']-0.5,('copying from the visible same-fact sentences must lower the second-entity NLL',m)
print('LEAK PROBE SMOKE PASSED second-entity NLL',{k:round(v,2) for k,v in m.items()},'first-entity',{k:round(v,2) for k,v in f.items()})
PYE
rm -rf $O/smoke/states
for s in 231 232 233 234 235 236 237 238; do
  set +e
  $PY $D/e59_leak_probe.py --seed $s --out $O
  echo "RUN leak_probe seed $s rc=$?"
  rm -rf $O/states/s$s
  set -e
done
$PY $D/e59_leak_analyze.py --dir $O | tee $O/leak_probe_summary.txt
