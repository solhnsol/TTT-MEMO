#!/bin/bash
# JOB 1317 = E59 E2E seed addition: world seeds 73 / 74 (EXPLORATORY; same code as 1314: e59_srv_run.py; pre-set rule 'add seeds when the sign of a contrast differs across the first two seeds': in 1314 FACT-ISO, ISO-CHKF and I=(NAT-CHK)-(NATF-CHKF) differ between seeds 71 and 72; NOT queued until Hans' sentence; est. ~1.5 h): arms NAT (reference norms of the seed), NATF, CHK, CHKF, FACT, ISO per seed (NATREF / ISOORIG / noise-floor rerun of 1314 are not repeated). Output experiments/exp07_write_ladder/e59 (tag e59, same directory as seeds 71 / 72).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 15"; exit 15; }
D=scripts/exp07_write_ladder;O=$OUT/e59
sha256sum $D/e59_srv_core.py $D/e59_srv_run.py $D/e55_core.py $D/e55_iso.py $D/e55_run.py $D/e57_natmask_run.py $D/e57_qband_run.py
$PY $D/e59_srv_run.py --selftest | tail -2
for s in 73 74; do
  set +e
  $PY $D/e59_srv_run.py --seed $s --out $O --arms NAT,NATF,CHK,CHKF,FACT,ISO --tag e59
  echo "RUN e59 seed $s rc=$?"
  set -e
done
