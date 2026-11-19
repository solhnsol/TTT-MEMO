#!/bin/bash
# JOB 1325 = REVERSE-question re-read of the saved E2E E59 states (NO training for NAT / NATF / CHK / CHKF; Hans 11/19): seeds 71..74, saved weights at exposures 16 / 32 / 64 / 128 are scored with the reverse items (place -> person, 16 name candidates; scripts/exp07_write_ladder/e59_rev_reread.py) + ISO re-run with the inline reverse readout (ISO saves no states; seeds 71..74, direction rescaled to the saved NAT norms as in 1314; ~3 min per seed). EXPLORATORY; secondary readout; NOT queued until Hans' sentence; est. ~1.2 h (reverse re-read: model load ~1.5 min + 4 arms x 4 states ~ 1.5 min per state-read => ~12 min per seed; ISO re-run ~6 min per seed). Output experiments/exp07_write_ladder/e59_rev (files E59_e59_<arm>_s<seed>.rev.read.json).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 23"; exit 23; }
D=scripts/exp07_write_ladder;E=$OUT/e59;O=$OUT/e59_rev
mkdir -p $O
sha256sum $D/e59_rev.py $D/e59_rev_reread.py $D/e59_srv_run.py $D/e59_srv_core.py
$PY $D/e59_rev.py --selftest | tail -1; $PY $D/e59_rev_reread.py --selftest | tail -1
for s in 71 72 73 74; do
  set +e
  $PY $D/e59_rev_reread.py --seed $s --arms NAT,NATF,CHK,CHKF --e59-dir $E --tag e59 --out $O
  echo "RUN rev reread seed $s rc=$?"
  $PY $D/e59_srv_run.py --seed $s --out $O/iso --arms ISO --tag e59 --ref-dir $E --reverse
  echo "RUN rev iso seed $s rc=$?"
  set -e
done
