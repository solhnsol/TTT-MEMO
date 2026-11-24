#!/bin/bash
# JOB 1344m-RERUN = rerun of the seeds of job 1344m that failed (seed 237: AssertionError in the H prefix reads, the hidden answer tokens were not found as a token subsequence because the answer was tokenised differently inside the prefix sentence; fixed in e59_kvstep.py: the sentence of the same fact from the next candidate chunk is used, recorded in prefix_chunks). Same code, same 4 measured chunks 16, 48, 80, 112, SEEDS env var (default 237; add 238 if it failed too). The analysis at the end runs over all seeds present in the kvstep directory. NOT QUEUED: Hans decides. Est. ~70 min per seed.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 30"; exit 30; }
D=scripts/exp07_write_ladder;O=$OUT/kvstep;K=$OUT/kv1344k
mkdir -p $O/smoke
for s in ${SEEDS:-237}; do test -f $K/kv_s${s}_keys.npy || { echo "keys of job 1344b missing for seed $s: exit 31"; exit 31; }; done
sha256sum $D/e59_kvstep.py $D/e59_kvstep_analyze.py $D/e59_kvprobe.py $D/e59_shadow.py $D/e59_u12.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $D/e59_hidearms.py
$PY $D/e59_kvstep.py --selftest | tail -1
$PY $D/e59_kvprobe.py --selftest | tail -1
for s in ${SEEDS:-237}; do
  set +e
  $PY $D/e59_kvstep.py --seed $s --out $O --keys $K/kv_s${s}_keys.npy --chunks 16,48,80,112
  echo "RUN kvstep seed $s rc=$?"
  rm -rf $O/states/s${s}_*
  set -e
done
$PY $D/e59_kvstep_analyze.py --dir $O | tee $O/kvstep_summary.txt
