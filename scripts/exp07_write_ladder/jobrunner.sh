#!/bin/bash
# sequential GPU job runner (one job at a time): runs experiments/exp07_write_ladder/jobs/NN_*.sh in name order as they appear;
# each finished job is moved to jobs/done/. Jobs can be added or reordered (renamed) while the runner waits.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;J=$OUT/jobs;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
export OUT S=scripts/exp07_write_ladder PY=.venv/bin/python
while true; do
  j=$(ls $J/[0-9]*.sh 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then [ -f $J/STOP ] && { lg JOBRUNNER_STOP; exit 0; }; sleep 30; continue; fi
  n=$(basename $j);lg "JOB START $n";bash $j >> $OUT/jobs_$n.log 2>&1;rc=$?;lg "JOB END $n rc=$rc";mv $j $J/done/
done
