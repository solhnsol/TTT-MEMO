#!/bin/bash
# sequential GPU job runner (one job at a time), v2 = jobrunner.sh + event hooks (implementation thread 11/27; Hans 11/27 18:23 KST "안전장치 최대한 세워두자").
# Same behaviour as jobrunner.sh: runs experiments/exp07_write_ladder/jobs/NN_*.sh in name order as they appear, moves each finished job to jobs/done/, writes queue.log exactly as before,
# exits on jobs/STOP when the queue is empty. Added: one tab-separated line per event in queue_events.log (UTC time, event, job name, rc or detail):
#   RUNNER_START / QUEUED (a job file first seen in jobs/, while waiting or between jobs) / START / END (rc) / FAIL (rc != 0, a second line easy to grep) / RUNNER_STOP.
# JOBRUNNER_ROOT overrides the working tree (tests only).
cd "${JOBRUNNER_ROOT:-/home/solhnsol/workspace/TTT-MEMO}" || exit 2
OUT=experiments/exp07_write_ladder;J=$OUT/jobs;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
ev(){ printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:--}" "${3:--}" >> $OUT/queue_events.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
export OUT S=scripts/exp07_write_ladder PY=.venv/bin/python
declare -A SEEN
scan(){ local f n; for f in $(ls $J/[0-9]*.sh 2>/dev/null | sort); do n=$(basename $f); [ -n "${SEEN[$n]}" ] || { SEEN[$n]=1; ev QUEUED "$n"; }; done; }
ev RUNNER_START - "pid $$"
while true; do
  scan
  j=$(ls $J/[0-9]*.sh 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then [ -f $J/STOP ] && { lg JOBRUNNER_STOP; ev RUNNER_STOP; exit 0; }; sleep 30; continue; fi
  n=$(basename $j);lg "JOB START $n";ev START "$n";bash $j >> $OUT/jobs_$n.log 2>&1;rc=$?;lg "JOB END $n rc=$rc";ev END "$n" "rc=$rc";[ $rc -ne 0 ] && ev FAIL "$n" "rc=$rc";mv $j $J/done/;unset "SEEN[$n]"
done
