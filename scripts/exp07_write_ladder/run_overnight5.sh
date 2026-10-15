#!/bin/bash
# Filler (same pre-registered designs, more seeds) only if queue 4 ends early: start a job only if it can finish by ~00:00 UTC (09:00 KST).
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
until grep -q QUEUE4_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
deadline=$(date -u -d 'tomorrow 00:00' +%s); [ $(date -u +%H) -lt 12 ] && deadline=$(date -u -d 'today 00:00' +%s)
fits(){ [ $(( $(date -u +%s) + $1*60 )) -le $deadline ]; }
two(){ for mode in save eval; do echo "$(date -u +%FT%TZ) START $mode $*" >> $OUT/queue.log
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/run5_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then echo "$(date -u +%FT%TZ) DONE $mode $*" >> $OUT/queue.log; else echo "$(date -u +%FT%TZ) FAILED $mode $*" >> $OUT/queue.log; return; fi; done; }
ev(){ echo "$(date -u +%FT%TZ) START eval $*" >> $OUT/queue.log; $PY $S/ladder.py --mode eval "$@" >> $OUT/run5_eval_$(echo "$*" | tr ' -' '__').log 2>&1 && echo "$(date -u +%FT%TZ) DONE eval $*" >> $OUT/queue.log || echo "$(date -u +%FT%TZ) FAILED eval $*" >> $OUT/queue.log; }
# re-score L0mp s101 (first eval failed on a missing method; saved states are intact), then re-run the multi-pass analysis
ev --level L0mp --world-seed 101 --passes 8
$PY $S/analyze_multipass.py 101 > $OUT/analyze_multipass_s101.log 2>&1
for ws in 102 103; do fits 85 && two --level L0mp --world-seed $ws --passes 8; done
echo "$(date -u +%FT%TZ) QUEUE5_DONE" >> $OUT/queue.log
