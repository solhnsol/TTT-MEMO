#!/bin/bash
# Replication of the stage-0b result on world seeds 102 and 103 (reviewer requirement): L1 re-run with canaries and
# R1-alt, then L0mp re-scored with R1-alt; analysis per seed (PROTOCOL_stage01.md rules).
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
until grep -q STAGE23_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
run1(){ lg START "$@"; $PY $S/ladder.py "$@" >> $OUT/repl_$(echo "$*" | tr ' -' '__').log 2>&1 && lg DONE "$@" || lg FAILED "$@"; }
two(){ for mode in save eval; do lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/repl_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return; fi; done; }
for ws in 102 103; do
  run1 --level L1 --world-seed $ws --epochs 32 --tag ck
  two --level L0mp --world-seed $ws --passes 8 --tag alt
  $PY $S/analyze_stage01.py $ws > $OUT/analyze_stage01_s$ws.log 2>&1; lg ANALYZED repl $ws
done
lg REPL_DONE
