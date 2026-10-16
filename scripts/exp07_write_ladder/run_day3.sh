#!/bin/bash
# Day queue 3 (main-track must-haves, PROTOCOL_stage01.md 2026-10-16). Part 1: s103 pilot (states kept).
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
one(){ mode=$1;shift;lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/n3_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return 1; fi; }
P="--level L0mw --world-seed 103 --passes 8 --block 1024 --scale 2 --tag x2"
one save $P && one eval $P --keep && { $PY $S/analyze_pilot.py 103 > $OUT/analyze_pilot_s103.log 2>&1; lg ANALYZED pilot 103; }
lg DAY3_PART1_DONE
