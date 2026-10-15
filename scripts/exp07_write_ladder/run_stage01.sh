#!/bin/bash
# Stage 0/1 queue (2026-10-15, PROTOCOL_stage01.md). World seed 101.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
run1(){ lg START "$@"; $PY $S/ladder.py "$@" >> $OUT/st01_$(echo "$*" | tr ' -' '__').log 2>&1 && lg DONE "$@" || lg FAILED "$@"; }
two(){ for mode in save eval; do lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/st01_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return; fi; done; }
run1 --level L1 --world-seed 101 --epochs 32 --tag ck --save-epochs 1,2,4,8,32
two --level L0mp --world-seed 101 --passes 8 --tag alt
$PY $S/analyze_stage01.py 101 > $OUT/analyze_stage01_s101.log 2>&1; lg ANALYZED stage0
two --level L0or --world-seed 101 --passes 8
two --level L0fs --world-seed 101 --passes 8
$PY $S/analyze_stage01.py 101 > $OUT/analyze_stage01_s101.log 2>&1; lg ANALYZED stage01
lg STAGE01_DONE
