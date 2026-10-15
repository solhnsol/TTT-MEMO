#!/bin/bash
# Stage 2 (gradient components) and stage 3 (K sweep) after stage 0/1 (PROTOCOL_stage01.md).
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
until grep -q STAGE01_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
lg START grad_components; $PY $S/grad_components.py > $OUT/st2_grad_components.log 2>&1 && lg DONE grad_components || lg FAILED grad_components
two(){ for mode in save eval; do lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/st3_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return; fi; done; }
for K in 2 4 8; do two --level L0k --world-seed 101 --k $K --tag k$K; done
$PY $S/analyze_ksweep.py 101 > $OUT/analyze_ksweep_s101.log 2>&1; lg STAGE23_DONE
