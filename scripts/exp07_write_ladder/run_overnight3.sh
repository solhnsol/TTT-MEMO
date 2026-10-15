#!/bin/bash
# Queue 3 (approved 2026-10-14 14:55Z): repetition vs intensity in native TTT, world seed 101.
#  L0mp: native TTT re-reading the stream 8 times (eval after each pass)
#  L0x1: re-implemented native step, scale 1 (sanity: must match L0's end state readout)
#  L0x2/x4/x8: one pass with every step x2/x4/x8 (intensity control; overshoot checked by general-text NLL)
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
until grep -q QUEUE2_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
run(){ echo "$(date -u +%FT%TZ) START $*" >> $OUT/queue.log; $PY $S/ladder.py "$@" >> $OUT/run3_$(echo "$*" | tr ' -' '__').log 2>&1 && echo "$(date -u +%FT%TZ) DONE $*" >> $OUT/queue.log || echo "$(date -u +%FT%TZ) FAILED $*" >> $OUT/queue.log; }
run --level L0mp --world-seed 101 --passes 8
run --level L0x --world-seed 101 --scale 1 --tag x1
run --level L0x --world-seed 101 --scale 2 --tag x2
run --level L0x --world-seed 101 --scale 4 --tag x4
run --level L0x --world-seed 101 --scale 8 --tag x8
echo "$(date -u +%FT%TZ) QUEUE3_DONE" >> $OUT/queue.log
