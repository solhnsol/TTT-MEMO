#!/bin/bash
# Follow-up queue: re-run L0 s101 (first attempt OOM in the fp32 readout, fixed with GP=2), then re-analyse s101.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
until grep -q QUEUE_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
echo "$(date -u +%FT%TZ) START --level L0 --world-seed 101 (retry)" >> $OUT/queue.log
$PY $S/ladder.py --level L0 --world-seed 101 >> $OUT/run_L0_s101_retry.log 2>&1 && echo "$(date -u +%FT%TZ) DONE L0 s101 retry" >> $OUT/queue.log || echo "$(date -u +%FT%TZ) FAILED L0 s101 retry" >> $OUT/queue.log
for t in main shuf1; do $PY $S/analyze_ladder.py 101 $t > $OUT/analyze_s101_$t.log 2>&1; done
echo "$(date -u +%FT%TZ) QUEUE2_DONE" >> $OUT/queue.log
