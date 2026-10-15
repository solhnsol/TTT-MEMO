#!/bin/bash
# Overnight queue (2026-10-14): L0/L1 with fp32 packed R1 readout. Each step runs after the previous one; a failure
# is logged and the queue moves on. Results: experiments/exp07_write_ladder/
cd /home/solhnsol/workspace/TTT-MEMO
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export EXP07_OUT="$PWD/experiments/exp07_write_ladder"; OUT=$EXP07_OUT; mkdir -p $OUT; PY=.venv/bin/python; S=scripts/exp07_write_ladder
run(){ echo "$(date -u +%FT%TZ) START $*" >> $OUT/queue.log; $PY $S/ladder.py "$@" >> $OUT/run_$(echo "$*" | tr ' -' '__').log 2>&1 && echo "$(date -u +%FT%TZ) DONE $*" >> $OUT/queue.log || echo "$(date -u +%FT%TZ) FAILED $*" >> $OUT/queue.log; }
ana(){ $PY $S/analyze_ladder.py "$@" > $OUT/analyze_s$1_$2.log 2>&1; echo "$(date -u +%FT%TZ) ANALYZED $*" >> $OUT/queue.log; }
run --level L0 --world-seed 101
run --level L1 --world-seed 101 --epochs 32
ana 101 main
run --level L1 --world-seed 101 --epochs 32 --seed 1 --tag shuf1
ana 101 shuf1
for ws in 102 103; do run --level L0 --world-seed $ws; run --level L1 --world-seed $ws --epochs 32; ana $ws main; done
echo "$(date -u +%FT%TZ) QUEUE_DONE" >> $OUT/queue.log
