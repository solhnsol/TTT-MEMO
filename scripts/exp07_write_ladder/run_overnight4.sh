#!/bin/bash
# Queue 4 (replaces queues 2-3): every streaming L0-family run in two processes (save inner weights at the evaluation
# points; then score them with the fp32 packed readout), because stream + update + fp32 readout OOM in one process.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
# queue 1 was stopped after L1 s102 (its L0 s103 would OOM); wait for the running L1 s102 to finish
while pgrep -f "ladder.py --level L1 --world-seed 102" >/dev/null; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
two(){ for mode in save eval; do echo "$(date -u +%FT%TZ) START $mode $*" >> $OUT/queue.log
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/run4_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then echo "$(date -u +%FT%TZ) DONE $mode $*" >> $OUT/queue.log; else echo "$(date -u +%FT%TZ) FAILED $mode $*" >> $OUT/queue.log; return; fi; done; }
run1(){ echo "$(date -u +%FT%TZ) START $*" >> $OUT/queue.log; $PY $S/ladder.py "$@" >> $OUT/run4_$(echo "$*" | tr " -" "__").log 2>&1 && echo "$(date -u +%FT%TZ) DONE $*" >> $OUT/queue.log || echo "$(date -u +%FT%TZ) FAILED $*" >> $OUT/queue.log; }
two --level L0 --world-seed 101
run1 --level L1 --world-seed 103 --epochs 32
$PY $S/analyze_ladder.py 101 main > $OUT/analyze_s101_main.log 2>&1; $PY $S/analyze_ladder.py 101 shuf1 > $OUT/analyze_s101_shuf1.log 2>&1
two --level L0mp --world-seed 101 --passes 8
for sc in 1 2 4 8; do two --level L0x --world-seed 101 --scale $sc --tag x$sc; done
$PY $S/analyze_multipass.py 101 > $OUT/analyze_multipass_s101.log 2>&1
for ws in 102 103; do two --level L0 --world-seed $ws; $PY $S/analyze_ladder.py $ws main > $OUT/analyze_s${ws}_main.log 2>&1; done
echo "$(date -u +%FT%TZ) QUEUE4_DONE" >> $OUT/queue.log
