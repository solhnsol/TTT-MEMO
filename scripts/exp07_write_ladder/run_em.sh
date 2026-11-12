#!/bin/bash
# EM queue (PROTOCOL_stage01.md "Greedy EM"): after the N-sweep. L1 s101 ck EM; re-save native (L0x scale 1) and L0fs
# s101 weights (kept) and score EM on them.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
until grep -q NSWEEP_QUEUE_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
lg START em L1ck; $PY $S/em_eval.py L1ck > $OUT/em_L1ck.log 2>&1 && lg DONE em L1ck || lg FAILED em L1ck
lg START save emx1; $PY $S/ladder.py --mode save --level L0x --world-seed 101 --scale 1 --tag emx1 > $OUT/em_save_x1.log 2>&1 && lg DONE save emx1 || lg FAILED save emx1
lg START em L0x emx1; $PY $S/em_eval.py saved L0x emx1 > $OUT/em_L0x.log 2>&1 && lg DONE em L0x || lg FAILED em L0x
lg START save emfs; $PY $S/ladder.py --mode save --level L0fs --world-seed 101 --passes 8 --tag emfs > $OUT/em_save_fs.log 2>&1 && lg DONE save emfs || lg FAILED save emfs
lg START em L0fs emfs; $PY $S/em_eval.py saved L0fs emfs > $OUT/em_L0fs.log 2>&1 && lg DONE em L0fs || lg FAILED em L0fs
lg EM_DONE
