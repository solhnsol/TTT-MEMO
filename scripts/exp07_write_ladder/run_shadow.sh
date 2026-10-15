#!/bin/bash
# Shadowing controls (PROTOCOL_stage01.md, registered 22:00 KST): after the EM queue.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
until grep -q EM_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
run1(){ lg START "$@"; $PY $S/ladder.py "$@" >> $OUT/sh_$(echo "$*" | tr ' -' '__').log 2>&1 && lg DONE "$@" || lg FAILED "$@"; }
lg START ctx3; CTX3=1 $PY $S/grad_components.py > $OUT/ctx3.log 2>&1 && lg DONE ctx3 || lg FAILED ctx3
run1 --level L1 --world-seed 101 --epochs 8 --tag irr --l1-variant irr
for ws in 102 103; do run1 --level L1 --world-seed $ws --epochs 8 --tag ctx --l1-variant ctx; run1 --level L1 --world-seed $ws --epochs 8 --tag irr --l1-variant irr; done
for ws in 101 102 103; do $PY $S/analyze_l1variants.py $ws > $OUT/analyze_l1variants_s$ws.log 2>&1; done
lg SHADOW_DONE
