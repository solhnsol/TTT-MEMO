#!/bin/bash
# Re-run of the suppression screen (s101-103) with the OOM-fixed grad_components.py (PROTOCOL note 2026-10-16 15:50 KST).
# Waits until the day3d/day3c queue is finished (DAY3C_DONE) so no two 3B jobs share the GPU.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
lg "SCREEN2 waiting for the day3c end marker"
until grep -qE "Z DAY3C_DONE$" $OUT/queue.log; do sleep 60; done
for sd in 101 102 103; do [ -f $OUT/onestep2_screen_s$sd.log ] && mv $OUT/onestep2_screen_s$sd.log $OUT/onestep2_screen_s${sd}_oom.log; done
for sd in 101 102 103; do lg START screen2 s$sd; SEED=$sd ONESTEP2=1 STAGE=screen $PY $S/grad_components.py > $OUT/onestep2_screen_s$sd.log 2>&1 && lg DONE screen2 s$sd || lg FAILED screen2 s$sd; done
$PY $S/analyze_onestep2.py screen > $OUT/onestep2_screen.log 2>&1; lg "ANALYZED screen2 $(grep -o 'SCREEN [A-Z]*' $OUT/onestep2_screen.md | head -1)"
lg SCREEN2_DONE
