#!/bin/bash
cd /home/solhnsol/workspace/TTT-MEMO
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform
OUT=experiments/exp05_write_dynamics
# full seed-101 A and B (probes, V3, C5), then V1 determinism repeat of world A (probes, no carry/V3)
.venv/bin/python scripts/exp05_write_dynamics/trajectory.py --seed 101 --tag full --worlds AB >> $OUT/full.log 2>&1
.venv/bin/python scripts/exp05_write_dynamics/trajectory.py --seed 101 --tag repeat --worlds A --no-carry --no-v3 >> $OUT/repeat.log 2>&1
echo ALL_DONE >> $OUT/repeat.log
