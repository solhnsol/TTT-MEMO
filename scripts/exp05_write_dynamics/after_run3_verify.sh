#!/bin/bash
# waits for run3 to finish (ALL_DONE), then runs the packing equivalence check on the free GPU
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp05_write_dynamics/run3_readouts
until [ -f $OUT/ALL_DONE ] || [ -f $OUT/FAILED ]; do sleep 60; done
[ -f $OUT/FAILED ] && exit 1
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP05_OUT="$PWD/$OUT"
.venv/bin/python scripts/exp05_write_dynamics/verify_packing.py > $OUT/verify_packing.log 2>&1 && echo done > $OUT/VERIFY_DONE || echo failed > $OUT/VERIFY_FAILED
