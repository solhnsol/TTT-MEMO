#!/bin/bash
# Exp05-A run3: readout comparison + null calibration + perturbation replicate (seed 101, deterministic XLA ops).
# Stops at the first failure. Results: experiments/exp05_write_dynamics/run3_readouts/
cd /home/solhnsol/workspace/TTT-MEMO
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export EXP05_OUT="$PWD/experiments/exp05_write_dynamics/run3_readouts"
OUT=$EXP05_OUT; PY=.venv/bin/python; S=scripts/exp05_write_dynamics
mkdir -p $OUT
$PY $S/readouts.py --seed 101 --tag main --worlds AB > $OUT/main.log 2>&1 || { echo "main failed (see main.log; exit 2 = hash mismatch vs run2)" > $OUT/FAILED; exit 1; }
$PY $S/readouts.py --seed 101 --tag perturb --worlds AB --perturb 3e-3 --states 15,34,63 --no-short > $OUT/perturb.log 2>&1 || { echo "perturb failed" > $OUT/FAILED; exit 1; }
$PY $S/analyze_readouts.py 101 > $OUT/analyze.log 2>&1 || { echo "analysis failed" > $OUT/FAILED; exit 1; }
echo ALL_DONE > $OUT/ALL_DONE
