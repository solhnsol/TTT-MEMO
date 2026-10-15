#!/bin/bash
# Exp05-A re-run with deterministic XLA ops. Stops at the first failed gate. Results: experiments/exp05_write_dynamics/run2_detops/
cd /home/solhnsol/workspace/TTT-MEMO
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export EXP05_OUT="$PWD/experiments/exp05_write_dynamics/run2_detops"
OUT=$EXP05_OUT; PY=.venv/bin/python; T=scripts/exp05_write_dynamics/trajectory.py
$PY $T --seed 101 --max-chunks 4 --tag chk1 --worlds A --no-carry --no-v3 > $OUT/chk1.log 2>&1 || { echo "chk1 crashed" > $OUT/GATE_FAILED; exit 1; }
$PY $T --seed 101 --max-chunks 4 --tag chk2 --worlds A --no-carry --no-v3 > $OUT/chk2.log 2>&1 || { echo "chk2 crashed" > $OUT/GATE_FAILED; exit 1; }
$PY scripts/exp05_write_dynamics/v1_short_check.py > $OUT/short_gate.log 2>&1 || { echo "short gate failed (determinism)" > $OUT/GATE_FAILED; exit 1; }
echo "short gate PASSED" > $OUT/GATE1_PASSED
$PY $T --seed 101 --tag full --worlds AB > $OUT/full.log 2>&1 || { echo "full crashed" > $OUT/GATE_FAILED; exit 1; }
$PY $T --seed 101 --tag repeat --worlds A --no-carry > $OUT/repeat.log 2>&1 || { echo "repeat crashed" > $OUT/GATE_FAILED; exit 1; }
echo ALL_DONE > $OUT/ALL_DONE
