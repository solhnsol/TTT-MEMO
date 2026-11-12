#!/bin/bash
# label-free v2 selector diagnostic (W0 forwards only), after the 2x2 kappa checks
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
until grep -qE "Z S2X2CHECK_DONE$" $OUT/queue.log; do sleep 60; done
lg START seldiag; $PY $S/sel_diag.py > $OUT/v2_selector_diagnostic.log 2>&1 && lg "DONE seldiag $(grep -o 'SELECTOR [a-z]*' $OUT/v2_selector_diagnostic.log)" || lg FAILED seldiag
lg SELDIAG_DONE
