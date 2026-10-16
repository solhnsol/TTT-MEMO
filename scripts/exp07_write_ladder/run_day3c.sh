#!/bin/bash
# after run_day3b.sh: long-context NLL for the s103 masked x2 states (first attempt failed on a check bug: the check ran on
# the W0 state; fixed in longctx_nll.py). States were hard-linked to states_hold before the queue deleted them.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
until grep -q DAY3B_DONE $OUT/queue.log; do sleep 60; done
rm -rf $OUT/states/L0mw_x2_s103;mv $OUT/states_hold/L0mw_x2_s103 $OUT/states/;rm -f $OUT/longnll_L0mw_x2_s103.json
lg START longnll_L0mw_x2_s103 retry; $PY $S/longctx_nll.py 103 L0mw x2 > $OUT/n3_longnll_L0mw_x2_s103_retry.log 2>&1 && lg DONE longnll_L0mw_x2_s103 retry || lg FAILED longnll_L0mw_x2_s103 retry
rm -rf $OUT/states/L0mw_x2_s103;lg "DELETED states L0mw_x2_s103";lg DAY3C_DONE
