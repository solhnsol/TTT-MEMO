#!/bin/bash
# gradient-factor extraction (approved 22:36 KST; no verdict): last step of the overnight A100 chain, after NIGHT_DONE
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
until grep -qE "Z NIGHT_DONE$" $OUT/queue.log; do sleep 60; done
for sd in 101 102 103; do for kinds in ntp c2X,c2Y; do lg START gradfactors s$sd $kinds; SEED=$sd KINDS=$kinds $PY $S/gradfactors.py >> $OUT/gradfactors_s$sd.log 2>&1 && lg DONE gradfactors s$sd $kinds || { lg FAILED gradfactors s$sd $kinds; grep -q "RECONSTRUCTION CHECK FAILED" $OUT/gradfactors_s$sd.log && { lg "GRADFACTORS STOPPED: reconstruction check"; exit 1; }; }; done; done
lg GRADFACTORS_DONE
