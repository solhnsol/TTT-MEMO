#!/bin/bash
# After the day queue: re-run the unit checks on the current grad_components.py, then the N-sweep (PROTOCOL_stage01.md).
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python
lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
until grep -q DAY2_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
cp $OUT/grad_checks_s101.json $OUT/grad_checks_s101_run2.json
lg START grad_checks_rerun; GRAD_CHECK=1 $PY $S/grad_components.py > $OUT/st2_grad_checks_rerun.log 2>&1
if grep -q '"ALL_PASS": true' $OUT/grad_checks_s101.json; then lg "PASS grad_checks_rerun sha256 $(sha256sum $S/grad_components.py | cut -c1-64)"
  lg START nsweep; NSWEEP=1 $PY $S/grad_components.py > $OUT/nsweep.log 2>&1 && lg DONE nsweep || lg FAILED nsweep
else lg "FAIL grad_checks_rerun (nsweep not run)"; fi
lg NSWEEP_QUEUE_DONE
