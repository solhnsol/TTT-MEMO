#!/bin/bash
# restart the driver until it writes DRIVER_DONE or the deadline (2026-10-13 05:45 UTC) passes
cd /home/solhnsol/workspace/TTT-MEMO
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
EXP=experiments/exp04_what_ttt_retains
DEADLINE=$(date -u -d "2026-10-13 05:45:00" +%s)
while [ ! -f $EXP/DRIVER_DONE ] && [ $(date -u +%s) -lt $DEADLINE ]; do
  echo "$(date -u) start driver" >> $EXP/watchdog.log
  .venv/bin/python scripts/exp04_what_ttt_retains/driver.py >> $EXP/driver_stdout.log 2>&1
  echo "$(date -u) driver exited rc=$?" >> $EXP/watchdog.log
  sleep 20
done
echo "$(date -u) watchdog finished" >> $EXP/watchdog.log
