#!/bin/bash
# wait for the in-flight Stage-3 seed to finish, then restart the watchdog/driver so the new plan (Stage 1H, clean seeds, weights-only readout) loads
cd /home/solhnsol/workspace/TTT-MEMO
EXP=experiments/exp04_what_ttt_retains
until grep -q "stage3 seed done 3" $EXP/driver_events.log; do sleep 15; done
sleep 5
for p in $(ps -eo pid,cmd | awk '/exp04_what_ttt_retains\/watchdog.sh/ && !/awk/ {print $1}'); do kill $p; done
for p in $(ps -eo pid,cmd | awk '/exp04_what_ttt_retains\/driver.py/ && !/awk/ {print $1}'); do kill $p; done
sleep 10
echo "$(date -u) restarting with diagnostics plan" >> $EXP/watchdog.log
rm -f $EXP/DRIVER_DONE
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
setsid nohup bash scripts/exp04_what_ttt_retains/watchdog.sh > /dev/null 2>&1 < /dev/null &
