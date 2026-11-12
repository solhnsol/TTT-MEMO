#!/bin/bash
# GPU idle watchdog (design 09:03 KST): every 60 s read GPU utilisation; if it has been 0 for > 10 min and the chain has not
# logged NIGHT_DONE, write "ALERT idle" to queue.log (once per idle episode). Uses no GPU.
Q=/home/solhnsol/workspace/TTT-MEMO/experiments/exp07_write_ladder/queue.log;idle=0;alerted=0
while true; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$u" = 0 ]; then idle=$((idle+1)); else idle=0;alerted=0; fi
  if [ $idle -ge 10 ] && [ $alerted = 0 ] && ! grep -qE "Z NIGHT_DONE$" $Q; then echo "$(date -u +%FT%TZ) ALERT idle: GPU utilisation 0 for ${idle} min; last chain line: $(grep -vE 'ALERT' $Q | tail -1 | cut -c1-100)" >> $Q;alerted=1; fi
  sleep 60
done
