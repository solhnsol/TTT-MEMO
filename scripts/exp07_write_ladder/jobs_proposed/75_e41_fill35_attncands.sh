#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 55-65 min in total (4 passes, worlds A+B, save then eval); basis: E39 measured about 52 min for the oracle-type cells (job 70 estimate) [estimate]. Every chunk runs the masked path, as in prefixall/oracle.
# E42 optional U35e: oraclefill 35% with the fill drawn only from the attention cells' candidate set (positions >= 4 and < chunk start - 128), so U35 and A35 differ only in how the fill is chosen. (PROPOSED, NOT QUEUED; needs Hans's approval). Needs s3.py / s3_sets.py from the commit that adds the cells (git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- scripts/exp07_write_ladder/s3.py scripts/exp07_write_ladder/s3_sets.py); jobrunner env: $PY $S $OUT.
set -e
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-frac 0.35 --fill-attn-cands --world-seed 101 --passes 4 --worlds AB --keep --tag fill35e --mode $M
done
