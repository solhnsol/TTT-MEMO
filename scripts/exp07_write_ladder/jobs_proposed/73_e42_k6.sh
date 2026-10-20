#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 55-65 min in total (4 passes, worlds A+B, save then eval); basis: E39 measured about 52 min for the oracle-type cells (job 70 estimate) [estimate]. Every chunk runs the masked path, as in prefixall/oracle.
# E42 K6: chunks with an oracle set (19 of 63) -> oracle O; the other 44 chunks -> full masking of the window before the chunk (prefix-all-pos). (PROPOSED, NOT QUEUED; needs Hans's approval). Needs s3.py / s3_sets.py from the commit that adds the cells (git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- scripts/exp07_write_ladder/s3.py scripts/exp07_write_ladder/s3_sets.py); jobrunner env: $PY $S $OUT.
set -e
for M in save eval; do
  $PY $S/s3.py --cell k6  --world-seed 101 --passes 4 --worlds AB --keep --tag k6 --mode $M
done
