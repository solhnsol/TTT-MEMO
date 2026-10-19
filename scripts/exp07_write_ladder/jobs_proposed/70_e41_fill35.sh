#!/bin/bash
# E41 job 70: oraclefill 35% (O + uniform random fill) (PROPOSED, NOT QUEUED; needs Hans's approval). Save then eval, same arguments (jobrunner env: $PY $S $OUT).
set -e
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-frac 0.35 --world-seed 101 --passes 4 --worlds AB --keep --tag fill35 --mode $M
done
