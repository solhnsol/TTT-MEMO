#!/bin/bash
# E41 job 71: attnfill 35% (O + highest head-max attention positions outside O) (PROPOSED, NOT QUEUED; needs Hans's approval). Save then eval, same arguments (jobrunner env: $PY $S $OUT).
set -e
for M in save eval; do
  $PY $S/s3_attn.py --cell attnfill --fill-frac 0.35 --world-seed 101 --passes 4 --worlds AB --keep --tag attnfill35 --mode $M
done
