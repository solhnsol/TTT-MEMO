#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): NOT calibrated: same cost class as job 70 plus the attention readout for the fill (attention maps per chunk); rough guess 1.5-3 h in total [estimate]; read the job 70 timing first.
# E41 job 71: attnfill 35% (O + highest head-max attention positions outside O) (PROPOSED, NOT QUEUED; needs Hans's approval). Save then eval, same arguments (jobrunner env: $PY $S $OUT).
set -e
for M in save eval; do
  $PY $S/s3_attn.py --cell attnfill --fill-frac 0.35 --world-seed 101 --passes 4 --worlds AB --keep --tag attnfill35 --mode $M
done
