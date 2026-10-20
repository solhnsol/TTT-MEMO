#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): NOT calibrated: 4 passes x worlds A+B, save then eval, same cost class as the earlier s3.py oracle-mask cells; take sec_pass / sec_eval from those cells' logs (e.g. the job 41/42 saved.json). Rough guess 1-2 h in total [estimate].
# E41 job 70: oraclefill 35% (O + uniform random fill) (PROPOSED, NOT QUEUED; needs Hans's approval). Save then eval, same arguments (jobrunner env: $PY $S $OUT).
set -e
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-frac 0.35 --world-seed 101 --passes 4 --worlds AB --keep --tag fill35 --mode $M
done
