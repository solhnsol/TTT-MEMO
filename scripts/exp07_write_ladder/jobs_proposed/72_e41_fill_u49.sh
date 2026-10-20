#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): NOT calibrated: same cost as job 70 (the fill size comes from a saved.json, no extra compute); rough guess 1-2 h in total [estimate].
# E41 job 72 (optional): oraclefill with the per-chunk total taken from the attn-sel mask counts (S3_attnsel_p4_s101_{A,B}.saved.json) (PROPOSED, NOT QUEUED; needs Hans's approval). Save then eval, same arguments (jobrunner env: $PY $S $OUT).
set -e
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-size-from attnsel_p4 --world-seed 101 --passes 4 --worlds AB --keep --tag fillu49 --mode $M
done
