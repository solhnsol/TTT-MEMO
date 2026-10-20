#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 55-65 min in total (4 passes, worlds A+B, save then eval); basis: E39 about 52 min for oracle-type cells, but ALL 63 chunks are masked here (job 72 masks only the 19 O chunks) [estimate].
# E41 U49 paired with attn-sel (ALTERNATIVE to job 72, PROPOSED, NOT QUEUED; replacing 72 needs Hans's approval): oraclefill with the per-chunk total = n_masked of S3_attnsel_p4_s101_{A,B}.saved.json in EVERY chunk:
# chunks with O -> O + uniform fill up to the attn-sel count; chunks WITHOUT O -> the attn-sel count of uniform positions of the window (job 72 masks nothing there: oraclefill returns [] when O is empty). Needs s3.py / s3_sets.py with --fill-no-o.
set -e
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-size-from attnsel_p4 --fill-no-o --world-seed 101 --passes 4 --worlds AB --keep --tag fillu49all --mode $M
done
