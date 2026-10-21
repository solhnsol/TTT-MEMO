#!/bin/bash
# NOT APPROVED (Hans approved only SENT and NOO of E44) - DO NOT QUEUE. Prepared code only. EXPECTED RUN TIME: about 50-55 min per cell [estimate, K6' job 64: 53.9 min]; bg then unif70 = ~1.8 h.
# bg (BG) = O + the window minus all non-O exposure-sentence tokens (background masked, B/P/X sentences visible). unif70 (UNIF70) = O + uniform fill outside O up to BG's per-chunk n_masked (the existing oraclefill with --fill-size-from bg; needs S3_bg_s101_{A,B}.saved.json in $OUT, so bg first).
# Same arguments as 87 (s101, worlds AB, passes 4, --keep). Tests: e44_sets_test.py.
set -e
$PY $S/e44_sets_test.py
for M in save eval; do
  $PY $S/s3.py --cell bg --world-seed 101 --passes 4 --worlds AB --keep --tag bg --mode $M
done
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-size-from bg --world-seed 101 --passes 4 --worlds AB --keep --tag unif70 --mode $M
done
