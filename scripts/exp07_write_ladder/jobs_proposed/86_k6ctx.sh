#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 95-115 min GPU [estimate, not calibrated]: the K6' cell's own time (~70-90 min per the design note: 4 passes x worlds A+B, save then eval) + the window-context-gain reads at W0 on every pass stream, 63 chunks x (native read + chunk-alone read) ~3 min per pass and world = ~24 min.
# PROPOSED, NOT QUEUED (no approval). Label-free K6' (cell k6ctx): per chunk, delta = mean NLL with the 8K window context - mean NLL of the chunk alone, both read at W0 on the pass stream (the ctx_gain.py read); if delta < tau the chunk is masked whole (prefix-all, exactly the K6' mask), else nothing is masked. tau = -0.65 fixed before the run.
#   No oracle labels are used for the selection; the log records n_O / has_O / ctx_delta / selected per chunk so the agreement with the oracle chunks can be tabulated afterwards. Selection weights: W0 (not the written weights) so the rule is the one calibrated on job 77 (AUC 1.000 on s101-103, world A). chunk 0 is never selected.
#   Everything else (lr, clip, streams, passes, readouts) as the K6' job: s101, passes 4, worlds AB, --keep.
# Code: s3.py (cell k6ctx, --ctx-tau), s3_sets.py (select_by_ctx, pure; unit test e43_sets_test.py incl. 'a selected chunk is masked exactly like K6'' and the stored hashes of the existing cells). GPU path unrun.
set -e
$PY $S/e43_sets_test.py
for M in save eval; do
  $PY $S/s3.py --cell k6ctx --ctx-tau ${TAU:--0.65} --world-seed 101 --passes 4 --worlds AB --keep --tag k6ctx --mode $M
done
