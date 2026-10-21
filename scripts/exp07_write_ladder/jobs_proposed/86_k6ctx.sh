#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 95-115 min GPU per cell [estimate, not calibrated]: the K6' cell's own time (~70-90 min per the design note: 4 passes x worlds A+B, save then eval) + the window-context-gain reads at W0 on every pass stream, 63 chunks x (native read + chunk-alone read) ~3 min per pass and world = ~24 min.
# PROPOSED, NOT QUEUED (no approval). Label-free K6' (cell k6ctx): per chunk, delta = mean NLL with the 8K window context - mean NLL of the chunk alone, both read at W0 on the pass stream (the ctx_gain.py read); if delta < tau the chunk is masked whole (prefix-all, exactly the K6' mask), else nothing is masked. tau = -0.38 fixed before the run (the earlier value -0.65 was wrong: job 77 split gave O-chunk delta -1.603..-0.450 and non-O minimum -0.316, so -0.38 separates them with a margin on both sides).
#   Alternative cell k6ctxq (CELLS="k6ctx k6ctxq" or CELLS=k6ctxq): no fixed threshold; per pass stream the round(0.30 x 62) = 19 chunks (of 1..62) with the lowest delta are masked whole (about the number of oracle chunks, 19-20).
#   No oracle labels are used for the selection; the log records n_O / has_O / ctx_delta / selected per chunk so the agreement with the oracle chunks can be tabulated afterwards. Selection weights: W0 (not the written weights) so the rule is the one calibrated on job 77 (AUC 1.000 on s101-103, world A). chunk 0 is never selected.
#   Everything else (lr, clip, streams, passes, readouts) as the K6' job: s101, passes 4, worlds AB, --keep.
# Code: s3.py (cell k6ctx, --ctx-tau), s3_sets.py (select_by_ctx, pure; unit test e43_sets_test.py incl. 'a selected chunk is masked exactly like K6'' and the stored hashes of the existing cells). GPU path unrun.
set -e
$PY $S/e43_sets_test.py
for C in ${CELLS:-k6ctx}; do
  for M in save eval; do
    $PY $S/s3.py --cell $C --ctx-tau ${TAU:--0.38} --ctx-frac ${FRAC:-0.30} --world-seed 101 --passes 4 --worlds AB --keep --tag $C --mode $M
  done
done
