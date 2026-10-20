#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): precision check about 20-30 min [estimate: 3 chunks x 2 worlds x 3 factor extractions + 3 cells x 4 scales x 2 worlds readouts];
# main swap (c2, s101, worlds AB, scales 1,2,4,8, first-order on) as job 60 was planned, plus per-chunk window prefix if win32 is chosen (+9 prefix forwards per chunk, a few min). Total about the job-60 estimate + 30 min [estimate; job-60 time is the lab's number].
# E29 swap rerun after the job-60 OOM (fp32 64K prefix asked for 992 GiB). PROPOSED, NOT QUEUED. Needs commit 8efe605 or later + the prefcheck/win32 commit.
# Step 1 (prefcheck): s101, worlds A and B, chunks 0,1,2 (window = whole stream so far, so win32 differs from the bf16 64K prefix only by precision): (i) factor relative error win32 vs bf16 (<= 1e-2),
#   (ii) D(win32,win32) - D(bf16,bf16) must be <= 10% of |D(M,M) - D(A,A)| at every scale. Output OUT/e29_swap_s101_prefcheck.{json,md} (manifest.prefcheck_verdict, prefcheck_factor_pass_1e-2).
# Step 2: main swap with the prefix precision chosen from step 1 (bf16 if both pass, else win32; the choice is recorded in manifest pref_dtype_A/B). Note: win32 truncates the prefix receptive field at the 9216-token window start for chunks >= 9 (the 64K bf16 prefix does not); this is a stated difference, not a precision difference.
set -e
$PY $S/e29_factors.py swap --seed 101 --worlds AB --chunks 0,1,2 --scales 1,2,4,8 --prefcheck
PREF=$(cat $OUT/e29_prefcheck_choice_s101.txt)
echo "prefix precision for the main swap: $PREF"
E29_PREF=$PREF $PY $S/e29_factors.py swap --seed 101 --worlds AB --chunks c2 --scales 1,2,4,8
