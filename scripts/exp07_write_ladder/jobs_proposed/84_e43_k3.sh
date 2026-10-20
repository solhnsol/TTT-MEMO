#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): K3 alone about 50-60 min GPU [estimate, from the E41 U35 job 70 (52 min, same cell class); K3 masks fewer tokens than U35, so the cost is not larger]. Optional cells: K3b +~55 min, K3f +~55 min (each its own cell, same arguments); the CPU steps take ~2 min.
# E43 sentence-cover oracle (PROPOSED, NOT QUEUED; needs Hans's approval). K3 = on chunks WITH an oracle set: O plus every other token of the earlier-round sentences of the same triple (window, before the fact sentence's place token); chunks without O mask nothing. s101, worlds A+B, passes 1-4, --keep.
#   Cells: k3 (default), k3b = K3 plus the sentences of the same pair with another triple (CELLS="k3 k3b"), k3f = K3 plus a uniform fill up to A35's per-chunk masked count (needs S3_attnfill35_s101_{A,B}.saved.json in $OUT; fill candidates = the attention cells' candidates as A35: --fill-attn-cands) (CELLS="k3 k3f").
# Code: s3_sets.py (Sets.sentencecover / sentencecover_fill, sentence_table gets a 'toks' field = all tokens of the sentence; existing cells bit-identical, checked by e43_sets_test.py against stored hashes), s3.py (cells k3 k3b k3f; per-chunk log gets n_O, has_O, n_sent). Server needs the repo files s3_sets.py s3.py e43_sets_test.py k3_counts.py e42_k2_tokens.py.
# Step 1 (CPU, run BEFORE queueing; no GPU): unit tests, then the per-chunk masked counts of K3 / K3b on the real sentence table: python3 $S/k3_counts.py --seeds 101 --passes 4 --worlds AB --out-dir /tmp/k3_counts  -> k3_counts.md (report to the design thread before the GPU steps).
set -e
CELLS=${CELLS:-k3}
$PY $S/e43_sets_test.py
$PY $S/k3_counts.py --selftest
$PY $S/k3_counts.py --seeds 101 --passes 4 --worlds AB --out-dir ${K3_COUNTS_DIR:-/tmp/k3_counts}
for C in $CELLS; do
  EXTRA=""
  [ "$C" = "k3f" ] && EXTRA="--fill-size-from attnfill35 --fill-attn-cands"
  for M in save eval; do
    $PY $S/s3.py --cell $C --world-seed 101 --passes 4 --worlds AB --keep --tag $C $EXTRA --mode $M
  done
done
