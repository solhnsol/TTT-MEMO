#!/bin/bash
# JOB 242 = E55 32.12 TOKEN-SELECTION EXTENSIONS on the seed-61 sentence dump (CPU only, NO GPU, NO model, NO download). Methods: (1) surprisal filter / CE weighting of the candidate-1 votes, (2) learned head weighting (logistic regression, leave-one-chunk-out, LABELS USED IN TRAINING; frozen coefficients saved for later seeds), (3) neighbour expansion / pooling. Bar: lower mask fraction than candidate 1 at the same recall (cand1 seed 61: recall .90, mask ~.61). Seeds 62-64 have NO head dump (only masks), so frozen evaluation there needs a separate GPU dump job.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only, about 5-20 min (1.1 GB dump load, 2 states x 4 chunks, 12 regressions per state and chunk). Selftest on synthetic planted data takes 2 s.
set -eo pipefail
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder;COMMIT=9b9364319dde87d5366ac7c725e645c326f1ba8b;F=$D/e55_sentgrid_ext.py;OUT=${EXP07_OUT:-experiments/exp07_write_ladder}
if [ -z "$NO_INSTALL" ]; then git cat-file -e $COMMIT^{commit} || { echo "commit missing (exit 4)"; exit 4; }; git checkout $COMMIT -- $F; fi
G=$(sha256sum $F | cut -c1-12); [ "$G" = "7de0dd3b97f0" ] || { echo "VERSION MISMATCH $F: have $G (exit 5)"; exit 5; }
$PY $F --selftest | tail -1
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: selftest done, analysis skipped"; exit 0; }
[ -f $OUT/e55_sentdump_s61.npz ] && [ -f $OUT/e55_sentdump_labels_s61.npz ] || { echo "dump or label file missing (exit 7)"; exit 7; }
REP=$OUT/e55_sentgrid_ext_s61.md; [ -f $REP ] && [ -z "$RERUN" ] && { echo "report exists (exit 8)"; exit 8; }
$PY $F --dump $OUT/e55_sentdump_s61.npz --labels $OUT/e55_sentdump_labels_s61.npz --out $REP --save-coef $OUT/e55_sentgrid_ext_s61_coef.npz | tail -80
