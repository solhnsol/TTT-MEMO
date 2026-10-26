#!/bin/bash
# JOB 245 = E55 32.12b GROUPED CROSS-VALIDATION AND LEAKAGE CONTROLS of the learned head weighting (job 242 part 2; CPU only, NO GPU, NO model, NO download). (a) e55_sentframe.py regenerates the world of seed 61 and writes the sentence-frame id of every window sentence (fact ids asserted equal to the saved labels); (b) e55_sentgrid_lofo.py evaluates the logistic regression grouped by frame (K = 10 folds, no frame shared between train and test), grouped by fact, leave-one-chunk-out (reference), with PERMUTED training labels (control) and with features from non-fact / fact query rows only. Labels are used in training in every row; evaluation is out-of-fold.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, CPU only, about 5-15 min (frames about 1 min; 8 evaluations x 2 states x 10 folds of a 768-feature regression, job 242 needed 2.5 min for 24 evaluations of 4 folds). Commit 475ccbe908988dbf74d126b863914cac55e49960. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder;COMMIT=475ccbe908988dbf74d126b863914cac55e49960;OUT=${EXP07_OUT:-experiments/exp07_write_ladder}
if [ -z "$NO_INSTALL" ]; then git cat-file -e $COMMIT^{commit} || { echo "commit missing (exit 4)"; exit 4; }; git checkout $COMMIT -- $D/e55_sentframe.py $D/e55_sentgrid_lofo.py; git checkout 9b9364319dde87d5366ac7c725e645c326f1ba8b -- $D/e55_sentgrid_ext.py; fi
for x in "a4757d5ed117 e55_sentframe.py" "6659efd1d87a e55_sentgrid_lofo.py" "7de0dd3b97f0 e55_sentgrid_ext.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (exit 5)"; exit 5; }; done
$PY $D/e55_sentgrid_ext.py --selftest | tail -1
$PY $D/e55_sentgrid_lofo.py --selftest | tail -1
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: selftests done, analysis skipped"; exit 0; }
for f in e55_sentdump_s61.npz e55_sentdump_labels_s61.npz; do [ -f $OUT/$f ] || { echo "$f missing (exit 7)"; exit 7; }; done
REP=$OUT/e55_sentgrid_lofo_s61.md; [ -f $REP ] && [ -z "$RERUN" ] && { echo "report exists (exit 8)"; exit 8; }
$PY $D/e55_sentframe.py --seed 61 --out $OUT
$PY $D/e55_sentgrid_lofo.py --dump $OUT/e55_sentdump_s61.npz --labels $OUT/e55_sentdump_labels_s61.npz --frames $OUT/e55_sentdump_frames_s61.npz --out $REP | tail -50
