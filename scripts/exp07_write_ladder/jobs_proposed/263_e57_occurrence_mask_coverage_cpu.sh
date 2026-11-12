#!/bin/bash
# JOB 263 = E57 POST-HOC EXPLORATORY, CPU ONLY (no GPU): (1) occurrences of every answer string in the stream (variants, repeats inside an article) next to n_k; (2) coverage of the ORC oracle mask and what else it hides. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY"); cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; COMMIT=e6fd285ee566144555ae580a0970f9b7e4a9f91a
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; BUNDLE=${BUNDLE:-$OUT/e57_news}; PH=$OUT/e57_posthoc; mkdir -p "$PH"
git cat-file -e $COMMIT^{commit} || { echo "commit missing (exit 4)"; exit 4; }
git checkout $COMMIT -- $D/e57/e57_occurrence_count.py $D/e57/e57_mask_coverage.py
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
$PY $D/e57/e57_occurrence_count.py --bundle "$BUNDLE" --out "$PH/e57_occurrence_count.json"
$PY $D/e57/e57_mask_coverage.py --bundle "$BUNDLE" --out "$PH/e57_mask_coverage"
