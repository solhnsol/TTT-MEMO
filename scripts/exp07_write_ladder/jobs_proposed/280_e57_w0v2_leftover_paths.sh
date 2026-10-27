#!/bin/bash
# JOB 280 = E57 leftover answer paths in the ORC v2 deleted windows (CPU only, no GPU, no model; post-hoc, exploratory). Classification of what is left in the IN_DELORC_V2 windows of job 279: P1 other format, P2 referring, P3 near value, P4 indirect, P5 time cue, value sentences gold / distractor. 6 residual-gain questions + 6 controls.
set -eo pipefail
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
E=experiments/exp07_write_ladder
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY scripts/exp07_write_ladder/e57/e57_w0v2_paths.py --bundle $E/e57_news --w0mask $E/e57_w0mask/E57_w0mask.json --w0v2report $E/e57_w0v2/E57_w0v2_report.json --out ${OUT:-$E/e57_w0v2_paths/e57_w0v2_paths}
