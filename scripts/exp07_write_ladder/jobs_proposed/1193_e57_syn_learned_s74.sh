#!/bin/bash
# JOB 1193 = learned-selector arms on hold-out world s74 (v3_32_1), via job 1190 (Hans 11/07 approved; see 1190)
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=74 CELL=v3_32_1 bash scripts/exp07_write_ladder/jobs_proposed/1190_e57_syn_learned.sh
