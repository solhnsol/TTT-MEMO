#!/bin/bash
# JOB 1222 = online g-prime arms on tune world s72 (v3_32_1), via job 1220 (Hans 11/07 approved)
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=72 CELL=v3_32_1 bash scripts/exp07_write_ladder/jobs_proposed/1220_e57_syn_online_gprime.sh
