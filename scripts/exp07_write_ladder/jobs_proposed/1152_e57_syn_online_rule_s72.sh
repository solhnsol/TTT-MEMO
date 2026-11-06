#!/bin/bash
# JOB 1152 = job 1150 (online ABS2 rule) on the synthetic tune world s72 (v3_32_1)
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=72 CELL=v3_32_1 bash scripts/exp07_write_ladder/jobs_proposed/1150_e57_syn_online_rule.sh
