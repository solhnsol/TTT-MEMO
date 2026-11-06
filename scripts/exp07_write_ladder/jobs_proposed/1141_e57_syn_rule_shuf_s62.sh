#!/bin/bash
# JOB 1141 = job 1140 (pair-shuffle control of the rule arms) on the synthetic tune world s62 (v3_16_1)
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=62 CELL=v3_16_1 bash scripts/exp07_write_ladder/jobs_proposed/1140_e57_syn_rule_shuf.sh
