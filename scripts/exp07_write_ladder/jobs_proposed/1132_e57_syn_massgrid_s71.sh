#!/bin/bash
# JOB 1132 = job 1130 (mass-matched grid: top 10% w=8.5, top 15% w=6, top 30% w=3.5; extra loss mass 0.75; tune world s71 v3_4_1 only; exploratory, no rule is chosen from it)
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=71 CELL=v3_4_1 bash scripts/exp07_write_ladder/jobs_proposed/1130_e57_syn_massgrid.sh
