#!/bin/bash
# JOB 1072 = job 1070 (synthetic rule search, 6 arms) on the TUNE world s71 (v3_4_1); needs job 1060 (band-128 W0 pass, NAT norms). Hold-out worlds s65 / s73 / s74 are NOT run until Hans has chosen the rule.
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=71 CELL=v3_4_1 bash scripts/exp07_write_ladder/jobs_proposed/1070_e57_syn_rules.sh
