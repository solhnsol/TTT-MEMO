#!/bin/bash
# JOB 1123 = HOLD-OUT world s74 (v3_32_1) with the chosen rule ABS tau 2 only (Hans 11/06 10:29 KST, '\ABS2로 가자'): arms M3W1N05,M3W8N05,M3T05W16N05,M3TA2W16N05 via job 1070's script (TAUS=2). Run once; no other rule is put on the hold-out worlds.
set -eo pipefail
PY=$(realpath -s "$PY"); export PY
cd "$(dirname "$0")/../../.." 2>/dev/null || true
SEED=74 CELL=v3_32_1 TAUS=2 ARMS=M3W1N05,M3W8N05,M3T05W16N05,M3TA2W16N05 bash scripts/exp07_write_ladder/jobs_proposed/1070_e57_syn_rules.sh

