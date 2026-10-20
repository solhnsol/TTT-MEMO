#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 21 min in total (same cost as job 58) [estimate].
# E35 R2-iso CONTROL (OPTIONAL, PROPOSED, NOT QUEUED): the same step path as job 58 (ladder.step, SGD) but ONE segment per chunk, i.e. no isolation = R2 computed through the L1-style step instead of rt.update.
# R2 vs this = the path difference (no optimizer state, step-index handling, cross-chunk state); this vs job 58 = the isolation alone. Needs job 58's code. Tag r2step.
set -e
for W in A B; do
  $PY $S/e35_ladder.py --level L0iso --iso none --passes 4 --worlds $W --mode save --tag r2step
  $PY $S/e35_ladder.py --level L0iso --iso none --passes 4 --worlds $W --mode eval --tag r2step
done
