#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 55-65 min in total (4 passes, worlds A+B, save then eval; measured for R3 55 min and R4 65 min = jobs 54/55; the write time per chunk at P=2 is not slower than the measured 2.1 s/chunk at P=1) [estimate].
# E35 context-reset period P=2 (follow-up of R3/R4; PROPOSED, NOT QUEUED beyond Hans's approval). Same arguments as jobs 54/55: L0reset, passes 4, save -> eval, world A then B, --reset-period 2 --tag rp2. If job 51's real command line has further flags, add them here as in 54.
set -e
for W in A B; do
  $PY $S/e35_ladder.py --level L0reset --reset-period 2 --passes 4 --worlds $W --mode save --tag rp2
  $PY $S/e35_ladder.py --level L0reset --reset-period 2 --passes 4 --worlds $W --mode eval --tag rp2
done
