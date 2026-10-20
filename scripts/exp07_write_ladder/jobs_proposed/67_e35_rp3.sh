#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 1 h in total (4 passes, worlds A+B, save then eval; the same cost class as R3/R4 = jobs 54/55; the write time per chunk at P=3 is not slower than the measured 2.1 s/chunk at P=1) [estimate].
# E35 context-reset period P=3 (follow-up of R3/R4; PROPOSED, NOT QUEUED beyond Hans's approval). Same arguments as jobs 54/55: L0reset, passes 4, save -> eval, world A then B, --reset-period 3 --tag rp3. If job 51's real command line has further flags, add them here as in 54.
set -e
for W in A B; do
  $PY $S/e35_ladder.py --level L0reset --reset-period 3 --passes 4 --worlds $W --mode save --tag rp3
  $PY $S/e35_ladder.py --level L0reset --reset-period 3 --passes 4 --worlds $W --mode eval --tag rp3
done
