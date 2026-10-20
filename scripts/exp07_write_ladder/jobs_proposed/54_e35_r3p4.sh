#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): P=4, 4 passes, worlds A+B: writes about 10-20 min in total (job 35 measured 2.1 s/chunk at P=1, 63 chunks x 4 passes x 2 worlds; P=4 should not be slower) [estimate]; the eval of 4 states per world is not calibrated here: read sec_eval in job 50/51's json and multiply. Total estimate: under 1 h.
# E35 R3 (reset every 4 chunks) (PROPOSED, NOT QUEUED; needs Hans's approval; run right after job 51 = piece B). Same arguments as job 51 (E35 3.2: L0reset, passes 4, save -> eval, world A then B),
# except --reset-period 4 --tag r3p4. If job 51's actual command line has further flags (e.g. --log-clip, --world-seed), add them here too (diff against jobs/done/51_*.sh).
# Name chosen as 54_ so that it sorts after 53 (E35 D') and before 60 in both C and locale sort (a letter suffix like 51a could sort BEFORE 51_<name> under a locale that ignores punctuation).
set -e
for W in A B; do
  $PY $S/e35_ladder.py --level L0reset --reset-period 4 --passes 4 --worlds $W --mode save --tag r3p4
  $PY $S/e35_ladder.py --level L0reset --reset-period 4 --passes 4 --worlds $W --mode eval --tag r3p4
done
