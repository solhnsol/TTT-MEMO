#!/bin/bash
# E35 R4 (reset every 8 chunks) (PROPOSED, NOT QUEUED; needs Hans's approval; run right after job 51 = piece B). Same arguments as job 51 (E35 3.2: L0reset, passes 4, save -> eval, world A then B),
# except --reset-period 8 --tag r4p8. If job 51's actual command line has further flags (e.g. --log-clip, --world-seed), add them here too (diff against jobs/done/51_*.sh).
# Name chosen as 55_ so that it sorts after 53 (E35 D') and before 60 in both C and locale sort (a letter suffix like 51a could sort BEFORE 51_<name> under a locale that ignores punctuation).
set -e
for W in A B; do
  $PY $S/e35_ladder.py --level L0reset --reset-period 8 --passes 4 --worlds $W --mode save --tag r4p8
  $PY $S/e35_ladder.py --level L0reset --reset-period 8 --passes 4 --worlds $W --mode eval --tag r4p8
done
