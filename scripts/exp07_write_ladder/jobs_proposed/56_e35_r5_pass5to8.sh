#!/bin/bash
# E35 5.11 step 2: R5 (job 51) extended to passes 1-8, both worlds (PROPOSED, NOT QUEUED; needs Hans's approval together with R3/R4 = 54/55).
# There is no resume-from-state path in e35_ladder.py, so this is a full 8-pass run with a NEW tag (r5fresh8): its passes 1-4 must reproduce job 51 (same fresh pools: books are taken in the seeded
# order, so the pass 1-4 background does not depend on --passes; compare the per-pass inner_hash of pass 1-4 with the L0reset_P63_r5fresh_s101_{A,B}.json entries afterwards). Job 51's files are not touched.
# Needs >= 7 disjoint fresh books x ~3 books (multi-book pools): run `e35_ladder.py --cpu-checks` / the audit with --passes 8 first; the run aborts before any GPU work if the books run out or the audit fails.
# --log-clip is on: the E35 5.11 criterion (spike = clip ratio >= 0.4 per pass) needs per-chunk clip coefficients (one extra forward/backward per chunk). If job 51's actual command line differs (flags, --world-seed), diff against jobs/done/51_*.sh and copy it.
# Order: save -> eval per world (the 64K stream + update + fp32 readout can OOM together), world A then B. Estimated ~2x job 51's time per world.
set -e
for W in A B; do
  $PY $S/e35_ladder.py --level L0reset --reset-period 63 --passes 8 --worlds $W --mode save --tag r5fresh8 --log-clip
  $PY $S/e35_ladder.py --level L0reset --reset-period 63 --passes 8 --worlds $W --mode eval --tag r5fresh8 --log-clip
done
