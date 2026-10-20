#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): world A alone: writes about 15-25 min (job 35 P=64 measurement: pass 1 147 s, later passes ~97 s; 8 passes, plus --log-clip's extra forward/backward per chunk) + eval of 9 states (2x job 51's per-world eval time, read sec_eval there) + first-chunk NLL of 3 states (job 62 estimates well under 1 h for 2 states) [estimates]. Total estimate: about 1-2 h.
# E35 5.11 step 2, world A: R5 (job 51) extended to passes 1-8, tag r5fresh8 (PROPOSED, NOT QUEUED; needs Hans's approval together with R3/R4 = 54/55; the other world is 57_e35_r5p8_B.sh). Split by world to fit the 1-2 h limit.
# No resume-from-state path exists, so this is a full 8-pass run; its passes 1-4 must reproduce job 51 (the pass 1-4 fresh pools do not depend on --passes): compare the per-pass inner_hash with L0reset_P63_r5fresh_s101_A.json afterwards.
# --log-clip on (E35 5.11 criterion: clip ratio >= 0.4 per pass needs per-chunk clip coefficients; one extra forward/backward per chunk). If job 51's real command line differs (flags), diff against jobs/done/51_*.sh and copy it.
# The run aborts before any GPU work if the fresh books run out for 8 passes or the audit fails. Both worlds write the same meta file (identical content).
# --keep: the eval keeps the pass states (states/L0reset_P63_r5fresh8_s101/A/state_00k.npz, k = pass, 000 = W0) because the first-chunk NLL read below needs passes 4, 6, 8 (about 8 states of inner weights per world on disk; delete state_001/002/003/005/007 afterwards if space is short).
# The first-chunk NLL (E35 design thread, 'longnll' read for passes 4, 6, 8 = e37_firstchunk_nll.py on the 200 DCLM samples as job 62) is a separate step that does NOT fail the job; the canary (gen pg11 + neutral tail) is in the eval itself.
set -e
$PY $S/e35_ladder.py --level L0reset --reset-period 63 --passes 8 --worlds A --mode save --tag r5fresh8 --log-clip
$PY $S/e35_ladder.py --level L0reset --reset-period 63 --passes 8 --worlds A --mode eval --tag r5fresh8 --log-clip --keep
D=$OUT/states/L0reset_P63_r5fresh8_s101/A
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
if [ -f "$IDS" ] && [ -f "$D/state_004.npz" ] && [ -f "$D/state_006.npz" ] && [ -f "$D/state_008.npz" ]; then
  $PY $S/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e35r5p8_A --state r5p4_A=$D/state_004.npz --state r5p6_A=$D/state_006.npz --state r5p8_A=$D/state_008.npz || echo "first-chunk NLL read failed (canary-only result stands)"
else
  echo "first-chunk NLL skipped: missing $IDS or pass states under $D (canary-only result stands)"
fi
