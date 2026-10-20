#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): well under 1 h (stated below) [estimate].
# E37b read job (PROPOSED, NOT QUEUED): the same 200 DCLM samples (first 1024 tokens, no context) on W0 and on the chunk-only masked pass-4 states written by job 61.
# Output: $OUT/e37_firstchunk_e37b.json / .csv (columns sample, token_offset, W0, mwfs_s101_A, mwfs_s101_B, W0_repeat). Must run after 61 (name order). Expected well under 1 h.
set -e
A=$OUT/states/L0mwfs_mwfskeep_s101/A/state_004.npz;B=$OUT/states/L0mwfs_mwfskeep_s101/B/state_004.npz
[ -f "$A" ] && [ -f "$B" ] || { echo "missing state files from job 61: $A $B"; exit 3; }
$PY $S/e37_firstchunk_nll.py --ids data/dclm_val_subset/val_c0_200x8193.u32le.bin --seed 101 --tag e37b --state mwfs_s101_A=$A --state mwfs_s101_B=$B
