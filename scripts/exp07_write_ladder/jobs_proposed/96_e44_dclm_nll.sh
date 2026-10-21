#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 14 min GPU for SENT [estimate from the design thread; model load ~5 min + W0 and 2 states]; with NOO=1 (+2 states, only after job 87 has produced states/S3_noo_s101) ~ +4 min. No training, no weights written.
# E44 first-chunk damage read (E37c method, e37_firstchunk_nll.py): per-sample NLL of the first 1024-token chunk of the DCLM validation subset under W0 (always included) and the saved PASS-4 states of the SENT cell (s101, worlds A and B); optional NOO pass-4 states. PROPOSED, NOT QUEUED (waits for Hans's approval). Tag e37g.
# The script is pinned to commit 061c9412015bb8b8ae9ab49540cc99735fe56e06: step 0 checks out e37_firstchunk_nll.py from that commit (NO_INSTALL=1 skips) and step 1 verifies its sha256 (first 12 hex e167e94bc935), exit 5 on a mismatch.
# Output: $OUT/e37_firstchunk_e37g.{json,csv} (labels SENT_s101_A_p4, SENT_s101_B_p4, optional NOO_s101_A_p4, NOO_s101_B_p4; columns W0 and W0_repeat).
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
if [ -z "$NO_INSTALL" ]; then git fetch origin claude/project-thread-rwhdac && git checkout 061c9412015bb8b8ae9ab49540cc99735fe56e06 -- $D/e37_firstchunk_nll.py; fi
G=$(sha256sum $D/e37_firstchunk_nll.py | cut -c1-12); [ "$G" = "e167e94bc935" ] || { echo "VERSION MISMATCH e37_firstchunk_nll.py: have $G, expected e167e94bc935"; exit 5; }
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
[ -f "$IDS" ] || { echo "$IDS missing"; exit 3; }
ARGS=""
for W in A B; do
  F=$OUT/states/S3_sent_s101/$W/state_004.npz; [ -f "$F" ] || { echo "missing $F (job sent not run or states not kept)"; exit 3; }
  ARGS="$ARGS --state SENT_s101_${W}_p4=$F"
done
if [ -n "$NOO" ]; then
  for W in A B; do
    F=$OUT/states/S3_noo_s101/$W/state_004.npz; [ -f "$F" ] || { echo "missing $F (job noo not finished)"; exit 3; }
    ARGS="$ARGS --state NOO_s101_${W}_p4=$F"
  done
fi
$PY $S/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e37g $ARGS
