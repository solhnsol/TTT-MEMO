#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 40-50 min GPU [estimate as K3 save + eval, ~2 x 20 min for worlds A+B, 4 passes, --keep]. CPU steps ~3 min.
# E46 K3 token-kind split, cell 'frm' (O + FRM: the earlier same-triple sentences without their occupation words (frame words) are masked in addition to name and place) on seed 101 (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread). Masked set, chunks WITH an oracle set only (same-triple earlier sentences inside the window, before the fact sentence's place token): O = name + place tokens, OCC = occupation tokens of those sentences, FRM = their remaining tokens; O + OCC + FRM = K3 (asserted per chunk on the real table in the CPU step).
# The job installs its own code (step 0 fetches the branch, step 1 checks sha256 against the manifest below, exit 5 on mismatch; NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check). Output: $OUT/S3_frm_s101_{A,B}.json (.saved.json), meta_S3_frm_s101.json, states under $OUT/states/S3_frm_s101/{A,B}/. The DCLM first-chunk damage read of the pass-4 states is a separate proposal.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
FILES="$D/s3.py $D/s3_sets.py $D/e43_sets_test.py $D/e44_sets_test.py $D/e46_sets_test.py $D/e46_counts.py $D/e42_k2_tokens.py $D/e37_firstchunk_nll.py "
if [ -z "$NO_INSTALL" ]; then git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- $FILES; fi
if [ -z "$SKIP_VERIFY" ]; then
  while read H F; do
    [ -n "$H" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$H" ] || { echo "VERSION MISMATCH $F: have $G, expected $H"; exit 5; }
  done <<'MANIFEST'
77ce2c79464e s3.py
5072cfae0d81 s3_sets.py
a256dbbcb9ec e43_sets_test.py
3dc2e49e33fc e44_sets_test.py
0b7f4b351d23 e46_sets_test.py
b8e94c1cfb4d e46_counts.py
9153e937b8c1 e42_k2_tokens.py
e167e94bc935 e37_firstchunk_nll.py

MANIFEST
  echo "installed code verified against the manifest"
fi
# CPU (run BEFORE queueing, the same lines): unit tests and the per-chunk masked counts by kind on the real sentence table
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/e46_sets_test.py
$PY $S/e46_counts.py --selftest
$PY $S/e46_counts.py --seeds 101 --passes 4 --worlds AB --out-dir ${E46_COUNTS_DIR:-/tmp/e46_counts}
[ -n "$CPU_ONLY" ] && exit 0
for M in save eval; do
  $PY $S/s3.py --cell frm --world-seed 101 --passes 4 --worlds AB --keep --tag frm --mode $M
done
