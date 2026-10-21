#!/bin/bash
# generator (not a job): writes 97_e46_occ.sh, 98_e46_frm.sh, 99_e46_unif_occ.sh from one template with the current sha256 manifest
cd "$(dirname "$0")/.."
D=scripts/exp07_write_ladder
FILES="s3.py s3_sets.py e43_sets_test.py e44_sets_test.py e46_sets_test.py e46_counts.py e42_k2_tokens.py e37_firstchunk_nll.py"
FL=""; for F in $FILES; do FL="$FL\$D/$F "; done
MAN=""; for F in $FILES; do MAN="$MAN$(sha256sum $F | cut -c1-12) $F
"; done
gen() { # num cell name label dclm
cat > jobs_proposed/$1_e46_$3.sh <<JOB
#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 40-50 min GPU [estimate as K3 save + eval, ~2 x 20 min for worlds A+B, 4 passes, --keep]$5. CPU steps ~3 min.
# E46 K3 token-kind split, cell '$2' ($4) on seed 101 (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread). Masked set, chunks WITH an oracle set only (same-triple earlier sentences inside the window, before the fact sentence's place token): O = name + place tokens, OCC = occupation tokens of those sentences, FRM = their remaining tokens; O + OCC + FRM = K3 (asserted per chunk on the real table in the CPU step).
# The job installs its own code (step 0 fetches the branch, step 1 checks sha256 against the manifest below, exit 5 on mismatch; NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check). Output: \$OUT/S3_$2_s101_{A,B}.json (.saved.json), meta_S3_$2_s101.json, states under \$OUT/states/S3_$2_s101/{A,B}/. The DCLM first-chunk damage read of the pass-4 states is a separate proposal.
set -e
cd "\$(dirname "\$0")/../../.." 2>/dev/null || true
D=$D
FILES="$FL"
if [ -z "\$NO_INSTALL" ]; then git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- \$FILES; fi
if [ -z "\$SKIP_VERIFY" ]; then
  while read H F; do
    [ -n "\$H" ] || continue
    G=\$(sha256sum \$D/\$F | cut -c1-12)
    [ "\$G" = "\$H" ] || { echo "VERSION MISMATCH \$F: have \$G, expected \$H"; exit 5; }
  done <<'MANIFEST'
$MAN
MANIFEST
  echo "installed code verified against the manifest"
fi
# CPU (run BEFORE queueing, the same lines): unit tests and the per-chunk masked counts by kind on the real sentence table
\$PY \$S/e43_sets_test.py
\$PY \$S/e44_sets_test.py
\$PY \$S/e46_sets_test.py
\$PY \$S/e46_counts.py --selftest
\$PY \$S/e46_counts.py --seeds 101 --passes 4 --worlds AB --out-dir \${E46_COUNTS_DIR:-/tmp/e46_counts}
[ -n "\$CPU_ONLY" ] && exit 0
for M in save eval; do
  \$PY \$S/s3.py --cell $2 --world-seed 101 --passes 4 --worlds AB --keep --tag $2 --mode \$M
done
JOB
}
gen 97 occ occ "O + OCC: the occupation words of the earlier same-triple sentences are masked in addition to name and place" ""
gen 98 frm frm "O + FRM: the earlier same-triple sentences without their occupation words (frame words) are masked in addition to name and place" ""
gen 99 unif_occ unif_occ "O + the same number of tokens as OCC drawn uniformly at random from the window outside O and OCC: size control for occ" ""
