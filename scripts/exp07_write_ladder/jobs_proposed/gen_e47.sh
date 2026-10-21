#!/bin/bash
# generator (not a job): writes 101_e47_kp.sh, 102_e47_kx.sh, 103_e47_kp_u.sh, 104_e47_kx_u.sh, 105_e47_kx_s.sh, 106_e47_kx_o.sh from one template; code pinned to commit ab3018b6decab6aea7bfe1ede55c3883659ce643 (local checkout, no fetch)
cd "$(dirname "$0")/.."
H=ab3018b6decab6aea7bfe1ede55c3883659ce643
D=scripts/exp07_write_ladder
FILES="s3.py s3_sets.py e43_sets_test.py e44_sets_test.py e46_sets_test.py e47_sets_test.py e46_counts.py e47_counts.py k3_counts.py e42_k2_tokens.py suppress.py ladder.py margins_split.py margins_split_test.py"
FL="";MAN="";for F in $FILES;do FL="$FL\$D/$F ";MAN="$MAN$(git show $H:$D/$F | sha256sum | cut -c1-12) $F
";done
LAD=$(git show $H:$D/ladder.py | sha256sum | cut -c1-12)
gen() { # num cell label
cat > jobs_proposed/$1_e47_$2.sh <<JOB
#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 45-55 min GPU [estimate: K3 40 min to SENT 50-55 min, the masked set is K3 plus P or X tokens, worlds A+B, 4 passes, --keep]. CPU steps ~3 min.
# E47 SENT-gain split, cell '$2' ($3) on seed 101 (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread; queue order: after the E46 jobs 97 / 98). Masked set, chunks WITH an oracle set only: K3 (O + the other tokens of the earlier same-triple sentences, before the fact sentence's place token) plus the stated extra set from the exposure sentences inside the window before the chunk. P = tokens of sentences of the same pair, another triple (E44 type P); X = tokens of sentences of other pairs (E44 type X); the two do not overlap K3 (asserted per chunk, with SENT == K3 + P + X + later same-triple window tokens).
# PINNED INSTALL, NO FETCH: step 0 checks the code out of the local repository at commit \$COMMIT (exit 4 if the commit is not in the local clone: fetch it first by hand); step 1 verifies sha256 (first 12 hex) of every file against the manifest and stops (exit 5) on a mismatch. NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py is VERIFIED ONLY (the lab copy is used, a difference prints a WARNING).
# Output: \$OUT/S3_$2_s101_{A,B}.json (.saved.json), meta_S3_$2_s101.json, states under \$OUT/states/S3_$2_s101/{A,B}/ . The DCLM first-chunk damage read of the pass-4 states is a separate proposal (one run for several states).
# Gold / foil (E47): this commit also carries ladder.py (the lab version b9fd20049420 + an additive patch) and margins_split.py: the result files additionally hold margins['_gold'], ['_foil'], ['_alt_gold'], ['_alt_foil'] (summed span log-probs; margin = gold - foil); margins and every old field are unchanged (margins_split_test.py in the CPU step), GPU cost unchanged. The step 0 guard stops with exit 6 if the lab ladder.py is neither the base nor the patched file.
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e47 FETCH_HEAD && ln -s "\$(pwd)/data" /tmp/dry_e47/data && cd /tmp/dry_e47 && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 E47_COUNTS_DIR=/tmp/dry_e47_counts bash scripts/exp07_write_ladder/jobs_proposed/$1_e47_$2.sh   (afterwards: git worktree remove --force /tmp/dry_e47)
set -e
cd "\$(dirname "\$0")/../../.." 2>/dev/null || true
D=$D
COMMIT=$H
FILES="$FL"
LADDER_OK="b9fd20049420 $LAD"
if [ -z "\$NO_INSTALL" ]; then
  GL=\$(sha256sum \$D/ladder.py | cut -c1-12); case " \$LADDER_OK " in *" \$GL "*) ;; *) echo "lab ladder.py (\$GL) is neither the lab version the E47 gold/foil patch was made on (b9fd20049420) nor the patched file: not overwritten (exit 6), the code thread must rebase"; exit 6;; esac
  git cat-file -e \$COMMIT^{commit} || { echo "commit \$COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout \$COMMIT -- \$(for F in \$FILES; do [ "\$F" = "\$D/suppress.py" ] || echo \$F; done)
fi
if [ -z "\$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "\$HH" ] || continue
    G=\$(sha256sum \$D/\$F | cut -c1-12)
    if [ "\$F" = suppress.py ]; then [ "\$G" = "\$HH" ] || echo "WARNING suppress.py differs from the repo copy: have \$G, repo \$HH (the lab copy is used)"; continue; fi
    [ "\$G" = "\$HH" ] || { echo "VERSION MISMATCH \$F: have \$G, expected \$HH"; exit 5; }
  done <<'MANIFEST'
$MAN
MANIFEST
  echo "installed code verified against the manifest"
fi
# CPU (the same lines in the dry run): unit tests and the per-chunk masked counts by kind (P, X, same-frame share) on the real sentence table, seed 101, worlds A and B, passes 1-4
\$PY \$S/e43_sets_test.py
\$PY \$S/e44_sets_test.py
\$PY \$S/e47_sets_test.py
\$PY \$S/margins_split_test.py
\$PY \$S/e47_counts.py --selftest
\$PY \$S/e47_counts.py --seeds 101 --passes 4 --worlds AB --out-dir \${E47_COUNTS_DIR:-/tmp/e47_counts}
[ -n "\$CPU_ONLY" ] && exit 0
for M in save eval; do
  \$PY \$S/s3.py --cell $2 --world-seed 101 --passes 4 --worlds AB --keep --tag $2 --mode \$M
done
JOB
}
gen 101 kp "K3 + P: sentences of the same pair, another triple (the swap partner's other-person sentences) masked in addition to K3"
gen 102 kx "K3 + X: sentences of other pairs masked in addition to K3"
gen 103 kp_u "size control for kp: K3 + |P| tokens drawn uniformly from the window outside K3 and P"
gen 104 kx_u "size control for kx: K3 + |X| tokens drawn uniformly from the window outside K3 and X"
gen 105 kx_s "optional: K3 + the X sentences that use the same sentence frame as a fact sentence of the chunk"
gen 106 kx_o "optional: K3 + the X sentences that use another frame"
