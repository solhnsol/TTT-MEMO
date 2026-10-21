#!/bin/bash
# generator (not a job): writes 111_e47_k3hc.sh from the same template (earlier jobs stay pinned to their own commits); code pinned to commit 1f32a9f42545514586fee133068ff59b086595e6 (local checkout, no fetch)
cd "$(dirname "$0")/.."
H=1f32a9f42545514586fee133068ff59b086595e6
D=scripts/exp07_write_ladder
FILES="s3.py s3_sets.py e43_sets_test.py e44_sets_test.py e46_sets_test.py e47_sets_test.py e46_counts.py e47_counts.py e47_p_split.py e47_k3h_test.py e47_k3h_counts.py e47_triple_da.py e47_k3all_test.py e47_k3all_counts.py e47_k3hc_test.py e47_dump.py e47_native_da.py k3_counts.py e42_k2_tokens.py suppress.py"
FL="";MAN="";for F in $FILES;do FL="$FL\$D/$F ";MAN="$MAN$(git show $H:$D/$F | sha256sum | cut -c1-12) $F
";done
gen() { # num cell label
cat > jobs_proposed/$1_e47_$2.sh <<JOB
#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 40 min GPU [estimate as K3 save + eval, worlds A+B, 4 passes, --keep; k3hc masks about half of the K3 tokens]. CPU steps ~3 min.
# E47 SENT-gain split, cell '$2' ($3) on seed 101 (PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread; queue order: after the E46 jobs 97 / 98). Chunks WITH an oracle set only; the K3 set (O + the other tokens of the earlier same-triple sentences, before the fact sentence's place token) restricted as stated for the cell.
# PINNED INSTALL, NO FETCH: step 0 checks the code out of the local repository at commit \$COMMIT (exit 4 if the commit is not in the local clone: fetch it first by hand); step 1 verifies sha256 (first 12 hex) of every file against the manifest and stops (exit 5) on a mismatch. NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py is VERIFIED ONLY (the lab copy is used, a difference prints a WARNING).
# Output: \$OUT/S3_$2_s101_{A,B}.json (.saved.json), meta_S3_$2_s101.json, states under \$OUT/states/S3_$2_s101/{A,B}/ . The DCLM first-chunk damage read of the pass-4 states is a separate proposal (one run for several states).
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e47k3hc FETCH_HEAD && ln -s "\$(pwd)/data" /tmp/dry_e47k3hc/data && cd /tmp/dry_e47k3hc && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 E47_COUNTS_DIR=/tmp/dry_e47k3hc_counts bash scripts/exp07_write_ladder/jobs_proposed/$1_e47_$2.sh   (afterwards: git worktree remove --force /tmp/dry_e47k3hc)
set -e
cd "\$(dirname "\$0")/../../.." 2>/dev/null || true
D=$D
COMMIT=$H
FILES="$FL"
if [ -z "\$NO_INSTALL" ]; then
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
# CPU (the same lines in the dry run): unit tests and the masked-token counts (K3, k3h, k3hc) on the real sentence table, seed 101, worlds A and B, passes 1-4
\$PY \$S/e43_sets_test.py
\$PY \$S/e44_sets_test.py
\$PY \$S/e47_sets_test.py
\$PY \$S/e47_k3hc_test.py
\$PY \$S/e47_k3all_test.py
\$PY \$S/e47_k3all_counts.py --selftest
\$PY \$S/e47_dump.py --selftest
\$PY \$S/e47_k3h_counts.py --selftest
\$PY \$S/e47_triple_da.py --selftest >/dev/null
\$PY \$S/e47_k3h_counts.py --seeds 101 --passes 4 --worlds AB --out-dir \${E47_COUNTS_DIR:-/tmp/e47_k3hc_counts}
[ -n "\$CPU_ONLY" ] && exit 0
for M in save eval; do
  \$PY \$S/s3.py --cell $2 --world-seed 101 --passes 4 --worlds AB --keep --tag $2 --mode \$M
done
JOB
}
gen 111 k3hc "k3hc: the complement of k3h: the K3 share of the OTHER person of each pair (the one half_person_of did not choose); k3h and k3hc are disjoint and their union is K3 (asserted per chunk)"
