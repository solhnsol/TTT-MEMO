#!/bin/bash
# generator (not a job): writes 120_e47_regold.sh; code pinned to the commit H below (local checkout, no fetch)
cd "$(dirname "$0")/.."
H=df5d39cc4ce97cecf0e18f46567f1473eb74b6b5
D=scripts/exp07_write_ladder
FILES="s3.py s3_sets.py e43_sets_test.py e44_sets_test.py e46_sets_test.py e47_sets_test.py e42_k2_tokens.py k3_counts.py suppress.py ladder.py margins_split.py margins_split_test.py e47_triple_da.py e47_terms.py"
FL="";MAN="";for F in $FILES;do FL="$FL\$D/$F ";MAN="$MAN$(git show $H:$D/$F | sha256sum | cut -c1-12) $F
";done
LAD=$(git show $H:$D/ladder.py | sha256sum | cut -c1-12)
cat > jobs_proposed/120_e47_regold.sh <<JOB
#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): RE-SCORING ONLY, no streaming. Cells k3all, k3h, k3hc, SENT (10 saved states each) + native fskeep (18 states). Measured original eval times (sec_eval, s101): K3 183 s (A) + 149 s (B), k3h 182 + 149, SENT 182 + 148 = about 5.5 min per S3 cell (k3all and k3hc not read: assumed the same), 4 cells about 22 min; the evaluated L0fs_main (native) took 460 + 313 s = about 13 min; plus the model load of each s3.py / ladder.py call (about 1-2 min each, not measured). Total about 45 min (estimate from those figures). CPU steps ~1 min.
# E47 (design request: the four log-prob terms behind the binding margin): re-score the SAVED states of finished S3 s101 cells with the current ladder.evaluate, which additionally stores the gold / foil span log-probs (margins['_gold'], ['_foil'], ['_alt_gold'], ['_alt_foil']); margin = gold - foil unchanged. PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread.
# What it does: s3.py --mode eval --regold --keep for every cell in REGOLD_CELLS (default "k3all k3h k3hc sent": k3all (K3-all, job 110/112) is the reference cell instead of K3, per Hans 2026-10-21). Reads states/S3_<tag>_s101/{A,B}/ (never removed), writes S3_<tag>_s101_{A,B}.regold.json NEXT TO the untouched original result files (originals and .saved.json are not changed; the meta file is rewritten with identical content) and prints the max |difference| of every margin against the original result (stored as regold_check). NATIVE_TAG (default fskeep; NATIVE_TAG= empty skips it) additionally re-scores ladder.py --level L0fs --tag fskeep (states L0fs_fskeep_s101, 8 passes kept) -> L0fs_fskeep_s101_{A,B}.regold.json; fskeep was never scored before (only L0fs_fskeep_s101_{A,B}.saved.json and the states exist), so there is no original to compare; whether fskeep equals the earlier L0fs_main run is reported by the --compare step (max |alt margin difference| of fskeep@regold vs L0fs_main).
# PINNED INSTALL, NO FETCH: step 0 checks the lab ladder.py (must be the lab version b9fd20049420 the patch was made on, or the patched file; exit 6 otherwise), then checks the code out of the local repository at commit \$COMMIT (exit 4 if the commit is not in the local clone: fetch it first by hand); step 1 verifies sha256 (first 12 hex) of every file against the manifest and stops (exit 5) on a mismatch. NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py is VERIFIED ONLY (the lab copy is used).
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e47rg FETCH_HEAD && ln -s "\$(pwd)/data" /tmp/dry_e47rg/data && cd /tmp/dry_e47rg && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/120_e47_regold.sh   (afterwards: git worktree remove --force /tmp/dry_e47rg)
set -e
cd "\$(dirname "\$0")/../../.." 2>/dev/null || true
D=$D
COMMIT=$H
FILES="$FL"
LADDER_OK="b9fd20049420 $LAD"
REGOLD_CELLS="\${REGOLD_CELLS:-k3all k3h k3hc sent}"
NATIVE_TAG="\${NATIVE_TAG-fskeep}"
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
# CPU: unit tests, and per cell the original eval time and the saved states (read-only; skipped in the worktree dry run when OUT has no such files)
\$PY \$S/e43_sets_test.py
\$PY \$S/e44_sets_test.py
\$PY \$S/e47_sets_test.py
\$PY \$S/margins_split_test.py
\$PY \$S/e47_triple_da.py --selftest >/dev/null
\$PY \$S/e47_terms.py --selftest >/dev/null
for T in \$REGOLD_CELLS; do
  for W in A B; do
    echo "\$T \$W states: \$(ls \$OUT/states/S3_\${T}_s101/\$W 2>/dev/null | wc -l) files"
    \$PY -c "import json,sys;print('\$T \$W original sec_eval',json.load(open('\$OUT/S3_\${T}_s101_\$W.json')).get('sec_eval'))" || echo "\$T \$W: original result not readable here"
  done
done
[ -n "\$CPU_ONLY" ] && exit 0
for T in \$REGOLD_CELLS; do for W in A B; do N=\$(ls \$OUT/states/S3_\${T}_s101/\$W 2>/dev/null | wc -l); [ "\$N" -ge 5 ] || { echo "no saved states for \$T world \$W (\$N files): exit 7"; exit 7; }; done; done
[ -z "\$NATIVE_TAG" ] || [ "\$(ls \$OUT/states/L0fs_\${NATIVE_TAG}_s101/A 2>/dev/null | wc -l)" -ge 5 ] || { echo "no saved native states: exit 7"; exit 7; }
for T in \$REGOLD_CELLS; do
  \$PY \$S/s3.py --cell \$T --world-seed 101 --passes 4 --worlds AB --keep --tag \$T --mode eval --regold
done
if [ -n "\$NATIVE_TAG" ]; then
  \$PY \$S/ladder.py --level L0fs --tag \$NATIVE_TAG --world-seed 101 --worlds AB --keep --mode eval --regold
fi
# CPU after the GPU step: margin check and the four log-prob terms (the first cell is the reference of the change table)
CELLS=""; CMP=""; for T in \$REGOLD_CELLS; do CELLS="\$CELLS,\$T@regold"; CMP="\$CMP,\$T@regold=\$T"; done
if [ -n "\$NATIVE_TAG" ]; then CELLS="\$CELLS,L0fs:\$NATIVE_TAG@regold"; CMP="\$CMP,L0fs:\$NATIVE_TAG@regold=L0fs:main"; fi
\$PY \$S/e47_terms.py --cells "\${CELLS#,}" --ref "\${REGOLD_CELLS%% *}@regold" --seeds 101 --epochs 4 --compare "\${CMP#,}" --out-dir \${E47_TERMS_DIR:-/tmp/e47_terms}
JOB
