#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): RE-SCORING ONLY, no streaming: per S3 cell 10 saved states (worlds A and B, passes 0-4) are scored again with the fp32 readout; the original eval of the cell took the 'sec_eval' of its result file (the CPU step prints it for every cell; use that sum as the estimate; my guess before reading it: 10-20 min per cell, NOT measured). Native (NATIVE_TAG=fskeep): 18 states (8 passes kept), about 1.8x one S3 cell. CPU steps ~1 min.
# E47 (design request: the four log-prob terms behind the binding margin): re-score the SAVED states of finished S3 s101 cells with the current ladder.evaluate, which additionally stores the gold / foil span log-probs (margins['_gold'], ['_foil'], ['_alt_gold'], ['_alt_foil']); margin = gold - foil unchanged. PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread.
# What it does: s3.py --mode eval --regold --keep for every cell in REGOLD_CELLS (default "k3 k3h sent"; k3hc only after job 111 has finished: REGOLD_CELLS="k3hc"). Reads states/S3_<tag>_s101/{A,B}/ (never removed), writes S3_<tag>_s101_{A,B}.regold.json NEXT TO the untouched original result files (originals and .saved.json are not changed; the meta file is rewritten with identical content) and prints the max |difference| of every margin against the original result (stored as regold_check). NATIVE_TAG=fskeep additionally re-scores ladder.py --level L0fs --tag fskeep (states L0fs_fskeep_s101, 8 passes kept) -> L0fs_fskeep_s101_{A,B}.regold.json; whether fskeep equals the earlier L0fs_main run is reported by the --compare step (max |alt margin difference| fskeep@regold vs main).
# PINNED INSTALL, NO FETCH: step 0 checks the lab ladder.py (must be the lab version b9fd20049420 the patch was made on, or the patched file; exit 6 otherwise), then checks the code out of the local repository at commit $COMMIT (exit 4 if the commit is not in the local clone: fetch it first by hand); step 1 verifies sha256 (first 12 hex) of every file against the manifest and stops (exit 5) on a mismatch. NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. suppress.py is VERIFIED ONLY (the lab copy is used).
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e47rg FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e47rg/data && cd /tmp/dry_e47rg && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/120_e47_regold.sh   (afterwards: git worktree remove --force /tmp/dry_e47rg)
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=ab3018b6decab6aea7bfe1ede55c3883659ce643
FILES="$D/s3.py $D/s3_sets.py $D/e43_sets_test.py $D/e44_sets_test.py $D/e46_sets_test.py $D/e47_sets_test.py $D/e42_k2_tokens.py $D/k3_counts.py $D/suppress.py $D/ladder.py $D/margins_split.py $D/margins_split_test.py $D/e47_triple_da.py $D/e47_terms.py "
LADDER_OK="b9fd20049420 714b08ea3b90"
REGOLD_CELLS="${REGOLD_CELLS:-k3 k3h sent}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); case " $LADDER_OK " in *" $GL "*) ;; *) echo "lab ladder.py ($GL) is neither the lab version the E47 gold/foil patch was made on (b9fd20049420) nor the patched file: not overwritten (exit 6), the code thread must rebase"; exit 6;; esac
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $(for F in $FILES; do [ "$F" = "$D/suppress.py" ] || echo $F; done)
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    if [ "$F" = suppress.py ]; then [ "$G" = "$HH" ] || echo "WARNING suppress.py differs from the repo copy: have $G, repo $HH (the lab copy is used)"; continue; fi
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH"; exit 5; }
  done <<'MANIFEST'
5442628d45b2 s3.py
93503d06d5ea s3_sets.py
a256dbbcb9ec e43_sets_test.py
3dc2e49e33fc e44_sets_test.py
0b7f4b351d23 e46_sets_test.py
99da9b569218 e47_sets_test.py
9153e937b8c1 e42_k2_tokens.py
fe0ba0b228a6 k3_counts.py
9dac9fdda99e suppress.py
714b08ea3b90 ladder.py
2638a505ef0c margins_split.py
1daebd776335 margins_split_test.py
900e1318361b e47_triple_da.py
7028ce5acb50 e47_terms.py

MANIFEST
  echo "installed code verified against the manifest"
fi
# CPU: unit tests, and per cell the original eval time and the saved states (read-only; skipped in the worktree dry run when OUT has no such files)
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/e47_sets_test.py
$PY $S/margins_split_test.py
$PY $S/e47_triple_da.py --selftest >/dev/null
$PY $S/e47_terms.py --selftest >/dev/null
for T in $REGOLD_CELLS; do
  for W in A B; do
    echo "$T $W states: $(ls $OUT/states/S3_${T}_s101/$W 2>/dev/null | wc -l) files"
    $PY -c "import json,sys;print('$T $W original sec_eval',json.load(open('$OUT/S3_${T}_s101_$W.json')).get('sec_eval'))" || echo "$T $W: original result not readable here"
  done
done
[ -n "$CPU_ONLY" ] && exit 0
for T in $REGOLD_CELLS; do
  $PY $S/s3.py --cell $T --world-seed 101 --passes 4 --worlds AB --keep --tag $T --mode eval --regold
done
if [ -n "$NATIVE_TAG" ]; then
  $PY $S/ladder.py --level L0fs --tag $NATIVE_TAG --world-seed 101 --worlds AB --keep --mode eval --regold
fi
# CPU after the GPU step: margin check and the four log-prob terms (the first cell is the reference of the change table)
CELLS=""; CMP=""; for T in $REGOLD_CELLS; do CELLS="$CELLS,$T@regold"; CMP="$CMP,$T@regold=$T"; done
if [ -n "$NATIVE_TAG" ]; then CELLS="$CELLS,L0fs:$NATIVE_TAG@regold"; CMP="$CMP,L0fs:$NATIVE_TAG@regold=L0fs:$NATIVE_TAG,L0fs:$NATIVE_TAG@regold=L0fs:main"; fi
$PY $S/e47_terms.py --cells "${CELLS#,}" --ref "${REGOLD_CELLS%% *}@regold" --seeds 101 --epochs 4 --compare "${CMP#,}" --out-dir ${E47_TERMS_DIR:-/tmp/e47_terms}
