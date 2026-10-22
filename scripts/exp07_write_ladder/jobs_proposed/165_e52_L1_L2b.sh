#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 0.7-0.9 h [same call as job 162: 41-49 min measured for 162-type L1 calls on 10/21-22 (162 itself 03:37-04:18Z = 41 min), + CPU steps and the pair analysis (~2-3 min); slack ~30%].
# E52 Level-2, offline L1 control on layout L2b (PROPOSED, NOT QUEUED; needs Hans's approval written in the experiment-run thread). Same settings as job 162 (L1 doc rows = each exposure unit of the stream 9201 alone, plain SGD, worlds A and B, 8 epochs, evaluation at W0 and epochs 1, 2, 3, 8, default l1-variant doc), only the layout differs: L2b instead of L2a, tag e52l1b. Purpose: within-pair Delta of L1 (L1 on L2a minus L1 on L2b), computed exactly like L0's (e52_analyze.py, cell L1), to separate the placement effect from the shadowing share (E52 section 13).
# Needs the output of job 162 in $OUT (L1_e52l1_s9201_A.json / _B.json, meta_nat_e52l1_s9201.json); without them the final analysis step is skipped with a message.
# NO OVERWRITE: exit 8 if $OUT/L1_e52l1b_s9201_A.json or _B.json already exists unless RERUN=1. Output: $OUT/L1_e52l1b_s9201_{A,B}.json, meta_nat_e52l1b_s9201.json, $OUT/e52_L1_pair_analysis.md. No filler (no pass-2 pool). No sealed stream is read.
# CPU-only dry run: CPU_ONLY=1 bash <this file> (checkout, world sha check, selftests, per-person table; then exits before any GPU step).
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
NAT_DIR=${NAT_DIR:-data/natural-text/worlds/e52}
mkdir -p $OUT
TOK=${TOKENIZER_JSON:-data/tokenizer/llama3/tokenizer.json}
COMMIT=bd03b998986bdcf4d24743c3fa1909a848c619d5
FILES="scripts/exp07_write_ladder/nat_world.py scripts/exp07_write_ladder/nat_filler.py scripts/exp07_write_ladder/nat_pass.py scripts/exp07_write_ladder/nat_ladder.py scripts/exp07_write_ladder/nat_analyze.py scripts/exp07_write_ladder/nat_dryrun.py scripts/exp07_write_ladder/nat_signal.py scripts/exp07_write_ladder/e51_nat_table.py scripts/exp07_write_ladder/e52_table.py scripts/exp07_write_ladder/e52_analyze.py"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $FILES
  git diff --quiet $COMMIT -- $FILES || { echo "checked-out files differ from commit $COMMIT"; exit 4; }
fi
sha() { [ "$(sha256sum "$NAT_DIR/$1" | cut -d' ' -f1)" = "$2" ] || { echo "world file $1 does not match the pinned sha256 $2: not run (exit 5)"; exit 5; }; }
sha nat_9201_L2a.json 93c9347e1c212fbc54e2594cb85268d471343c9e30873af556dfcbbbc34aa735
sha nat_9201_L2b.json d3f2669d698787071ed30c492ef6626f031911aa55950708ae4d7fcefb7efa01
[ -z "$RERUN" ] && for W in A B; do [ -f $OUT/L1_e52l1b_s9201_$W.json ] && { echo "$OUT/L1_e52l1b_s9201_$W.json exists: not overwritten (exit 8; RERUN=1 to redo)"; exit 8; }; done
$PY $S/nat_filler.py --selftest
$PY $S/nat_pass.py --selftest
$PY $S/e52_analyze.py --selftest | tail -1
$PY $S/e52_table.py --nat-dir $NAT_DIR --stream 9201 --require-real --tokenizer-file $TOK --out $OUT/e52_person_table_9201.tsv
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before GPU"; exit 0; }
COMMON="--nat 9201:L2b --nat-dir $NAT_DIR --cells L1 --epochs 8 --l1-eval-epochs 1,2,3,8 --worlds AB --tag e52l1b"
$PY $S/nat_ladder.py $COMMON --mode save
$PY $S/nat_ladder.py $COMMON --mode eval
if [ -f $OUT/L1_e52l1_s9201_A.json ] && [ -f $OUT/L1_e52l1_s9201_B.json ]; then
  $PY $S/e52_analyze.py --dir $OUT --nat-dir $NAT_DIR --stream 9201 --tags e52l1,e52l1b --cell L1 --out $OUT/e52_L1_pair_analysis.md
else
  echo "analysis skipped: the job-162 outputs (L1 on L2a) are not in $OUT"
fi
