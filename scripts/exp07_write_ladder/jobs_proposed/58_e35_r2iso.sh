#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 21 min in total (4 passes x 63 chunks, worlds A+B, save then eval; E35 design estimate; basis: R2 measured 133 s/pass in job 35, so 4 passes x 2 worlds ~ 18 min + the eval of 4 states per world) [estimate].
# E35 R2-iso (PROPOSED, NOT QUEUED beyond Hans's approval of the follow-ups): the R2 schedule (L0reset P=1: each 1024-token chunk of the fresh-background stream is one SGD step, fresh suffix state, chunk-relative positions, same order) but with attention
# blocked between rows (exposure sentences / background paragraphs) in prefix AND suffix layers; targets that cross a row boundary are not scored. R2 vs R2-iso = effect of the neighbouring rows inside the chunk; R1 vs R2-iso = bundling / fixed order / step composition.
# Runs through ladder.step (SGD, no optimizer state) like the L1 rows, not rt.update like R2; the optional control job 62 (--iso none) runs R2 through the same path to separate the path difference.
# Gate: the CPU row-segmentation and attention-mask check (no key of another row in the same chunk, expo sentences one row each) must pass (exit code 3 otherwise). Code: e35_ladder.py from the commit that adds --level L0iso (git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- scripts/exp07_write_ladder/e35_ladder.py); jobrunner env: $PY $S $OUT.
set -e
$PY $S/e35_ladder.py --iso-check --world-seed 101 --passes 4
for W in A B; do
  $PY $S/e35_ladder.py --level L0iso --iso rows --passes 4 --worlds $W --mode save --tag r2iso
  $PY $S/e35_ladder.py --level L0iso --iso rows --passes 4 --worlds $W --mode eval --tag r2iso
done
