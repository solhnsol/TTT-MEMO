#!/bin/bash
# E55 design 27.8 Q1-A (can the inner step make the binding rise at fewer exposures?): the ISOLATED-ROW training of stage 0 / job 185 (same world v3_16_1 of job 181: 16 facts, 128 exposures per fact, 300-frame bank, pooled background; every exposure sentence its own row "\n\n" + sentence, loss on its tokens, shuffled within a round, packed as in job 185; cumulative update norm of every segment matched to the NAT run of the same seed (job 181 files); same reads, same canary), seeds 61 62 63, reads at a = 16 / 32 / 64 / 128 (out-of-window, 16 candidates, W0 subtracted, unexposed question frames). Five arms in one process (one model load, one W0 read per seed):
#   ISO-STD   the plain inner SGD step, RERUN on this pin so that the comparison is paired (not reused from job 185)
#   ISO-SUR1  token weight proportional to the token's own CE (stop-gradient), normalised to mean 1 over the chunk's loss tokens, capped at 5 (cap after the normalisation), loss = sum(w nll)/n
#   ISO-SUR2  the same with CE^2
#   ISO-ADAM  Adam-style inner state (beta 0.9 / 0.999, eps 1e-8, bias correction) over the whole run on the clipped gradient; direction only, the step size is set by the cumulative-norm matching (each step rescaled to the NAT increment of its segment / number of steps), as for STD
#   ISO-MOM   momentum 0.9 (heavy-ball sum) on the clipped gradient, same rescaling
# Report $OUT/e55_q1a_report.md (NEW name): u(a) per arm and seed + pooled, minimum exposure count with u >= 0.15, paired difference to ISO-STD, canary rise and clip share, and the pre-set success rule (u(32) >= 0.15 in at least 2 of 3 seeds AND pooled u(128) >= STD - 0.05 AND clip share < 0.9, canary rise <= +0.10). Exploratory (3 seeds, 16 facts).
# The attention diagnostic of design 27.8 is NOT in this file (separate job file, same commit family).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 1.2-1.8 h: CPU part 5-10 min; model + fp32 reader load 7-20 min once; per seed W0 read 1-2 min; per arm 53 packed training steps (2-3 min incl. the surprise-weight backward) + 4 frozen-weight reads (about 2-6 min) + canary = about 6-10 min per (arm, seed); 5 arms x 3 seeds = 15 runs. (Job 185: 3 seeds x 1 arm took about 15 min of GPU work after loading.)
# Needs job 181 finished (NAT files, tag v3a, seeds 61-63; exit 7 if missing). Outputs: $OUT/E55_q1a_<ISO-STD|ISO-SUR1|ISO-SUR2|ISO-ADAM|ISO-MOM>_s<seed>.saved.json / .read.json, e55_q1a_report.md. NO OVERWRITE: finished (arm, seed) files are reused; exit 8 if the report exists unless RERUN=1.
# CPU-only dry run (also runs the arm / argument validation of q1a through --check-only): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_q1a FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_q1a/data && cd /tmp/dry_q1a && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/192_e55_q1a_inner_step.sh   (then git worktree remove --force /tmp/dry_q1a)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: after job 190 (needs the NAT files of job 181). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=085b2813d503def954a3c3ac8920ef07c389bbc1
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="a16e6dd0d4c9"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  [ "$S3H" = "$S3_OLD" ] || [ "$S3H" = "$S3NEW" ] || { echo "lab s3.py ($S3H) is neither the committed old version ($S3_OLD) nor the new one ($S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
535d927e2004 e55_run.py
00fbe5110e78 e55_core.py
9827e34dcde6 e55_gpu.py
71eb12a3913f e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
d82b51b8e199 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
f261120d0f03 s3.py
1746ce4c159c ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in 61 62 63; do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_q1a_s$SD.md > $OUT/e55_gates_q1a_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_q1a_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_q1a_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py q1a --cell v3_16_1 --tag q1a --bg pool --out $OUT --seeds 61,62,63 --read-apps 16,32,64,128 --norm-files "$OUT/E55_v3a_NAT_s{seed}.saved.json" --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_q1a_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py q1a --cell v3_16_1 --tag q1a --bg pool --out $OUT --seeds 61,62,63 --read-apps 16,32,64,128 --norm-files "$OUT/E55_v3a_NAT_s{seed}.saved.json"
$PY $S/e55_ic_report.py --q1a --dir $OUT --seeds 61,62,63 --tag q1a --out $REP
tail -30 $REP
