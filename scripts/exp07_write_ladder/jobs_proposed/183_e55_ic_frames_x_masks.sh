#!/bin/bash
# E55 design 19 (R2 + R3), one seed (61), 6 cells in ONE process: frames {unique = every fact its own 64 of the 300 frames, shared64 = the same 64 frames for every fact (Latin assignment: each fact uses each frame once, the 16 sentences of a chunk have 16 different frames)} x arms {NAT = real window, KMASK = a query token inside the sentence of fact f does not see the EARLIER sentences of the same fact f in the 8-chunk window (background and other facts stay visible), SENT = no query sees any earlier fact sentence in the window (background stays visible; the sentences of the chunk being written are never hidden)}. Masks act on attention only (write path as in arm MSK); the written chunk is the real chunk in every arm. 16 facts, one sentence per fact in every chunk, 64 chunks (65.5k tokens), non-repeating pooled background (as job 181).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED. About 1.5-2.5 h on one A100 (the design thread's ~1 h is optimistic because this is 6 cells, each with its own K,V recording of 64 chunks): CPU part (selftests, 2 gate checks, mask dump, dry runs) 5-10 min; GPU unit checks 25-35 min (model load 13-23 min + 12 chunks x 6 runs; includes the mask checks U6); ONE run process (model + fp32 reader load 20-35 min once) then per cell 64 chunks at about 1.3 s (NAT, measured in 180a; the masked arms are guessed at 1.3-2 s) = 1.5-2.5 min plus K,V recording of the stream (unmeasured, minutes) plus the frozen-weight reads: W0, out-of-window after exposure 16, 32, 64 and the in-window read after 16, 32, 64, 30-90 s each (assumed) = 4-10 min per cell. Host memory about 15 GiB per stream (estimate).
# Reads (frozen weights, 16 candidates = gold + 15 places of other facts, chance mean rank 8.5; question frame = fact id mod 3, never an exposure frame): out-of-window (neutral 8192-token context) after exposure 16, 32, 64 and in-window after 16, 32, 64. NOT in this job: a read on an exposure frame (the design relay of 12:49Z does not ask for it). Exploratory: ONE seed, no preregistered verdict; the report shows u = (W0 rank - rank)/15 with fact bootstrap CIs and paired differences only.
# Gates before any GPU step (exit 7 on failure): same background gates as job 181, for both cells. Masks: `maskdump` checks on the real world that SENT hides only fact-sentence tokens of the 8 window chunks (hidden keys == sentence tokens), none of the current chunk, KMASK keys are a subset of SENT keys; it also prints the frame statistics (frame repeats per fact 0, same-frame sentences in the window: unique about 0.45, shared64 about 1.95).
# Outputs: $OUT/E55_icu_<ARM>_s61.* (unique frames) and E55_ics_<ARM>_s61.* (shared64) .saved.json / .read.json (+ states/ at exposure 64, canary at 64), e55_gates_ic*_s61.md, e55_maskdump_*.json, E55_unit_icu.json, e55_ic_report.md. NO OVERWRITE: finished files are reused (restart-safe); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_ic FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_ic/data && cd /tmp/dry_ic && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/183_e55_ic_frames_x_masks.sh   (then git worktree remove --force /tmp/dry_ic)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: any time (independent of 181/182).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=69f5cc7e9e76628dc626c54f4458b6343ff55690
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
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
1ca0bfe4f1ea e55_run.py
ca9f3280f5d6 e55_core.py
9827e34dcde6 e55_gpu.py
5179c07fb43e e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
97747552b3a4 e55_ic_report.py
f261120d0f03 s3.py
e7f22e6ade3f ../exp04_what_ttt_retains/gen_facts.py
52aafe88d682 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
ee82bf19b0b3 e55_gates.py
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_analyze.py --selftest | tail -1
$PY $S/e55_gates.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for C in "ic_u64 icu" "ic_s64 ics"; do set -- $C
  $PY $S/e55_gates.py --cell $1 --seed 61 --default-refs --require-refs --out $OUT/e55_gates_$2_s61.md > $OUT/e55_gates_$2_s61.log 2>&1 || { tail -30 $OUT/e55_gates_$2_s61.log; echo "background gates failed for cell $1 (exit 7)"; exit 7; }
  tail -14 $OUT/e55_gates_$2_s61.md
  $PY $S/e55_run.py maskdump --cell $1 --seed 61 --bg pool --arms NAT > $OUT/e55_maskdump_$2_s61.json || { echo "mask dump failed for cell $1 (exit 7)"; exit 7; }
  tail -22 $OUT/e55_maskdump_$2_s61.json | head -14
  $PY $S/e55_run.py dry --cell $1 --seed 61 --bg pool --arms NAT,KMASK,SENT | tail -3
done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_ic_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py unit --cell ic_u64 --seed 61 --bg pool --tag icu --out $OUT --unit-chunks 12 --unit-masks
$PY $S/e55_run.py run --cells ic_u64:icu,ic_s64:ics --bg pool --out $OUT --save-apps 64 --canary-apps 64 --seeds 61 --arms NAT,KMASK,SENT --plan "61=16,32,64|16,32,64"
$PY $S/e55_ic_report.py --dir $OUT --seed 61 --out $REP
tail -70 $REP
