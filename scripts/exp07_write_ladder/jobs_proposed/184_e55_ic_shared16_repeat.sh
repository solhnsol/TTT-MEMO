#!/bin/bash
# E55 design 21.2: repeats allowed. 16 facts, one sentence per fact in EVERY chunk, 64 chunks (65.5k tokens); frames = shared16rep: the first 16 frames of the bank are shared by all facts; in every chunk the 16 sentences use the 16 frames exactly once (Latin assignment), each fact has its own random order of the 16 frames and repeats the SAME order every 16 chunks, i.e. 64 exposures = 16 frames x 4 laps; the same (fact, frame) pair therefore returns every 16 chunks (outside the 8-chunk window). Same-frame sentences of other facts in the window: 8.0 on average. Arms NAT (real window) and SENT (no query sees any earlier fact sentence in the window; background and the chunk being written stay visible; attention only, the written chunk is real). Seeds 61, 62, 63 = 6 cells in ONE process. Non-repeating pooled background as in jobs 181 / 183.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED (183 has not run yet either; same per-cell work as 183). Without the GPU unit check (default, see below): about 1.3-2.1 h = CPU part 5-10 min + model and fp32 reader load 20-35 min once + 6 cells x (K,V recording of 64 chunks, unmeasured, minutes + 64 write steps at about 1.3-2 s + W0 and 6 frozen-weight reads at 30-90 s each = 4-10 min) = 40-80 min. With UNIT=1 (GPU unit checks incl. the mask checks on this world, a second model load): +25-35 min.
# GPU unit check: default SKIPPED. Reason: the write path, mask code and read code are byte-identical to job 183 (hash manifest below: only gen_facts.py / gen_world_ctl.py / the report differ, which build the text and the frame assignment); the frame assignment is fully checked on CPU (maskdump + selftests: 16 distinct frames per chunk, every fact 16 frames x 4 laps, same-frame mean 8, SENT hidden keys == fact-sentence tokens). If 183's GPU unit checks (U1-U6) have not passed, run this with UNIT=1. U1 is not re-measured by default.
# Reads (frozen weights, 16 candidates = gold + 15 places of other facts, chance mean rank 8.5; question frame = fact id mod 3, never an exposure frame): out-of-window after exposure 16, 32, 64 and in-window after 16, 32, 64, plus W0. Exploratory: no preregistered verdict; report = per seed and pooled u with fact bootstrap CIs, paired SENT - NAT.
# Gates before any GPU step (exit 7 on failure): background gates for the 3 seeds; maskdump for seed 61 (frame statistics and mask checks); dry run.
# Outputs: $OUT/E55_icr_<NAT|SENT>_s<61|62|63>.* (.saved.json, .read.json, states/ at exposure 64, canary at 64), e55_gates_icr_s*.md, e55_maskdump_icr_s61.json, e55_icr_report.md. NO OVERWRITE: finished files are reused (restart-safe); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_icr FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_icr/data && cd /tmp/dry_icr && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/184_e55_ic_shared16_repeat.sh   (then git worktree remove --force /tmp/dry_icr)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: any time.
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
for SD in 61 62 63; do
  $PY $S/e55_gates.py --cell ic_r64 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_icr_s$SD.md > $OUT/e55_gates_icr_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_icr_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -14 $OUT/e55_gates_icr_s$SD.md
done
$PY $S/e55_run.py maskdump --cell ic_r64 --seed 61 --bg pool --arms NAT > $OUT/e55_maskdump_icr_s61.json || { echo "mask dump failed (exit 7)"; exit 7; }
tail -22 $OUT/e55_maskdump_icr_s61.json | head -14
$PY $S/e55_run.py dry --cell ic_r64 --seed 61 --bg pool --arms NAT,SENT | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_icr_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
[ -n "$UNIT" ] && $PY $S/e55_run.py unit --cell ic_r64 --seed 61 --bg pool --tag icr --out $OUT --unit-chunks 12 --unit-masks
$PY $S/e55_run.py run --cell ic_r64 --tag icr --bg pool --out $OUT --save-apps 64 --canary-apps 64 --seeds 61,62,63 --arms NAT,SENT --plan "61=16,32,64|16,32,64;62=16,32,64|16,32,64;63=16,32,64|16,32,64"
$PY $S/e55_ic_report.py --rep --dir $OUT --seeds 61,62,63 --tag icr --out $REP
tail -60 $REP
