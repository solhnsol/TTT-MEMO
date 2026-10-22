#!/bin/bash
# E55 design 22.3 stage 0: offline training on ISOLATED rows (ladder.py run_L1 "plain" variant) on the E55 v3 world of job 181 (16 facts, 128 exposures per fact, 300-frame bank, pooled background): every exposure sentence of the real stream is its own row ("\n\n" + sentence, loss on all its tokens, no other sentence, no background, empty suffix KV), rows of exposure round r = the 16 sentences of chunk r shuffled, rounds in order, so after a rounds every fact has exactly a exposures; rows packed into 1024-token chunks (exp05 packing, segment masks), ONE inner-rule step (SGD, config lr x ilr multiplier, global-norm clip) per packed chunk. Reads at a = 16, 32, 64, 128 exposures per fact: 16 candidates, W0 subtracted, out-of-window neutral context, question frames never seen in the stream. Cumulative update norm matched to the NAT run of the same seed (job 181 files E55_v3a_NAT_s<seed>.saved.json): within a read segment every step is rescaled after clipping so that the segment's update norm equals the NAT increment (direction unchanged, as arm NATM); raw norms logged. Canary at a = 128. Seeds 61, 62, 63.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED. About 45-75 min on one A100: CPU part 5-10 min; model + fp32 reader load 20-35 min once; per seed W0 read 0.5-1.5 min, 53 packed training steps in total (2048 rows of about 25 tokens, about 40 rows per chunk) at about 1.5-3 s = 1.5-3 min, 4 frozen-weight reads at 30-90 s = 2-6 min, i.e. 5-11 min per seed. No GPU unit check: the isolated-row step is the lab ladder.py step (packing verified there); the isolation property, the row bookkeeping and the norm matching are CPU-tested (U26); the first segment of seed 61 is the first GPU execution of this code path.
# Needs job 181 finished (NAT files, tag v3a, seeds 61-63; exit 7 if missing). Outputs: $OUT/E55_iso_ISO_s<seed>.saved.json (+ .read.json), e55_iso_report.md (NAT vs ISO, per seed and pooled, paired ISO - NAT). NO OVERWRITE: finished seeds are reused (restart-safe); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_iso FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_iso/data && cd /tmp/dry_iso && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/185_e55_stage0_isolated_rows.sh   (then git worktree remove --force /tmp/dry_iso)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: after job 181 (needs its NAT files, tag v3a, seeds 61-63).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=ab198e6f9de77fc3ba8357d9c75c75ee748b80a3
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
1ca0bfe4f1ea e55_run.py
ca9f3280f5d6 e55_core.py
9827e34dcde6 e55_gpu.py
5179c07fb43e e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
535d11775b7d e55_ic_report.py
f261120d0f03 s3.py
1746ce4c159c ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
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
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in 61 62 63; do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_iso_s$SD.md > $OUT/e55_gates_iso_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_iso_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_iso_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_iso_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py iso --cell v3_16_1 --tag iso --bg pool --out $OUT --seeds 61,62,63 --read-apps 16,32,64,128 --norm-files "$OUT/E55_v3a_NAT_s{seed}.saved.json"
$PY $S/e55_ic_report.py --iso --dir $OUT --seeds 61,62,63 --out $REP
tail -30 $REP
