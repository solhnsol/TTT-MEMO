#!/bin/bash
# E55 design 25.3: (A) arm SENT (the online stream WITH the window, every earlier fact sentence in the 8-chunk window hidden from attention, background visible, attention-only mask exactly as job 183, no norm matching) on the 181 world (16 facts, 128 chunks = 128 exposures per fact, 300-frame bank, pooled background), seeds 61 62 63, reads at a = 64 and 128 (out-of-window, 16 candidates, W0 subtracted, unexposed question frames); (B) the arms of job 188 (ISOB, ISOD, NATB, FOS, definitions in job 188 and in e55_iso.py) for the seeds 62 and 63. Seed 61 of 188 is reused as is: the run skips every (arm, seed) whose E55_dc_<ARM>_s<seed>.saved.json already exists, so only SENT is new for seed 61.
# Report: $OUT/e55_dc2_report.md (NEW name; job 188's e55_dc_report.md is not touched): u per arm and seed + pooled over the 3 seeds, paired contrasts ISOB-ISO, ISOD-ISO, NATB-NAT, FOS-ISO, ISOB-NAT, ISO-NAT and the new SENT-NAT, SENT-ISOB, SENT-ISO. The pooled 3-seed table needs 188 seed 61 + this job.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY. About 0.8-1.2 h: CPU part 5-10 min; model + fp32 reader load 7-20 min once (188 seed 61 took 19 min in total for 4 arms incl. load); seeds 62, 63: W0 read + 4 arms about 10-15 min each; SENT 3 seeds x (128 chunks + 30 GiB K,V recording + 2 reads) about 6-10 min each. 3-seed hard limit if the NAT files of 181 are missing: exit 7.
# No GPU unit check (the SENT write path is job 183's, the row arms are job 188's; U22/U23/U24/U28 on CPU).
# Outputs: $OUT/E55_dc_SENT_s<seed>.saved.json / .read.json, E55_dc_<ISOB|ISOD|NATB|FOS>_s62/s63.*, e55_dc2_report.md. NO OVERWRITE; exit 8 if e55_dc2_report.md exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_dc2 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_dc2/data && cd /tmp/dry_dc2 && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/189_e55_decomp_sent_seeds23.sh   (then git worktree remove --force /tmp/dry_dc2)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: after 188 seed 61 (reuses its files). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=8104bd87c6e5e855196f6aaf53c7f78955fb3c85
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
a8aa3e708ddf e55_run.py
ca9f3280f5d6 e55_core.py
9827e34dcde6 e55_gpu.py
5179c07fb43e e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
77154d115c29 e55_ic_report.py
5723bbc93aa9 e55_iso.py
f261120d0f03 s3.py
1746ce4c159c ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
SEEDS="${SEEDS:-61,62,63}"
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_dc2_s$SD.md > $OUT/e55_gates_dc2_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_dc2_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_dc2_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_dc2_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp --cell v3_16_1 --tag dc --bg pool --out $OUT --seeds $SEEDS --arms ISOB,ISOD,FOS,NATB,SENT --norm-files "$OUT/E55_v3a_NAT_s{seed}.saved.json"
$PY $S/e55_ic_report.py --decomp --dir $OUT --seeds $SEEDS --out $REP
tail -50 $REP
