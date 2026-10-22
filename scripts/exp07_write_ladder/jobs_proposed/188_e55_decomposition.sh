#!/bin/bash
# E55 design 24.3 (+ logic-check arm d): decomposition between ISO (stage 0, job 185) and NAT on the E55 v3 world of job 181 (16 facts, 128 exposures per fact, 300-frame bank, pooled background), all reads at a = 64 and 128 (out-of-window, 16 candidates, W0 subtracted, question frames never seen), seeds 61 62 63 (SEEDS=61 for seed 61 first).
# ISOB (a) "ISO + background": the tokens of every stream chunk (fact sentences AND that chunk's non-repeating background, fact share 31-37%) cut into isolated rows (each fact sentence a row, each background run a row, no row sees another), one packed chunk per exposure round = 128 steps, per-step update norm = the NAT per-chunk norm.
# NATB (b) "NAT-bigstep": the online NAT stream with the window; every chunk step rescaled to the size of ONE stage-0 ISO step of its segment (NAT norm x chunks / packed ISO steps of the segment, segments (0,64], (64,128]; about x2.4).
# ISOD (c) "ISO-density": isolated sentences only, ONE round (16 sentences) per chunk, 128 steps, per-step norm = the NAT per-chunk norm.
# FOS (d) "fact-only stream": the stage-0 sentences as ORDINARY continuous text in 1024-token chunks (about 52 chunks, no isolation, the 8-chunk SWA window present = the online write path, so shadowing inside the ISO text can act); each read segment is padded with newline tokens to a multiple of 1024 (at most 1023 pad tokens per segment, logged) so the reads fall on chunk boundaries; norm per chunk = the NAT segment increment spread equally (cumulative norm at the read points = NAT's, as in stage 0). The first 8 chunks of the stream have a short window.
# Interpretation choices of the code thread (design thread may overrule): ISOB keeps 128 steps (one stream chunk per step; 'step count the same' read as NAT's and ISOD's 128, ISO itself has about 53); ISO from job 185 (read segments 16/32/64/128) is only the comparison column.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED. All 3 seeds about 1.5-2.5 h (SEEDS=61 only: about 0.6-1.1 h): CPU part 5-10 min; model + fp32 reader load 20-35 min once; per seed W0 read 0.5-1.5 min, ISOB 128 steps 4-6 min, ISOD 128 steps 3-5 min, FOS (about 52 chunks + K,V recording) 3-6 min, NATB (128 chunks + recording 30 GiB host) 5-9 min, 2 reads per arm at 30-90 s = 4-12 min, i.e. 20-40 min per seed.
# No GPU unit check: row construction, packing, per-step norm matching, the NATM norm path and the bigstep / FOS targets are CPU-tested (U28, earlier U26); the first ISOB step of seed 61 is the first GPU execution of the row code (stage 0 job 185 shares the packed-row step).
# Needs job 181 NAT files (tag v3a, per seed; exit 7 if missing). Job 185's ISO files (tag iso) are used for the comparison column if present. Outputs: $OUT/E55_dc_<ISOB|ISOD|NATB|FOS>_s<seed>.saved.json (+ .read.json), e55_dc_report.md (u at a=64, 128 per arm and seed, pooled, paired contrasts ISOB-ISO, ISOD-ISO, NATB-NAT, FOS-ISO, ISOB-NAT, ISO-NAT). NO OVERWRITE: finished (arm, seed) files are reused; exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_dc FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_dc/data && cd /tmp/dry_dc && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/188_e55_decomposition.sh   (then git worktree remove --force /tmp/dry_dc)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: after job 181 (and 185 for the comparison column). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=c1acae3d62c31aa55e18258ba9dd1c6e656f9074
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
e34f4d4740fd e55_run.py
ca9f3280f5d6 e55_core.py
9827e34dcde6 e55_gpu.py
5179c07fb43e e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
5fafecc9e06b e55_ic_report.py
91a471c61d21 e55_iso.py
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
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_dc_s$SD.md > $OUT/e55_gates_dc_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_dc_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_dc_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_dc_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp --cell v3_16_1 --tag dc --bg pool --out $OUT --seeds $SEEDS --arms ISOB,ISOD,FOS,NATB --norm-files "$OUT/E55_v3a_NAT_s{seed}.saved.json"
$PY $S/e55_ic_report.py --decomp --dir $OUT --seeds $SEEDS --out $REP
tail -40 $REP
