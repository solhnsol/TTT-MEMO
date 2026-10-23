#!/bin/bash
# E55 design 31 (Hans 22 Oct 21:45 KST, approved in the core thread): ONE small job. Only the NAME, OCCUPATION and PLACE tokens of ALL earlier fact sentences in the 8-chunk window are hidden from the fact-sentence queries (arm ALLSPAN; per-query convention exactly as SENTPQ of job 190: background queries and the sentence being written see everything), compared with hiding the whole sentences (SENTPQ and SENT, jobs 189/190) and with NAT (job 181). World v3_16_1 (the 181 world: 16 facts, 128 chunks = 128 exposures, 300-frame bank, pooled background), seeds 61 62 63, window kept, reads out-of-window at a = 64 and 128, canary at a = 128, the stream hash is checked against the SENTPQ file of job 190 in the report.
# Report $OUT/e55_all_report.md (NEW name): g per arm, R = g / g(SENTPQ), gain share (g - g(NAT)) / (g(SENTPQ) - g(NAT)), hidden keys % of the window, canary rise, contrasts ALLSPAN - SENTPQ, ALLSPAN - NAT, stream hash check. Pre-set reading: R >= 0.7 sufficient, <= 0.3 not sufficient, else partial.
# Hidden share: about 4.7% of the window keys (CPU estimate with the stub tokenizer on the same world; the exact share with the real tokenizer is printed by the maskdump step of this job and measured in the report), versus about 30% for SENTPQ.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 35-65 min: CPU part (selftests, 3 background gate checks, dry run, maskdump) 5-10 min; model load 7-20 min once; 3 runs (one per seed) of 128 chunks + 30 GiB K,V recording about 6-10 min each plus 2 out-of-window reads and the canary read about 1-3 min = 21-39 min. Not run on a GPU: the reads, the canary, the 3B paths.
# Outputs (NEW names): $OUT/E55_all_ALLSPAN_s<seed>.* , $OUT/e55_all_report.md; exit 8 if the report exists unless RERUN=1. Nothing is overwritten.
# CPU-only dry run (validates the arm through decomp --check-only and the real-tokenizer name / occupation / place span search through maskdump): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_all FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_all/data && cd /tmp/dry_all && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/196_e55_name_occ_place_mask.sh   (then git worktree remove --force /tmp/dry_all)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). Lab ladder.py must be the patched W1 version (exit 6); lab s3.py must be f261120d0f03, a16e6dd0d4c9 or the new version (exit 6).
# Needs: SENTPQ and SENT files of jobs 189/190 (tag dc) and the NAT file of job 181 (tag v3a) per seed. Independent of jobs 194 and 195 (own tag all). Queue position: after 195 (core thread).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=406b49a82aba4ab0f981a68c8c317f854ffc9d8f
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"   # lab s3.py after job 189 (2-D mask version); the pre-189 version a16e6dd0d4c9 is accepted too
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  [ "$S3H" = "$S3_OLD" ] || [ "$S3H" = "a16e6dd0d4c9" ] || [ "$S3H" = "$S3NEW" ] || { echo "lab s3.py ($S3H) is neither the committed old version ($S3_OLD) nor the new one ($S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
13a1fbdedc2a e55_run.py
b95933f52762 e55_core.py
9827e34dcde6 e55_gpu.py
f240fac19f43 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
4c8679add232 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
558d7974f051 s3.py
f14491a9ddf7 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEEDS="${SEEDS:-61,62,63}"
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_all_s$SD.md > $OUT/e55_gates_all_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_all_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_all_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
  [ -f $OUT/E55_dc_SENTPQ_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "SENTPQ file of job 190 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp --cell v3_16_1 --tag all --bg pool --out $OUT --seeds $SEEDS --arms ALLSPAN --norm-files "$NORM" --check-only
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool > $OUT/e55_maskdump_all.json
grep -A16 'answer_span_miss' $OUT/e55_maskdump_all.json | grep -i 'ALLSPAN\|answer_span_miss' || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_all_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp --cell v3_16_1 --tag all --bg pool --out $OUT --seeds $SEEDS --arms ALLSPAN --norm-files "$NORM"
$PY $S/e55_ic_report.py --all --dir $OUT --seeds $SEEDS --tag all --out $REP
tail -40 $REP
