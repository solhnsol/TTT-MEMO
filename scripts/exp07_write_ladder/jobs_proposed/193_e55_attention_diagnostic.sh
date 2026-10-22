#!/bin/bash
# E55 design 27.8 attention diagnostic (FORWARD ONLY, no backward, no weight change by the diagnostic): in the NAT stream of the world v3_16_1 (job 181: 16 facts, 128 chunks, 300-frame bank, pooled background), seed 61, the NAT write path is run for chunks 0..127 and, at the chunks 16, 32, 64 and 127 (0-based chunk index), BEFORE the write of that chunk, the attention that the fact-sentence query tokens of the chunk (every token inside a fact sentence) put on the keys of the 8-chunk window is recorded (prefix layers only: the layers that read the window K,V through s3 mode "att"; head- and layer-averaged probabilities, also for the first / middle / last third of the layers) and, for the top p % (1 / 2 / 5 / 10 / 20) of the in-window keys per query, the share of tokens of earlier sentences of the SAME fact, of any earlier fact sentence and of background, and the recall of the name and place tokens of the same fact, are reported against the random baseline (category share of the window keys; recall p %).
# Report $OUT/e55_attdiag_report.md + raw numbers $OUT/e55_attdiag_s61.json (NEW names). Exploratory, one seed.
# NEEDS an s3.py change (new mode "att", repo version installed from $COMMIT; the lab s3.py must be the version of the previous jobs f261120d0f03 or already the new one, otherwise exit 6) and the new file e55_diag.py.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 25-45 min: CPU part 5-10 min; model load 7-20 min; the NAT stream of 128 chunks with the 30 GiB K,V recording about 6-10 min; 4 diagnostic forwards (eager, 24 prefix layers, about 500 query rows x 9216 keys per layer, host copy of the probabilities) about 1-3 min each. No reads, no canary, no GPU unit check (U33 on the tiny model: hidden states of the att path == production prefix path, rows are proper distributions, the run with the diagnostic == the run without).
# Outputs: $OUT/e55_attdiag_s61.json, e55_attdiag_report.md; the run is skipped if the json exists (RERUN=1 to redo); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run (also validates the arguments through attdiag --check-only and the real-tokenizer answer-span search through maskdump): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_ad FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_ad/data && cd /tmp/dry_ad && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/193_e55_attention_diagnostic.sh   (then git worktree remove --force /tmp/dry_ad)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: after job 192 or any time (needs the NAT file of job 181 seed 61 only for the gates check). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=0e10c090b52f7d55ce4bfbee3107ed5aeda25ba1
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
24ce16d29b5d e55_run.py
5aa88707abf4 e55_core.py
9827e34dcde6 e55_gpu.py
71eb12a3913f e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
d82b51b8e199 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
f7199dbfee23 e55_diag.py
8111393db79e s3.py
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
for SD in 61; do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_ad_s$SD.md > $OUT/e55_gates_ad_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_ad_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_ad_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py attdiag --cell v3_16_1 --seed 61 --bg pool --out $OUT --diag-chunks 16,32,64,127 --check-only
$PY $S/e55_diag.py --selftest
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool > $OUT/e55_maskdump_ad.json
grep -A6 'answer_span_miss' $OUT/e55_maskdump_ad.json || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_attdiag_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py attdiag --cell v3_16_1 --seed 61 --bg pool --out $OUT --diag-chunks 16,32,64,127
tail -30 $REP
