#!/bin/bash
# E55 REVERSE READING (place -> name), READ ONLY on the saved states of job 210 (world v3_64_4, seed 61, read point a = 128 appearances: arms NAT, SENTPQ, SENTPQADAMH = SENTPQ with Adam, 0.5 N) and W0 (E55 section 37; GPU, NO training, NO new states). Families read in ONE model load: 'rev' = a place is given, the model completes a person name, 16 candidates (the gold name + the 15 names of the other candidate facts of the forward item of the same fact: 7 of the same chunk group, 8 of other groups), >= 3 reverse frames (the first 3 of 6 that pass the CPU checks: slot-free 5-gram overlap 0 with the 300 training frames and the forward question frames, no word-order permutation of a training frame, no 13-gram shared with the training stream or the reference texts), and 'fwd' = the forward items (name -> place, the original question frames) of the same facts as paired reference F. CPU checks first (E55_d64_rev_check_s61.md): frame overlaps, facts that share a place (excluded from the reverse analysis), token length and first-token collisions of the candidate names, same / other chunk group split. Analysis (CPU): per-fact ranks and u (u = (16 - rank) / 15) minus W0, strata same-group / other-group names, bootstrap over facts, the classification rule of section 37.3 (R / F, NAT as control).
# NOT bundled with the section 36.3 new-format reads (question-answer, paraphrase, dialogue): those frames do not exist in code yet and wait for Hans's answer; the reader of this job (e55_run.py rev) takes any family of item rows, so adding them later is a small change that reuses one model load. Judgement: bundle them when their frames and overlap checks are written (saves one 13-23 min load), not before.
# Needs: job 210 done for seed 61 (E55_d64_<arm>_s61.saved.json and the state files under $OUT/states), job 209 books (pool path). Outputs (NEW names): $OUT/E55_d64_rev_check_s61.md / .json, $OUT/E55_d64_rev_<arm>_s61.json, $OUT/e55_rev_d64_report.md. Exit 7 if a saved.json is missing, exit 8 if the report exists unless RERUN=1.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, about 20-30 min GPU: model load 13-23 min (E55 measured, load only) + W0 and 3 states x 2 families x 1024 rows (64 items x 16 candidates; E55 reads took about 40-46 s per 1024-row read in the 128-chunk worlds) = 8 reads about 6 min, loading each state a few minutes more. CPU checks 1-5 min. Server load (~80) up to 2x.
# CPU-only dry run: PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/224_e55_reverse_read_s61.sh   (checks, unit tests and the CPU frame / place / name checks with the real tokenizer and pool, then exits). PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 210). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=a1dd3917612c3214c7fadfcae9f530d76a011b80
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt $D/rope_fly.py $D/e55_bgbooks.py $D/e55_bgbooks_sha.json $D/e55_rev.py"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"   # lab s3.py after job 189 (2-D mask version); the pre-189 version a16e6dd0d4c9 is accepted too
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  S3OK=$( (echo "$S3_OLD"; echo a16e6dd0d4c9; for C in $(git log --format=%H --all -- $D/s3.py); do git show $C:$D/s3.py | sha256sum | cut -c1-12; done) )   # every s3.py version ever committed in the local clone (all branches, so a later job that left a newer committed s3.py on the lab does not block this one) is accepted (a local edit never is)
  echo "$S3OK" | grep -qx "$S3H" || { echo "lab s3.py ($S3H) is not any committed version of s3.py up to $COMMIT (new one: $S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
303732b85991 e55_run.py
f05caa0a4811 e55_core.py
9827e34dcde6 e55_gpu.py
b82e7ab635c2 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
e8c32d9ffa65 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
6d9e9b62cecd e55_uv.py
a123948130e4 s3.py
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
eb8bc59bf9a0 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
76802d72e2c4 rope_fly.py
3d8657814bbb e55_bgbooks.py
75ff3cee6437 e55_bgbooks_sha.json
2215511cee2a e55_rev.py
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
ARMS="${ARMS:-NAT,SENTPQ,SENTPQADAMH}"
$PY $S/e55_bgbooks.py verify --dest data/narrative_gutenberg --pins $S/e55_bgbooks_sha.json || { echo "the extra background books are missing or differ from the pins: run job 209 first (exit 7)"; exit 7; }
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_rev.py --selftest | tail -1
$PY $S/e55_bgbooks.py --selftest
$PY $S/e55_run.py revcheck --cell v3_64_4 --seed 61 --bg pool --tag d64 --arms NAT --out $OUT
grep -E "^- (items|frames|candidate)|^\| [0-9] " $OUT/E55_d64_rev_check_s61.md
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
for AR in $(echo $ARMS | tr ',' ' '); do [ -f $OUT/E55_d64_${AR}_s61.saved.json ] || { echo "$OUT/E55_d64_${AR}_s61.saved.json missing: job 210 must have run for seed 61 (exit 7)"; exit 7; }; done
REP=$OUT/e55_rev_d64_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py rev --cell v3_64_4 --seed 61 --seeds 61 --bg pool --tag d64 --arms $ARMS --apps 128 --out $OUT ${RERUN:+--rerun}
$PY $S/e55_rev.py --analyze --files $(for AR in $(echo $ARMS | tr ',' ' '); do printf "%s," $OUT/E55_d64_rev_${AR}_s61.json; done | sed 's/,$//') --out $REP
tail -40 $REP
