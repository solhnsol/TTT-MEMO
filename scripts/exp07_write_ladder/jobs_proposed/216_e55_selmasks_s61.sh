#!/bin/bash
# JOB 216 = E55 32.11 MASK STAGE (FORWARD ONLY, no write step, one model load; GPU): label-free sentence masks of the top candidates of the 213 sentence grid (candidate file $OUT/e55_sentgrid_report_s61_candidates.json from the seed-61 run of job 213; ranks RANKS, default 1,2,3,5), for EVERY chunk 1..127 of the stream of SEED (default 61; 62 / 63 use the SAME candidate file, no new selection). Why a separate stage: the attention the candidates use is the attention of the frozen PREFIX layers, which does not depend on the write state, so the mask of chunk k depends on the stream tokens only; it is computed once here and read by the write arms SELC<rank> (job 217) and RSELC<rank> (job 218). CPU test U48 shows on the tiny model that the selection recomputed online from the CURRENT weights before every Adam write equals the stored mask at every chunk.
# Output $OUT/e55_selmasks_s<SEED>.npz (hidden window sentences per chunk and rank), .json, report $OUT/e55_selmasks_report_s<SEED>.md (NEW names): per candidate and chunk tier (8-31, 32-63, 64-127, 8-127) recall, precision, F1, accuracy, hidden share, background false-mask and chance (label statistics: the fact table is used for these numbers only). This is a free full-stream check of the candidates (the 213 grid looked at 4 chunks only) before the write arms spend GPU hours.
# Only token-vote candidates are supported (V6 tok / V7 tok: heads by sharp / copy / induction, optional rare-token query filter, sink / previous-chunk exclusion, k-times trigger); a candidate that needs the sentence-level arrays (AL / AV / q_ce) is refused with its name (exit 7 on the check).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 0.5-1.1 h per seed: model load 7-20 min; recording the natural K,V of 128 chunks 6-10 min; 127 forward-only attention passes (all 1024 query tokens, per layer only the maximum window mass and its key index are kept; guessed 5-15 s each) 11-32 min; CPU selection and label statistics 1-3 min. Contended server up to 2x. Disk about 1 MB per seed.
# Needs: $OUT/e55_sentgrid_report_s61_candidates.json (exit 7 if missing). Not run on a GPU: the real 3B lean attention pass (memory: one [heads, 1024, 9216] float32 score tensor per layer, as in job 213), the real-tokenizer sentence layout (checked by the CPU_ONLY part: selmasks --check-only).
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_sm FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_sm/data && cd /tmp/dry_sm && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/216_e55_selmasks_s61.sh   (then git worktree remove --force /tmp/dry_sm)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 213). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=34841f101cf49d46f94fff4fa7fb73bbed125ef1
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D/rope_fly.py $D/e55_sentdump.py $D/e55_sentdump_labels.py $D/e55_sentsel.py $D/e55_sentgrid.py $D/e55_selmask.py $D/e55_gates.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D4/frames300.txt"
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
bf121294e3ce e55_run.py
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
76802d72e2c4 rope_fly.py
16bf1b4d7808 e55_sentdump.py
df41b7d13739 e55_sentdump_labels.py
3728d3954e71 e55_sentsel.py
24b0b4970d22 e55_sentgrid.py
21324f7fcdb7 e55_selmask.py
eb8bc59bf9a0 e55_gates.py
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEED="${SEED:-61}"
case "$SEED" in 61|62|63) ;; *) echo "SEED must be 61, 62 or 63" >&2; exit 3;; esac
RANKS="${RANKS:-1,2,3,5}"
CAND=$OUT/e55_sentgrid_report_s61_candidates.json
[ -f $CAND ] || { echo "candidate file $CAND missing: job 213 with SEED=61 must have run (exit 7)"; exit 7; }
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_sentgrid.py --selftest | tail -1
$PY $S/e55_gates.py --cell v3_16_1 --seed $SEED --default-refs --require-refs --out $OUT/e55_gates_sm_s$SEED.md > $OUT/e55_gates_sm_s$SEED.log 2>&1 || { tail -30 $OUT/e55_gates_sm_s$SEED.log; echo "background gates failed (exit 7)"; exit 7; }
tail -4 $OUT/e55_gates_sm_s$SEED.md
$PY $S/e55_run.py dry --cell v3_16_1 --seed $SEED --bg pool --arms NAT | tail -3
$PY $S/e55_run.py selmasks --cell v3_16_1 --seed $SEED --bg pool --sel-cands $CAND --sel-ranks $RANKS --out $OUT --check-only || { echo "selmasks check failed (exit 7)"; exit 7; }
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_selmasks_report_s$SEED.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py selmasks --cell v3_16_1 --seed $SEED --bg pool --sel-cands $CAND --sel-ranks $RANKS --out $OUT ${RERUN:+--rerun}
tail -40 $REP
