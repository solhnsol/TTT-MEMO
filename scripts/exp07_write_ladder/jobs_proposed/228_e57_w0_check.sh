#!/bin/bash
# E57-LITE W0 CHECK (GPU, FORWARD PASS ONLY, no training, no weight update, one model load). Does the base model (weights W0, no news read) already guess the PLACE answer from the question alone? Input: place_clean.json of job 227. Stage 1 (CPU): e57_w0.py --prepare builds per item (event, S) 16 candidate places of the SAME token length (answer A + 15 distractors: other places of the event, answers of other events, gazetteer places; rules in the docstring of e57_w0.py) -> w0_items.json. Stage 2 (GPU): summed span log-probs at W0 with the neutral 8192-token SQuAD tail as in E55 -> w0_scores.json, w0_items_kept.json, w0_report.md. EXCLUSION RULE fixed before the result: drop an item if u(W0) >= 0.9 (u = (16 - rank of A)/15) or the top-1 candidate leads the 2nd by >= 2 nats; both reasons counted separately.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 10-25 min: CPU prepare under 1 min; model load 5-7 min; scoring (one packed 8192-token-tail pass per span length, 16 rows per item, a few lengths, a few hundred items at most) 3-15 min. Contended server up to 2x.
# Needs: job 227 finished ($OUT/e57_news/e57_lite/place_clean.json; exit 7 if missing), data/tokenizer/llama3/tokenizer.json, data/squad/dev-v1.1.json. Not run on a GPU: the whole scoring path (it mirrors the fam command of e55_run.py: Reader55, stage2.item_ids, out_tail_8192; CPU-tested here: prepare, the decision rule). exit 8 if the report exists (RERUN=1).
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 224). CPU_ONLY=1 = checks, selftests, prepare (if place_clean.json exists), exit 0 before any GPU step.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=2df6ba5971cc8340e5e0e3749f4b5e1c64cde4a3
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt $D/rope_fly.py $D/e55_bgbooks.py $D/e55_bgbooks_sha.json $D/e57/e57_w0.py $D/e57/e57_placeclean.py"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  S3OK=$( (echo "$S3_OLD"; echo a16e6dd0d4c9; for C in $(git log --format=%H --all -- $D/s3.py); do git show $C:$D/s3.py | sha256sum | cut -c1-12; done) )
  echo "$S3OK" | grep -qx "$S3H" || { echo "lab s3.py ($S3H) is not any committed version of s3.py up to $COMMIT (new one: $S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
46f1065a3762 e55_run.py
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
ce74cd244004 e57/e57_w0.py
faa878cec667 e57/e57_placeclean.py
MANIFEST
  echo "installed code verified against the manifest"
fi
LITE="$OUT/e57_news/e57_lite"
$PY $D/e57/e57_placeclean.py --selftest | tail -1
$PY $D/e57/e57_w0.py --selftest | tail -1
if [ -f "$LITE/place_clean.json" ]; then
  $PY $D/e57/e57_w0.py --prepare --rerun --out "$LITE"
else
  [ -n "$CPU_ONLY" ] || { echo "$LITE/place_clean.json missing: job 227 first (exit 7)"; exit 7; }
  echo "place_clean.json not here yet (job 227 first): prepare skipped in this CPU_ONLY check"
fi
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$LITE/w0_report.md" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo): $LITE/w0_report.md"; exit 8; }
$PY $D/e57/e57_w0.py --score --rerun --out "$LITE"
tail -30 "$LITE/w0_report.md"
