#!/bin/bash
# E57 NAMED-EVENTS SURVEY v2 (CPU only, NO GPU, NO model, NO download; Hans 2026-10-23 17:51Z in the news thread: confirm the changed rules and rerun job 220). Same 12 fixed events and measurements as job 220 with four changes fixed before the result: (a) the ethics filter works on the (S, A) target (the sentence where S and A occur together plus one before and one after; rejects: e-mail / phone / street address, age with a violence keyword, minor with a violence keyword in any window; address words, family words, health words, victim / suspect words, or a private-individual proxy (age marker or role word) in >= 20 % of the windows); event-level rule a only as a report column; (b) strict title matching with word boundaries (the old 'ncert' matched 'concert'), top 5 matched titles and a seeded sample of 20 per event for a manual precision check; (c) per event, the repeated occurrences whose previous occurrence is outside the 8192-token window; (d) rejected targets saved with counts per reason and examples in e57_namedevents2_rejects.json. The rule change belongs in E57 section 6.
# Needs $OUT/titles.pkl of job 202 (exit 7 if missing) and the parquet shards in $RAW. Outputs $OUT/e57_namedevents2/e57_namedevents2_report.md, .json, _rejects.json (NEW names). Exit 8 if the report exists unless RERUN=1. No article text in the report.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. About 0.5-2 h, the same work as job 220 plus the target filter (regex over 3 sentences per occurrence, minutes) and one per-event stream pass; server load (~80) up to 2x. Job 220 ran at this scale; its measured time is the best guide.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing). Writes only under $OUT/e57_namedevents2. PINNED INSTALL, NO FETCH (exit 4 / 5). CPU_ONLY=1 (check run): install check, manifest, selftests and the input checks only, then exit 0 without the full run. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=709231ca4e2c72538b7149d57ed21114a304904d
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py $D/e57/e57_events.py $D/e57/e57_namedevents.py $D/e57/e57_namedevents2.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news}"; mkdir -p "$OUT"; : "${RAW:=$OUT/news_raw}"; : "${WORKERS:=8}"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
f09dcb7e69eb e56/e56_text.py
a0590cd53539 e57/e57_common.py
9f8d8d79bbea e57/e57_news.py
9bef0d97a08a e57/e57_news96.py
a39189e927f7 e57/e57_news10.py
29fd1f69fb0d e57/e57_rope.py
de41671acd28 e57/e57_freeze.py
433589d20efc e57/e57_bigclusters.py
b9b3ae99f2ef e57/e57_events.py
fe7309299152 e57/e57_namedevents.py
38b621622a9b e57/e57_namedevents2.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_namedevents2/e57_namedevents2_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: titles.pkl of job 202 is needed (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_namedevents.py --selftest | tail -1
$PY $D/e57/e57_namedevents2.py --selftest | tail -1
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, full run skipped"; exit 0; }
$PY $D/e57/e57_namedevents2.py --out "$OUT" --raw "$RAW" --workers $WORKERS ${RERUN:+--rerun}
tail -120 "$REP"
