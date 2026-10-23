#!/bin/bash
# E57 NAMED-EVENTS SURVEY (CPU only, NO GPU, NO model, NO download; Hans 2026-10-23 17:30Z in the news thread: check the events that are clearly events). 214-style survey for a FIXED list of 12 events (regex groups fixed in e57_namedevents.py before any result; every group must match a title, case-insensitive, over all titles of titles.pkl): Rashmika-Vijay wedding, Spider-Man Brand New Day, Venezuela-Machado (Machado title with Venezuela / Maduro / Nobel / opposition / Trump; the tanker stories are NOT merged, only counted as a side line with overlap), Mamdani inauguration, Nipah virus, El Paso airspace, HDFC Bank chairman resignation, IDFC First Bank fraud, Hannah Montana 20th anniversary, Mayweather-Pacquiao, Ranji Trophy J&K, NCERT textbooks. Salim Khan health is excluded. Per event: independent articles and domains after near-duplicate removal (one article per hostname), target (S, A) count distribution c >= 3/5/10/20/50 for the lead variant and the full-article variant, candidate-pool satisfaction (strict / middle / relaxed), the combined date-ordered stream of all events (total tokens, chunks, distance to the previous occurrence, share within 8192 tokens; CLUSTER / SPREAD references) and the share of S whose A changes over time. The ethics filter is NOT changed (ETH-v1 stays a report column only); the change of the event-selection rule (fixed list instead of the seeded draw) belongs in E57 section 6.
# Needs $OUT/titles.pkl of job 202 (exit 7 if missing) and the parquet shards in $RAW. Outputs $OUT/e57_namedevents/e57_namedevents_report.md and .json (NEW names). Exit 8 if the report exists unless RERUN=1. No article text in the report.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. About 0.5-2 h: title matching over all titles 1-3 min; reading the matched articles from the shards (one pass, a few thousand articles per event at most) 0.1-0.5 h; analyze per event (near-duplicate removal, regex mining of (S, A)) minutes per event with 8 workers; combined stream and layouts a few minutes. Server load (~80) up to 2x.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing). Writes only under $OUT/e57_namedevents. PINNED INSTALL, NO FETCH (exit 4 / 5). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=f3a8b5535cc12298c9a5f13f07179985e79a23c3
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py $D/e57/e57_events.py $D/e57/e57_namedevents.py"
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
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_namedevents/e57_namedevents_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: titles.pkl of job 202 is needed (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_namedevents.py --selftest | tail -1
$PY $D/e57/e57_namedevents.py --out "$OUT" --raw "$RAW" --workers $WORKERS ${RERUN:+--rerun}
tail -120 "$REP"
