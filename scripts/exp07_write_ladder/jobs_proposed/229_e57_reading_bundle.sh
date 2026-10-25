#!/bin/bash
# E57 READING BUNDLE for hand-written questions (CPU only, NO GPU, NO model, NO download). Idea of Hans 2026-10-24: famous events, the questions are written per article set by Claude, not by code. Same analysis as job 223 (one article per hostname, near-duplicate removal Jaccard 0.5) for the 223 events + the next N_EXTRA (default 20) clusters of the job-219 list + optional hand-chosen keyword events (KEYWORD_EVENTS=file.json {name: [regex, ...]}, a title matches when ALL regexes match). Outputs under $OUT/e57_bundle/: reading_bundle.json (SERVER ONLY, publisher text: up to 40 independent article excerpts per event: event, article id, hostname, date, title, first sentences up to 700 characters; seeded shuffle, seed 57241), bundle_facts.json (no article text: top proper nouns / dates / numbers / place-person-name proxies per event with the number of independent articles holding them anywhere and in the lead), news-lite/<event>.json (one file per event with the same excerpts, for /mnt/project-files/data/news-lite/: private internal research only, Hans 2026-10-24; copied by hand, this job copies nothing), bundle_index.json (no text: articles before / after near-duplicate removal, selected ids), bundle_store.pkl (SERVER ONLY: all independent articles with sentences, input of the helper), e57_bundle_report.md (counts only). The helper e57_verify.py (run it afterwards, seconds): per hand-written question (event, S alternatives, A alternatives) the number of independent articles holding A (anywhere / in the lead) by article id, the share where A and S are in one sentence and where S lies within +-64 tokens, near-duplicates removed.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. About 1-3.5 h: the analysis of job 223 (0.7-2.5 h estimated, use its measured time) for about 30 instead of 20 events plus the bundle files (minutes). Server load (~80) up to 2x.
# Needs: titles.pkl (job 202), e57_bigclusters/clusters_big.pkl (job 215), e57_bigclusters/e57_biglist.json (job 219), the parquet shards in $RAW (exit 7 if missing; under CPU_ONLY only reported). Excerpts may go to the project folder (private, internal research only, no publication or redistribution; Hans 2026-10-24) but this job copies nothing itself. PINNED INSTALL, NO FETCH (exit 4 / 5). CPU_ONLY=1 = install check, manifest, selftests, input check, exit 0 without the full run. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=5c835409e9a32b77829fdb20e4022a6547f06705
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py $D/e57/e57_events.py $D/e57/e57_namedevents.py $D/e57/e57_namedevents2.py $D/e57/e57_namedevents3.py $D/e57/e57_lite.py $D/e57/e57_bundle.py $D/e57/e57_verify.py"
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
485f5fb0f8c4 e57/e57_namedevents3.py
f66cea288f61 e57/e57_lite.py
765c674da589 e57/e57_bundle.py
4f6ad6e59e4a e57/e57_verify.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_bundle/e57_bundle_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl e57_bigclusters/clusters_big.pkl e57_bigclusters/e57_biglist.json; do [ -f "$OUT/$F" ] || { [ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: $OUT/$F missing here (needed for the full run)"; continue; }; echo "$OUT/$F missing: titles.pkl of job 202, clusters_big.pkl of job 215 and e57_biglist.json of job 219 are needed (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { [ -n "$CPU_ONLY" ] && echo "CPU_ONLY: parquet shards missing here" || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_namedevents.py --selftest | tail -1
$PY $D/e57/e57_namedevents2.py --selftest | tail -1
$PY $D/e57/e57_namedevents3.py --selftest | tail -1
$PY $D/e57/e57_lite.py --selftest | tail -1
$PY $D/e57/e57_bundle.py --selftest | tail -2
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, full run skipped"; exit 0; }
$PY $D/e57/e57_bundle.py --out "$OUT" --raw "$RAW" --workers $WORKERS --n-extra ${N_EXTRA:-20} ${KEYWORD_EVENTS:+--keyword-events "$KEYWORD_EVENTS"} ${RERUN:+--rerun}
tail -120 "$REP"
