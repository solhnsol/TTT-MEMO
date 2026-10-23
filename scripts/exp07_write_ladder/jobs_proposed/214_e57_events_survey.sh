#!/bin/bash
# E57 news EVENTS SURVEY (CPU only, NO GPU, NO model, NO download; Hans approved the survey in the news thread 2026-10-23 11:54Z: "유명 사건들이 기사 수 확보가 되는지부터"). Reuses titles.pkl, clusters.pkl and the monthly shards in $RAW of the job-202 run (exit 7 if missing); fetch is NOT called.
# Answers: (1) independent-article distribution per event (title-stage clusters, then near-duplicate removal + one article per hostname on the SCAN=60 clusters with the most hostnames and on a seeded sample of 200 others) and the top-20 events with independent domains, first / last date, ethics filter ETH-v1; (2) per top event the number of (S, A) targets that appear in >= c independent articles, c >= 3, 5, 10, 20, 50, lead variant and full-article variant; (3) the top N = 5 / 10 / 15 ETH-passing events once in publication order: total tokens, chunks (= updates), distance to the previous occurrence of the same target, rho_win (share of repeated occurrences with the previous occurrence inside the 8192 tokens before the chunk), mixing of simultaneous events, with the CLUSTER and SPREAD layouts of the same articles as references; (4) share of targets whose value A has a competitor for the same S and whether the competitor sits at a different time.
# PROXIES as in all E57 CPU jobs: regex (S, A), title-token clusters. The report names events by the title of one article and lists no article text. Report $OUT/e57_events/e57_events_report.md (+ .json). Exit 8 if the report exists unless RERUN=1.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. 0.3-1.2 h with WORKERS=8 (text stage for about 260 clusters; the big clusters of up to 1500 articles cost the most: near-duplicate removal is pairwise and the conflict check loops over all (S, A) pairs; use the lab log of job 202 for per-cluster speed); the stream part takes seconds. Server CPU is contended (load ~80): up to 2x. Memory as the measure stage of job 202.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing), the repo tokenizer data/tokenizer/llama3/tokenizer.json. Writes only under $OUT.
# PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch).
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=358065ad914ce18ff879a9ca6f674d64566fcb94
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_events.py"
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
b9b3ae99f2ef e57/e57_events.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_events/e57_events_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl clusters.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: the job-202 run must have finished the titles and cluster stages (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_freeze.py --selftest
$PY $D/e57/e57_events.py --selftest
$PY $D/e57/e57_events.py --out "$OUT" --raw "$RAW" --workers ${WORKERS:-8} --scan ${SCAN:-60} --sample ${SAMPLE:-200} ${RERUN:+--rerun}
tail -150 "$REP"
