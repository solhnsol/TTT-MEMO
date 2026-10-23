#!/bin/bash
# E57 section 10 (10.3 item 6, part 1) CPU recount for the lead30 design (lead variant = first ~350 tokens of each article, 30 events, fixed-order cyclic SPREAD without filler, CLUSTER as contrast). CPU only, NO GPU, NO model, NO download. Extends the news CPU check approved by Hans (2026-10-23 04:28Z, "둘 다 승인할게 cpu로 점검해봐"); no GPU approval is involved. Reuses titles.pkl, clusters.pkl in $OUT and the shards in $RAW of the job-202 run (exit 7 if missing); fetch is NOT called.
# What it does: (a) whether 30 events can be drawn under the rules (independent articles >= 8, domains >= 8, exclusions, dedup, >= 1 / 3 / 5 lead targets), extrapolated from the seeded sample to all title-stage clusters, and the target count n for random 30-event draws (20 draws) and the 30 richest events; (b) how many facts occur in 3-4 (and more) different articles, by c, type and number of distinct sentence forms f (f = 1 = same form everywhere is counted too); (c) M4' for lead30: CLUSTER vs fixed-order SPREAD (i)/(ii)/(iii), gaps, round length, updates at 128 exposures, all targets and c in {3,4} only; plus M5' for c in {3,4} (middle candidates >= 10 / >= 15) and power for 24 / 30 events with the measured targets per event. If fewer than 30 events exist the layout rows use all available events and the report says so.
# Report $OUT/e57_news10_report.md (+ .json). PROXIES as in job 202. The text stage is redone once (f >= 1 rule, cache analyze_cache_f1.pkl; f >= 2 targets are a subset, so the bank rule f >= 2 is applied by filtering).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. The text stage dominates and equals the measure stage of job 202 (my estimate 20-60 min; the lab log of job 202 has the real time: use that number), plus 1-5 min for the layouts (20 draws x 2 rows). Memory as the measure stage of job 202. Reruns with the cache: 1-5 min.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing), the repo tokenizer data/tokenizer/llama3/tokenizer.json. Writes only under $OUT. Exit 8 if the report exists unless RERUN=1 (RERUN=1 also redoes the text stage).
# PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). Exit 8 if the report exists unless RERUN=1.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=7188c75b780f444b60839d43719d6a04e625f9fa
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news}"; mkdir -p "$OUT"; : "${RAW:=$OUT/news_raw}"; : "${WORKERS:=16}"; : "${MAXCL:=3000}"
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
9061ee969cb1 e57/e57_news.py
9bef0d97a08a e57/e57_news96.py
a39189e927f7 e57/e57_news10.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_news10_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl clusters.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: the job-202 run must have finished the titles and cluster stages (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_news96.py --selftest
$PY $D/e57/e57_news10.py --out "$OUT" --raw "$RAW" --workers $WORKERS --max-clusters $MAXCL ${RERUN:+--rerun}
tail -90 "$REP"
