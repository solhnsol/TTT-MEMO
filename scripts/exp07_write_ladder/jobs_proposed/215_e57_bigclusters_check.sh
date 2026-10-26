#!/bin/bash
# E57 BIG-CLUSTER CHECK (CPU only, NO GPU, NO model, NO download; Hans approved "215 돌려보자" in the news thread 2026-10-23 14:04Z). Why: the title-stage clustering of job 202 refused every cluster above 1500 members ('size>max (chained)'); Hans asked why large events were thrown away. This job redoes ONLY the title-stage clustering with the limit lifted (default 20000) into a SEPARATE folder $OUT/e57_bigclusters/ (the job-202 clusters.pkl is NOT touched) from the job-202 titles.pkl (exit 7 if missing), and reports: size distribution, number of clusters above the old limit and their article records; for the top 50 clusters by members: first title and 3 seeded example titles, hostnames, first / last date, over-merge checks (most frequent title tokens and core coverage, stricter re-split with title Jaccard >= 0.35 on a sample of at most 3000 members: largest component share, components >= 8, singletons; date concentration: span, densest 7-day share, days holding >= 5 %), and ETH-v1 pass / reasons (REPORT COLUMN ONLY, the filter rule is not changed) on the first 1500 characters of 40 seeded members per cluster (one pass over the shards for about 2000 articles).
# PROXY: title-token single-link clusters (the chaining risk is what the check looks at). The report names clusters by article titles and writes no article text. Report $OUT/e57_bigclusters/e57_bigclusters_report.md (+ .json). Exit 8 if the report exists unless RERUN=1.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. Clustering reads the title table only (same work as the cluster stage of job 202; the larger limit does not change it): about 0.1-0.5 h; the strict re-split of 50 clusters (sample <= 3000 members each) a few minutes; the 2000-article text sample reads each shard once: 0.05-0.3 h. Total about 0.2-0.8 h; server load (~80) up to 2x. Memory as the title / cluster stage of job 202.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing). Writes only under $OUT/e57_bigclusters.
# PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch).
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=fdac636bdff5f4d80d22c2fc01e6d62c8f0b83fd
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py"
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
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_bigclusters/e57_bigclusters_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: the job-202 run must have finished the titles and cluster stages (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_bigclusters.py --selftest | tail -1
$PY $D/e57/e57_bigclusters.py --out "$OUT" --raw "$RAW" --max-cluster ${MAXCLUSTER:-20000} ${RERUN:+--rerun}
tail -150 "$REP"
