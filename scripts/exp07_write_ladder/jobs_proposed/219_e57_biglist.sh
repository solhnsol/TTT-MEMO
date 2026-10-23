#!/bin/bash
# E57 BIG-CLUSTER LIST (CPU only, NO GPU, NO model, NO download, NO new clustering; Hans 2026-10-23 17:16Z in the news thread: list only the clusters that are high on both checks). Re-processes the output of job 215 ($OUT/e57_bigclusters/clusters_big.pkl, exit 7 if missing): from all clusters of the clustering without the 1500 limit takes those with >= 50 members, computes the core coverage (share of ALL members whose title has >= 2 of the 3 most frequent title tokens) and, for core >= 0.5, the strict re-split (title Jaccard >= 0.35, date gap <= 3 days, sample <= 3000 members) and keeps the clusters whose largest-component share is >= 0.5. Columns: rank by members, members, hostnames, first / last date, core, largest component, densest 7-day share, ETH-v1 pass / reasons (report column only; first 1500 characters of 20 seeded members per listed cluster, one pass over the shards), 3 title examples, recurring-type suspect and product-type suspect (>= 30 % keyword share over the titles; screening aid only). Top of the report: funnel and counts per size band 50-100 / 100-300 / 300+.
# Outputs $OUT/e57_bigclusters/e57_biglist_report.md and e57_biglist.csv (all selected clusters, NEW names). Exit 8 if the report exists unless RERUN=1. No article text in the report.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. About 0.2-1.0 h: core coverage over the titles of all clusters with >= 50 members 1-5 min; the strict re-split of the clusters that pass the core test (sample <= 3000 members each, up to a few seconds per cluster, count unknown until 215 has run); the ETH text sample (about 20 members per listed cluster) reads each shard once, 0.05-0.3 h. Server load (~80) up to 2x.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing). Writes only under $OUT/e57_bigclusters. PINNED INSTALL, NO FETCH (exit 4 / 5). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=e85d60c6829a1f752cdaa359f6436b5d598c40c8
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py $D/e57/e57_biglist.py"
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
161757fdb4c4 e57/e57_biglist.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_bigclusters/e57_biglist_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl e57_bigclusters/clusters_big.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: titles.pkl of job 202 and clusters_big.pkl of job 215 are needed (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_biglist.py --selftest | tail -1
$PY $D/e57/e57_biglist.py --out "$OUT" --raw "$RAW" ${RERUN:+--rerun}
tail -120 "$REP"
