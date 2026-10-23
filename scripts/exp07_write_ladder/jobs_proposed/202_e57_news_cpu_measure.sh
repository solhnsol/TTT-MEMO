#!/bin/bash
# E57 section 3 M1-M7: CPU-only measurement on news (ruggsea/infini-news-corpus, CC-News extraction, English, 2026-01..04). NO GPU, NO model. Approved by Hans for a CPU check (2026-10-23 04:28Z, "둘 다 승인할게 cpu로 점검해봐").
# What it does: fetch (HF, about 36.5 GB of monthly parquet shards for 2026-01..04, listed 483+370+394+88 files) -> titles stage (6 columns only) -> event clusters from title tokens + date proximity (>= 8 articles, >= 8 hostnames, recurring / sports / victim titles excluded) -> text stage on a seeded sample of the clusters (near-duplicate removal, one article per domain) -> regex fact proxy and (S, A) mining (c >= 3, f >= 2, conflict proxy), lead share M3, CLUSTER / SPREAD layouts and rho_win M4, candidate pools M5, power M6, license text M7. Report $OUT/e57_news/e57_news_report.md (+ .json); pass criteria are the fixed ones of E57 section 3 and are printed next to each number.
# PROXIES: (S, A) are regular-expression candidates (numbers, money, percent, dates, capitalised names), clusters come from title tokens, not embeddings; the numbers are indicative of the final bank, not equal to it. M1 / M2 are extrapolated from a seeded sample (default 3000 title-stage clusters) of the title-stage clusters.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured on the lab server. Roughly 2-6 h of CPU, no GPU: download 36.5 GB (10-60 min depending on the link; skipped when the shards are already in $RAW); titles stage 30-90 min (reads 6 columns of about 1,350 shards); cluster stage 30-60 min and 20-60 GB RAM (about 19 M English titles, bucket dictionary); text stage 20-60 min (each shard is read once for the sampled clusters, then WORKERS processes analyse the clusters). The stages write files and are skipped when their file exists (RERUN=1 redoes them). Smoke test on 3 shards (no download of the rest): SMOKE=1 (about 2 minutes).
# Needs: the lab python with pyarrow, numpy, tokenizers, huggingface_hub (exit 3 if one is missing; nothing is installed), network to huggingface.co, the repo tokenizer data/tokenizer/llama3/tokenizer.json. Writes only under $OUT and $RAW ($OUT default experiments/exp07_write_ladder/e57_news, $RAW default $OUT/news_raw). Does not touch any other file or job. NOT APPROVED to run until Hans confirms the time.
# PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). Exit 8 if the report exists unless RERUN=1.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=10853fe4aa571c1b7e5276c482da2b70e21dce3a
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py"
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
5a813562330b e57/e57_news.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers huggingface_hub; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_news_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $D/e57/e57_common.py --selftest
if [ -n "$SMOKE" ]; then
  # 3 shards, relaxed thresholds: exercises every stage; the numbers mean nothing
  mkdir -p "$RAW"; $PY $D/e57/e57_news.py fetch --raw "$RAW" --months 01 --max-files 3 2>/dev/null || true
  EXTRA="--max-files 3 --min-articles 3 --min-domains 3 --min-c 2"; OUT="$OUT/smoke"; mkdir -p "$OUT"
fi
[ -d "$RAW/data" ] || $PY $D/e57/e57_news.py fetch --raw "$RAW" --months 01,02,03,04
$PY $D/e57/e57_news.py all --raw "$RAW" --out "$OUT" --workers $WORKERS --max-clusters $MAXCL $EXTRA
tail -80 "$OUT/e57_news_report.md"
