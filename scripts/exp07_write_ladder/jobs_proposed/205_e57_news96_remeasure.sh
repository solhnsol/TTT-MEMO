#!/bin/bash
# E57 section 9.6 news re-measurement (CPU only, NO GPU, NO model, NO download). Extends the news CPU check approved by Hans (2026-10-23 04:28Z, "둘 다 승인할게 cpu로 점검해봐"); no GPU approval sentence is involved. Reuses the files of the job-202 run: titles.pkl, clusters.pkl in $OUT and the parquet shards in $RAW (exit 7 if they are missing); fetch is NOT called.
# What it does: redoes the text stage of the same seeded cluster sample (same seed as 202; results cached in $OUT/analyze_cache.pkl, later runs skip it) and writes $OUT/e57_news96_report.md (+ .json): (1) rho_win split into (i) another article, (ii) same article same round, (iii) copy of the same article in the previous round, plus reproduction of the relayed M4 numbers with the original shuffled layouts, round-order and article-length facts; (2) M4' for CLUSTER and a deterministic fixed-order SPREAD (uniform filler tokens per pass, only counted) for full articles with 12 events and the lead variant with 12 / 20 / 26 / 30 events, with gap min / 5th percentile, filler share (<= 0.30 required), round length, updates at 128 exposures; (3) M5' middle candidates (c' >= 2, |c-c'| <= 2) fulfilment and the c versus c' distribution; (4) power for 24-30 events with the DEFF table.
# PROXIES as in job 202 (regex facts, title-token clusters). The ICC of the outcome g cannot be measured on CPU; the report gives the implied DEFF instead.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured on the lab server. The text stage dominates and equals the measure stage of job 202 (about 20-60 min by my estimate: read each shard once for the sampled clusters, then WORKERS processes; the lab log of job 202 has the real time: use that number), plus 1-5 min for the layouts (5 draws x 10 configurations, pure Python). RAM as in the measure stage of job 202 (a few to a few tens of GB). Reruns with the cache take 1-5 min.
# Needs: the lab python with pyarrow, numpy, tokenizers (exit 3 if one is missing; nothing is installed), the repo tokenizer data/tokenizer/llama3/tokenizer.json. Writes only under $OUT. Exit 8 if the report exists unless RERUN=1.
# PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). Exit 8 if the report exists unless RERUN=1.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=eafed8fcd14863a3e1398f2060db4fd2dc65136d
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py"
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
dfb4de7621d0 e57/e57_news.py
c6714761bed4 e57/e57_news96.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_news96_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl clusters.pkl; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: the job-202 run must have finished the titles and cluster stages (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_news96.py --selftest
$PY $D/e57/e57_news96.py --out "$OUT" --raw "$RAW" --workers $WORKERS --max-clusters $MAXCL ${RERUN:+--rerun}
tail -90 "$REP"
