#!/bin/bash
# E57 NAMED-EVENTS SURVEY v3 (CPU only, NO GPU, NO model, NO download; Hans 2026-10-23 18:07Z in the news thread). Job 221 with: (1) family words reject a target only together with the private-individual proxy of S (Hans approved); (2) 16 events = the 12 fixed regex events + the next 4 clusters of the job-219 list (e57_biglist.json; not recurring / product suspects with keyword share >= 0.3, not overlapping with an event already taken (>= 20 % of the members shared or >= 2 of 3 top title tokens equal), in the order of members; the list's own ETH-v1 is ignored, the ethics filter is applied per target only); (3) per event and answer type (NAME / NUM / DATE): targets, c distribution, candidate-pool sizes inside the event (strict +-1 / middle / relaxed: median and share with >= 15); (4) cap of 20 / 30 / 40 independent articles per event (seeded random subsets, mean of 5 draws): targets with c >= 3 left. Rest as in job 221 (target-level ethics filter, strict title matching, outside-window repeats, rejects saved).
# Needs $OUT/titles.pkl of job 202 and $OUT/e57_bigclusters/clusters_big.pkl (job 215) and e57_biglist.json (job 219) (exit 7 if missing), the parquet shards in $RAW. Outputs $OUT/e57_namedevents3/e57_namedevents3_report.md, .json, _rejects.json (NEW names). Exit 8 if the report exists unless RERUN=1. No article text in the report.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. About 0.6-2.5 h: the work of job 221 (0.5-2 h estimate; use its measured time if available) plus 4 clusters that can be large (articles read once, one per hostname) plus the cap draws (5 draws x 3 caps per event, minutes). Server load (~80) up to 2x.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing). Writes only under $OUT/e57_namedevents3. PINNED INSTALL, NO FETCH (exit 4 / 5). CPU_ONLY=1 (check run): install check, manifest, selftests and the input checks only, then exit 0 without the full run. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=76c1db506a2ac8abab9c34ac404255c6ff291d45
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py $D/e57/e57_events.py $D/e57/e57_namedevents.py $D/e57/e57_namedevents2.py $D/e57/e57_namedevents3.py"
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
88c925d75c39 e57/e57_namedevents3.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_namedevents3/e57_namedevents3_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl e57_bigclusters/clusters_big.pkl e57_bigclusters/e57_biglist.json; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: titles.pkl of job 202, clusters_big.pkl of job 215 and e57_biglist.json of job 219 are needed (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_namedevents.py --selftest | tail -1
$PY $D/e57/e57_namedevents2.py --selftest | tail -1
$PY $D/e57/e57_namedevents3.py --selftest | tail -1
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, full run skipped"; exit 0; }
$PY $D/e57/e57_namedevents3.py --out "$OUT" --raw "$RAW" --workers $WORKERS ${RERUN:+--rerun}
tail -120 "$REP"
