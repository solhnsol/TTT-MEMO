#!/bin/bash
# E57-LITE FREEZE (CPU only, NO GPU, NO model, NO download; E57 sections 16.1 / 16.10 requests 1-2; replaces job 222 which was built before the 16.1 correction). (1) Event list fixed: the 12 regex events of job 220 / 221 WITHOUT Nipah virus and IDFC First Bank fraud + the next 6 clusters of the job-219 list (not recurring / product suspects with keyword share >= 0.3, not overlapping with an event already taken (>= 20 % of the members shared or >= 2 of 3 top title tokens equal), in the order of members) = 16 events. (2) Ethics filter per target only (family words reject only together with the private-individual proxy of S). (3) Frozen bank of the LEAD variant (c >= 3): per target the question for S (fixed frame per answer type), the candidates (same type, same event, lead exposure c'; fill order strict > middle > relaxed, at most 16, G >= 8 for the main analysis), 4 seeded S' for the cross null; strata table (c 3-4 / 5-9 / 10-19 / 20+) per event, answer type and main targets; and the article-cap tables N = 10 / 15 / 20 per event (main rule: articles with the most lead targets first, ties by publication date; comparison rule: first N by date; selection frozen in lite_caps.json): targets c >= 3, strata, main share (G >= 8), targets with 4 S', round tokens (compared with 8192); and, for Hans's questions, (1) whether an article belongs to one event only (cluster partition check, overlap of the title matches per event pair, lead mentions of another event, 50 seeded articles saved for manual reading) and (2) binding sentences per article lead (0/1/2/3/4+ distribution, mean, median, distinct targets, articles without target, the same target repeated inside one article, 10 examples; all articles and the N = 10 / 20 main-rule selections); and event attribute candidates per event (place / person proxies, date, number: top 5 strings and the share of the event's lead articles that hold them). No article text in any output (ids, sentence indices, strings).
# Needs $OUT/titles.pkl of job 202, $OUT/e57_bigclusters/clusters_big.pkl (job 215) and e57_biglist.json (job 219) (exit 7 if missing), the parquet shards in $RAW. Outputs $OUT/e57_lite/lite_events.json, lite_bank.json, lite_manifest.json, e57_lite_report.md (NEW names). Exit 8 if the report exists unless RERUN=1.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. About 0.5-2 h: the analysis of job 221 (use its measured time) plus 6 added clusters (can be large) plus bank building (seconds to minutes). Server load (~80) up to 2x.
# Needs: lab python with pyarrow, numpy, tokenizers (exit 3 if missing). Writes only under $OUT/e57_lite. PINNED INSTALL, NO FETCH (exit 4 / 5). CPU_ONLY=1 (check run): install check, manifest, selftests and the input checks only, then exit 0 without the full run. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=f4d2f1a0c99be0b38f478fcecb3177e3402a67ec
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py $D/e57/e57_events.py $D/e57/e57_namedevents.py $D/e57/e57_namedevents2.py $D/e57/e57_namedevents3.py $D/e57/e57_lite.py"
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
a35230989359 e57/e57_namedevents3.py
01178c42aa12 e57/e57_lite.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in pyarrow numpy tokenizers; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_lite/e57_lite_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for F in titles.pkl e57_bigclusters/clusters_big.pkl e57_bigclusters/e57_biglist.json; do [ -f "$OUT/$F" ] || { echo "$OUT/$F missing: titles.pkl of job 202, clusters_big.pkl of job 215 and e57_biglist.json of job 219 are needed (exit 7)"; exit 7; }; done
[ -d "$RAW/data" ] || { echo "parquet shards missing in $RAW (exit 7)"; exit 7; }
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_namedevents.py --selftest | tail -1
$PY $D/e57/e57_namedevents2.py --selftest | tail -1
$PY $D/e57/e57_namedevents3.py --selftest | tail -1
$PY $D/e57/e57_lite.py --selftest | tail -1
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, full run skipped"; exit 0; }
$PY $D/e57/e57_lite.py --out "$OUT" --raw "$RAW" --workers $WORKERS ${RERUN:+--rerun}
tail -120 "$REP"
