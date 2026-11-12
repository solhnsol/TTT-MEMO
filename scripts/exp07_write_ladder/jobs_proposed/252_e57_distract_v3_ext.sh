#!/bin/bash
# E57 QUESTION-BANK TOOLS (CPU only, NO GPU, NO model, NO download; design E57 section 22). One job file, ONE STEP per approval (STEP=packet | validate | distract | freeze); each step reads only the output of job 229 ($OUT/e57_bundle/bundle_store.pkl, bundle_index.json) and the files of the earlier steps, takes seconds to a minute, no second analysis. Private, internal research only: nothing is copied to the project folder by the job.
#  STEP=packet   -> $OUT/e57_bank/packets/<event>.packet.json (12 independent articles per event, seed 57251 + md5(event name); lead sentences numbered a01.. / s001..; NO statistics; packet_hash) + packets_index.json (events left out: Kärcher and Umar Khalid by default (EXCLUDE), events failing the event-level ETH-v1, events without analysis). The person writes the questions from the packets (not code).
#  STEP=validate -> BANK=<jsonl file of the written questions> : schema, evidence ids (>= 2 articles, answer inside), answer / S overlap, relative time, frame 5-gram overlap, c >= 10, c_SA >= .5, same-type competitor <= half; validated.jsonl, validation.json, validation_report.md. Thresholds fixed before any result (E57 22.3).
#  STEP=distract -> easy list (answers of other events, same type, >= 8 or flagged) and hard list (up to 8 frequent same-type strings of the event lead + answers of other questions of the event) with c' and c'_win: distractors.json.
#  STEP=freeze   -> VERSION=v1 (never overwritten; exit 8), optional NOMINATIONS=<jsonl of the person's hard-distractor nominations>: bank_<VERSION>.json (+ .sha256) with hash, frozen_at (KST), authors, packet hashes.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, CPU only. Each step seconds to about 5 min (the validator scans the lead sentences of the event once per question). Needs job 229 finished (exit 7 if its store is missing; under CPU_ONLY only reported). PINNED INSTALL, NO FETCH (exit 4 / 5). CPU_ONLY=1 = install check, manifest, selftests, exit 0. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=5c835409e9a32b77829fdb20e4022a6547f06705; BANKCOMMIT=f6ee962f821e4394e5ca11a10d1a632c17386e49
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_news.py $D/e57/e57_news96.py $D/e57/e57_news10.py $D/e57/e57_rope.py $D/e57/e57_freeze.py $D/e57/e57_bigclusters.py $D/e57/e57_events.py $D/e57/e57_namedevents.py $D/e57/e57_namedevents2.py $D/e57/e57_namedevents3.py $D/e57/e57_lite.py $D/e57/e57_bundle.py $D/e57/e57_verify.py $D/e57/e57_bank.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news}"; mkdir -p "$OUT"; : "${RAW:=$OUT/news_raw}"; : "${WORKERS:=8}"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git cat-file -e $BANKCOMMIT^{commit} || { echo "commit $BANKCOMMIT missing (exit 4)"; exit 4; }
  git checkout $COMMIT -- ${FILES% $D/e57/e57_bank.py}; git checkout $BANKCOMMIT -- $D/e57/e57_bank.py
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
611e53d0164a e57/e57_bank.py
MANIFEST
  echo "installed code verified against the manifest"
fi
for M in numpy; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3)"; exit 3; }; done
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_bundle.py --selftest | tail -2
$PY $D/e57/e57_bank.py --selftest | tail -2
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, full run skipped"; exit 0; }
[ -f "$OUT/e57_bundle/bundle_store.pkl" ] || { echo "$OUT/e57_bundle/bundle_store.pkl missing: job 229 first (exit 7)"; exit 7; }
STEP=distract; V="$OUT/e57_bank/validated.jsonl"; [ -f "$V" ] || { echo "validated.jsonl missing (exit 7)"; exit 7; }; [ -f "$OUT/e57_bank/distractors.json" ] && [ ! -f "$OUT/e57_bank/distractors_v3_37_job248.json" ] && cp -p "$OUT/e57_bank/distractors.json" "$OUT/e57_bank/distractors_v3_37_job248.json"; echo "distract on $(wc -l < "$V") validated questions"
case "$STEP" in
  packet)   $PY $D/e57/e57_bank.py packet --out "$OUT" ${N_PACKET:+--n $N_PACKET} ${EXCLUDE:+--exclude "$EXCLUDE"};;
  validate) [ -f "$BANK" ] || { echo "BANK=<questions.jsonl> missing (exit 7)"; exit 7; }; $PY $D/e57/e57_bank.py validate --out "$OUT" --bank "$BANK";;
  distract) $PY $D/e57/e57_bank.py distract --out "$OUT";;
  freeze)   $PY $D/e57/e57_bank.py freeze --out "$OUT" --version "${VERSION:-v1}" ${NOMINATIONS:+--nominations "$NOMINATIONS"};;
  *) echo "unknown STEP $STEP (exit 3)"; exit 3;;
esac
