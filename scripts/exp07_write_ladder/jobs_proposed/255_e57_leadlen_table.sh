#!/bin/bash
# JOB 255 = E57 LEAD-LENGTH TABLE (CPU only, NO GPU, NO model, NO download; read-only on the bank files; writes leadlen_table.json / .md). Every independent article truncated by one fixed rule (first 1, 2, 3 sentences; first 64, 128, 192 tokens; current lead as reference): updates per cycle, c per question, questions below c = 10, min and median exposure and total updates for R = 1..8.
# EXPECTED RUN TIME (Hans confirms before it runs): measured in a scratch run 4.2 s. Commit ba08820012fdbdafbd73e9cd19e6961b29a8d54b.
set -eo pipefail
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder;COMMIT=ba08820012fdbdafbd73e9cd19e6961b29a8d54b;OUT=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news
if [ -z "$NO_INSTALL" ]; then git cat-file -e $COMMIT^{commit} || { echo "commit missing (exit 4)"; exit 4; }; git checkout $COMMIT -- $D/e57/e57_leadlen.py; git checkout 5c835409e9a32b77829fdb20e4022a6547f06705 -- $D/e56/e56_text.py $D/e57/e57_common.py; git checkout 2c05884a35c84a2c9906b8beebd9d2a8d58d5979 -- $D/e57/e57_bank.py; fi
for x in "55aa897f59b0 e57/e57_leadlen.py" "7843a666e1a8 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (exit 5)"; exit 5; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: install check done"; exit 0; }
for f in e57_bank/validated.jsonl e57_bundle/bundle_store.pkl; do [ -f "$OUT/$f" ] || { echo "$f missing (exit 7)"; exit 7; }; done
[ -f "$OUT/e57_bank/leadlen_table.json" ] && [ -z "$RERUN" ] && { echo "leadlen_table.json exists (exit 8)"; exit 8; }
$PY $D/e57/e57_leadlen.py --out "$OUT" --tokenizer data/tokenizer/llama3/tokenizer.json
