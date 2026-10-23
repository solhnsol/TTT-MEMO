#!/bin/bash
# E57 section 8: CPU-only measurement on the Federal Register. NO GPU, NO model. Approved by Hans for a CPU check (2026-10-23 04:28Z, "둘 다 승인할게 cpu로 점검해봐").
# What it does: lists proposed-rule / rule / correction documents of 2026-01-01..2026-10-19 from the Federal Register API (federalregister.gov/api/v1, JSON metadata + full-text), groups them by docket id or RIN (union-find; chain groups with more than 40 documents are excluded), and counts for each group the facts (regex proxy: numbers, money, percent, dates, capitalised names) repeated across >= 2 documents of the group, with the same M-style numbers as the news job (repeat count c, spread f, lead share, conflict proxy, power) and the pass criteria printed next to each number. Report $OUT/e57_fr_report.md (+ .json).
# PROXIES: facts are regular-expression candidates; grouping is by docket / RIN metadata, so documents without either are not grouped.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. Tens of minutes up to about 1 h, CPU only, a few GB RAM; it is limited by the API speed (about 10-20 k documents; metadata pages of 1000, text fetched per document and cached in $OUT). The api path was NOT run by the author (the author container cannot reach federalregister.gov); only the milo stand-in source (--source milo, a local parquet copy) was run.
# Needs: the lab python with numpy, tokenizers (pyarrow only for SOURCE=milo), network to federalregister.gov, the repo tokenizer data/tokenizer/llama3/tokenizer.json. Writes only under $OUT. Exit 8 if the report exists unless RERUN=1. NOT APPROVED to run until Hans confirms the time.
# PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch).
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=10853fe4aa571c1b7e5276c482da2b70e21dce3a
FILES="$D/e56/e56_text.py $D/e57/e57_common.py $D/e57/e57_fr.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_fr}"; mkdir -p "$OUT"; : "${SOURCE:=api}"
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
dd65a4c921bf e57/e57_fr.py
MANIFEST
  echo "installed code verified against the manifest"
fi
MODS="numpy tokenizers"; [ "$SOURCE" = milo ] && MODS="$MODS pyarrow"
for M in $MODS; do $PY -c "import $M" 2>/dev/null || { echo "python module $M missing in $PY (exit 3); nothing is installed by this job"; exit 3; }; done
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing (exit 3)"; exit 3; }
REP=$OUT/e57_fr_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $D/e57/e57_common.py --selftest
$PY $D/e57/e57_fr.py --source "$SOURCE" --out "$OUT" ${RERUN:+--rerun}
tail -60 "$REP"
