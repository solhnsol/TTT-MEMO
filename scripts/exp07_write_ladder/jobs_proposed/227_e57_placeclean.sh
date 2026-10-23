#!/bin/bash
# E57-LITE PLACE CLEAN-UP (CPU only, NO GPU, NO model, NO download; reads ONLY the output files of job 223: lite_attributes.json, lite_place.json, lite_bank.json, lite_events.json (+ lite_place_sample.json for the review table); takes seconds). Rules fixed before the result (full text in the docstring of e57_placeclean.py): drop events without place / with c_SA = 0 / whose answer A is not a place (small embedded gazetteer of countries, states, Indian states, large cities, or a place-suffix rule; person-like and non-place words rejected; no NER); 2nd place = best OTHER place after removing persons and non-places, s / s2 < 1.5 -> separate list 'hard' (not main); items = (event, S) with S sharing no core word with A. Output $OUT/e57_news/e57_lite/place_clean.json (+ .sha), place_clean_report.md: numbers of main / hard / dropped events and items, reasons, per-event tables, the 20 seeded extractions of job 223 with the verdicts, the places unknown to the gazetteer. No article text.
# EXPECTED RUN TIME (Hans confirms before it runs): seconds (under 1 min). Needs job 223 finished (exit 7 if its files are missing); exit 8 if the report exists (RERUN=1). PINNED INSTALL, NO FETCH (exit 4 / 5). CPU_ONLY=1 = install check, manifest, selftest, exit 0.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=34ed0920721b4fbb1eabeeec5a8ef62f6bb342e7
FILES="$D/e57/e57_placeclean.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news}"
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
faa878cec667 e57/e57_placeclean.py
MANIFEST
  echo "installed code verified against the manifest"
fi
$PY $D/e57/e57_placeclean.py --selftest | tail -1
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, full run skipped"; exit 0; }
$PY $D/e57/e57_placeclean.py --out "$OUT" ${RERUN:+--rerun}
