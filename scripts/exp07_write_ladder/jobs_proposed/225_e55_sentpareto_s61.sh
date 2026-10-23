#!/bin/bash
# E55 LABEL-FREE MASKING: recall vs hidden-share Pareto of the sentence-selection grid on the FINISHED dump of seed 61 (CPU only, NO GPU, NO model, NO download, no training). Reads $OUT/e55_sentdump_s61.npz and $OUT/e55_sentdump_labels_s61.npz (written by job 213), evaluates the 589 configurations of e55_sentgrid.py plus a threshold sweep (every top-q% rule at q = 5/10/20/40/60/80/90, every z rule at z = 0/.5/1/2/3; about 1500 configurations), and reports per state (SENTPQADAMH, NAT) and per dump-chunk range (8-31 / 32-63 / 64-127 / all): the rule with the LOWEST hidden share at recall >= .85 / .90 / .95 and the recall-vs-hidden Pareto front (plot as png if matplotlib exists). Labels are used only for the metrics. Exploration on one seed: the minimum over many rules is optimistic; nothing here is a confirmed rule.
# Outputs $OUT/e55_sentpareto_s61.md (+ .json with the token counts of every configuration and chunk, + _<state>.png). Exit 8 if the report exists (RERUN=1 to redo), exit 7 if the dump or label file is missing.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 0.2-0.6 h: the grid of job 213 was estimated at 3-10 min for 589 configurations (V7 part about 1 min on synthetic arrays of real size), here about 2.5x the configurations over --workers 4 (WORKERS=4 default). Contended server up to 2x.
# Needs: lab python with numpy (exit 3 if missing). Writes only the files above. PINNED INSTALL, NO FETCH (exit 4 / 5). CPU_ONLY=1 = install check, manifest, selftest, input check, then exit 0.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=8ec14bae3282df6b8f9ea0f0ea11781c0f9b6d5f
FILES="$D/e55_sentpareto.py $D/e55_sentgrid.py $D/e55_sentsel.py"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; mkdir -p "$OUT"; : "${WORKERS:=4}"; : "${SEED:=61}"
$PY -c "import numpy" || { echo "numpy missing (exit 3)"; exit 3; }
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
862e0a43ab04 e55_sentpareto.py
24b0b4970d22 e55_sentgrid.py
3728d3954e71 e55_sentsel.py
MANIFEST
  echo "installed code verified against the manifest"
fi
REP="$OUT/e55_sentpareto_s$SEED.md"
[ -z "$RERUN" ] && [ -e "$REP" ] && { echo "report exists (exit 8; RERUN=1 to redo): $REP"; exit 8; }
DUMP="$OUT/e55_sentdump_s$SEED.npz"; LAB="$OUT/e55_sentdump_labels_s$SEED.npz"
[ -f "$DUMP" ] && [ -f "$LAB" ] || { echo "dump or labels missing: $DUMP $LAB (job 213 first) (exit 7)"; exit 7; }
$PY $D/e55_sentpareto.py --selftest | tail -1
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, full run skipped"; exit 0; }
$PY $D/e55_sentpareto.py --dump "$DUMP" --labels "$LAB" --out "$REP" --workers $WORKERS | head -150
