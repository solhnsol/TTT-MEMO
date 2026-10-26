#!/bin/bash
# JOB 275 = (274 refused: duplicate hard surfaces counted twice, fixed) E57 T1/T2 FREEZE (STEP=freeze, VERSION=v1; inputs = the files of job 272 in $OUT/e57_t1t2; never overwrites) (STEP=check_verify: check with decisions D, E plus verify of nominations-draft-2; no freeze) (269 plus: own-event titles out of the title easy list, film names out of the PERSON easy lists) (268 crashed with a NameError, fixed) (re-run with decisions D, E: cleaned PERSON easy lists, title/team easy pools, easy_extra of the nominations, draft-v4) E57 T1/T2 CANDIDATE PREPARATION, CPU ONLY (no GPU, no model, no download; E57 sections 72-75; private, internal research only). NOT APPROVED yet (Hans's own sentence with job number + commit is needed). Never touches jobs 264/265 or the RunPod pod; bank_v1 and its code are read only (hash recomputed), every output goes to the NEW dir $OUT/e57_t1t2.
#  STEP=check   (default) : (1) entity-masked 6-gram overlap (k=1 strict, 2 normal, 3 loose) of the 150 frames of $DRAFT with every sentence of the 871-article stream, the original bank frame and the sibling frames; (2) the 12 T2 golds: aliases / surface counts, fixed surface, c' ; (3) T2 easy (<= 20 per item, same class) and hard (same event; the other T2 golds of the event auto-included; gold aliases removed) lists with c' per candidate; T1 candidates read from bank_v1 (sha256 9374e1f5... recomputed).
#  STEP=verify  NOMINATIONS=<json of the news thread's 2-4 nominations per T2 item (+ optional easy_extra)> : stream presence / alias check, nominations_freeze.jsonl, e57_t1t2_extras_check.json (no packet evidence needed).
#  STEP=check_verify (default of job 271) = check with the nominations file for the easy extras, then verify of the hard nominations; NO freeze.
#  STEP=freeze  VERSION=v1 : t1t2_<VERSION>.json + .sha256 (never overwritten, exit 8).
# Outputs (small, CPU, seconds to a few minutes; EXPECTED RUN TIME is an ESTIMATE, unmeasured): $OUT/e57_t1t2/e57_t1t2_{overlap,t2_golds,t2_distractors,t2_distractors_ext,extras_check,key_numbers}.json. The server cannot write project subfolders: copy them FLAT to the project root with the e57_ prefix (already in the names), plus e57_t1t2_key_numbers.json as the small key-number file.
# CPU_ONLY=1 = code check, selftests, hash check of bank_v1, exit 0 (no stream is read).
set -eo pipefail
PY=$(realpath -s "$PY"); cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; COMMIT=b3ab898
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news}"; : "${DRAFT:=data/news-lite/t1t2_draft_v1.json}"; : "${STEP:=freeze}"; case "$STEP" in check|check_verify) : "${NOMINATIONS:=data/news-lite/t1t2_nominations_draft1.json}";; esac; T=$OUT/e57_t1t2
git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT missing (exit 4)"; exit 4; }
git diff --quiet $COMMIT -- $D/e57/e57_t1t2.py $D/e57/e57_t1t2_easy_pools.json || { echo "e57_t1t2.py differs from commit $COMMIT (exit 5)"; exit 5; }
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
$PY -I $D/e57/e57_t1t2.py --selftest
$PY -I $D/e57/e57_bank.py --selftest | tail -1
$PY -I - <<PYEOF
import sys,json;sys.path.insert(0,'$D/e57');import e57_t1t2 as T
b,c=T.bank_check('$OUT/e57_bank');print('bank_v1 sha256',c['sha256'][:16],'ok' if c['ok'] else 'MISMATCH');sys.exit(0 if c['ok'] else 6)
PYEOF
[ -z "$CPU_ONLY" ] || { echo "CPU_ONLY: checks done, stream not read"; exit 0; }
[ -f "$OUT/e57_bundle/bundle_store.pkl" ] || { echo "$OUT/e57_bundle/bundle_store.pkl missing (exit 7)"; exit 7; }
for f in e57_t1t2_t2_distractors_ext.json e57_t1t2_t2_golds.json e57_t1t2_overlap.json e57_bank/t2_nominations_freeze.jsonl; do [ -f "$T/$f" ] || { echo "$T/$f missing: run check+verify first (exit 7)"; exit 7; }; sha256sum "$T/$f"; done
mkdir -p "$T"; [ -e "$T/e57_bundle" ] || ln -s ../e57_bundle "$T/e57_bundle"
case "$STEP" in
  check)  $PY -I $D/e57/e57_t1t2.py check --out "$T" --draft "$DRAFT" --bank-dir "$OUT/e57_bank" ${NOMINATIONS:+--nominations "$NOMINATIONS"};;
  check_verify) $PY -I $D/e57/e57_t1t2.py check --out "$T" --draft "$DRAFT" --bank-dir "$OUT/e57_bank" --nominations "$NOMINATIONS" && $PY -I $D/e57/e57_t1t2.py verify --out "$T" --nominations "$NOMINATIONS";;
  verify) [ -f "$NOMINATIONS" ] || { echo "NOMINATIONS=<json> missing (exit 7)"; exit 7; }; $PY -I $D/e57/e57_t1t2.py verify --out "$T" --nominations "$NOMINATIONS";;
  freeze) $PY -I $D/e57/e57_t1t2.py freeze --out "$T" --draft "$DRAFT" --bank-dir "$OUT/e57_bank" --version "${VERSION:-v1}";;
  *) echo "unknown STEP $STEP (exit 3)"; exit 3;;
esac
