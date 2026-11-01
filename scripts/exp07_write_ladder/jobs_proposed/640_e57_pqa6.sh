#!/bin/bash
# JOB 640 = E57 PUBLIC QA STAGE 6 (E57 122-V / 123; Hans 'group ~10 paragraphs: write the originals, then re-description 1, ... and score all questions of those paragraphs'): 33 paragraph-group episodes (10-11 paragraphs each, fixed permutation of jobs 610 / 630), each from W0: sections original, re-description 1..8 in one stream (10-16 chunks), arms NATQ and ALLQ128 (NAT only as the norm source; ALLQ128 mask applied from chunk 0), states at n = 1, 3, 5, 9 versions, reads A / B / C of all questions of the group (455 items), OUT = stage-1 tail, FastOUT with packing per group. New file e57_pqa6_run.py (uses e57_pqa2_run.py, e57_pqa3_stream.py, e57_pqa4_run.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa6_run.py e57_pqa4_run.py e57_pqa3_stream.py e57_pqa2_run.py e57_pqa2_stream.py e57_kvreuse.py e57_qband_run.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa6_run.py --selftest | tail -1
D610=$OUT/e57_public_qa_stage3; D620=$OUT/e57_public_qa_stage4; D650=$OUT/e57_public_qa_stage5; TOUT=${TOUT:-$OUT/e57_public_qa_stage6}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; PAR=$D610/data/paragraphs.json; R8=$D650/data/rewrites_all8.json; ASG=$D620/E57_pqa4_assignB.json
[ -f "$ITEMS" ] && [ -f "$BANK" ] && [ -f "$ST1" ] && [ -f "$PAR" ] && [ -f "$R8" ] && [ -f "$ASG" ] || { echo "input missing (exit 7)"; exit 7; }
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
chk $PAR 18aaec5b1fbf8135f61791d2d4fc93725525aa6b31c114fd01c2ccd720a3b281
chk $R8 eb08d0aacc78a512a19003bbc27cf084357b8ab008b84e037eb82a0f494cdb72
[ -f "$TOUT/e57_pqa6_groups.json" ] || $PY $S/e57_pqa6_run.py --stage build --para $PAR --rewrites8 $R8 --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa6_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa6_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_pqa6_run.py --stage write --out "$TOUT"
$PY $S/e57_pqa6_run.py --stage read --out "$TOUT" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa6_run.py --stage analyze --out "$TOUT"
echo "outputs: $TOUT/E57_pqa6_keys.json $TOUT/E57_pqa6_rows_*.json $TOUT/e57_pqa6_groups.json $TOUT/e57_pqa6_writes.json $TOUT/e57_pqa6_read_meta.json"
