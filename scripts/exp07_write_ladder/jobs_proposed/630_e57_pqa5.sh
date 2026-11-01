#!/bin/bash
# JOB 630 = E57 PUBLIC QA STAGE 5 (E57 122-U / 123; Hans 'go, NATQ and ALLQ only'): nine-version accumulating stream (filler / original / re-description 1..8, ten sections, 453 chunks, one write) by NATQ and ALLQ128 (NAT is written only as the norm source and not read), reads A (own question) at all ten section ends, B (question of another item) and C ('Answer:' only) at the ends of re-description 4 and 8; W0 rows from jobs 610 / 620. FastOUT, 640-row packing, the 1,000 frozen stage-1 items. New files e57_pqa5_stream.py / e57_pqa5_run.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa5_run.py e57_pqa5_stream.py e57_pqa4_run.py e57_pqa3_run.py e57_pqa3_stream.py e57_pqa2_run.py e57_kvreuse.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa5_run.py --selftest | tail -1
D610=$OUT/e57_public_qa_stage3; D620=$OUT/e57_public_qa_stage4; TOUT=${TOUT:-$OUT/e57_public_qa_stage5}; SIN=$TOUT/in; DAT=$TOUT/data; mkdir -p "$TOUT" "$SIN"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; PAR=$D610/data/paragraphs.json
[ -f "$ITEMS" ] && [ -f "$BANK" ] && [ -f "$ST1" ] && [ -f "$PAR" ] && [ -f "$D620/E57_pqa4_assignB.json" ] && [ -f "$D620/E57_pqa4_rows_B_W0.json" ] && [ -f "$D620/E57_pqa4_rows_C_W0.json" ] && [ -f "$D610/E57_pqa3_rows_W0.json" ] || { echo "input missing (exit 7)"; exit 7; }
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
chk $PAR 18aaec5b1fbf8135f61791d2d4fc93725525aa6b31c114fd01c2ccd720a3b281
chk $DAT/rewrites_all8.json eb08d0aacc78a512a19003bbc27cf084357b8ab008b84e037eb82a0f494cdb72
[ -f "$SIN/streamP5_ids.npy" ] || $PY $S/e57_pqa5_stream.py --para $PAR --rewrites8 $DAT/rewrites_all8.json --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json --out "$SIN" --ref610 $D610/in
$PY $S/e57_pqa5_stream.py --check --out "$SIN" --ref610 $D610/in
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa5_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa5_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_pqa5_run.py --stage write --stream-dir "$SIN" --out "$TOUT"
RD="--out $TOUT --dir610 $D610 --dir620 $D620 --items-file $ITEMS --bank $BANK --stage1 $ST1"
for C in A B C; do
  $PY $S/e57_pqa5_run.py --stage manifest --cond $C --out "$TOUT" --manifest "$TOUT/e57_pqa5_manifest_$C.json"
  [ -f "$TOUT/E57_pqa5_meta_$C.json" ] || $PY $S/e57_pqa5_run.py --stage read --cond $C --manifest "$TOUT/e57_pqa5_manifest_$C.json" $RD
done
$PY $S/e57_pqa5_run.py --stage analyze --out "$TOUT" --dir610 "$D610" --dir620 "$D620" --stream-dir "$SIN" --paragraphs "$PAR" --items-file $ITEMS
echo "outputs: $TOUT/E57_pqa5_keys.json $TOUT/E57_pqa5_item_rows.json $TOUT/E57_pqa5_meta_{A,B,C}.json $TOUT/E57_pqa5_rows_*.json"
