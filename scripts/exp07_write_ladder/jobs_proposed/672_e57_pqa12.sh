#!/bin/bash
# JOB 690 = E57 PUBLIC QA STAGE 12 (E57 122-AB / 123; Hans (a)+(b) of the synthetic / news / QA comparison thread, read only): bundle-matrix reads of the stored states of jobs 640 (NATQ, ALLQ128) and 650 (GAINQ512 p.10) at n = 1 / 3 / 5 / 9 and W0: the candidates of the question of item i are the gold strings of the other items of the same group (related / duplicate strings left out); rank, top-1, gold-minus-candidate difference, A and B_bundle (the same string under the other questions of the group); occurrence counts of every candidate string in the written text. FastOUTG packing per group. New file e57_pqa12_run.py (uses e57_pqa6_run.py, e57_pqa8_run.py, e57_pqa4_run.py, e57_pqa2_run.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa12_run.py e57_pqa8_run.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e57_kvreuse.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa12_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; S650=$OUT/e57_public_qa_stage7; D610=$OUT/e57_public_qa_stage3; D650=$OUT/e57_public_qa_stage5; TOUT=${TOUT:-$OUT/e57_public_qa_stage12}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; PAR=$D610/data/paragraphs.json; R8=$D650/data/rewrites_all8.json
for f in $ITEMS $BANK $ST1 $PAR $R8 $S640/e57_pqa6_writes.json $S650/e57_pqa7_writes.json $S640/e57_pqa6_groups.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa12_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa12_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_pqa12_read_meta.json" ] || $PY $S/e57_pqa12_run.py --stage read --out "$TOUT" --src640 "$S640" --src650 "$S650" --items-file $ITEMS --bank "$BANK" --stage1 $ST1 --para $PAR --rewrites8 $R8
$PY $S/e57_pqa12_run.py --stage analyze --out "$TOUT" --src640 "$S640" --src650 "$S650"
echo "outputs: $TOUT/E57_pqa12_keys.json $TOUT/E57_pqa12_rows_*.json $TOUT/e57_pqa12_layout.json $TOUT/e57_pqa12_counts.json $TOUT/e57_pqa12_read_meta.json"
