#!/bin/bash
# JOB 660 = E57 PUBLIC QA STAGE 8 (E57 122-X / 123; Hans 'a + b + c', read only): control reads of the stored states of jobs 640 (NATQ, ALLQ128) and 650 (GAINQ512 p.10) at n = 1 / 3 / 5 / 9 and W0. (a) 536 of the 545 non-items455 items whose gold string / variants do not occur in the written text of their assigned group, A / B / C reads; (b) mean NLL of 200 fixed unrelated SQuAD train paragraphs under every state, generation canary of the n = 9 states; (c) CPU analysis by occurrence strata of the gold string. FastOUTG packing per group. New file e57_pqa8_run.py (uses e57_pqa6_run.py, e57_pqa4_run.py, e57_pqa2_run.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa8_run.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e57_pqa3_stream.py e57_kvreuse.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa8_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; S650=$OUT/e57_public_qa_stage7; D620=$OUT/e57_public_qa_stage4; D610=$OUT/e57_public_qa_stage3; D650=$OUT/e57_public_qa_stage5; TOUT=${TOUT:-$OUT/e57_public_qa_stage8}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json; PAR=$D610/data/paragraphs.json; R8=$D650/data/rewrites_all8.json
for f in $ITEMS $BANK $ST1 $ASG $PAR $R8 $S640/e57_pqa6_writes.json $S650/e57_pqa7_writes.json $S650/E57_pqa7_keys.json data/squad/train-v1.1.json data/squad/dev-v1.1.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$TOUT/e57_pqa8_selection.json" ] || $PY $S/e57_pqa8_run.py --stage select --out "$TOUT" --src640 "$S640" --items-file $ITEMS --para $PAR --rewrites8 $R8
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa8_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa8_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_pqa8_read_meta.json" ] || $PY $S/e57_pqa8_run.py --stage read --out "$TOUT" --src640 "$S640" --src650 "$S650" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1 --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json --canary 9
$PY $S/e57_pqa8_run.py --stage analyze --out "$TOUT" --src640 "$S640" --src650 "$S650" --items-file $ITEMS --para $PAR --rewrites8 $R8
echo "outputs: $TOUT/E57_pqa8_keys.json $TOUT/E57_pqa8_generic.json $TOUT/e57_pqa8_selection.json $TOUT/E57_pqa8_rows_*.json $TOUT/e57_pqa8_read_meta.json"
