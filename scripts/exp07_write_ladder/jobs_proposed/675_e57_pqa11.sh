#!/bin/bash
# JOB 675 = E57 PUBLIC QA STAGE 11 (E57 122-AA / 123; Hans 'also try re-framed questions', read only): the stored states of jobs 640 (NATQ, ALLQ128), 650 (GAINQ512 p.10) and 670 (RANDQ512 p.10) at n = 1 / 3 / 5 / 9 and W0 read on the items455 with two paraphrased questions per item (A'v own variant, B'v variant of a paired item of items455, C re-used). FastOUTG packing per group. New file e57_pqa11_run.py (uses e57_pqa6_run.py, e57_pqa4_run.py, e57_pqa2_run.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa11_run.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e57_kvreuse.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa11_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; S650=$OUT/e57_public_qa_stage7; S670=$OUT/e57_public_qa_stage9; D620=$OUT/e57_public_qa_stage4; TOUT=${TOUT:-$OUT/e57_public_qa_stage11}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json; QV=$TOUT/data/qvariants_all.json
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
for f in $ITEMS $BANK $ST1 $ASG $QV $S640/e57_pqa6_writes.json $S650/e57_pqa7_writes.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
chk $QV 899d62b9b4619d16fcf21e73306ed86ebb8e102db56734f606899ebe9bf58543
[ -f "$S670/e57_pqa7_writes.json" ] && echo "job 670 states found: RANDQ512 is read too" || echo "job 670 writes not found: RANDQ512 is skipped"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa11_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa11_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_pqa11_read_meta.json" ] || $PY $S/e57_pqa11_run.py --stage read --out "$TOUT" --src640 "$S640" --src650 "$S650" --src670 "$S670" --qvariants $QV --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa11_run.py --stage analyze --out "$TOUT" --src640 "$S640" --src650 "$S650" --src670 "$S670"
echo "outputs: $TOUT/E57_pqa11_keys.json $TOUT/E57_pqa11_rows_*.json $TOUT/e57_pqa11_pairs.json $TOUT/e57_pqa11_read_meta.json"
