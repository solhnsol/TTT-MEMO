#!/bin/bash
# JOB 680 = E57 PUBLIC QA STAGE 10 (E57 122-Z / 123; Hans 'unrelated-text control'): the 33 paragraph-group episodes of job 640 with the same structure (section ends, chunk counts, filler completion, read chunks) but UNRELATED SQuAD train text (no gold / variant / long candidate string of the items455 in it; e57_pqa10_build.py), arms NATQ and ALLQ128 (NAT as norm source), n = 1 / 3 / 5 / 9, A / B / C reads of the items455 with the assignment of job 620; FastOUTG packing as job 640. Runs the write / read / analyze stages of e57_pqa6_run.py unchanged on the control stream directory; outputs are linked as e57_pqa10_* names.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa10_build.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e57_kvreuse.py e57_qband_run.py e57_closure_masks.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa10_build.py --selftest | tail -1
$PY $S/e57_pqa6_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; D620=$OUT/e57_public_qa_stage4; D610=$OUT/e57_public_qa_stage3; TOUT=${TOUT:-$OUT/e57_public_qa_stage10}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json; PAR=$D610/data/paragraphs.json
for f in $ITEMS $BANK $ST1 $ASG $PAR $S640/e57_pqa6_groups.json $S640/e57_pqa6_group_ids.npz data/squad/train-v1.1.json data/squad/dev-v1.1.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$TOUT/e57_pqa6_groups.json" ] || $PY $S/e57_pqa10_build.py --src640 "$S640" --items-file $ITEMS --paragraphs $PAR --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa10_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa10_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_pqa6_run.py --stage write --out "$TOUT"
$PY $S/e57_pqa6_run.py --stage read --out "$TOUT" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa6_run.py --stage analyze --out "$TOUT"
for f in "$TOUT"/E57_pqa6_* "$TOUT"/e57_pqa6_*; do b=$(basename "$f"); ln -f "$f" "$TOUT/${b/pqa6/pqa10}"; done
echo "outputs: $TOUT/E57_pqa10_keys.json $TOUT/E57_pqa10_rows_*.json $TOUT/e57_pqa10_writes.json $TOUT/e57_pqa10_build_meta.json"
