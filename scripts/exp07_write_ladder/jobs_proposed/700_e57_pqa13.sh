#!/bin/bash
# JOB 700 = E57 PUBLIC QA STAGE 13 (E57 122-AC / 123; Hans card '690 + QA n=9 / W0 re-read', read only): per-token re-read of the n = 9 states (NATQ, ALLQ128 of job 640, GAINQ512 p.10 of job 650) and W0 on the items455 with the original 16 candidates and A / B / C; token log-probs stored; branch-token, first-token and full-sum ranks and A-B gains. FastOUTG packing per group (reproduces the stored sums of jobs 640 / 650). New file e57_pqa13_run.py (uses e57_pqa12_run.py, e57_pqa6_run.py, e57_pqa4_run.py, e57_pqa2_run.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa13_run.py e57_pqa12_run.py e57_pqa8_run.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e57_kvreuse.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa13_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; S650=$OUT/e57_public_qa_stage7; D620=$OUT/e57_public_qa_stage4; TOUT=${TOUT:-$OUT/e57_public_qa_stage13}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
for f in $ITEMS $BANK $ST1 $ASG $S640/e57_pqa6_writes.json $S650/e57_pqa7_writes.json $S640/E57_pqa6_rows_A_W0.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa13_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa13_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_pqa13_read_meta.json" ] || $PY $S/e57_pqa13_run.py --stage read --out "$TOUT" --src640 "$S640" --src650 "$S650" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa13_run.py --stage analyze --out "$TOUT" --src640 "$S640" --src650 "$S650"
echo "outputs: $TOUT/E57_pqa13_keys.json $TOUT/E57_pqa13_rows_*.json $TOUT/e57_pqa13_tokens.json $TOUT/e57_pqa13_read_meta.json"
