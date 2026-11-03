#!/bin/bash
# JOB 740 = E57 PUBLIC QA STAGE 14, READ-FORMAT COMPARISON (thread 'QA에서 가림이 안 되는 이유 규명', Hans '진행해' 11/03 11:29 KST; read only, NO weight update): the n = 9 states (NATQ, ALLQ128 of job 640; GAINQ512 p.10 of job 650) and W0 read on the items455 with the 16 candidates of job 674 (e57_public_qa_stage13) in six formats: Q (question style, reproduces job 674 condition A), C-orig (original paragraph, sentence start up to just before the answer), C-rw3 / C-rw6 / C-rw9 (version 3 / 6 / 9, job 720 numbering), C-short (last 6 words of the C-orig prompt); per-token log-probs; first / second / later-token log-prob, rank ratio, top-1, DeltaA vs W0, masking extras with item-level CIs, by answer type. New file e57_pqa14_run.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa14_run.py e57_pqa13_run.py e57_pqa12_run.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa14_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; S650=$OUT/e57_public_qa_stage7; S13=$OUT/e57_public_qa_stage13; TOUT=${TOUT:-$OUT/e57_qa_readformat}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json
[ "$(sha256sum $OUT/e57_public_qa_stage3/data/paragraphs.json | cut -d' ' -f1)" = 18aaec5b1fbf8135f61791d2d4fc93725525aa6b31c114fd01c2ccd720a3b281 ] || { echo "paragraphs.json sha256 differs (exit 9)"; exit 9; }
[ "$(sha256sum $OUT/e57_public_qa_stage5/data/rewrites_all8.json | cut -d' ' -f1)" = eb08d0aacc78a512a19003bbc27cf084357b8ab008b84e037eb82a0f494cdb72 ] || { echo "rewrites_all8.json sha256 differs (exit 9)"; exit 9; }
for f in $ITEMS $BANK $ST1 $S640/e57_pqa6_writes.json $S650/e57_pqa7_writes.json $S13/E57_pqa13_rows_A_W0.json $S13/E57_pqa13_rows_A_GAINQ512P10_n9.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_qa_readformat.json" ] && [ -z "$RERUN" ] && { echo "e57_qa_readformat.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_pqa14_read_meta.json" ] || $PY $S/e57_pqa14_run.py --stage read --out "$TOUT" --src640 "$S640" --src650 "$S650" --items-file $ITEMS --bank "$BANK" --stage1 $ST1 --ref674 "$S13" --root "$OUT"
$PY $S/e57_pqa14_run.py --stage analyze --out "$TOUT" --items-file $ITEMS --json "$TOUT/e57_qa_readformat.json"
echo "outputs: $TOUT/e57_qa_readformat.json $TOUT/E57_pqa14_rows_*.json $TOUT/e57_pqa14_tokens.json $TOUT/e57_pqa14_prompts.json"
