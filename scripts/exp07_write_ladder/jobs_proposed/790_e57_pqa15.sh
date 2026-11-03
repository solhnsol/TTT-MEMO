#!/bin/bash
# JOB 790 = E57 PUBLIC QA STAGE 15, BUNDLE EPISODES WITH 17 EXPOSURES (QA thread; Hans 11/03 13:21 KST '노출 더 늘려볼까 ... 일단 일부 문단만 빠르게 재서술' and 23:34 KST '병렬 작성으로 처리해'): the bundles 5 / 8 / 9 / 10 / 12 of job 640 (50 paragraphs, 73 items) with 16 re-descriptions per paragraph (rewrites_16_subset.json, sha256 9ee1c6da...): orig -> rw1 -> ... -> rw16 (17 sections of 10 paragraphs, SQuAD filler completion), arms NATQ, ALLQ128 (NAT norm source) (GAINQ512 p.10 dropped on Hans' instruction 'GAINQ는 빼자'); read points n = 1 / 3 / 5 / 9 / 13 / 17; reads: 16-candidate question read (A, B) and the closed same-bundle read (A, B_bundle), per-token log-probs; reproduction checks against job 640 (update norms before the n = 9 chunk, W0 rows). New file e57_pqa15_run.py (uses the stages of e57_pqa6_run.py / e57_pqa7_run.py unchanged on the new directory); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa15_run.py e57_pqa12_run.py e57_pqa7_run.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa15_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; D13=$OUT/e57_public_qa_stage13; D620=$OUT/e57_public_qa_stage4; D610=$OUT/e57_public_qa_stage3; D650=$OUT/e57_public_qa_stage5; IN=$OUT/e57_pqa15_in; TOUT=${TOUT:-$OUT/e57_public_qa_n17}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
[ "$(sha256sum $IN/rewrites_16_subset.json | cut -d' ' -f1)" = 9ee1c6da057089f68f627410733b8524f281307eaf3182e04f759813a0a92c35 ] || { echo "rewrites_16_subset.json sha256 differs (exit 9)"; exit 9; }
for f in $ITEMS $BANK $ST1 $ASG $IN/subset.json $D610/data/paragraphs.json $D650/data/rewrites_all8.json $S640/e57_pqa6_writes.json $S640/E57_pqa6_rows_A_W0.json data/squad/train-v1.1.json data/squad/dev-v1.1.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$TOUT/e57_pqa6_groups.json" ] || $PY $S/e57_pqa15_run.py --stage build --out "$TOUT" --src640 "$S640" --para $D610/data/paragraphs.json --rewrites8 $D650/data/rewrites_all8.json --rew16 $IN/rewrites_16_subset.json --subset $IN/subset.json --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_qa_n17.json" ] && [ -z "$RERUN" ] && { echo "e57_qa_n17.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_pqa6_writes.json" ] || $PY $S/e57_pqa6_run.py --stage write --out "$TOUT"
[ -f "$TOUT/e57_pqa15_read_meta.json" ] || $PY $S/e57_pqa15_run.py --stage read --out "$TOUT" --src640 "$S640" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa15_run.py --stage analyze --out "$TOUT" --items-file $ITEMS --json "$TOUT/e57_qa_n17.json"
echo "outputs: $TOUT/e57_qa_n17.json $TOUT/E57_pqa15_rows_*.json $TOUT/e57_pqa15_tokens.json $TOUT/e57_pqa15_read_meta.json"
