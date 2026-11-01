#!/bin/bash
# JOB 610 = E57 PUBLIC QA STAGE 3 (E57 122-R / 123; coordinator request, Hans 'drop the control stream to the todo list, run the core quickly'): accumulating re-description stream (filler / original 332 paragraphs / re-description 1..4; 280 chunks; one write), arms NATQ and ALLQ128 (+ NAT: its per-chunk update norms are the NATQ / ALLQ128 norm targets, its states are read last), state + canary (measured only) at the end of each of the 6 sections, OUT reads of the 1,000 frozen stage-1 items (FastOUT, 640-row packing of stage 1 / job 600) for W0 and 18 states; evaluation subsets items455 / main373 of the 332 paragraphs. GAINQ / RANDQ and the control stream are NOT run. New files e57_pqa3_stream.py / e57_pqa3_run.py (+ e57_pqa2_run.py of job 600, unchanged); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa3_run.py e57_pqa3_stream.py e57_pqa2_run.py e57_pqa2_stream.py e57_kvreuse.py e57_qband_run.py e57_closure_masks.py e57_squad_w0_reads.py e55_core.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa3_stream.py --selftest | tail -1
$PY $S/e57_pqa3_run.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_public_qa_stage3}; SIN=$TOUT/in; DAT=$TOUT/data; mkdir -p "$TOUT" "$SIN"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
[ -f "$ITEMS" ] && [ -f "$BANK" ] && [ -f "$ST1" ] && [ -f data/squad/train-v1.1.json ] && [ -f data/squad/dev-v1.1.json ] || { echo "input missing (exit 7)"; exit 7; }
chk $DAT/paragraphs.json 18aaec5b1fbf8135f61791d2d4fc93725525aa6b31c114fd01c2ccd720a3b281
chk $DAT/rewrites_all.json 5ae0c6861cd77ef171c131fc77d668dfa301abe6ffff0ab6a54f97f1d5fb9220
[ "$(md5sum data/squad/train-v1.1.json | cut -c1-32)" = 981b29407e0affa3b1b156f72073b945 ] || { echo "train-v1.1.json md5 differs (exit 9)"; exit 9; }
[ -f "$SIN/streamP3_ids.npy" ] || $PY $S/e57_pqa3_stream.py --para $DAT/paragraphs.json --rewrites $DAT/rewrites_all.json --train data/squad/train-v1.1.json --dev data/squad/dev-v1.1.json --out "$SIN"
$PY $S/e57_pqa3_stream.py --check --out "$SIN"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa3_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa3_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_pqa3_run.py --stage write --stream-dir "$SIN" --out "$TOUT" --arms NAT,NATQ,ALLQ128
$PY $S/e57_pqa3_run.py --stage manifest --out "$TOUT" --manifest "$TOUT/e57_pqa3_manifest.json"
$PY $S/e57_pqa3_run.py --stage read --out "$TOUT" --manifest "$TOUT/e57_pqa3_manifest.json" --paragraphs $DAT/paragraphs.json --items-file $ITEMS --bank "$BANK" --stage1 $ST1
echo "outputs: $TOUT/E57_pqa3_keys.json $TOUT/E57_pqa3_rows_*.json $TOUT/E57_pqa3_{NAT,NATQ,ALLQ128}.arms.json"
