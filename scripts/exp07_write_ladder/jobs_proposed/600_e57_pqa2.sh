#!/bin/bash
# JOB 600 = E57 PUBLIC QA STAGE 2 (E57 122-Q; Hans 'reuse included, 2 cycles, no canary limit: experiment approved'): stream S = SQuAD dev without Super_Bowl_50 / Warsaw (46 docs, 1,964 paragraphs, doc order shuffled per cycle, 311 chunks per cycle, 622 chunks), arms NAT / NATQ / ALLQ128 / GAINQ128P10 / RANDQ128P10 (same engines and settings as jobs 570 / 585), 2 cycles in one continuous stream, state + canary (measured only, no stop threshold) at the end of cycle 1 and cycle 2; reads = the 1,000 frozen stage-1 items x 16 candidates, OUT (the stage-1 8192-token tail) with the KV-reuse scorer FastOUT. Order: (1) validation of FastOUT against the reference scorer (stored stage-1 OUT rows for all 1,000 items, a fresh reference run on 100 items, W0 and one trained state); (2) W0 score pass; (3) masks; (4) writes; (5) cycle-1 reads; (6) cycle-2 reads. New files e57_pqa2_stream.py / e57_pqa2_run.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa2_run.py e57_pqa2_stream.py e57_kvreuse.py e57_t1t2v2_lowp_run.py e57_t1t2v2_gainq_run.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e57_squad_w0_reads.py e57_main_run.py e55_core.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa2_stream.py --selftest | tail -1
$PY $S/e57_pqa2_run.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_public_qa_stage2}; SIN=$TOUT/in; mkdir -p "$TOUT" "$SIN"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; SMAN=$OUT/e57_t1t2v2_lowp/e57_t1t2v2_states_manifest.json
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
[ -f "$ITEMS" ] && [ -f "$BANK" ] && [ -f "$ST1" ] && [ -f "$SMAN" ] && [ -f data/squad/dev-v1.1.json ] || { echo "input missing (exit 7)"; exit 7; }
chk $ITEMS 79e842dbec77fe84ac7152bb26471d4a8127e47398301f998f32de277b1cb4ba
[ -f "$SIN/streamS_ids.npy" ] || $PY $S/e57_pqa2_stream.py --squad data/squad/dev-v1.1.json --out "$SIN"
$PY $S/e57_pqa2_stream.py --check --out "$SIN"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa2_keys_cycle2.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa2_keys_cycle2.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/E57_pqa2_validate.json" ] || $PY $S/e57_pqa2_run.py --stage validate --out "$TOUT" --items-file $ITEMS --bank "$BANK" --stage1 $ST1 --state-manifest $SMAN --val-states GAINQ128P10_B
[ -f "$TOUT/E57_v2B20_score_W0.npz" ] || $PY $S/e57_pqa2_run.py --stage score --stream-dir "$SIN" --out "$TOUT"
[ -f "$TOUT/e57_pqa2_masks.npz" ] || $PY $S/e57_pqa2_run.py --stage mask --stream-dir "$SIN" --out "$TOUT" --p 0.10
$PY $S/e57_pqa2_run.py --stage write --stream-dir "$SIN" --out "$TOUT"
for C in 1 2; do
  $PY $S/e57_pqa2_run.py --stage manifest --out "$TOUT" --manifest "$TOUT/e57_pqa2_manifest_cycle$C.json" --cycle $C
  [ -f "$TOUT/E57_pqa2_keys_cycle$C.json" ] || $PY $S/e57_pqa2_run.py --stage read --out "$TOUT" --manifest "$TOUT/e57_pqa2_manifest_cycle$C.json" --tag cycle$C --items-file $ITEMS --bank "$BANK"
done
echo "outputs: $TOUT/E57_pqa2_validate.json $TOUT/E57_pqa2_reads_cycle{1,2}.json $TOUT/E57_pqa2_keys_cycle{1,2}.json $TOUT/E57_pqa2_{NAT,NATQ,ALLQ128,GAINQ128P10,RANDQ128P10}.arms.json"
