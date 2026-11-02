#!/bin/bash
# JOB 650 = E57 PUBLIC QA STAGE 7 (E57 122-W / 123; Hans 'run GAINQ 512 10% too'): GAINQ512 p.10 on the paragraph-group episodes of job 640 (33 groups, original -> re-description 1..8, from W0, reads A / B / C at n = 1, 3, 5, 9, per-group packing). Stage score = forward-only W0 pass per group (natural and band-512 NLL per row; masks GAINQ512P10 and RANDQ512P10 are both built and saved, RANDQ512 is NOT written or read here unless ARMS=GAINQ512P10,RANDQ512P10 is set), stage write = SELC1 with the selected rows banded to 512 (mask from chunk 0, 0.5 x the NAT norm of job 640), stage read = A / B / C of the group's questions. NATQ / ALLQ128 / W0 values and NAT norm targets are taken from job 640. New file e57_pqa7_run.py (uses e57_pqa6_run.py, e57_pqa4_run.py, e57_pqa2_run.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa7_run.py e57_pqa6_run.py e57_pqa4_run.py e57_pqa2_run.py e57_kvreuse.py e57_qband_run.py e57_closure_masks.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa7_run.py --selftest | tail -1
S640=$OUT/e57_public_qa_stage6; D620=$OUT/e57_public_qa_stage4; TOUT=${TOUT:-$OUT/e57_public_qa_stage7}; mkdir -p "$TOUT"; ARMS=${ARMS:-GAINQ512P10}
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
for f in $ITEMS $BANK $ST1 $ASG $S640/E57_pqa6_keys.json $S640/e57_pqa6_writes.json $S640/e57_pqa6_groups.json $S640/e57_pqa6_read_meta.json $S640/E57_pqa6_rows_A_W0.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa7_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa7_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_pqa7_masks.npz" ] || $PY $S/e57_pqa7_run.py --stage score --out "$TOUT" --src640 "$S640"
$PY $S/e57_pqa7_run.py --stage write --out "$TOUT" --src640 "$S640" --arms $ARMS
$PY $S/e57_pqa7_run.py --stage read --out "$TOUT" --src640 "$S640" --arms $ARMS --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa7_run.py --stage analyze --out "$TOUT" --src640 "$S640" --arms $ARMS
echo "outputs: $TOUT/E57_pqa7_keys.json $TOUT/E57_pqa7_rows_*.json $TOUT/e57_pqa7_mask_stats.json $TOUT/e57_pqa7_writes.json $TOUT/e57_pqa7_read_meta.json"
