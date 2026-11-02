#!/bin/bash
# JOB 670 = E57 PUBLIC QA STAGE 9 (E57 122-Y / 123; Hans 'RANDQ with the same budget'): RANDQ512 p.10 (a random 10% of the eligible query rows per chunk, band 512, from chunk 0) on the paragraph-group episodes of job 640, written with the masks saved by job 650 (e57_pqa7_masks.npz), read A / B / C at n = 1 / 3 / 5 / 9; NATQ / ALLQ128 / GAINQ512 / W0 values and the NAT norm targets come from jobs 640 / 650. Uses e57_pqa7_run.py unchanged; the outputs are the E57_pqa7_* names of that script, linked as E57_pqa9_*.
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
S640=$OUT/e57_public_qa_stage6; S650=$OUT/e57_public_qa_stage7; D620=$OUT/e57_public_qa_stage4; TOUT=${TOUT:-$OUT/e57_public_qa_stage9}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$D620/E57_pqa4_assignB.json
for f in $ITEMS $BANK $ST1 $ASG $S640/e57_pqa6_writes.json $S640/e57_pqa6_read_meta.json $S650/e57_pqa7_masks.npz $S650/e57_pqa7_mask_stats.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa9_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa9_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
cp -n $S650/e57_pqa7_masks.npz $S650/e57_pqa7_mask_stats.json "$TOUT"/
$PY $S/e57_pqa7_run.py --stage write --out "$TOUT" --src640 "$S640" --arms RANDQ512P10
$PY $S/e57_pqa7_run.py --stage read --out "$TOUT" --src640 "$S640" --arms RANDQ512P10 --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
$PY $S/e57_pqa7_run.py --stage analyze --out "$TOUT" --src640 "$S640" --arms RANDQ512P10
for f in "$TOUT"/E57_pqa7_* "$TOUT"/e57_pqa7_*; do b=$(basename "$f"); ln -f "$f" "$TOUT/${b/pqa7/pqa9}"; done
echo "outputs: $TOUT/E57_pqa9_keys.json $TOUT/E57_pqa9_rows_*.json $TOUT/e57_pqa9_writes.json $TOUT/e57_pqa9_read_meta.json"
