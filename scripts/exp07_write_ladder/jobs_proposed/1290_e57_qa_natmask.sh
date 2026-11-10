#!/bin/bash
# JOB 1290 = E57 PUBLIC QA 65-VERSION STREAM, ORIGINAL-NAT (clipped SGD) WRITE ARMS (Hans 11/09 22:54 KST; EXPLORATORY per Council): bundles 5/8/9/10/12, arms NATS, NATMASK (ALLQ128 from chunk 0), NATMASKC (same number of hidden keys, random keys), DIRmaskNORMfull, DIRfullNORMmask via e57_pqa_natmask_run.py; reads at n=65 (same protocol as job 1251b). Output $OUT/e57_public_qa_n65_natmask/. ESTIMATE ONLY (job 640: 3.4 h for 3 arms incl. reads; swap arms double the backward passes): ~6-7 h. NOT YET SMOKE-TESTED on GPU (new runner, CPU selftest only) -> run a 1-bundle smoke first. NOTE: QA n=65 is EXPLORATORY (Council 11/09).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
$PY $S/e57_pqa_natmask_run.py --selftest | tail -1
N17=$OUT/e57_public_qa_n17; N65=$OUT/e57_public_qa_n65; RD=${RD:-$OUT/e57_public_qa_n65_natmask}; mkdir -p "$RD"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; ASG=$OUT/e57_public_qa_stage4/E57_pqa4_assignB.json
for f in $N65/e57_pqa6_groups.json $N65/e57_pqa6_group_ids.npz $N65/e57_dwg_qa_nll.npz $N17/e57_pqa6_groups.json $ITEMS $BANK $ST1 $ASG; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
for f in e57_pqa6_groups.json e57_pqa6_group_ids.npz e57_dwg_qa_nll.npz; do [ -f "$RD/$f" ] || cp "$N65/$f" "$RD/$f"; done
$PY $S/e57_pqa_natmask_run.py --stage write --out "$RD"
$PY $S/e57_pqa_natmask_run.py --stage read --out "$RD" --src640 "$N17" --items-file $ITEMS --bank "$BANK" --assign $ASG --stage1 $ST1
echo "outputs: $RD/e57_pqa6_writes.json, $RD/E57_pqa15_rows_*_n65.json"
