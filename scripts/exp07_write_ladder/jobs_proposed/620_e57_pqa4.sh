#!/bin/bash
# JOB 620 = E57 PUBLIC QA STAGE 4 (E57 122-S / 123; Hans 'run this plan, add the variant work too'): binding-gain control reads of the job-610 states (NATQ, ALLQ128, NAT at the end of the six sections of the accumulating re-description stream) and W0, read only, no write. Conditions: B = the same 16 candidates under the question of another item (same answer type, different document, seeded), C = 'Answer:' only, A = own question re-computed for W0 and the three final states as a check against the stored 610 rows. FastOUT, 640-row packing, the 1,000 frozen stage-1 items. New file e57_pqa4_run.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_pqa4_run.py e57_pqa3_run.py e57_pqa2_run.py e57_kvreuse.py e57_squad_w0_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_pqa4_run.py --selftest | tail -1
D610=$OUT/e57_public_qa_stage3; TOUT=${TOUT:-$OUT/e57_public_qa_stage4}; mkdir -p "$TOUT"
ITEMS=$OUT/e57_public_qa/e57_public_qa_squad_items.json; BANK=${BANK:-$OUT/e57_news/e57_bank/bank_v1.json}; ST1=$OUT/e57_squad_w0/E57_squad_w0_reads.json; MAN=$D610/e57_pqa3_manifest.json; PAR=$D610/data/paragraphs.json
[ -f "$ITEMS" ] && [ -f "$BANK" ] && [ -f "$ST1" ] && [ -f "$MAN" ] && [ -f "$PAR" ] && [ -f "$D610/E57_pqa3_keys.json" ] || { echo "input missing (exit 7)"; exit 7; }
$PY - "$MAN" <<'PYE'
import json,sys,os
m=json.load(open(sys.argv[1]));miss=[k for k,v in m.items() if not os.path.exists(v['file'])];print('610 states:',len(m),'missing:',miss or 'none');sys.exit(1 if miss else 0)
PYE
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_pqa4_keys.json" ] && [ -z "$RERUN" ] && { echo "E57_pqa4_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
for C in B C A; do
  [ -f "$TOUT/E57_pqa4_meta_$C.json" ] || $PY $S/e57_pqa4_run.py --stage read --cond $C --out "$TOUT" --manifest "$MAN" --dir610 "$D610" --items-file $ITEMS --bank "$BANK" --stage1 $ST1
done
$PY $S/e57_pqa4_run.py --stage analyze --out "$TOUT" --dir610 "$D610" --manifest "$MAN" --stream-dir $D610/in --paragraphs "$PAR" --items-file $ITEMS
echo "outputs: $TOUT/E57_pqa4_keys.json $TOUT/E57_pqa4_item_rows.json $TOUT/E57_pqa4_assignB.json $TOUT/E57_pqa4_meta_{A,B,C}.json $TOUT/E57_pqa4_rows_*.json"
