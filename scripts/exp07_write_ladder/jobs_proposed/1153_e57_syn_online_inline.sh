#!/bin/bash
# JOB 1153 = E57 ONLINE ABS2 RULE WITH INLINE READS (GPM thread / core-claim thread, Hans 11/06 '1153, 1160 승인'; world s62 = v3_16_1 by default, the second world s72 (1154) is NOT approved): the arm M3OA2W16N05 of job 1151 rewritten with the inline reads (in-window 'in_g0', out-of-window 'out', canary) at a = 16 / 32 / 64 / 128, tag synqoi; the end-state hash is compared with the end state of job 1151 (tag synqog, same arm, no inline reads) and the report e57_online_inwindow_report.py compares the reads with ALLQ256, M3 w=8, top 5% and the W0-g ABS2 of job 1071.
set -eo pipefail
export PYTHONUNBUFFERED=1
: "${SEED:=62}"; : "${CELL:=v3_16_1}"; export SEED CELL
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_mrun.py e57_online_inwindow_report.py e57_rules.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_online_inwindow_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=$OUT/e57_syn_onlinerule_s$SEED; mkdir -p "$TT"
if [ -f "$OUT/E55_v3a_NAT_s$SEED.saved.json" ]; then NTG=$OUT/E55_v3a_NAT_s$SEED.saved.json; else NTG=$SQ/E55_synqnat_NATSGD_s$SEED.saved.json; fi
for f in $NTG $SQ/syn_s${SEED}_online.npz $SQ/E55_synqog_M3OA2W16N05_s$SEED.saved.json $SQ/E55_synqrl_M3TA2W16N05_s$SEED.read.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqoi --m30 "$SQ/syn_s${SEED}_online.npz" --ntg-nat $NTG --arms M3OA2W16N05 --save-apps 16,32,64,96,128
$PY - <<'PYE'
import json,os
O=os.environ['EXP07_OUT']+'/e57_synq/';sd=os.environ.get('SEED','62')
a=json.load(open(O+f'E55_synqog_M3OA2W16N05_s{sd}.saved.json'))['reads']['127']['_hash'];b=json.load(open(O+f'E55_synqoi_M3OA2W16N05_s{sd}.saved.json'))['reads']['127']['_hash']
print('end-state hash equal to job 1151:',a==b,a[:12],b[:12])
PYE
$PY $S/e57_online_inwindow_report.py --seed $SEED --out "$TT/e57_online_inwindow_report_s$SEED.json"
echo "outputs: $TT/e57_online_inwindow_report_s$SEED.json"
