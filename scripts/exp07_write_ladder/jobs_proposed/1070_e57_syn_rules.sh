#!/bin/bash
# JOB 1070 = E57 SYNTHETIC RULE SEARCH, ONE WORLD (Hans 11/05 approved: ABS-NAT first with tau {2,4,6} nats (may change after the g128 quantiles), w = 16 and 8; g = nll_band128 - nll_win; ALLQ256 on every row as in M3). Usage: SEED=62 CELL=v3_16_1 PY=/abs/python bash 1070_e57_syn_rules.sh  (tuning worlds: 62 v3_16_1, 71 v3_4_1, 72 v3_32_1; held-out: 65 v3_16_1, 73 v3_4_1, 74 v3_32_1). Arms: M3W1N05 (ALLQ256), M3W8N05 (M3), M3T05W16N05 (fixed top 5%, g128), M3TA2W16N05, M3TA4W16N05, M3TA6W16N05. Add M3TA*W8N05 or other rules with ARMS=... Inputs from job 1060: e57_syn_s$SEED/syn_s${SEED}_nll128.npz and the NAT norm file ($NTG). Exploratory; the mean share per rule is reported in the mask json.
set -eo pipefail
export PYTHONUNBUFFERED=1
: "${SEED:?set SEED}"; : "${CELL:?set CELL}"
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${ARMS:=M3W1N05,M3W8N05,M3T05W16N05,M3TA2W16N05,M3TA4W16N05,M3TA6W16N05}"
: "${TAUS:=2,4,6}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_rules.py e57_syn_rmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_rules.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=$OUT/e57_syn_rules_s$SEED; mkdir -p "$TT"
if [ -f "$OUT/E55_v3a_NAT_s$SEED.saved.json" ]; then NTG=$OUT/E55_v3a_NAT_s$SEED.saved.json; else NTG=$SQ/E55_synqnat_NATSGD_s$SEED.saved.json; fi
for f in $NTG $OUT/e57_syn_s$SEED/syn_s${SEED}_nll128.npz; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s${SEED}_rules.npz" ] || $PY $S/e57_syn_rmasks.py --seed $SEED --cell $CELL --taus $TAUS --out "$SQ/syn_s${SEED}_rules.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqrl --m30 $SQ/syn_s${SEED}_rules.npz --ntg-nat $NTG --arms $ARMS --save-apps 16,32,64,96,128
$PY $S/e57_syn_tokread.py --stage read --seed $SEED --cell $CELL --out "$TT" --tag synqrl
$PY $S/e57_curve_report.py --kind syn --seed $SEED --tag synqrl --tokread $TT/e57_syn_tokread_s$SEED.json --out $TT/e57_curve_rules_s$SEED.json
echo "outputs: $TT/e57_curve_rules_s$SEED.json; mask stats $SQ/syn_s${SEED}_rules.json (mean share per rule)"
