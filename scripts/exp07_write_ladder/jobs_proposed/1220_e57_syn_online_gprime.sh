#!/bin/bash
# JOB 1220 = E57 ONLINE g' ABS2 RULE and W0 g' ABS2 RULE on ONE synthetic tune world (GPM thread, Hans 11/07 '온라인 쪽 실험부터' / '미리 승인할게'): arm M3OS2W16N05 = ALLQ256 on every row, loss weight 16 on the rows with g' > 2 nats where g' = nll_suffixband128 - nll_win is computed AT WRITE TIME with the CURRENT fast weights (prefix outputs = the natural window; only the 8 suffix layers get the band); arm M3TS2W16N05 = the same rule with the W0 g' of job 1210 (fixed masks). Both with inline reads (in-window / out-of-window) and canary. Usage: SEED=62 CELL=v3_16_1 PY=/abs/python bash 1220_e57_syn_online_gprime.sh
set -eo pipefail
export PYTHONUNBUFFERED=1
: "${SEED:?set SEED}"; : "${CELL:?set CELL}"
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_mrun.py e57_gprime_masks.py e57_gprime_report.py e57_syn_tokread.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_gprime_report.py --selftest | tail -1
ARMS=M3OS2W16N05,M3TS2W16N05
SQ=$OUT/e57_synq; TT=$OUT/e57_syn_gprime_s$SEED; mkdir -p "$TT"
if [ -f "$OUT/E55_v3a_NAT_s$SEED.saved.json" ]; then NTG=$OUT/E55_v3a_NAT_s$SEED.saved.json; else NTG=$SQ/E55_synqnat_NATSGD_s$SEED.saved.json; fi
for f in $NTG $OUT/e57_syn_s$SEED/syn_s${SEED}_nll128.npz $OUT/e57_suffixband/s$SEED/syn_s${SEED}_nllsuf.npz $SQ/syn_s${SEED}_online.npz $OUT/e57_syn_rules_s$SEED/e57_syn_tokread_s$SEED.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s${SEED}_gprime.npz" ] || $PY $S/e57_gprime_masks.py --seed $SEED --online "$SQ/syn_s${SEED}_online.npz" --out "$SQ/syn_s${SEED}_gprime.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqgp --m30 "$SQ/syn_s${SEED}_gprime.npz" --ntg-nat $NTG --arms $ARMS --save-apps 16,32,64,96,128
$PY $S/e57_syn_tokread.py --stage read --seed $SEED --cell $CELL --out "$TT" --tag synqgp --arms $ARMS
$PY $S/e57_gprime_report.py --seed $SEED --out "$TT/e57_gprime_report_s$SEED.json"
echo "outputs: $TT/e57_gprime_report_s$SEED.json"
