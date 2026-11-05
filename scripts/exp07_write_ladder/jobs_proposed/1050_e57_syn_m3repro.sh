#!/bin/bash
# JOB 1050 = E57 SYNTHETIC REPRODUCTION of the narrow / adaptive weighted sets on FRESH 16-fact seeds (cell v3_16_1, 128 chunks, same world as s64). Hans 11/05 23:28 KST "진행해봐" (T05/T10/T15 w=16 vs M3 w=8 on other seeds). Usage: SEED=62 PY=/abs/python bash 1050_e57_syn_m3repro.sh   (then SEED=65). Arms (all ALLQ256 on every row): M3W1N05 (= ALLQ256), M3W8N05 (M3), M3T05W16N05, M3T10W16N05, M3T15W16N05, M3TA05W16N05 (global-threshold rule of job 1040). Prerequisites for the seed (server): E55_v3a_NAT_s$SEED.saved.json, e57_syn_s$SEED/syn_s$SEED_nll.npz (W0 pass), e57_syn_tokread for the seed (the stage must accept this seed). Missing input -> exit 7. Exploratory; per-seed numbers are reported next to s64, no pooled claim.
set -eo pipefail
export PYTHONUNBUFFERED=1
: "${SEED:?set SEED}"
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${ARMS:=M3W1N05,M3W8N05,M3T05W16N05,M3T10W16N05,M3T15W16N05,M3TA05W16N05}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_cmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_cmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=$OUT/e57_syn_repro_s$SEED; mkdir -p "$TT"
NTG=$OUT/E55_v3a_NAT_s$SEED.saved.json
for f in $NTG $OUT/e57_syn_s$SEED/syn_s${SEED}_nll.npz; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s${SEED}_repro_c.npz" ] || $PY $S/e57_syn_cmasks.py --seed $SEED --out "$SQ/syn_s${SEED}_repro_c.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_syn_mrun.py --seed $SEED --out $SQ --tag synqrp --m30 $SQ/syn_s${SEED}_repro_c.npz --ntg-nat $NTG --arms $ARMS --save-apps 16,32,64,96,128
$PY $S/e57_syn_tokread.py --stage read --seed $SEED --out "$TT" --tag synqrp
$PY $S/e57_curve_report.py --kind syn --seed $SEED --tag synqrp --tokread $TT/e57_syn_tokread_s$SEED.json --out $TT/e57_curve_repro_s$SEED.json
echo "outputs: $TT/e57_curve_repro_s$SEED.json; mask stats $SQ/syn_s${SEED}_repro_c.json"
