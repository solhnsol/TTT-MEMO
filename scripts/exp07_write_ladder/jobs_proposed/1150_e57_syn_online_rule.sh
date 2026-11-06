#!/bin/bash
# JOB 1150 = E57 ONLINE ABS2 RULE (GPM thread, Hans 11/06 '승인할게 시작해봐', after '실제 바인딩 점수를 봐야 아는 거 아냐?'): one synthetic tune world per call (SEED=62 CELL=v3_16_1 or SEED=72 CELL=v3_32_1): arm M3OA2W16N05 = ALLQ256 on every row, loss weight 16 on the rows with g128 > 2 nats among the eligible rows where g128 = nll_band128 - nll_win is computed AT WRITE TIME with the current fast weights (window and band-128 forwards of the chunk before its write, e57_syn_mrun.py UVOnline / install_online_dec), update norm 0.5 x NAT; compared with the arms of jobs 1071 / 1073 (ALLQ256, M3 w=8, top 5% w=16, W0-g ABS2 w=16): e57_online_rule_report.py.
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
for f in e57_syn_mrun.py e57_syn_rmasks.py e57_syn_tokread.py e57_online_rule_report.py e57_rules.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_online_rule_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=$OUT/e57_syn_onlinerule_s$SEED; mkdir -p "$TT"
if [ -f "$OUT/E55_v3a_NAT_s$SEED.saved.json" ]; then NTG=$OUT/E55_v3a_NAT_s$SEED.saved.json; else NTG=$SQ/E55_synqnat_NATSGD_s$SEED.saved.json; fi
for f in $NTG $OUT/e57_syn_s$SEED/syn_s${SEED}_nll128.npz $OUT/e57_syn_rules_s$SEED/e57_syn_tokread_s$SEED.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s${SEED}_online.npz" ] || $PY $S/e57_syn_rmasks.py --seed $SEED --cell $CELL --taus 2 --out "$SQ/syn_s${SEED}_online.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqog --m30 "$SQ/syn_s${SEED}_online.npz" --ntg-nat $NTG --arms M3OA2W16N05 --save-apps 16,32,64,96,128 --no-inline
$PY $S/e57_syn_tokread.py --stage read --seed $SEED --cell $CELL --out "$TT" --tag synqog --arms M3OA2W16N05
$PY $S/e57_online_rule_report.py --seed $SEED --out "$TT/e57_online_rule_report_s$SEED.json"
echo "outputs: $TT/e57_online_rule_report_s$SEED.json"
