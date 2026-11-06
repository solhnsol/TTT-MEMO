#!/bin/bash
# JOB 1140 = E57 PAIR-SHUFFLE CONTROL OF THE RULE ARMS, SYNTHETIC (GPM thread, Hans 11/06 '1번 2번 먼저 하자'): same method as job 960 (within-chunk place derangement, E57_SYN_SHUF=perm1) for ONE world per call: SEED=62 CELL=v3_16_1 or SEED=72 CELL=v3_32_1 (s71 is at the rank-ratio floor and is skipped). On the SHUFFLED stream actually written: (1) band-128 window-gain pass (e57_syn_w0pass.py --bands 128, own directory), (2) rule masks from that g128 (e57_syn_rmasks.py --nll: ABS2 = g128 > 2 nats [qtA2], fixed top 5% [qt05], top 30% [qb_TOP30], eligible rows of the shuffled stream), (3) NAT norms of the shuffled stream (arm NATSGD, --ntg-nat auto), (4) arms ALLQ256 (M3W1N05), M3 w=8 (M3W8N05), top 5% w=16 (M3T05W16N05), ABS2 w=16 (M3TA2W16N05), update norm 0.5 x NAT, ALLQ256 on every row, save points 64 / 128; (5) per-token reads with the ORIGINAL pairs and the report e57_rule_shuf_report.py against the original-stream arms of jobs 1071-1073.
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
for f in e57_syn_shuf.py e57_syn_w0pass.py e57_syn_rmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_rule_shuf_report.py e57_rules.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_w0pass.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_rule_shuf_report.py --selftest | tail -1
SQ=$OUT/e57_synq; PL=$OUT/e57_syn_shufr_s$SEED; TT=$OUT/e57_syn_ruleshuf_s$SEED; mkdir -p "$PL" "$TT"
[ -f "$OUT/e57_syn_rules_s$SEED/e57_syn_tokread_s$SEED.json" ] || { echo "input missing: original-stream tokread (exit 7)"; exit 7; }
E57_SYN_SHUF=perm1 $PY $S/e57_syn_shuf.py --check --seed $SEED --cell $CELL --out "$TT/e57_shuf_check_s$SEED.json"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
export E57_SYN_SHUF=perm1
[ -f "$PL/syn_s${SEED}_nll.npz" ] || $PY $S/e57_syn_w0pass.py --seed $SEED --cell $CELL --out "$PL" --bands 128
[ -f "$SQ/syn_s${SEED}_ruleshuf.npz" ] || $PY $S/e57_syn_rmasks.py --seed $SEED --cell $CELL --taus 2 --nll "$PL/syn_s${SEED}_nll.npz" --out "$SQ/syn_s${SEED}_ruleshuf.npz"
$PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqsr --m30 "$SQ/syn_s${SEED}_ruleshuf.npz" --ntg-nat auto --arms NATSGD,M3W1N05,M3W8N05,M3T05W16N05,M3TA2W16N05 --save-apps 64,128 --no-inline
unset E57_SYN_SHUF
$PY $S/e57_syn_tokread.py --stage read --seed $SEED --cell $CELL --out "$TT" --tag synqsr --arms M3W1N05,M3W8N05,M3T05W16N05,M3TA2W16N05
$PY $S/e57_rule_shuf_report.py --seed $SEED --out "$TT/e57_rule_shuf_report_s$SEED.json"
echo "outputs: $TT/e57_rule_shuf_report_s$SEED.json; masks and their shares $SQ/syn_s${SEED}_ruleshuf.json"
