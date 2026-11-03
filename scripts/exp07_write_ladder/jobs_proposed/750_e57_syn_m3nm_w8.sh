#!/bin/bash
# JOB 750 = E57 SYNTHETIC WEIGHT-ONLY CONTROL (thread 'ALLQ보다 나은 가림 방법 찾기', Hans card '둘 다 진행' cmsg_01BzMhNUxkYTC1nqycfzS3Bk11UQsbqHkH7LSfmavFdobw; seed 64): NATQ (no mask, natural window, update norm 0.5 x NAT, Adam direction) with loss weight 8 on the deciding rows (top 30% of the eligible rows of every chunk by window gain = 307 rows); same reads and per-token store as jobs 710 / 730. Arm M3NMW8N05 added to e57_syn_mrun.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_mrun.py e57_syn_mmasks.py e57_syn_m_analyze.py e57_syn_tokread.py e57_qband_run.py e57_closure_masks.py e55_run.py e55_core.py e55_gpu.py s3.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_mmasks.py --selftest | tail -1; $PY $S/e57_syn_m_analyze.py --selftest | tail -1
ARMS=${ARMS:-M3NMW8N05}; SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_mtokread3}; TA=${TA:-$OUT/e57_syn_m_analyze3}; mkdir -p "$TT" "$TA"
for f in $SQ/syn_s64_m30_masks.npz $OUT/E55_v3a_NAT_s64.saved.json $SQ/E55_synq_ALLQ256_s64.saved.json $OUT/e57_syn_tokread/e57_syn_tokread_s64.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TA/e57_syn_m_keys.json" ] && [ -z "$RERUN" ] && { echo "e57_syn_m_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --m30 $SQ/syn_s64_m30_masks.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --ntg-allq $SQ/E55_synq_ALLQ256_s64.saved.json --arms $ARMS
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --arms $ARMS
$PY $S/e57_syn_m_analyze.py --old $OUT/e57_syn_tokread/e57_syn_tokread_s64.json --new $TT/e57_syn_tokread_s64.json --out $TA/e57_syn_m_keys.json
echo "outputs: $TA/e57_syn_m_keys.json $TT/e57_syn_tokread_s64.json $SQ/E55_synq_{M3NMW8N05}_s64.{saved,read}.json"
