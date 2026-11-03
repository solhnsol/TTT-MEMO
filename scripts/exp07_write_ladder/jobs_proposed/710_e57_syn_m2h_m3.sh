#!/bin/bash
# JOB 710 = E57 SYNTHETIC M2h / M3 ARMS AGAINST ALLQ256 (thread 'ALLQ보다 나은 가림 방법 찾기', design confirmed by Hans 11/03 09:39 KST; seed 64 only): deciding rows = the top 30% of the eligible rows of every chunk by window gain (e57_syn_mmasks.py); M2H64 = the deciding rows see only the previous 64 tokens (Hans' replacement of M2H, card decision cmsg_01BzMhNUxkYTC1nqycfzS3Bk3pov7wRF1XPBL1UKwaXTU9), the other rows are ALLQ256; M3W2 / M3W4 = ALLQ256 mask on every row and loss weight 2 / 4 on the deciding rows; step size 0.5 x the NAT norm of the chunk (= the norm ALLQ256 really used: the ratio is 0.5000 in all 128 chunks, so the two norm criteria coincide and one run per arm suffices); inline reads, end states, canary as in job 360b (e57_syn_mrun.py); per-token end-state reads of the new arms (e57_syn_tokread.py --arms) and the analysis against ALLQ256 / NATQ (e57_syn_m_analyze.py). New files; no existing e57_*.py edited, no git checkout.
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
ARMS=${ARMS:-M2H64N05,M3W2N05,M3W4N05}; SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_mtokread}; TA=${TA:-$OUT/e57_syn_m_analyze}; mkdir -p "$TT" "$TA"
for f in $SQ/syn_s64_m30_masks.npz $OUT/E55_v3a_NAT_s64.saved.json $SQ/E55_synq_ALLQ256_s64.saved.json $OUT/e57_syn_tokread/e57_syn_tokread_s64.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TA/e57_syn_m_keys.json" ] && [ -z "$RERUN" ] && { echo "e57_syn_m_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --m30 $SQ/syn_s64_m30_masks.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --ntg-allq $SQ/E55_synq_ALLQ256_s64.saved.json --arms $ARMS
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --arms $ARMS
$PY $S/e57_syn_m_analyze.py --old $OUT/e57_syn_tokread/e57_syn_tokread_s64.json --new $TT/e57_syn_tokread_s64.json --out $TA/e57_syn_m_keys.json
echo "outputs: $TA/e57_syn_m_keys.json $TT/e57_syn_tokread_s64.json $SQ/E55_synq_{M2H64N05,M3W2N05,M3W4N05}_s64.{saved,read}.json"
