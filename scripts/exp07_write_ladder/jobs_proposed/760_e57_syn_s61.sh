#!/bin/bash
# JOB 760 = E57 SYNTHETIC SEED 61 REPLICATION (thread 'ALLQ보다 나은 가림 방법 찾기', Hans card '둘 다 진행' cmsg_01BzMhNUxkYTC1nqycfzS3Bk11UQsbqHkH7LSfmavFdobw; cell v3_64_4 = 64 facts, 512 chunks, read chunks 63 / 127 / 255 / 511): (1) W0 window pass over chunks 8..511 (natural window and band 256; e57_syn_w0pass.py), (2) deciding-row masks (top 30% of the eligible rows of every chunk by window gain; e57_syn_mmasks.py), (3) two writes in e57_syn_mrun.py: M3 weight 1 (= ALLQ256: ALLQ256 mask from chunk 8, ordinary loss; the equivalence is checked on seed 64 by job 755) and M3 weight 8 on the deciding rows, update norm 0.5 x the per-chunk norm of the plain NAT write of seed 61 (E55_d64_NAT_s61), same inline reads (appearances 16 / 32 / 64 / 128) and canary as the seed-64 runs, (4) divergence-token read of the two end states (e57_syn_tokread.py --seed 61) and the paired analysis against M3 weight 1 and the NAT state. No existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_w0pass.py e57_syn_mmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_syn_m_analyze.py e57_qband_run.py e57_closure_masks.py e55_run.py e55_core.py e55_gpu.py rope_fly.py s3.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_w0pass.py --selftest | tail -1; $PY $S/e57_syn_mmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_m_analyze.py --selftest | tail -1
P61=$OUT/e57_syn_s61; SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_mtokread61}; TA=${TA:-$OUT/e57_syn_m_analyze61}; mkdir -p "$P61" "$TT" "$TA"
ARMS=${ARMS:-M3W1N05,M3W8N05}
for f in $OUT/E55_d64_NAT_s61.saved.json $OUT/e57_syn_tokread/e57_syn_tokread_s61.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TA/e57_syn_m_keys.json" ] && [ -z "$RERUN" ] && { echo "e57_syn_m_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$P61/syn_s61_nll.npz" ] || $PY $S/e57_syn_w0pass.py --seed 61 --cell v3_64_4 --out "$P61" --bands 256
[ -f "$SQ/syn_s61_m30_masks.npz" ] || $PY $S/e57_syn_mmasks.py --seed 61 --cell v3_64_4 --dir "$P61" --out "$SQ/syn_s61_m30_masks.npz"
$PY $S/e57_syn_mrun.py --seed 61 --cell v3_64_4 --out $SQ --m30 $SQ/syn_s61_m30_masks.npz --ntg-nat $OUT/E55_d64_NAT_s61.saved.json --arms $ARMS
[ -f "$TT/e57_syn_tokread_s61.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 61 --out "$TT" --arms $ARMS
$PY $S/e57_syn_m_analyze.py --old $OUT/e57_syn_tokread/e57_syn_tokread_s61.json --new $TT/e57_syn_tokread_s61.json --out $TA/e57_syn_m_keys.json --seed 61 --refs M3W1N05,NAT
echo "outputs: $P61/syn_s61_nll.npz $SQ/syn_s61_m30_masks.npz $TA/e57_syn_m_keys.json $TT/e57_syn_tokread_s61.json $SQ/E55_synq_{M3W1N05,M3W8N05}_s61.{saved,read}.json"
