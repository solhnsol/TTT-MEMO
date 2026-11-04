#!/bin/bash
# JOB 895 = E57 SYNTHETIC P2 PRE-CHECK OF THE UPDATE DIRECTION (investigation thread 'binding learned fast at few exposures', Hans 11/04 12:58 KST '2번 사전 점검부터'; design E57 122-AB addendum 16; seed 64, W0 weights, no stream training): at the chunks 16 / 32 / 64 / 96 / 127 the inner-MLP gradient of the ALLQ256 plain loss and of the M3 w=8 weighted loss; directions raw, sign, input-side whitening (rho 0.1 and 0.01), Muon (NS 5); each scaled to 0.5 x the NAT norm of the chunk and applied once to W0; per-token read of the 16 facts and the canary; report with 95% CIs over the facts (five chunks merged). New file e57_syn_p2dirs.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_p2dirs.py e57_syn_tokread.py e57_syn_pass.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_p2dirs.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1
TT=${TT:-$OUT/e57_syn_p2}; mkdir -p "$TT"
for f in $OUT/e57_synq/syn_s64_m30_masks.npz $OUT/E55_v3a_NAT_s64.saved.json $OUT/e57_synq/E55_synq_ALLQ256_s64.read.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_syn_p2_report.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TT/e57_syn_p2dirs.json" ] || $PY $S/e57_syn_p2dirs.py --stage run --out "$TT"
$PY $S/e57_syn_p2dirs.py --stage analyze --out "$TT"
echo "outputs: $TT/e57_syn_p2_report.json $TT/e57_syn_p2dirs.json"
