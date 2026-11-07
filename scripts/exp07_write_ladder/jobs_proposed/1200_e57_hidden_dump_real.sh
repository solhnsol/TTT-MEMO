#!/bin/bash
# JOB 1200 = E57 W0 PREFIX HIDDEN-STATE DUMP of the QA bundles 5/8/9/10/12 and news stream A (GPM thread, Hans 11/07 '1~2단계 먼저'); forward only, W0, no write. About 0.005 GiB per chunk (4.1 GB in all).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_hidden_dump_real.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_hidden_dump_real.py --selftest | tail -1
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_hidden_dump_real.py --out "$OUT/e57_hidden_dump_real"
echo "outputs: $OUT/e57_hidden_dump_real/hid_qa_g{5,8,9,10,12}.npz hid_news_A.npz"
