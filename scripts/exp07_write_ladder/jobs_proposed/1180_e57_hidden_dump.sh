#!/bin/bash
# JOB 1180 = E57 W0 PREFIX HIDDEN-STATE DUMP of the six synthetic worlds (core-claim / GPM thread, Hans 11/07 '은닉 상태 먼저, 결과를 보고 확장'): train worlds s62 (v3_16_1), s71 (v3_4_1), s72 (v3_32_1), evaluation worlds s65 (v3_16_1), s73 (v3_4_1), s74 (v3_32_1); the last prefix layer output of the natural-window forward of the whole stream, bf16 bits as uint16 [T, 2560]; forward only, W0, no write. About 0.7 GiB per world.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_hidden_dump.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_hidden_dump.py --selftest | tail -1
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_hidden_dump.py --worlds 62:v3_16_1,71:v3_4_1,72:v3_32_1,65:v3_16_1,73:v3_4_1,74:v3_32_1 --out "$OUT/e57_hidden_dump"
echo "outputs: $OUT/e57_hidden_dump/hid_s{62,71,72,65,73,74}.npz"
