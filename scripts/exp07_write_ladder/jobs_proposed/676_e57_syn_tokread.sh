#!/bin/bash
# JOB 676 = E57 SYNTHETIC WORLD PER-TOKEN RE-READ (comparison thread card 'synthetic + news T1/T2', Hans 22:11 KST; read only, no write): the saved end states of the 15 seed-64 query-band arms of job 360b (cell v3_16_1) and of the E55 210s arms NAT / SENTPQ / SENTPQADAMH (seed 61, cell v3_64_4, states/E55_d64s_*_s61) are read again with the out-of-window reader of the original runs (world, items and rows rebuilt with the original arguments, reference packed scorer) with the log-prob of every candidate token stored; the summed values are compared with the stored read[128]['out'] / W0['out'] matrices. Only the end read (d = 128) can be repeated (the earlier reads were inline). Analysis: branch-token / first-token / full-sum ranks, A / B (first token and full sum), masked arms against NATQ (NAT for seed 61). New file e57_syn_tokread.py; no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_tokread.py e55_run.py e55_read.py e55_gpu.py e55_core.py s3.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_tokread.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_syn_tokread}; mkdir -p "$TOUT"
for f in $OUT/e57_synq/E55_synq_NATQ_s64.saved.json $OUT/E55_d64_NAT_s61.saved.json $OUT/states/E55_d64s_NAT_s61/state_dcNAT_511.npz $OUT/e57_synq/states/E55_synq_NATQ_s64/state_NATQ_s64_127.npz data/narrative_gutenberg; do [ -e "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_syn_tokread_keys.json" ] && [ -z "$RERUN" ] && { echo "e57_syn_tokread_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
for SEED in 64 61; do [ -f "$TOUT/e57_syn_tokread_s$SEED.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed $SEED --out "$TOUT"; done
$PY $S/e57_syn_tokread.py --stage analyze --out "$TOUT"
echo "outputs: $TOUT/e57_syn_tokread_keys.json $TOUT/e57_syn_tokread_s64.json $TOUT/e57_syn_tokread_s61.json"
