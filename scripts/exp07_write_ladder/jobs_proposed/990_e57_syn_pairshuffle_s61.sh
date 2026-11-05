#!/bin/bash
# JOB 990 = E57 SYNTHETIC 64-FACT WORLD (seed 61, cell v3_64_4), PAIR-SHUFFLED STREAM (Hans '970 990 먼저 돌리자', ~4.5 h): as job 960 (seed 64): within every chunk the place strings of the 16 sentences exchange by a random derangement (e57_syn_shuf.py, E57_SYN_SHUF=perm1; names / frames / filler / positions unchanged, token lengths kept, total exposures per place string = original); reads, 16 candidates and question frames unchanged = scored with the original pairs. Arms NATQ (M3NMW1N05), ALLQ256 (M3W1N05), M3 w=8 (M3W8N05), update norm 0.5 x NAT (NATSGD on the shuffled stream first); window-gain rows recomputed on the shuffled stream (W0 pass over 512 chunks + e57_syn_mmasks.py). Save points appearances 16 / 32 / 64 / 96 / 128 (the original states: job 970 NATQ, job 810 ALLQ256 / M3). Report e57_syn_shuf_report.py --seed 61.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_shuf.py e57_syn_shuf_report.py e57_syn_w0pass.py e57_syn_mmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_w0pass.py --selftest | tail -1; $PY $S/e57_syn_mmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_syn_shuf_report.py --selftest | tail -1
PL=$OUT/e57_syn_shuf_s61; SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_shuf_curve_s61}; mkdir -p "$PL" "$TT"
for f in $OUT/e57_syn_natq_s61/e57_syn_tokread_s61.json $OUT/e57_syn_curve_s61/e57_syn_tokread_s61.json $OUT/E55_d64_NAT_s61.saved.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7; job 970 first)"; exit 7; }; done
E57_SYN_SHUF=perm1 $PY $S/e57_syn_shuf.py --check --seed 61 --cell v3_64_4 --out "$TT/e57_shuf_check_s61.json"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_syn_shuf_report_s61.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
export E57_SYN_SHUF=perm1
[ -f "$PL/syn_s61_nll.npz" ] || $PY $S/e57_syn_w0pass.py --seed 61 --cell v3_64_4 --out "$PL" --bands 256
[ -f "$PL/syn_s61_shuf_m30_masks.npz" ] || $PY $S/e57_syn_mmasks.py --seed 61 --cell v3_64_4 --dir "$PL" --out "$PL/syn_s61_shuf_m30_masks.npz"
$PY $S/e57_syn_mrun.py --seed 61 --cell v3_64_4 --out $SQ --tag synqsh --m30 "$PL/syn_s61_shuf_m30_masks.npz" --ntg-nat auto --arms NATSGD,M3NMW1N05,M3W1N05,M3W8N05 --save-apps 16,32,64,96,128 --no-inline
unset E57_SYN_SHUF
[ -f "$TT/e57_syn_tokread_s61.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 61 --out "$TT" --tag synqsh --arms M3NMW1N05,M3W1N05,M3W8N05
$PY $S/e57_syn_shuf_report.py --seed 61 --shuf $TT/e57_syn_tokread_s61.json --orig $OUT/e57_syn_natq_s61/e57_syn_tokread_s61_all.json --out $TT/e57_syn_shuf_report_s61.json
echo "outputs: $TT/e57_syn_shuf_report_s61.json $TT/e57_shuf_check_s61.json $TT/e57_syn_tokread_s61.json $PL/syn_s61_shuf_m30_masks.npz"
