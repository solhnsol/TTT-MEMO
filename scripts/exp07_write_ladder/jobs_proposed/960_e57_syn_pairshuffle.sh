#!/bin/bash
# JOB 960 = E57 SYNTHETIC PAIR-SHUFFLED STREAM, SEED 64 (Hans card '합성 짝 섞기' 11/05 11:54 KST via coordinator; queued behind job 950, ~3 h): cell v3_16_1 (16 facts, 128 chunks, one sentence per fact in every chunk); WITHIN EVERY CHUNK the place strings of the 16 sentences are permuted by a random derangement (e57_syn_shuf.py, E57_SYN_SHUF=perm1; names, frames, filler and all positions unchanged; place strings re-encoded in the form of the sentence they land in, token lengths of all spans kept, so total exposures per place string = original; no per-chunk permutation is repeated across chunks, name->place pairs vary from chunk to chunk); reads, 16 candidates (gold + 15 other places) and question frames unchanged = scored with the ORIGINAL pairs. Arms: NATQ (M3NMW1N05), ALLQ256 (M3W1N05), M3 w=8 (M3W8N05), update norm 0.5 x NAT (NAT run on the shuffled stream: NATSGD first, --ntg-nat auto); masks as in jobs 850 / 881 (K0 = 8; deciding rows = top 30% of the eligible rows by window gain, RECOMPUTED on the shuffled stream: W0 pass + e57_syn_mmasks.py). Save points appearances 16 / 17 / 32 / 33 / 64 / 65 / 96 / 128 (17 / 33 / 65 match the NATQ states of job 850, 16 / 32 / 64 / 96 / 128 match the states of job 800). Per-token reads (e57_syn_tokread.py) and report (e57_syn_shuf_report.py): first-token dA, dA-dB, branch rank ratio, full-sum rank ratio, W0 gains, canary; shuffled minus original of the same arm and chunk. New files e57_syn_shuf.py, e57_syn_shuf_report.py, this script; e57_syn_mrun.py / e57_syn_mmasks.py / e57_syn_w0pass.py (own files) get one import line (no effect without E57_SYN_SHUF).
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
PL=$OUT/e57_syn_shuf; SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_shuf_curve}; mkdir -p "$PL" "$TT"
for f in $OUT/e57_syn_ncurve_s64/e57_syn_tokread_s64.json $OUT/e57_syn_curve_s64/e57_syn_tokread_s64.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
# CPU invariants of the shuffled stream (also CPU_ONLY)
E57_SYN_SHUF=perm1 $PY $S/e57_syn_shuf.py --check --out "$TT/e57_shuf_check.json"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_syn_shuf_report_s64.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
export E57_SYN_SHUF=perm1
[ -f "$PL/syn_s64_nll.npz" ] || $PY $S/e57_syn_w0pass.py --seed 64 --out "$PL" --bands 256
[ -f "$PL/syn_s64_shuf_m30_masks.npz" ] || $PY $S/e57_syn_mmasks.py --seed 64 --dir "$PL" --out "$PL/syn_s64_shuf_m30_masks.npz"
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --tag synqsh --m30 "$PL/syn_s64_shuf_m30_masks.npz" --ntg-nat auto --arms NATSGD,M3NMW1N05,M3W1N05,M3W8N05 --save-apps 16,17,32,33,64,65,96,128 --no-inline
unset E57_SYN_SHUF
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqsh --arms M3NMW1N05,M3W1N05,M3W8N05
$PY $S/e57_syn_shuf_report.py --shuf $TT/e57_syn_tokread_s64.json --orig $OUT/e57_syn_ncurve_s64/e57_syn_tokread_s64.json --orig $OUT/e57_syn_curve_s64/e57_syn_tokread_s64.json --out $TT/e57_syn_shuf_report_s64.json
echo "outputs: $TT/e57_syn_shuf_report_s64.json $TT/e57_shuf_check.json $TT/e57_syn_tokread_s64.json $PL/syn_s64_shuf_m30_masks.npz"
