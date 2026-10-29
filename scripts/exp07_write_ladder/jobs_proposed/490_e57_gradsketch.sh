#!/bin/bash
# JOB 490 = E57 GRADIENT-SKETCH DUMP (prepared; Hans 14:49Z 'go'; brainstorm thread 'Brainstorm new method ideas for the paper'): (A) one NAT news cycle (280 chunks, the stream and fast-weight layers of job 430, native write_step unchanged, no mask), per chunk the 256-dim count-sketch of the raw NAT write gradient of feed_forward_prime.w2 per layer [280,8,256]; (B) the 1,999 strict_gold fragments of e57_stream_sentences.jsonl (job 281; fid list file e57_gradsketch_in/e57_strict_gold_fids.json) each as one isolated chunk at W0, same loss / sketch [1999,8,256]; (C) per chunk the token-mean input of the last suffix layer's down_proj sketched to 256 dims; canary W0 / after chunk 279 (weights are updated in A); no pass criterion. Output $OUT/e57_gradsketch/E57_gradsketch.npz + E57_gradsketch_meta.json. ESTIMATE ONLY: load 15 min + record 5 min + 280 chunks x ~(7 s) ~35 min + 1,999 fragments x ~1-1.5 s ~35-50 min = ~1.5-1.9 h.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 77926fb5612e13e444c1caa19e7ce89180adc8f2^{commit} || { echo "commit 77926fb5612e13e444c1caa19e7ce89180adc8f2 missing (exit 4)"; exit 4; }; git checkout 77926fb5612e13e444c1caa19e7ce89180adc8f2 -- $D/e57_gradsketch.py; fi
sha256sum $D/e57_gradsketch.py | cut -c1-16,65-
$PY $S/e57_gradsketch.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_gradsketch}; FRAGS=${FRAGS:-$OUT/e57_gradsketch_in/e57_strict_gold_fids.json}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] && [ -f "$FRAGS" ] || { echo "input missing: $NATARMS or $FRAGS (exit 7)"; exit 7; }
[ "$(sha256sum "$FRAGS" | cut -c1-64)" = "4addd53e6f67f9beae3d1bb2f0b8a00b308134476683b45d8a3ca939588cdcf6" ] || { echo "fragment list differs from the frozen copy (exit 9)"; exit 9; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_gradsketch.npz" ] && [ -z "$RERUN" ] && { echo "E57_gradsketch.npz exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_gradsketch.py --frags "$FRAGS" --nat-ref "$NATARMS" --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --read-at 279 --canary-npos 1 --arms NAT --stage arms
echo "outputs: $MOUT/E57_gradsketch.npz $MOUT/E57_gradsketch_meta.json (copy to the shared root as e57_gradsketch.npz / e57_gradsketch_meta.json)"
