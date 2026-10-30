#!/bin/bash
# JOB 530 = E57 SHADOW-FREE ORACLE GRADIENT SKETCHES (prepared; Hans 00:56Z via the idea thread: queue right after 520 and run without asking again): job 500 ORACLE setup (NAT write path, 2048-dim count-sketch seed 490001, leaf feed_forward_prime.w2, loss = mean CE on the strict_gold fragment target rows) in three variants per chunk: A ORACLE (recomputed; check against job 500), B ORACLE_SHADOWFREE (the query rows of each gold fragment do not attend to the key positions of the fragments in mask_same_fact_fids of e57_shadow_mask_table.json; per-query keep, prefix and suffix layers), C ORACLE_RANDMASK (mask_random_other_event_fids). No thresholds. Output $OUT/e57_gradsketch_shadow/E57_gradsketch_shadow.npz + E57_gradsketch_shadow_meta.json. ESTIMATE ONLY: load 8 min + 280 chunks x ~5 s ~23 min = ~35 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e c731db3d69ba47b5adedb56dd005f0dc4cf07561^{commit} && git cat-file -e 919d8e56b13069493c9aad889b95e11e7f59a0dd^{commit} && git cat-file -e 1c708fe04ff861459c3c1c4f794da813c3de8954^{commit} || { echo "commit c731db3 / 919d8e5 / 1c708fe missing (exit 4)"; exit 4; }; git checkout c731db3d69ba47b5adedb56dd005f0dc4cf07561 -- $D/e57_gradsketch.py; git checkout 919d8e56b13069493c9aad889b95e11e7f59a0dd -- $D/e57_gradsketch_narrow.py; git checkout 1c708fe04ff861459c3c1c4f794da813c3de8954 -- $D/e57_gradsketch_shadow.py; fi
sha256sum $D/e57_gradsketch.py $D/e57_gradsketch_narrow.py $D/e57_gradsketch_shadow.py | cut -c1-16,65-
$PY $S/e57_gradsketch_shadow.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_gradsketch_shadow}; SHT=${SHT:-$OUT/e57_shadow_in/e57_shadow_mask_table.json}; STR=${STR:-$OUT/e57_shadow_in/e57_stream_sentences.jsonl}; ORACLE500=${ORACLE500:-$OUT/e57_gradsketch_narrow/E57_gradsketch_narrow.npz}; FRAGS=${FRAGS:-$OUT/e57_gradsketch_in/e57_strict_gold_fids.json}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] && [ -f "$FRAGS" ] || { echo "input missing: $NATARMS or $FRAGS (exit 7)"; exit 7; }
[ "$(sha256sum "$FRAGS" | cut -c1-64)" = "4addd53e6f67f9beae3d1bb2f0b8a00b308134476683b45d8a3ca939588cdcf6" ] || { echo "fragment list differs from the frozen copy (exit 9)"; exit 9; }
[ -f "$SHT" ] && [ -f "$STR" ] || { echo "input missing: $SHT or $STR (exit 7)"; exit 7; }
echo "shadow mask table sha256 $(sha256sum "$SHT" | cut -c1-64)"
[ "$(sha256sum "$SHT" | cut -c1-64)" = "f94e698b9e5f0cdca306e86e7ac704f7d99c1d674a00d8e04a7901a3986eba63" ] || { echo "shadow mask table differs from the frozen copy (exit 9)"; exit 9; }
[ "$(sha256sum "$STR" | cut -c1-64)" = "38ef483747a28a9334e5fba9fc7c12bc6fa2bdd92f8919a33faebe007cb4e8b6" ] || { echo "stream fragment file differs from the frozen copy (exit 9)"; exit 9; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_gradsketch_shadow.npz" ] && [ -z "$RERUN" ] && { echo "E57_gradsketch_shadow.npz exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_gradsketch_shadow.py --frags "$FRAGS" --shadow "$SHT" --stream-frags "$STR" --oracle-ref "$ORACLE500" --nat-ref "$NATARMS" --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --read-at 279 --canary-npos 1 --arms NAT --stage arms
echo "outputs: $MOUT/E57_gradsketch_shadow.npz $MOUT/E57_gradsketch_shadow_meta.json (copy to the shared root as e57_gradsketch_shadow*)"
