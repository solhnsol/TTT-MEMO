#!/bin/bash
# JOB 500 = E57 NARROW GRADIENT-SKETCH DUMP (prepared, NOT APPROVED to run; Hans 17:11Z via the idea thread 'go', replaces job 495): one NAT cycle (280 chunks, the weights follow the NAT write path, native write_step untouched, gradients recorded not applied), per chunk at the weights before the write the 2048-dim count-sketch (job 490 method, seed 490001) of the feed_forward_prime.w2 gradient [8 layers] for four losses: NAT (whole chunk), ORACLE (loss only on the strict_gold fragment tokens of the chunk, in context), GAINQ (GAINQ512 p=.26 query-row masking, whole-chunk loss), GAINQSEL (same masked forward, loss only on the banded query rows); chunks without fragments / banded rows: zeros + flag; per-chunk token and row counts, gradient norms, CE; exact-vs-sketch cosine check; update-norm comparison with the saved NAT run; canary W0 / after chunk 279. No pass criterion. Output $OUT/e57_gradsketch_narrow/E57_gradsketch_narrow.npz + E57_gradsketch_narrow_meta.json. ESTIMATE ONLY: load 8 min + record + 280 chunks x ~7 s ~33 min + check = ~45-50 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e c731db3d69ba47b5adedb56dd005f0dc4cf07561^{commit} && git cat-file -e 919d8e56b13069493c9aad889b95e11e7f59a0dd^{commit} || { echo "commit c731db3d69ba47b5adedb56dd005f0dc4cf07561 or 919d8e56b13069493c9aad889b95e11e7f59a0dd missing (exit 4)"; exit 4; }; git checkout c731db3d69ba47b5adedb56dd005f0dc4cf07561 -- $D/e57_gradsketch.py; git checkout 919d8e56b13069493c9aad889b95e11e7f59a0dd -- $D/e57_gradsketch_narrow.py; fi
sha256sum $D/e57_gradsketch.py $D/e57_gradsketch_narrow.py $D/e57_qband_run.py $D/e57_closure_masks.py | cut -c1-16,65-
for x in "091506dbe7b8 e57_closure_masks.py" "c48a57e093ed e57_qband_run.py"; do set -- $x; H=$(sha256sum $D/$2 | cut -c1-12); [ "$H" = "$1" ] || { echo "VERSION MISMATCH $2: have $H, expected $1 (exit 5)"; exit 5; }; done
$PY $S/e57_gradsketch_narrow.py --selftest | tail -1
$PY $S/e57_qband_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_gradsketch_narrow}; QM=${QM:-$OUT/e57_gainq_p0.26/e57_qband_masks.npz}; FRAGS=${FRAGS:-$OUT/e57_gradsketch_in/e57_strict_gold_fids.json}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] && [ -f "$FRAGS" ] || { echo "input missing: $NATARMS or $FRAGS (exit 7)"; exit 7; }
[ "$(sha256sum "$FRAGS" | cut -c1-64)" = "4addd53e6f67f9beae3d1bb2f0b8a00b308134476683b45d8a3ca939588cdcf6" ] || { echo "fragment list differs from the frozen copy (exit 9)"; exit 9; }
[ "$(sha256sum "$QM" | cut -c1-64)" = "e97176d146a98714754f2a26d2734038ef4549a8c1fee644efb17e84771def0f" ] || { echo "GAINQ p=.26 mask file differs from the frozen copy (exit 9)"; exit 9; }
grep -q f0e2285fc6d10565473cce6c9b7b27af1d2f392e75f050d9c280ae27995f1f47 "$QM.contenthash.json" || { echo "mask content hash differs (exit 9)"; exit 9; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_gradsketch_narrow.npz" ] && [ -z "$RERUN" ] && { echo "E57_gradsketch_narrow.npz exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_gradsketch_narrow.py --frags "$FRAGS" --qmasks "$QM" --nat-ref "$NATARMS" --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --read-at 279 --canary-npos 1 --arms NAT --stage arms
echo "outputs: $MOUT/E57_gradsketch_narrow.npz $MOUT/E57_gradsketch_narrow_meta.json (copy to the shared root as e57_gradsketch_narrow.npz / e57_gradsketch_narrow_meta.json)"
