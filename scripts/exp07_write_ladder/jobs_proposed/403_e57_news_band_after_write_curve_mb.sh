#!/bin/bash
# JOB 403b (multi-band: 512 + 256 + 128 + 64 + 32) = E57 NEWS STREAM BAND-512 / WINDOW NLL OF THE STORED CURVE STATES (post-hoc; Hans via the discussion thread 05:05Z, step C): the states of NAT, NATADAMH, ORC2, RAND2 at chunks 99 / 179 / 279 of experiments/exp07_write_ladder/e57_curve/states (job 286, 14 positions each, kept); runs e57_windowless_state_band_run.py (job 345 as is): forward only, NO weight update: per-token NLL under the natural window and under the band 512 over the chunks 8..279 of the one-cycle news stream (the whole stream, not only the chunks up to the state), per state -> $OUT/e57_band_states_curve_mb/E57_band_<LABEL>.npz. NOT APPROVED yet. ESTIMATE ONLY, unmeasured, lab A100: load 15 min + 12 states x about 7 min = about 100 min
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_band_states_curve_mb}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
J=$($PY - "$OUT" <<'PYE'
import json,sys
o=sys.argv[1];r=[]
for arm in ('NAT','NATADAMH','ORC2','RAND2'):
    A=json.load(open(o+'/e57_curve/E57_main_'+arm+'.arms.json'))
    for k in ('99','179','279'):
        st=A['states'][k];r.append(f"curve{arm}_c{int(k):03d}={st['file']}:{st['hash']}@512+256+128+64+32")
print(','.join(r))
PYE
)
WST=${WSTATES:-$J}
for F in $(echo "$WST" | tr ',' '\n' | sed 's/^[^=]*=//; s/:.*//; s/@.*//' | grep .); do [ -f "$F" ] || { echo "state missing: $F (exit 7)"; exit 7; }; done
echo "states: $WST"
[ -f "$MOUT/E57_band_$(echo "$WST" | cut -d= -f1).npz" ] && [ -z "$RERUN" ] && { echo "first output exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --wstates $WST"
$PY $S/e57_windowless_state_band_run.py $ARGS --stage arms
