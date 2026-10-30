#!/bin/bash
# JOB 565 = E57 GAINQ512 p=.10 FOCAL-LOSS WRITE ARMS (prepared; Hans 05:37Z via the idea thread: queue after 560 and 561): the GAINQ512 p=.10 arm of job 293 (SELC1 engine, Adam direction, update norm 0.5 x NAT, same selected rows and masked forward, one cycle) with the write loss reweighted on the SELECTED query rows only: tokens of selected rows with masked-context per-token CE < tau are dropped (weight 0), mean over the remaining loss rows, non-selected rows unchanged, update rescaled to the GAINQ512 norm. Arms GAINQ_FOCAL_NONE (control: nothing dropped; must reproduce the existing GAINQ512 p=.10 state hash 2a6836fcba5f at chunk 279 and gives the states at 99 / 179), GAINQ_FOCAL_T05 (tau .5 nats), GAINQ_FOCAL_T10 (tau 1.0), GAINQ_FOCAL_RAND (same number of selected-row tokens as T05 dropped per chunk, at random). OUT reads of the 42 questions at 99 / 179 / 279 + canary, then the T1/T2 150-frame OUT reads (W0 + chunk-279 states) as in jobs 410 / 550. Per chunk drop fractions are written to E57_focal_<ARM>_drop.json. No thresholds. ESTIMATE ONLY: load + record ~13 min + 4 arms x ~9 min + reads ~15 min + T1/T2 reads ~35 min = ~1.7-1.9 h.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 15666eb^{commit} && git cat-file -e a2cd49e8e9b76328ba0031350fb2d0ad1a6bc051^{commit} || { echo "commit 15666eb or a2cd49e8e9b76328ba0031350fb2d0ad1a6bc051 missing (exit 4)"; exit 4; }; git checkout a2cd49e8e9b76328ba0031350fb2d0ad1a6bc051 -- $D/e57_gainq_focal_run.py $D/e57_t1t2_focal_run.py; git checkout 15666eb -- $D/e57_t1t2_run.py; fi
sha256sum $D/e57_gainq_focal_run.py $D/e57_t1t2_focal_run.py $D/e57_qcurve_run.py $D/e57_qband_run.py $D/e57_closure_masks.py | cut -c1-16,65-
for x in "4123c9da29d6 e57_qcurve_run.py" "c48a57e093ed e57_qband_run.py" "091506dbe7b8 e57_closure_masks.py"; do set -- $x; H=$(sha256sum $D/$2 | cut -c1-12); [ "$H" = "$1" ] || { echo "VERSION MISMATCH $2: have $H, expected $1 (exit 5)"; exit 5; }; done
$PY $S/e57_gainq_focal_run.py --selftest | tail -1
for x in "3419b588f180 e57_t1t2_run.py" "d13a0ac9ee4a e57_main_run.py" "b19e986904ba e57_reads.py" "9ee4658ee168 e57_trial_run.py" "88076d1f3dd1 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2: have $G, expected $1 (exit 5)"; exit 5; }; done
$PY $S/e57_t1t2_focal_run.py --selftest | tail -1
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_gainq_focal}; QM=${QM:-$OUT/e57_gainq_p0.10/e57_qband_masks.npz}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] && [ -f "$QM" ] || { echo "input missing: $NATARMS or $QM (exit 7)"; exit 7; }
[ "$(sha256sum "$QM" | cut -c1-64)" = "84ac5f4f5e1056f4fb168538753ab192c07ad7f39fc54a7b003732b5ea23ceb9" ] || { echo "GAINQ p=.10 mask file differs from the frozen copy (exit 9)"; exit 9; }
grep -q 5d6af783d91893a4d68ecc39a8536636644a86ae2097fd7e872f2506a2a18a4e "$QM.contenthash.json" || { echo "mask content hash differs (exit 9)"; exit 9; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_GAINQ_FOCAL_NONE.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_GAINQ_FOCAL_NONE.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARMS=${ARMS:-GAINQ_FOCAL_NONE,GAINQ_FOCAL_T05,GAINQ_FOCAL_T10,GAINQ_FOCAL_RAND}
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --smoke 0 --arms $ARMS --qband-masks $QM --ntg-from $NATARMS"
$PY $S/e57_gainq_focal_run.py $ARGS --stage arms
$PY - "$MOUT" <<'PYE'
import json,sys
o=sys.argv[1];h=json.load(open(o+'/E57_main_GAINQ_FOCAL_NONE.arms.json'))['states']['279']['hash'];print('GAINQ_FOCAL_NONE state hash at 279',h[:12],'| existing GAINQ512 p=.10 2a6836fcba5f | identical:',h.startswith('2a6836fcba5f'))
PYE
$PY $S/e57_gainq_focal_run.py $ARGS --stage reads
for A in $(echo "$ARMS" | tr ',' ' '); do rm -f "$MOUT"/states/E57_main_$A/*_099.npz "$MOUT"/states/E57_main_$A/*_179.npz; done; ls "$MOUT"/states/*/ | head -20
T1OUT=${T1OUT:-$OUT/e57_gainq_focal_t1t2}; FROZEN=${FROZEN:-$BUNDLE/e57_t1t2/t1t2_v1.json}; T1MAN=$T1OUT/e57_t1t2_states_manifest.json; mkdir -p "$T1OUT"
[ -f "$FROZEN" ] || { echo "frozen T1/T2 file missing (exit 7)"; exit 7; }
$PY - "$MOUT" "$T1MAN" <<'PYE'
import json,sys
o,mf=sys.argv[1:3];m={}
for arm in ('GAINQ_FOCAL_NONE','GAINQ_FOCAL_T05','GAINQ_FOCAL_T10','GAINQ_FOCAL_RAND'):
    st=json.load(open(f'{o}/E57_main_{arm}.arms.json'))['states']['279'];m[arm]=dict(file=st['file'],hash=st['hash'])
json.dump(m,open(mf,'w'),indent=1);print('T1/T2 manifest',mf,{k:v['hash'][:12] for k,v in m.items()})
PYE
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2_run.py --cpu-check --frozen "$FROZEN" --manifest "$T1MAN" --out "$T1OUT"
[ -f "$T1OUT/E57_t1t2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2_scores.json exists (exit 8)"; exit 8; }
$PY $S/e57_t1t2_focal_run.py --bundle "$BUNDLE" --out "$T1OUT" --bank "$BUNDLE/e57_bank/bank_v1.json" --cycles 1 --npos 1 --canary-npos 1 --arms GAINQ_FOCAL_NONE,GAINQ_FOCAL_T05,GAINQ_FOCAL_T10,GAINQ_FOCAL_RAND --stage reads --frozen "$FROZEN" --manifest "$T1MAN" ${STATE_DIR:+--state-dir "$STATE_DIR"}
echo "T1/T2 outputs: $T1OUT/E57_t1t2_scores.json $T1OUT/E57_t1t2_score_keys.json (copy to the shared root as e57_gainq_focal_t1t2_scores.json / e57_gainq_focal_t1t2_score_keys.json)"
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy to the shared root as e57_gainq_focal_scores.json / e57_gainq_focal_<ARM>.arms.json); states of the chunk 279 kept in $MOUT/states/"
