#!/bin/bash
# JOB 540 = E57 NAT SGD AT 2x AND 4x THE NAT UPDATE NORM (prepared; Hans 01:25Z via the Rethink-momentum thread: queue after 530): no mask, no Adam, one cycle, 1 seed: arm NATM (the NAT clipped-gradient direction at the current weights) with the per-chunk update norm = 2x (NATX2) / 4x (NATX4) the per-chunk norms of the saved NAT run (E57_main_NAT.arms.json), OUT reads of the 42 questions at chunks 99 / 179 / 279 + canary. Output $OUT/e57_natsgd/. Hans 01:55Z (news primary metric = T1/T2): afterwards the T1/T2 reads (frozen t1t2_v1.json, 150 frames, OUT, method of jobs 410 / 450) of W0 and of the chunk-279 states of NATX2 / NATX4 -> $OUT/e57_natsgd_t1t2/. ESTIMATE ONLY: load + record ~13 min + 2 arms x ~7 min + reads ~12 min + T1/T2 reads (load 15 min + 4551 scorer rows ~15 min + 3 reads) ~35 min = ~75 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e 31d3f6683be1f91b967a828cde05e7d52398cb41^{commit} && git cat-file -e bae5f22ebfa72badf78fc296aa4767c7b5cf3fd3^{commit} && git cat-file -e e834636b00e92959ad2615a5687db816baf86773^{commit} && git cat-file -e 15666eb^{commit} || { echo "commit 31d3f66 or bae5f22ebfa72badf78fc296aa4767c7b5cf3fd3 missing (exit 4)"; exit 4; }; git checkout 31d3f6683be1f91b967a828cde05e7d52398cb41 -- $D/e57_adamdecomp_run.py; git checkout e834636b00e92959ad2615a5687db816baf86773 -- $D/e57_t1t2_natsgd_run.py; git checkout 15666eb -- $D/e57_t1t2_run.py; git checkout bae5f22ebfa72badf78fc296aa4767c7b5cf3fd3 -- $D/e57_natsgd_run.py; fi
sha256sum $D/e57_adamdecomp_run.py $D/e57_natsgd_run.py $D/e57_qcurve_run.py | cut -c1-16,65-
$PY $S/e57_natsgd_run.py --selftest | tail -1
for x in "3419b588f180 e57_t1t2_run.py" "d13a0ac9ee4a e57_main_run.py" "b19e986904ba e57_reads.py" "9ee4658ee168 e57_trial_run.py" "88076d1f3dd1 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2: have $G, expected $1 (exit 5)"; exit 5; }; done
$PY $S/e57_t1t2_natsgd_run.py --selftest | tail -1
$PY $S/e57_qcurve_run.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_natsgd}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
[ -f "$NATARMS" ] || { echo "input missing: $NATARMS (exit 7)"; exit 7; }
mkdir -p "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 1 --arms NAT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_main_NATX2.arms.json" ] && [ -z "$RERUN" ] && { echo "E57_main_NATX2.arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
ARMS=${ARMS:-NATX2,NATX4}
ARGS="--bundle $BUNDLE --out $MOUT --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms $ARMS --ntg-from $NATARMS"
$PY $S/e57_natsgd_run.py $ARGS --stage arms
$PY $S/e57_natsgd_run.py $ARGS --stage reads
for A in $(echo "$ARMS" | tr ',' ' '); do rm -f "$MOUT"/states/E57_main_$A/*_099.npz "$MOUT"/states/E57_main_$A/*_179.npz; done; ls "$MOUT"/states/*/ | head -20
T1OUT=${T1OUT:-$OUT/e57_natsgd_t1t2}; FROZEN=${FROZEN:-$BUNDLE/e57_t1t2/t1t2_v1.json}; T1MAN=$T1OUT/e57_t1t2_states_manifest.json; mkdir -p "$T1OUT"
[ -f "$FROZEN" ] || { echo "frozen T1/T2 file missing (exit 7)"; exit 7; }
$PY - "$MOUT" "$T1MAN" <<'PYE'
import json,sys
o,mf=sys.argv[1:3];m={}
for arm in ('NATX2','NATX4'):
    st=json.load(open(f'{o}/E57_main_{arm}.arms.json'))['states']['279'];m[arm]=dict(file=st['file'],hash=st['hash'])
json.dump(m,open(mf,'w'),indent=1);print('T1/T2 manifest',mf,{k:v['hash'][:12] for k,v in m.items()})
PYE
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2_run.py --cpu-check --frozen "$FROZEN" --manifest "$T1MAN" --out "$T1OUT"
[ -f "$T1OUT/E57_t1t2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2_scores.json exists (exit 8)"; exit 8; }
$PY $S/e57_t1t2_natsgd_run.py --bundle "$BUNDLE" --out "$T1OUT" --bank "$BUNDLE/e57_bank/bank_v1.json" --cycles 1 --npos 1 --canary-npos 1 --arms NATX2,NATX4 --stage reads --frozen "$FROZEN" --manifest "$T1MAN" ${STATE_DIR:+--state-dir "$STATE_DIR"}
echo "T1/T2 outputs: $T1OUT/E57_t1t2_scores.json $T1OUT/E57_t1t2_score_keys.json (copy to the shared root as e57_natsgd_t1t2_scores.json / e57_natsgd_t1t2_score_keys.json)"
echo "outputs: $MOUT/E57_qcurve_scores.json, E57_main_<ARM>.arms.json (copy to the shared root as e57_natsgd_scores.json / e57_natsgd_<ARM>.arms.json); states of the chunk 279 kept in $MOUT/states/"
