#!/bin/bash
# JOB 1040 = E57 SYNTHETIC ADAPTIVE-THRESHOLD WEIGHTED SET (Hans 11/05 23:28 KST 'adaptive'): ALLQ256 on every row; the deciding rows are those with g above ONE global threshold tau per run (tau = pooled 95% / 90% quantile of g over the eligible rows of all chunks, so the per-chunk share varies with the data), loss weight 16 / 8. Arms M3TA05W16N05, M3TA10W16N05, M3TA05W8N05, M3TA10W8N05. Compare with the fixed-share T05 / T10 of job 1030 (same average share by construction). Seed 64, 16 facts, exploratory. New qtA05 / qtA10 arrays in e57_syn_cmasks.py (stats json gets tau and adapt_share).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${ARMS:=M3TA05W16N05,M3TA10W16N05,M3TA05W8N05,M3TA10W8N05}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_cmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_cmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_adapt}; T0=$OUT/e57_syn_curve_s64; mkdir -p "$TT"
for f in $OUT/E55_v3a_NAT_s64.saved.json $T0/e57_syn_tokread_s64.json $OUT/e57_syn_tokread/e57_syn_tokread_s64.json $OUT/e57_syn_s64/syn_s64_nll.npz; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s64_adapt_c.npz" ] || $PY $S/e57_syn_cmasks.py --out "$SQ/syn_s64_adapt_c.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_adapt_s64.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --tag synqad --m30 $SQ/syn_s64_adapt_c.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --arms $ARMS --save-apps 16,32,64,96,128
[ -f "$TT/e57_syn_tokread_s64.json" ] && [ -z "$RERUN" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqad
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);a=json.load(open(O/'e57_syn_curve_s64/e57_syn_tokread_s64.json'));b=json.load(open(O/'e57_syn_adapt/e57_syn_tokread_s64.json'));c=json.load(open(O/'e57_syn_tokread/e57_syn_tokread_s64.json'))
d0=json.load(open(O/'e57_syn_cont/e57_syn_tokread_s64.json'));a['arms'].update(d0['arms']);g0=json.load(open(O/'e57_syn_grid/e57_syn_tokread_s64.json'));a['arms'].update(g0['arms']);n0=json.load(open(O/'e57_syn_narrow/e57_syn_tokread_s64.json'));a['arms'].update(n0['arms']);assert a['tokids']==b['tokids']==c['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['arms']['NATQ@c127']=c['arms']['NATQ'];a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx;json.dump(a,open(O/'e57_syn_adapt/e57_syn_tokread_s64_all.json','w'));print('merged arms',sorted(a['arms']))
PYE
$PY $S/e57_curve_report.py --kind syn --seed 64 --tag synqad --tokread $TT/e57_syn_tokread_s64_all.json --out $TT/e57_curve_adapt_s64.json
echo "outputs: $TT/e57_curve_adapt_s64.json $TT/e57_syn_tokread_s64.json; masks stats: $SQ/syn_s64_adapt_c.json; saved: $SQ/E55_synqad_*_s64.saved.json (canary, summary)"
$PY $S/e57_syn_partial_report.py --tokread $TT/e57_syn_tokread_s64_all.json --extra M3TA05W16N05:synqad,M3TA10W16N05:synqad,M3TA05W8N05:synqad,M3TA10W8N05:synqad,M3T05W16N05:synqnr,M3T10W16N05:synqnr --out $TT/e57_syn_inwindow_adapt_s64.json
