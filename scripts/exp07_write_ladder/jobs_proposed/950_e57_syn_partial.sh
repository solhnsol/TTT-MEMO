#!/bin/bash
# JOB 950 = E57 SYNTHETIC PARTIAL MASKING, M3-D30 / M3-D50 / M3-R30 (method thread, Hans card 11/05 09:2x KST; seed 64, 16 facts, 128 chunks): only the selected rows get the ALLQ256 band (the row sees only the previous 256 tokens), all other rows keep the natural window; the selected rows carry the loss weight 8; update norm 0.5 x NAT. D30 = top 30% of the eligible rows by the window gain (the deciding rows of jobs 710 / 730), D50 = top 50%, R30 = random 30% (same number as D30). States at appearances 16 / 32 / 64 / 96 / 128 with the canary and the inline reads of jobs 710 / 730 (out-of-window a = 16 / 32 / 64 / 128, in-window a = 128); per-token reads of all states; baselines reused: ALLQ256 (M3W1N05) and M3 w=8 from the curve job 800, NATQ at the end state from the token read of job 676. Rows masked per chunk and share of the eligible rows in syn_s64_partial_masks.json. New files e57_syn_pmasks.py, e57_syn_partial_report.py; arms added to e57_syn_mrun.py; no other existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_pmasks.py e57_syn_partial_report.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_pmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_partial}; T0=$OUT/e57_syn_curve_s64; mkdir -p "$TT"
for f in $SQ/syn_s64_m30_masks.npz $OUT/E55_v3a_NAT_s64.saved.json $T0/e57_syn_tokread_s64.json $OUT/e57_syn_tokread/e57_syn_tokread_s64.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s64_partial_masks.npz" ] || $PY $S/e57_syn_pmasks.py --out "$SQ/syn_s64_partial_masks.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_syn_partial_report.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --tag synqpm --m30 $SQ/syn_s64_partial_masks.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --arms M3D30W8N05,M3D50W8N05,M3R30W8N05 --save-apps 16,32,64,96,128
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqpm
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);a=json.load(open(O/'e57_syn_curve_s64/e57_syn_tokread_s64.json'));b=json.load(open(O/'e57_syn_partial/e57_syn_tokread_s64.json'));c=json.load(open(O/'e57_syn_tokread/e57_syn_tokread_s64.json'))
assert a['tokids']==b['tokids']==c['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['arms']['NATQ@c127']=c['arms']['NATQ'];a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx;json.dump(a,open(O/'e57_syn_partial/e57_syn_tokread_s64_all.json','w'));print('merged arms',sorted(a['arms']))
PYE
$PY $S/e57_curve_report.py --kind syn --seed 64 --tag synqpm --tokread $TT/e57_syn_tokread_s64_all.json --out $TT/e57_curve_partial_s64.json
$PY $S/e57_syn_partial_report.py --tokread $TT/e57_syn_tokread_s64_all.json --out $TT/e57_syn_partial_report.json
echo "outputs: $TT/e57_curve_partial_s64.json $TT/e57_syn_partial_report.json $TT/e57_syn_tokread_s64.json"
