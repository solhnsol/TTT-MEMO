#!/bin/bash
# JOB 970 = E57 SYNTHETIC 64-FACT WORLD (seed 61, cell v3_64_4: 64 facts, 512 chunks, 128 exposures per fact), NATQ ARM (Hans '970 990 먼저 돌리자' 11/05 13:54 KST via coordinator, ~1 h): M3NMW1N05 = natural window, no mask, loss weight 1, Adam direction, update norm 0.5 x the per-chunk norm of E55_d64_NAT_s61 (same recipe as NATQ on seed 64, job 850); save points appearances 16 / 32 / 64 / 96 / 128 (as job 810), inline reads at 16 / 32 / 64 / 128, canary; per-token reads; curve report together with the stored ALLQ256 / M3 w=8 states of job 810. No existing e57_*.py edited.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_natq_s61}; T0=$OUT/e57_syn_curve_s61; mkdir -p "$TT"
for f in $SQ/syn_s61_m30_masks.npz $OUT/E55_d64_NAT_s61.saved.json $T0/e57_syn_tokread_s61.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_natq_s61.json" ] && [ -z "$RERUN" ] && { echo "curve json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 61 --cell v3_64_4 --out $SQ --tag synqn --m30 $SQ/syn_s61_m30_masks.npz --ntg-nat $OUT/E55_d64_NAT_s61.saved.json --arms M3NMW1N05 --save-apps 16,32,64,96,128
[ -f "$TT/e57_syn_tokread_s61.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 61 --out "$TT" --tag synqn --arms M3NMW1N05
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);a=json.load(open(O/'e57_syn_curve_s61/e57_syn_tokread_s61.json'));b=json.load(open(O/'e57_syn_natq_s61/e57_syn_tokread_s61.json'))
assert a['tokids']==b['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx;json.dump(a,open(O/'e57_syn_natq_s61/e57_syn_tokread_s61_all.json','w'));print('merged arms',sorted(a['arms']))
PYE
$PY $S/e57_curve_report.py --kind syn --seed 61 --tag synqn --tokread $TT/e57_syn_tokread_s61_all.json --out $TT/e57_curve_natq_s61.json
echo "outputs: $TT/e57_curve_natq_s61.json $TT/e57_syn_tokread_s61.json $SQ/E55_synqn_M3NMW1N05_s61.{saved,read}.json"
