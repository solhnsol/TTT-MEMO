#!/bin/bash
# JOB 1020 = E57 SYNTHETIC M3 TUNING GRID (method thread "연속 가림 GPM 테스트", Hans 11/05 21:06 KST "카나리 상한은 따로 두지 말고 격자를 돌려보자; 상위일수록 w를 강하게; 띠 길이는 합성에서 의미 없음"; seed 64, 16 facts, 128 chunks). One factor at a time around M3 w=8 (ALLQ256 on every row, loss weight 8 on the top 30% rows by g): M3T15W8N05 (top 15%), M3T50W8N05 (top 50%), M3W32N05 (w = 32), and GRADED weights with the continuous s of job 1000 (s = clip(g,0,g90)/g90): M3GWW10N05 (w_i = 1 + 9 s_i) and EXPONENTIAL graded weights that follow M3's step (Hans 11/05 21:09 KST): M3GEW8N05 / M3GEW16N05 with f = (exp(3 t) - 1) / (exp(3) - 1), t = clip(g, 0, gc) / gc, gc = the smallest g among the chunk's top-30% rows, weight 1 + (W - 1) f (the top-30% rows get the full W = 8 / 16 exactly as in M3, lower rows ramp up exponentially from 1); band 256 on every row in all arms (no band arms: band length is not informative on this synthetic data, Hans). Update norm 0.5 x NAT, engine SELC1, save points 16 / 32 / 64 / 96 / 128 with canary and inline reads as job 950 / 1000. Baselines from existing jobs: ALLQ256 (M3W1N05) and M3 w=8 from job 800, D30 / D50 / R30 from job 950, GPM-C and its control from job 1000, NATQ. Output also the in-window comparison (e57_syn_partial_report.py --extra). No selection rule, no canary cap (Hans); exploratory, one seed. New: qb_TOP30 / qt15 / qt50 / qc_s arrays in e57_syn_cmasks.py, kinds M3T15 / M3T50 / M3GW in e57_syn_mrun.py.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${ARMS:=M3T15W8N05,M3T50W8N05,M3W32N05,M3GWW10N05,M3GEW8N05,M3GEW16N05}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_cmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_cmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_grid}; T0=$OUT/e57_syn_curve_s64; mkdir -p "$TT"
for f in $OUT/E55_v3a_NAT_s64.saved.json $T0/e57_syn_tokread_s64.json $OUT/e57_syn_tokread/e57_syn_tokread_s64.json $OUT/e57_syn_s64/syn_s64_nll.npz; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s64_cont_c.npz" ] || $PY $S/e57_syn_cmasks.py --out "$SQ/syn_s64_cont_c.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_grid_s64.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --tag synqgr --m30 $SQ/syn_s64_cont_c.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --arms $ARMS --save-apps 16,32,64,96,128
[ -f "$TT/e57_syn_tokread_s64.json" ] && [ -z "$RERUN" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqgr
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);a=json.load(open(O/'e57_syn_curve_s64/e57_syn_tokread_s64.json'));b=json.load(open(O/'e57_syn_grid/e57_syn_tokread_s64.json'));c=json.load(open(O/'e57_syn_tokread/e57_syn_tokread_s64.json'))
d0=json.load(open(O/'e57_syn_cont/e57_syn_tokread_s64.json'));a['arms'].update(d0['arms']);assert a['tokids']==b['tokids']==c['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['arms']['NATQ@c127']=c['arms']['NATQ'];a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx;json.dump(a,open(O/'e57_syn_grid/e57_syn_tokread_s64_all.json','w'));print('merged arms',sorted(a['arms']))
PYE
$PY $S/e57_curve_report.py --kind syn --seed 64 --tag synqgr --tokread $TT/e57_syn_tokread_s64_all.json --out $TT/e57_curve_grid_s64.json
echo "outputs: $TT/e57_curve_grid_s64.json $TT/e57_syn_tokread_s64.json; masks stats: $SQ/syn_s64_cont_c.json; saved: $SQ/E55_synqgr_*_s64.saved.json (canary, summary)"
$PY $S/e57_syn_partial_report.py --tokread $TT/e57_syn_tokread_s64_all.json --extra M3T15W8N05:synqgr,M3T50W8N05:synqgr,M3W32N05:synqgr,M3GWW10N05:synqgr,M3GEW8N05:synqgr,M3GEW16N05:synqgr,M3CGW10N05:synqcg,M3CGSHW10N05:synqcg --out $TT/e57_syn_inwindow_grid_s64.json
