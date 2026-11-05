#!/bin/bash
# JOB 1000 = E57 SYNTHETIC CONTINUOUS MASKING, M3-CG (method thread, Hans 11/05 10:31 KST "g에 비례해 직전 토큰 수를 줄이고 x8도 g에 비례"; seed 64, 16 facts, 128 chunks). Per row i of chunk k >= 8: s_i = clip(g_i, 0, g90) / g90 with g = nll_band256 - nll_win of the W0 pass (e57_syn_s64/syn_s64_nll.npz), g90 = 90th percentile of g over the eligible rows of the chunk (0 for non-eligible rows); the row sees only the previous b_i = 8192 x (16/8192)^s_i tokens (s = 0: natural window; s = 1: 16 tokens; window AND chunk keys) and carries the loss weight 1 + 9 s_i (max x10; Hans 11/05 10:37 KST, between the w = 8 and w = 16 of jobs 730). Update norm 0.5 x NAT, engine SELC1, states at appearances 16 / 32 / 64 / 96 / 128 with canary and inline reads as job 950. Default arms M3CGW10N05 and the control M3CGSHW10N05 (the same s permuted among the eligible rows; Hans 11/05 10:38 KST); optional via ARMS=...: M3CWW10N05 (continuous weight only, natural window), M3CBW10N05 (continuous band only, weight 1); the w = 8 variants M3CG*W8N05 stay defined. Baselines reused as in job 950: ALLQ256 (M3W1N05) and M3 w=8 from job 800, NATQ at the end state from job 676, D30/D50/R30 from job 950. New files e57_syn_cmasks.py; arms and cont_keep added to e57_syn_mrun.py; no other existing e57_*.py edited.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${ARMS:=M3CGW10N05,M3CGSHW10N05}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_cmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_cmasks.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_cont}; T0=$OUT/e57_syn_curve_s64; mkdir -p "$TT"
for f in $OUT/E55_v3a_NAT_s64.saved.json $T0/e57_syn_tokread_s64.json $OUT/e57_syn_tokread/e57_syn_tokread_s64.json $OUT/e57_syn_s64/syn_s64_nll.npz; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s64_cont_masks.npz" ] || $PY $S/e57_syn_cmasks.py --out "$SQ/syn_s64_cont_masks.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_cont_s64.json" ] && [ -z "$RERUN" ] && { echo "report exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --tag synqcg --m30 $SQ/syn_s64_cont_masks.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --arms $ARMS --save-apps 16,32,64,96,128
[ -f "$TT/e57_syn_tokread_s64.json" ] && [ -z "$RERUN" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqcg
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);a=json.load(open(O/'e57_syn_curve_s64/e57_syn_tokread_s64.json'));b=json.load(open(O/'e57_syn_cont/e57_syn_tokread_s64.json'));c=json.load(open(O/'e57_syn_tokread/e57_syn_tokread_s64.json'))
assert a['tokids']==b['tokids']==c['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['arms']['NATQ@c127']=c['arms']['NATQ'];a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx;json.dump(a,open(O/'e57_syn_cont/e57_syn_tokread_s64_all.json','w'));print('merged arms',sorted(a['arms']))
PYE
$PY $S/e57_curve_report.py --kind syn --seed 64 --tag synqcg --tokread $TT/e57_syn_tokread_s64_all.json --out $TT/e57_curve_cont_s64.json
echo "outputs: $TT/e57_curve_cont_s64.json $TT/e57_syn_tokread_s64.json; masks stats: $SQ/syn_s64_cont_masks.json; saved: $SQ/E55_synqcg_*_s64.saved.json (canary, summary)"
