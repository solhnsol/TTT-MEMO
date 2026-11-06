#!/bin/bash
# JOB 1134 = finish of the mass-matched grid on s71 and s72: the job 1130 script called e57_syn_tokread.py without --cell, so the per-token reads of s71 (v3_4_1) / s72 (v3_32_1) failed after the writes had finished (states of tag synqmg exist). Here: reads with --cell, merge with the tokread of jobs 1072 / 1073, curve report (as in 1130).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for W in 71:v3_4_1 72:v3_32_1; do
  SEED=${W%%:*}; CELL=${W##*:}; TT=$OUT/e57_syn_massgrid_s$SEED; mkdir -p "$TT"
  for a in M3T10W8P5N05 M3T15W6N05 M3W3P5N05; do [ -f "$OUT/e57_synq/E55_synqmg_${a}_s$SEED.saved.json" ] || { echo "missing state $a s$SEED (exit 7)"; exit 7; }; done
  $PY $S/e57_syn_tokread.py --stage read --seed $SEED --cell $CELL --out "$TT" --tag synqmg
  SEED=$SEED $PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);sd=os.environ['SEED']
a=json.load(open(O/f'e57_syn_rules_s{sd}/e57_syn_tokread_s{sd}.json'));b=json.load(open(O/f'e57_syn_massgrid_s{sd}/e57_syn_tokread_s{sd}.json'))
assert a['tokids']==b['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx
json.dump(a,open(O/f'e57_syn_massgrid_s{sd}/e57_syn_tokread_s{sd}_all.json','w'));print('merged arms',sorted(a['arms']))
PYE
  $PY $S/e57_curve_report.py --kind syn --seed $SEED --tag synqmg --tokread $TT/e57_syn_tokread_s${SEED}_all.json --out $TT/e57_curve_massgrid_s$SEED.json
done
echo "done"
