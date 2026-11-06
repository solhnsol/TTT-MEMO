#!/bin/bash
# JOB 1130 = E57 SYNTHETIC MASS-MATCHED GRID (Hans 11/06 15:41 KST 'ok, do it'): does the share or the extra loss mass share x (w-1) decide performance? Extra mass fixed at 0.75: (5%, w=16) [exists from job 1071-1073: M3T05W16N05], (10%, w=8.5), (15%, w=6), (30%, w=3.5); ALLQ256 on every row; rows by g128 = nll_band128 - nll_win. New arms per world: M3T10W8P5N05, M3T15W6N05, M3W3P5N05. Usage: SEED=62 CELL=v3_16_1 PY=/abs/python bash 1130_e57_syn_massgrid.sh (tuning worlds s62 v3_16_1, s71 v3_4_1, s72 v3_32_1; same worlds as job 1071-1073, whose tokread is merged for the paired report with ALLQ256, M3 w=8, T05 W16, ABS2). Exploratory; no new rule. Expected: about 12 min per arm, 3 arms per world.
set -eo pipefail
export PYTHONUNBUFFERED=1
: "${SEED:?set SEED}"; : "${CELL:?set CELL}"
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${ARMS:=M3T10W8P5N05,M3T15W6N05,M3W3P5N05}"
: "${TAUS:=2,4,6}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_rules.py e57_syn_rmasks.py e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_rules.py --selftest | tail -1; $PY $S/e57_syn_mrun.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=$OUT/e57_syn_massgrid_s$SEED; mkdir -p "$TT"
if [ -f "$OUT/E55_v3a_NAT_s$SEED.saved.json" ]; then NTG=$OUT/E55_v3a_NAT_s$SEED.saved.json; else NTG=$SQ/E55_synqnat_NATSGD_s$SEED.saved.json; fi
for f in $NTG $OUT/e57_syn_s$SEED/syn_s${SEED}_nll128.npz; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -f "$SQ/syn_s${SEED}_massgrid.npz" ] || $PY $S/e57_syn_rmasks.py --seed $SEED --cell $CELL --taus 2 --out "$SQ/syn_s${SEED}_massgrid.npz"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_syn_mrun.py --seed $SEED --cell $CELL --out $SQ --tag synqmg --m30 $SQ/syn_s${SEED}_massgrid.npz --ntg-nat $NTG --arms $ARMS --save-apps 16,32,64,96,128
$PY $S/e57_syn_tokread.py --stage read --seed $SEED --cell $CELL --out "$TT" --tag synqmg
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);sd=os.environ['SEED']
a=json.load(open(O/f'e57_syn_rules_s{sd}/e57_syn_tokread_s{sd}.json'));b=json.load(open(O/f'e57_syn_massgrid_s{sd}/e57_syn_tokread_s{sd}.json'))
assert a['tokids']==b['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx
json.dump(a,open(O/f'e57_syn_massgrid_s{sd}/e57_syn_tokread_s{sd}_all.json','w'));print('merged arms',sorted(a['arms']))
PYE
$PY $S/e57_curve_report.py --kind syn --seed $SEED --tag synqmg --tokread $TT/e57_syn_tokread_s${SEED}_all.json --out $TT/e57_curve_massgrid_s$SEED.json
echo "outputs: $TT/e57_curve_massgrid_s$SEED.json; mask stats $SQ/syn_s${SEED}_massgrid.json (mean share per rule)"
