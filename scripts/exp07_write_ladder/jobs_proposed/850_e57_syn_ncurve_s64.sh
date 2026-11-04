#!/bin/bash
# JOB 850 = E57 SYNTHETIC EXPOSURE CURVE OF THE MASKING EXTRA, SEED 64 (QA thread, Hans '1번만: 합성 n별 재읽기'; NOT QUEUED until Hans confirms the order and the scope): NATQ (new arm M3NMW1N05 = no mask, natural window, loss weight 1, Adam direction, update norm 0.5 x NAT; end-state hash compared with the stored NATQ where it exists) and ALLQ256 (M3W1N05, bit-identical to the stored ALLQ256) are written again with the state saved at the exposures a = 9 / 17 / 33 / 65 / 128 per fact, read per token like job 676 and reported per a with e57_syn_ncurve_report.py. No existing e57_*.py edited, no git checkout.

for f in e57_syn_mrun.py e57_syn_tokread.py e57_syn_ncurve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_syn_ncurve_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_ncurve_s64}; mkdir -p "$TT"
for f in $SQ/syn_s64_m30_masks.npz $OUT/E55_v3a_NAT_s64.saved.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_syn_n_curve_s64.json" ] && [ -z "$RERUN" ] && { echo "curve json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64  --out $SQ --tag synqn --m30 $SQ/syn_s64_m30_masks.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --arms M3NMW1N05,M3W1N05 --save-apps 9,17,33,65,128
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);res={}
def end(f):
    r=json.load(open(f))['reads'];ch=str(max(int(k) for k in r));return r[ch]['_hash']
for new,old in (('M3NMW1N05','e57_synq/E55_synq_NATQ_s64.saved.json'),('M3W1N05','e57_synq/E55_synq_ALLQ256_s64.saved.json')):
    if old is None:continue
    h=end(O/f'e57_synq/E55_synqn_{new}_s64.saved.json');o=end(O/old);res[new]=dict(end_state_hash_new=h,end_state_hash_original=o,equal=h==o)
print('end-state hash check',res);json.dump(res,open(O/'e57_syn_ncurve_s64'/'e57_hash_check_s64.json','w'))
PYE
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqn
$PY $S/e57_syn_ncurve_report.py --seed 64 --tag synqn --tokread $TT/e57_syn_tokread_s64.json --ref676 $OUT/e57_syn_tokread/e57_syn_tokread_s64.json --out $TT/e57_syn_n_curve_s64.json
echo "outputs: $TT/e57_syn_n_curve_s64.json $TT/e57_syn_tokread_s64.json $TT/e57_hash_check_s64.json"
