#!/bin/bash
# JOB 910 = E57 SYNTHETIC P1 CAPPED CONTRAST WRITE (Hans '고고' 11/04 19:55 KST after the read of job 905: the uncapped term of job 900 diverged): the term is - beta * sum_{i in D} w_i min(nll'_i, c) with c = 8 (the W0 mean of nll' over the deciding rows, about 7.95), so the term has no gradient above c; beta 0.5 and 1.0; states at appearances 16 / 32 / 64 / 96 / 128 (past the exposure 33); rules and stop conditions as in job 900. Previous text: E57 SYNTHETIC P1 CONTRAST WRITE (investigation thread 'binding learned fast at few exposures', Hans 11/04 13:02 KST card: band 1, beta 0.5 and 1.0; design E57 122-AB addendum 16; seed 64): M3 w=8 (ALLQ256 mask from chunk 8, loss weight 8 on the deciding rows = top 30% by window gain, update norm 0.5 x NAT) plus the term  - beta * sum_{i in D} w_i nll'_i  where nll' = the NLL of the deciding rows in a second suffix forward in which every row sees only its own position (band 1); arms M3PB05W8N05 (beta 0.5) and M3PB10W8N05 (beta 1.0); states at appearances 16 / 32 / 64 / 96 / 128 with the canary and the inline reads of jobs 710 / 730; an arm is stopped when the mean nll' over D is above 100 or not finite, or when the generation canary rises by more than 0.5; the mean nll' over D per chunk (contrast_log) and the update norms (log) are in the saved json. Per-token reads of all states and the curve against ALLQ256 and M3 w=8 of job 800. Arms added to e57_syn_mrun.py; no existing e57_*.py edited, no git checkout.
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
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_p2c}; T0=$OUT/e57_syn_curve_s64; mkdir -p "$TT"
for f in $SQ/syn_s64_m30_masks.npz $OUT/E55_v3a_NAT_s64.saved.json $T0/e57_syn_tokread_s64.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_p2c_s64.json" ] && [ -z "$RERUN" ] && { echo "curve json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64 --out $SQ --tag synqp2 --m30 $SQ/syn_s64_m30_masks.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --arms M3PBC05W8N05,M3PBC10W8N05 --save-apps 16,32,64,96,128
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqp2
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);a=json.load(open(O/'e57_syn_curve_s64/e57_syn_tokread_s64.json'));b=json.load(open(O/'e57_syn_p2c/e57_syn_tokread_s64.json'))
assert a['tokids']==b['tokids'];mx=max(abs(sum(x)-sum(y)) for x,y in zip(a['W0'],b['W0']));assert mx<1e-3,mx
a['arms'].update(b['arms']);a['reproduction'].update(b['reproduction']);a['W0_old_vs_new_max_abs_diff_sum']=mx;json.dump(a,open(O/'e57_syn_p2c/e57_syn_tokread_s64_all.json','w'));print('merged arms',sorted(a['arms']))
for arm in ('M3PBC05W8N05','M3PBC10W8N05'):
    f=O/f'e57_synq/E55_synqp2_{arm}_s64.saved.json'
    if not f.exists():print(arm,'no file');continue
    x=json.load(open(f))
    if 'aborted' in x:print(arm,'ABORTED',x['aborted']);continue
    w0=x['canary_w0'];print(arm,'canary',{k:(round(v['canary']['_gen_nll']-w0['_gen_nll'],3),round(v['canary']['_tail_nll']-w0['_tail_nll'],3)) for k,v in x['reads'].items() if 'canary' in v})
    cl=x.get('contrast_log') or [];print(arm,'mean nll prime over D per chunk: first',cl[:3],'last',cl[-3:],'max',max(cl) if cl else None)
    print(arm,'mean update norm',x['summary']['mean_update_norm'])
PYE
$PY $S/e57_curve_report.py --kind syn --seed 64 --tag synqp2 --tokread $TT/e57_syn_tokread_s64_all.json --out $TT/e57_curve_p2c_s64.json
echo "outputs: $TT/e57_curve_p2c_s64.json $TT/e57_syn_tokread_s64.json $SQ/E55_synqp2_{M3PBC05W8N05,M3PBC10W8N05}_s64.{saved,read}.json"
