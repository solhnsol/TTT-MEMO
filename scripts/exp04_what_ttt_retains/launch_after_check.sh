#!/bin/bash
# wait for the probe-path smoke test, judge it, then start the overnight watchdog (full run if healthy, Stage-1-only otherwise)
cd /home/solhnsol/workspace/TTT-MEMO
EXP=experiments/exp04_what_ttt_retains
while ps -eo cmd | grep -q "[s]moke_probe.py"; do sleep 10; done
ok=$(.venv/bin/python -I - <<'PY'
import json,glob,math
fs=sorted(glob.glob('experiments/exp04_what_ttt_retains/smoke_probe/smoke_cal_*_ON.json'))
if len(fs)<2:print('FAIL:missing');raise SystemExit
bad=False;msg=[]
for f in fs:
    rows=json.load(open(f))['rows'];m=[r['gold']-r['foil'] for r in rows]
    g=[r['gold'] for r in rows]
    fin=all(math.isfinite(x) for x in g+m);nz=sum(1 for x in g if x!=0)/len(g);var=len(set(round(x,3) for x in m))
    msg.append(f"{f.split('_')[-2]}: n={len(rows)} nonzero_gold={nz:.2f} distinct_margins={var} mean_margin={sum(m)/len(m):+.2f}")
    if not fin or nz<0.95 or var<20:bad=True
print(('FAIL:' if bad else 'OK:')+' | '.join(msg))
PY
)
echo "$(date -u) probe check: $ok" >> $EXP/watchdog.log
rm -f $EXP/DRIVER_DONE
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
case "$ok" in OK:*) ;; *) export EXP04_SKIP_STAGE2=1;; esac
exec bash scripts/exp04_what_ttt_retains/watchdog.sh
