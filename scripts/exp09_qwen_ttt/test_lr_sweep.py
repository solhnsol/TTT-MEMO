"""Test 10: run_lr_sweep.py end to end on the tiny model (CPU): 2 lrs x {L0mp, L0mw}, 2 passes, truncated stream; the
report covers every lr and the selection equals the rule recomputed independently from the raw logs."""
from _testutil import *
import json,shutil,subprocess,os
SW=ROOT/'experiments/exp09_qwen_ttt/smoke_lr_sweep'
if SW.exists():shutil.rmtree(SW)
LRS=[1e-3,3e2]
cmd=[sys.executable,str(HERE/'run_lr_sweep.py'),'--out',str(SW),'--lrs',','.join(map(str,LRS)),'--passes','2','--',
     '--tiny','--device','cpu','--max-chunks','2','--chunk','64','--tail-len','128','--canary-len','128','--readout-batch','16','--threads','8']
p=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
check('10 run_lr_sweep.py ran (tiny, 2 lrs x 2 levels x 2 worlds)',p.returncode==0,p.stderr[-600:] if p.returncode else 'ok')
S=json.load(open(SW/'lr_sweep_s101.json'))
def rise1(lev,lr):
    t='lr'+f'{lr:g}';r=-1e9
    for w in 'AB':
        ev=[e for e in json.load(open(SW/f'{lev}_{t}_s101_{w}.json'))['log'] if 'margins' in e]
        for k in ('_gen_nll','_tail_nll'):r=max(r,ev[1]['margins'][k]-ev[0]['margins'][k])
    return r
for lev in ('L0mp','L0mw'):
    rows=S[lev]['rows'];ind={lr:rise1(lev,lr) for lr in LRS};el=[lr for lr in LRS if ind[lr]<=0.10]
    ok=len(rows)==len(LRS) and all(abs(r['rise_pass1']-ind[r['lr']])<1e-12 for r in rows) and S[lev]['selected_lr']==(max(el) if el else None)
    check(f'10 {lev}: every lr reported; selection = largest lr with pass-1 canary rise <= 0.10',ok,
          ', '.join(f'lr {lr:g}: rise {ind[lr]:+.3f}' for lr in LRS)+f' -> selected {S[lev]["selected_lr"]}')
check('10 report markdown written',(SW/'lr_sweep_report_s101.md').exists())
# synthetic over-threshold condition: copy of the lr 300 logs with _gen_nll raised by 0.5 nat after pass 0
for w in 'AB':
    d=json.load(open(SW/f'L0mp_lr300_s101_{w}.json'))
    for e in d['log']:
        if 'margins' in e and e['epoch']>0:e['margins']['_gen_nll']+=0.5
    json.dump(d,open(SW/f'L0mp_lr1e+09_s101_{w}.json','w'))
import run_lr_sweep as RS
S2=RS.report(SW,['L0mp'],[1e-3,3e2,1e9],101)
r=S2['L0mp']['rows'];check('10 an lr with pass-1 canary rise > 0.10 is reported but not selected',S2['L0mp']['selected_lr']==3e2 and not r[2]['eligible'] and r[2]['rise_pass1']>0.10,f"lr 1e9 rise {r[2]['rise_pass1']:+.3f} -> selected {S2['L0mp']['selected_lr']}")
done()
