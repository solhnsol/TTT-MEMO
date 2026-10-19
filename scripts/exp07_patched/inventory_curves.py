"""Inventory of exposure-matched curves (design request 10/17 16:31 KST): per seed, which of {L1 offline, L0fs, L0or, L0mwfs} exist, pass range, last
undamaged pass k* (canary rise <= 0.10 on both worlds, all earlier passes undamaged too), D(R1)/D(R1-alt) at k*, R* = D(variant,k*)/D(L1,k*) (R1 and R1-alt, pair bootstrap 95% CI).
Usage: inventory_curves.py   (files only, CPU)"""
import sys,json,importlib,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
def mods(sd,tag):
    sys.argv=['x',str(sd),tag]
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def rb(num,den,n=4000):
    r=np.random.default_rng(0);k=len(num);b=[num[i].mean()/den[i].mean() for i in (r.integers(0,k,k) for _ in range(n))]
    return float(num.mean()/den.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
CELLS=[('L1 offline','L1','main'),('L1 offline (ck, alt re-eval)','L1','ck'),('L0fs (native fresh-filler)','L0fs','main'),('L0or (native oracle-filler-removed)','L0or','main'),('L0mwfs x1 (masked fresh-filler)','L0mwfs','x1'),('L0mp (native, same stream, repeated)','L0mp','main')]
out={}
for sd in (101,102,103):
    out[sd]={}
    meta=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))
    AL,S=mods(sd,'main');cur={}
    for name,lev,tag in CELLS:
        try:X=S.load(lev,tag)
        except Exception as e:X=None
        if not X:print(sd,name,'-- missing');continue
        A_,B_=X;b1=(A_[0],B_[0]);has_alt='_alt' in A_[0]['margins']
        alt=lambda e:{'epoch':e.get('epoch',e.get('state')),'margins':e['margins']['_alt']};b2=(alt(A_[0]),alt(B_[0])) if has_alt else None
        c=[]
        for a_,b_ in zip(A_,B_):
            ep=a_.get('epoch',a_.get('state'));d=max([a_['margins'][k]-A_[0]['margins'][k] for k in ('_gen_nll','_tail_nll') if k in a_['margins']]+[b_['margins'][k]-B_[0]['margins'][k] for k in ('_gen_nll','_tail_nll') if k in b_['margins']] or [float('nan')])
            c.append((ep,AL.stats(a_,b_,b1),AL.stats(alt(a_),alt(b_),b2) if has_alt else None,d))
        cur[name]=c
        eps=[e for e,*_ in c];dm=[d for *_,d in c]
        kstar=0
        for (ep,s1,sa,d) in c:
            if ep==0:continue
            if d!=d or d<=0.10:kstar=ep   # nan = canary not logged in this file (treated as undamaged, flagged)
            else:break
        print(sd,f"{name}: passes {eps[1]}-{eps[-1]} ({len(eps)-1} readouts), canary max {np.nanmax(dm) if not all(x!=x for x in dm) else float('nan'):+.3f}, k*={kstar}")
        out[sd][name]={'eps':eps,'kstar':kstar,'canary':dm}
    L1=cur.get('L1 offline')
    for name,c in cur.items():
        if name.startswith('L1'):continue
        k=out[sd][name]['kstar']
        if k<2 or not L1:print(sd,name,'R*: k*<2 or no L1');continue
        e1={ep:(s1,sa) for ep,s1,sa,d in L1};v={ep:(s1,sa) for ep,s1,sa,d in c}
        if k not in e1:print(sd,name,f'no L1 readout at pass {k}');continue
        r1=rb(v[k][0]['D'],e1[k][0]['D']);ra=rb(v[k][1]['D'],e1[k][1]['D']) if v[k][1] is not None and e1[k][1] is not None else None
        print(f"  {sd} {name} at k*={k}: D(R1) var {v[k][0]['D'].mean():+.3f} L1 {e1[k][0]['D'].mean():+.3f} R* {r1[0]:.3f} [{r1[1]:.3f},{r1[2]:.3f}]"+(f"; R1-alt var {v[k][1]['D'].mean():+.3f} L1 {e1[k][1]['D'].mean():+.3f} R* {ra[0]:.3f} [{ra[1]:.3f},{ra[2]:.3f}]" if ra else ''))
        out[sd][name]['R_star']={'R1':r1,'alt':ra}
json.dump(out,open(meta/'inventory_curves.json','w'),indent=1,default=str)
