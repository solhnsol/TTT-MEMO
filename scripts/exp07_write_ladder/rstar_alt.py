"""R1-alt R* at each variant's own k* (last undamaged pass; canary max(gen, tail) <= 0.10, all earlier passes undamaged too) vs L1 (ck file, R1-alt) at the SAME epoch,
masked-fs (L0mwfs x1) and native-fs (L0fs main), seeds 101-103, pair bootstrap 95% CI; registered rule: R* >= 0.70 with CI lower >= 0.50 (reproduces offline binding).
R1 numbers alongside (from the main L1 file). Files only."""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
def mods(sd):
    sys.argv=['x',str(sd),'main']
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def cur(AL,S,lev,tag):
    A,B=S.load(lev,tag);alt=lambda e:{'epoch':e.get('epoch'),'margins':e['margins']['_alt']};has='_alt' in A[0]['margins']
    b1=(A[0],B[0]);b2=(alt(A[0]),alt(B[0])) if has else None;out={}
    for a,b in zip(A,B):
        ep=a.get('epoch');ks=[k for k in ('_gen_nll','_tail_nll') if k in a['margins']]
        can=max([a['margins'][k]-A[0]['margins'][k] for k in ks]+[b['margins'][k]-B[0]['margins'][k] for k in ks]) if ks else float('nan')
        out[ep]=(AL.stats(a,b,b1)['D'],AL.stats(alt(a),alt(b),b2)['D'] if has else None,can)
    return out
def rb(n,d,N=8000):
    r=np.random.default_rng(0);k=len(n);b=[n[i].mean()/d[i].mean() for i in (r.integers(0,k,k) for _ in range(N))];return float(n.mean()/d.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
res={}
for sd in (101,102,103):
    AL,S=mods(sd);L1=cur(AL,S,'L1','ck');L1m=cur(AL,S,'L1','main')
    for name,lev,tag in (('masked-fs','L0mwfs','x1'),('native-fs','L0fs','main')):
        C=cur(AL,S,lev,tag);k=0
        for p in range(1,9):
            if C[p][2]<=0.10:k=p
            else:break
        a=rb(C[k][1],L1[k][1]);r1=rb(C[k][0],L1m[k][0]);ok=a[0]>=0.70 and a[1]>=0.50
        print(f'seed {sd} {name}: k*={k}; R1-alt D var {C[k][1].mean():+.3f} L1(ep{k}) {L1[k][1].mean():+.3f} R* {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}] -> registered rule (>=0.70, CI lower >=0.50): {"PASS" if ok else "no"}; R1: R* {r1[0]:.3f} [{r1[1]:.3f},{r1[2]:.3f}]')
        res[f'{sd}|{name}']=dict(kstar=k,alt=a,r1=r1,rule_pass=bool(ok))
json.dump(res,open(Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder/rstar_alt.json','w'),indent=1)
