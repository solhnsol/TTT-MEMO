"""S3 power check with damage truncation (design 16:50 KST 10/17), files only. Base cells: native-fs (L0fs main) and masked-fs (L0mwfs x1), s101-103, R1-alt D_alt per pair.
Per seed: p = earliest pass with canary (max over gen/tail NLL, both worlds) > 0.10 over the two cells; K_c = min(p - 1, 4). Area = mean D_alt over passes 1..K_c.
Pooled over seeds with K_c >= 2 (one 15-pair vector per seed, 45 pairs max): gain = area(masked-fs) - area(native-fs); relative SE = bootstrap SE / mean.
Also pass 3 and pass 4 (seeds with K_c >= pass). Q2 (long-context) is available for masked-fs only (longnll_L0mwfs_x1_*: beyond-window rise <= 0.041 everywhere); native-fs Q2 is not measured."""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
def mods(sd):
    sys.argv=['x',str(sd),'main']
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def cur(AL,S,lev,tag):
    A,B=S.load(lev,tag);alt=lambda e:{'epoch':e.get('epoch'),'margins':e['margins']['_alt']};b2=(alt(A[0]),alt(B[0]));out={}
    for a,b in zip(A,B):
        ep=a.get('epoch');can=max([a['margins'][k]-A[0]['margins'][k] for k in ('_gen_nll','_tail_nll')]+[b['margins'][k]-B[0]['margins'][k] for k in ('_gen_nll','_tail_nll')])
        out[ep]=(AL.stats(alt(a),alt(b),b2)['D'],can)
    return out
rng=np.random.default_rng(0);data={}
for sd in (101,102,103):
    AL,S=mods(sd);N=cur(AL,S,'L0fs','main');M=cur(AL,S,'L0mwfs','x1')
    dam=lambda C:next((p for p in range(1,9) if C[p][1]>0.10),99);pn,pm=dam(N),dam(M);p=min(pn,pm);Kc=min(p-1,4)
    data[sd]=dict(N=N,M=M,pn=pn,pm=pm,Kc=Kc);print(f'seed {sd}: first damaged pass native {pn if pn<99 else "none"}, masked {pm if pm<99 else "none"} -> K_c = {Kc}')
def area(C,K):return np.mean([C[p][0] for p in range(1,K+1)],axis=0)
def rel(vecs):
    v=np.concatenate(vecs);b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(8000)];return v.mean(),float(np.std(b)),float(np.std(b)/abs(v.mean())),float(np.percentile(b,2.5)),float(np.percentile(b,97.5)),len(v)
res={}
use=[sd for sd in data if data[sd]['Kc']>=2]
g=[area(data[sd]['M'],data[sd]['Kc'])-area(data[sd]['N'],data[sd]['Kc']) for sd in use];m,se,r,lo,hi,n=rel(g)
print(f'truncated area gain (masked-fs - native-fs), seeds {use} ({n} pairs): mean {m:+.3f}, SE {se:.3f}, relative SE {r:.2f}, 95% CI [{lo:+.3f},{hi:+.3f}]');res['area_trunc']=dict(seeds=use,mean=m,se=se,rel=r,ci=[lo,hi],n=n)
for k in (3,4):
    u=[sd for sd in data if data[sd]['Kc']>=k];g=[data[sd]['M'][k][0]-data[sd]['N'][k][0] for sd in u]
    if g:m,se,r,lo,hi,n=rel(g);print(f'pass {k} gain, seeds {u} ({n} pairs): mean {m:+.3f}, SE {se:.3f}, relative SE {r:.2f}, CI [{lo:+.3f},{hi:+.3f}]');res[f'pass{k}']=dict(seeds=u,mean=m,se=se,rel=r,ci=[lo,hi],n=n)
g=[area(data[sd]['M'],4)-area(data[sd]['N'],4) for sd in data];m,se,r,lo,hi,n=rel(g);print(f'(descriptive, untruncated 4-pass area, all seeds, {n} pairs): mean {m:+.3f}, SE {se:.3f}, relative SE {r:.2f}');res['area_untrunc']=dict(mean=m,se=se,rel=r,n=n)
json.dump(res,open(Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder/s3_power_trunc.json','w'),indent=1)
