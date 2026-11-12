"""(a) native fs x2 vs x1 (s101), descriptive curves + the canary-only selection reading (PROTOCOL 16:54:22 amendment). R1-alt primary, R1 alongside; pair-bootstrap 95% CI."""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.argv=['x','101','main']
AL=importlib.import_module('analyze_ladder');S=importlib.import_module('analyze_stage01')
def cur(lev,tag):
    A,B=S.load(lev,tag);alt=lambda e:{'epoch':e.get('epoch'),'margins':e['margins']['_alt']};b1=(A[0],B[0]);b2=(alt(A[0]),alt(B[0]));o={}
    for a,b in zip(A,B):
        ep=a.get('epoch');ks=[k for k in ('_gen_nll','_tail_nll') if k in a['margins']]
        can=max([a['margins'][k]-A[0]['margins'][k] for k in ks]+[b['margins'][k]-B[0]['margins'][k] for k in ks])
        o[ep]=(AL.stats(a,b,b1)['D'],AL.stats(alt(a),alt(b),b2)['D'],can,AL.stats(a,b,b1)['F_name'].mean()-AL.stats(a,b,b1)['F_null'].mean())
    return o
x2=cur('L0fs','fsx2');x1=cur('L0fs','main');L1=cur('L1','ck')
f=lambda D:(lambda c:f'{D.mean():+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')(AL.ci(D))
print('pass | canary x1 / x2 | D_alt x1 | D_alt x2 | D(R1) x1 | D(R1) x2 | L1 D_alt (ck) same epoch')
for p in range(1,9):print(f'{p} | {x1[p][2]:+.3f} / {x2[p][2]:+.3f} | {f(x1[p][1])} | {f(x2[p][1])} | {x1[p][0].mean():+.3f} | {x2[p][0].mean():+.3f} | {L1[p][1].mean():+.3f}')
def kstar(o):
    k=0
    for p in range(1,9):
        if o[p][2]<=0.10:k=p
        else:break
    return k
for nm,o in (('x1',x1),('x2',x2)):print(f'{nm}: k* (canary <= 0.10 at every pass up to k*) = {kstar(o)}; pass-3 canary {o[3][2]:+.3f}; passes 1-4 all <= 0.10: {all(o[p][2]<=0.10 for p in range(1,5))}; max canary over passes 1-4 {max(o[p][2] for p in range(1,5)):+.3f}')
