"""Design request 10/17 16:40 KST (files only): (1) L1 epoch-1 D, R1 and R1-alt, pair-bootstrap CI, seeds 101-103; (2) equal-FLOPs table: masked fresh-filler pass k (k = 1..4,
2.5F per chunk) vs native fresh-filler pass ceil(5k/3) (1.5F per chunk): D (R1, R1-alt) with CI and canary (max rise of gen / tail NLL over both worlds)."""
import sys,json,importlib,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
def mods(sd):
    sys.argv=['x',str(sd),'main']
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
def curve(AL,S,lev,tag):
    A,B=S.load(lev,tag);b1=(A[0],B[0]);alt=lambda e:{'epoch':e.get('epoch',e.get('state')),'margins':e['margins']['_alt']};has='_alt' in A[0]['margins'];b2=(alt(A[0]),alt(B[0])) if has else None;out={}
    for a,b in zip(A,B):
        ep=a.get('epoch',a.get('state'));ks=[k for k in ('_gen_nll','_tail_nll') if k in a['margins']]
        can=max([a['margins'][k]-A[0]['margins'][k] for k in ks]+[b['margins'][k]-B[0]['margins'][k] for k in ks]) if ks else float('nan')
        out[ep]=(AL.stats(a,b,b1)['D'],AL.stats(alt(a),alt(b),b2)['D'] if has else None,can)
    return out
def f(AL,D):
    if D is None:return 'n/a'
    c=AL.ci(D);return f'{D.mean():+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'
for sd in (101,102,103):
    AL,S=mods(sd);L1m=curve(AL,S,'L1','main');L1c=curve(AL,S,'L1','ck');M=curve(AL,S,'L0mwfs','x1');N=curve(AL,S,'L0fs','main')
    print(f'seed {sd}: L1 epoch 1: D(R1, main file) {f(AL,L1m[1][0])}; D(R1, ck file) {f(AL,L1c[1][0])}; D(R1-alt, ck file) {f(AL,L1c[1][1])}; canary main {L1m[1][2]:+.4f}, ck {L1c[1][2]:+.4f}')
    for k in (1,2,3,4):
        n=math.ceil(5*k/3);m=M[k];nn=N[n]
        print(f'  masked-fs pass {k}: R1 {f(AL,m[0])}; alt {f(AL,m[1])}; canary {m[2]:+.3f}   |  native-fs pass {n}: R1 {f(AL,nn[0])}; alt {f(AL,nn[1])}; canary {nn[2]:+.3f}')
