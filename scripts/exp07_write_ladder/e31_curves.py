"""E31 A1-A4 (CPU, read-only): pass-wise D_alt increments, exposures to reach the same D, common-pass ratio to L1, step/norm axes.
Usage: e31_curves.py SEED [--cells native=L0fs_main,masked=L0mwfs_x1,native2=L0x...,L1=L1_ck] [--kc 4] [--unit-test]
A cell NAME=PREFIX reads OUT/PREFIX_s{SEED}_{A,B}.json (same layout as ladder.py). L1 cell = last, named 'L1', exposure unit = epoch (10 SGD steps).
D_k = per-pair D_alt(k) - D_alt(0) (W0 baseline removed); the raw value is also printed for the unit test. 15 pairs, pair bootstrap 10000, seed 31.
Outputs OUT/e31_curves_s{SEED}.{md,json}. Never writes input files."""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('seed',type=int);ap.add_argument('--cells',default='native=L0fs_main,L1=L1_ck')
ap.add_argument('--kc',type=int,default=None,help='E20 damage-free pass count for this seed (reported next to the canary-based k*)');ap.add_argument('--targets',default='0.25,0.40')
ap.add_argument('--nboot',type=int,default=10000);ap.add_argument('--canary',type=float,default=0.10)
a=ap.parse_args();sys.argv=[sys.argv[0],str(a.seed),'ck']
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_ladder as AL   # AL.stats = pair-level D with the shared pairing; AL.meta from meta_L1_ck_s{seed}.json
OUT=AL.OUT;rng=np.random.default_rng(31);STEPS={'L1':10}   # SGD steps per exposure unit; native pass = 63 chunks
def load(prefix):
    A=[e for e in json.load(open(OUT/f'{prefix}_s{a.seed}_A.json'))['log'] if 'margins' in e];B=[e for e in json.load(open(OUT/f'{prefix}_s{a.seed}_B.json'))['log'] if 'margins' in e]
    assert len(A)==len(B);return A,B
def alt(e):return {'margins':e['margins']['_alt']}
def cell(name,prefix):
    A,B=load(prefix);b=(alt(A[0]),alt(B[0]));ex=[e.get('epoch',e.get('state')) for e in A]
    if ex!=list(range(len(A))):   # exposure axis must be pass/epoch 0..K; anything else (labels like -1, None, state counters) is replaced by the position and flagged
        print(f'WARNING {name}: point labels {ex} are not 0..{len(A)-1}; using position index as pass number',file=sys.stderr);WARN.append(f'{name}: labels {ex} replaced by 0..{len(A)-1}');ex=list(range(len(A)))
    D=np.stack([AL.stats(alt(x),alt(y),b)['D'] for x,y in zip(A,B)]);D1=np.stack([AL.stats(x,y,(A[0],B[0]))['D'] for x,y in zip(A,B)])
    can=np.array([[max(e['margins']['_gen_nll']-E[0]['margins']['_gen_nll'],e['margins']['_tail_nll']-E[0]['margins']['_tail_nll']) for e in E] for E in (A,B)]).max(0)
    return dict(name=name,ex=ex,Draw=D,D=D-D[0],R1=D1-D1[0],canary=can,cum=[e.get('cum_update_norm') for e in A],
                steps=[(x or 0)*(63 if name!='L1' else STEPS['L1']) for x in ex])
WARN=[]
cells=[cell(*kv.split('=')) for kv in a.cells.split(',')]
L1=next(c for c in cells if c['name']=='L1');nat=[c for c in cells if c['name']!='L1']
ci=lambda v:(float(np.mean(v)),*np.percentile(v,[2.5,97.5]).tolist())
f3=lambda t:f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
n=L1['D'].shape[1];BI=rng.integers(0,n,(a.nboot,n))   # shared pair resamples across all cells
md=[f'# E31 curves, seed {a.seed}, R1-alt D (baseline-removed), 15 pairs, pair bootstrap {a.nboot}\n'];res={}
md+= [f'AXIS WARNING: {w}\n' for w in WARN]
md.append(f"Raw D_alt at W0 (not removed): "+', '.join(f"{c['name']} {c['Draw'][0].mean():+.3f}" for c in cells)+'\n')
def kstar(c):
    k=0
    for i in range(1,len(c['ex'])):
        if c['canary'][i]>a.canary:break
        k=i
    return k
for c in cells:
    md.append(f"## {c['name']}: points {c['ex']}\n\n| k | D_alt | dD | canary rise | cum_update_norm | SGD steps | R1 D |\n|---|---|---|---|---|---|---|")
    for i,k in enumerate(c['ex']):
        dD=f3(ci(c['D'][i]-c['D'][i-1])) if i else '-'
        md.append(f"| {k} | {f3(ci(c['D'][i]))} | {dD} | {c['canary'][i]:+.3f} | {c['cum'][i] if c['cum'][i] is not None else 'NA'} | {c['steps'][i]} | {c['R1'][i].mean():+.3f} |")
    md.append('')
# A1: slope ratio S = mean increment of the second half / first half over the damage-free range
md.append('## A1 slope ratio S (later half / earlier half of increments; odd count: the middle increment is skipped)\n')
for c in cells:
    ks=len(c['ex'])-1 if c['name']=='L1' else kstar(c)
    for lab,K in ([('k*(canary<=%.2f)'%a.canary,ks)]+([('E20 kc',a.kc)] if a.kc and c['name']!='L1' else [])+([('epoch 1-4 vs 5-8',8)] if c['name']=='L1' else [])):
        if K is None or K<2:md.append(f"- {c['name']} {lab}: K={K}, S undefined");continue
        h=K//2;lo=slice(0,h);hi=slice(K-h,K);d=np.diff(c['D'][:K+1],axis=0)
        Sb=[];Sl=d[lo].mean(0);Sh=d[hi].mean(0)
        for b in BI:Sb.append(Sh[b].mean()/Sl[b].mean() if Sl[b].mean()>0 else np.nan)
        S=Sh.mean()/Sl.mean();lo_,hi_=np.nanpercentile(Sb,[2.5,97.5]);cls='saturating (<0.5)' if S<0.5 else ('near-linear' if S<=2 else 'accelerating (>2)')
        md.append(f"- {c['name']} {lab}: K={K} S={S:.2f} [{lo_:.2f},{hi_:.2f}] -> {cls}");res.setdefault('A1',{})[f"{c['name']}|{lab}"]=[float(S),float(lo_),float(hi_),K]
# A2: exposures to reach D*
md.append('\n## A2 exposures needed to first reach D* (linear interpolation; censored = not reached within the measured range, no extrapolation)\n')
def reach(curve,ex,t):
    for i in range(1,len(ex)):
        if curve[i]>=t and curve[i-1]<t:return ex[i-1]+(t-curve[i-1])/(curve[i]-curve[i-1])*(ex[i]-ex[i-1])
    return np.inf
for t in map(float,a.targets.split(',')):
    eL=reach(L1['D'].mean(0),L1['ex'],t);le='<=1 (upper bound)' if eL<=1 else f'{eL:.2f}'
    md.append(f"- D*={t}: e_L1 = {le}")
    for c in nat:
        ks=kstar(c);ex=c['ex'][:ks+1];e=reach(c['D'][:ks+1].mean(0),ex,t);assert np.isinf(e) or e>=ex[0],(c['name'],ex,e)
        eb=np.array([reach(c['D'][:ks+1][:,b].mean(1),ex,t) for b in BI]);cens=float(np.isinf(eb).mean())
        el=np.array([reach(L1['D'][:,b].mean(1),L1['ex'],t) for b in BI]);rb=eb/np.maximum(el,1e-9)
        md.append(f"  - {c['name']}: e = {'> %d (censored, damage-free range)'%ex[-1] if np.isinf(e) else f'{e:.2f}'}; boot censored fraction {cens:.2f}; ratio e/e_L1 (finite part) median {np.nanmedian(rb[np.isfinite(rb)]) if np.isfinite(rb).any() else float('nan'):.2f}")
        res.setdefault('A2',{})[f'{c["name"]}|{t}']={'e':None if np.isinf(e) else float(e),'e_L1':float(eL),'censored_frac':cens}
# A3: common-pass ratio
md.append('\n## A3 R_k = D_cell(k) / D_L1(epoch k) on common damage-free passes\n\n| cell | k | R_k [CI] |\n|---|---|---|')
for c in nat:
    ks=kstar(c);Rk={}
    for k in range(1,ks+1):
        i=c['ex'].index(k);j=L1['ex'].index(k);r=np.array([c['D'][i][b].mean()/L1['D'][j][b].mean() for b in BI]);Rk[k]=r;md.append(f"| {c['name']} | {k} | {ci(r)[0]:.3f} [{ci(r)[1]:.3f},{ci(r)[2]:.3f}] |")
    lo,hi=(4,2) if 4 in Rk and 2 in Rk else ((ks,2) if 2 in Rk and ks>2 else (None,None))
    if lo:
        q=Rk[lo]/Rk[hi];t=ci(q);md.append(f"| {c['name']} | R_{lo}/R_{hi} | {t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}] -> {'fixed ratio (CI within 0.7-1.3)' if t[1]>=0.7 and t[2]<=1.3 else 'CI not within 0.7-1.3'} |");res.setdefault('A3',{})[c['name']]=t
# A4: D vs steps / cum norm (descriptive, no rule)
md.append('\n## A4 D_alt vs cumulative SGD steps and cum_update_norm (descriptive)\n\n| cell | k | steps | cum_update_norm | D_alt |\n|---|---|---|---|---|')
for c in cells:
    for i,k in enumerate(c['ex']):
        if i:md.append(f"| {c['name']} | {k} | {c['steps'][i]} | {c['cum'][i] if c['cum'][i] is not None else 'NA'} | {c['D'][i].mean():+.3f} |")
if True:
    md.append('\n## Unit-test printout (compare with reported values: s101 native-fs pass4 D_alt 0.238, masked-fs pass4 1.330, L1 ep8 4.40, k* R* 0.074/0.169/0.335)\n')
    for c in cells:md.append(f"- {c['name']}: D_alt means by point: "+', '.join(f'{k}:{c["D"][i].mean():.3f}(raw {c["Draw"][i].mean():.3f})' for i,k in enumerate(c['ex'])))
json.dump(res,open(OUT/f'e31_curves_s{a.seed}.json','w'),indent=1);(OUT/f'e31_curves_s{a.seed}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
