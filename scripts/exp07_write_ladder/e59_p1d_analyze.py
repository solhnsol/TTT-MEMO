"""Analysis of job 1362 'p1d' (CPU; new file; implementation thread 11/28; descriptive, no threshold). R1 matched binding / familiarity / margin change (e59_kvmask_analyze.py definitions) of P1d against the 1348 streams P1s (C0) and P0s (SAMEVIS), same seeds. MAIN = matched binding at 128, P1d - P1s seed-paired; P1d - P0s; the curve 8..128 of all three; R2 at 128 (1-chunk context, as 1348).
  python e59_p1d_analyze.py --dir P1D_DIR --base KVMULTI_DIR"""
import sys,json,glob,random,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];B=a[a.index('--base')+1]
SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/kvmulti_s*.json'));RD={s:json.load(open(f'{D}/kvmulti_s{s}.json')) for s in SEEDS};RB={s:json.load(open(f'{B}/kvmulti_s{s}.json')) for s in SEEDS}
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def per_fact(dS,W0,seed):
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);mar=lambda S:S[:,0]-S[:,1:12].max(1);out=np.full((16,3),np.nan)
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];out[f,0]=dS[f,0]-np.mean(oth) if oth else np.nan
    out[:,1]=dS[:,1:12].mean(1)-dS[:,12:16].mean(1);out[:,2]=mar(W0+dS)-mar(W0);return out
src=lambda arm,s:RD[s]['arms']['P1d'] if arm=='P1d' else RB[s]['arms'][arm]
def col(arm,e,mi,key='dS1',w0='W0_R1'):
    out=[]
    for s in SEEDS:
        d=src(arm,s)['reads'].get(str(e),{});W0=(RD[s] if arm=='P1d' else RB[s])[w0];out.append(np.nanmean(per_fact(d[key],W0,s)[:,mi]) if key in d else np.nan)
    return np.array(out)
print(f'1362 P1d: seeds {SEEDS}; W0 R1 identical to 1348: {[float(np.abs(np.asarray(RD[s]["W0_R1"])-np.asarray(RB[s]["W0_R1"])).max())==0.0 for s in SEEDS]}')
print('== MAIN R1 matched binding at 128 (seed-paired)')
for x,y in (('P1d','P1s'),('P1d','P0s'),('P1s','P0s')):print(f'  {x} - {y}: {tci(col(x,128,0)-col(y,128,0))}')
for mi,mn in enumerate(('matched binding','familiarity','margin change')):
    print(f'== R1 curve [{mn}]')
    for arm in ('P1d','P1s','P0s'):print(f'  {arm}: '+' | '.join(f'e{e} {tci(col(arm,e,mi))}' for e in (8,16,32,64,96,128)))
print('== R2 at 128 (1-chunk context, as 1348) matched binding: '+' | '.join(f'{arm} {tci(col(arm,128,0,"dS2","W0_R2"))}' for arm in ('P1d','P1s','P0s')))
