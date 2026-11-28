"""Analysis of job 1363 'redo1' (S2628-REDO step 1; CPU; new file; implementation thread 11/28; descriptive, no threshold). Arms: ISO1 (cbo-format json, redo1/iso1/cbo_s<seed>.json), Dr and De (kvmulti-format, redo1/kvmulti_s<seed>.json), P1d (job 1362, p1d/), P1s and P0s (job 1348, kvmulti/). R1 matched binding / familiarity / margin change (e59_kvmask_analyze.py definitions; R1 = the neutral 8192-token tail). MAIN = matched binding at 128, seed-paired: P1d - ISO1, P1d - Dr, P1d - De; beside them Dr - De, ISO1 - P1s, Dr - P1s, De - P1s, and the curves at the read points common to all arms (8/16/32/64/96/128).
  python e59_redo1_analyze.py --redo REDO1_DIR --p1d P1D_DIR --base KVMULTI_DIR"""
import sys,json,glob,random,os,numpy as np
from scipy import stats
a=sys.argv;g=lambda k:a[a.index(k)+1];RE=g('--redo');PD=g('--p1d');BA=g('--base')
SEEDS=sorted(set(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(RE+'/kvmulti_s*.json'))|set(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(RE+'/iso1/cbo_s*.json')))
def load(p):return json.load(open(p)) if os.path.exists(p) else None
J={s:dict(re=load(f'{RE}/kvmulti_s{s}.json'),iso=load(f'{RE}/iso1/cbo_s{s}.json'),p1d=load(f'{PD}/kvmulti_s{s}.json'),base=load(f'{BA}/kvmulti_s{s}.json')) for s in SEEDS}
def rec(arm,s):
    j=J[s];src={'ISO1':j['iso'],'Dr':j['re'],'De':j['re'],'P1d':j['p1d'],'P1s':j['base'],'P0s':j['base']}[arm]
    return (src['arms'].get(arm),src['W0_R1']) if src else (None,None)
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
def col(arm,e,mi):
    out=[]
    for s in SEEDS:
        r,W0=rec(arm,s);d=(r or {}).get('reads',{}).get(str(e));out.append(np.nanmean(per_fact(d['dS1'],W0,s)[:,mi]) if d else np.nan)
    return np.array(out)
w0s=[float(np.abs(np.asarray(rec('Dr',s)[1] or rec('ISO1',s)[1])-np.asarray(J[s]['base']['W0_R1'])).max()) for s in SEEDS]
print(f'1363 redo step 1: seeds {SEEDS}; W0 R1 max abs vs 1348 {max(w0s):.2e}')
print('== MAIN R1 matched binding at 128 (seed-paired)')
for x,y in (('P1d','ISO1'),('P1d','Dr'),('P1d','De'),('Dr','De'),('ISO1','P1s'),('Dr','P1s'),('De','P1s'),('P1d','P1s')):print(f'  {x:4s} - {y:4s}: {tci(col(x,128,0)-col(y,128,0))}')
for mi,mn in enumerate(('matched binding','familiarity','margin change')):
    print(f'== R1 curve [{mn}]')
    for arm in ('ISO1','P1d','Dr','De','P1s','P0s'):print(f'  {arm:4s}: '+' | '.join(f'e{e} {tci(col(arm,e,mi))}' for e in (8,16,32,64,96,128)))
