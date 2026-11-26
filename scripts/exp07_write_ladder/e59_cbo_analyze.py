"""Analysis of job 1357 'cbo' (e59_cbo.py, + e59_kvwin.py --srcs for the in-window reads; CPU; new file; implementation thread 11/26; descriptive: values and intervals over seeds, same-direction seeds, no verdict wording, no threshold). Matched binding / familiarity / margin change exactly as e59_kvmask_analyze.py (same candidate rule kvpos2|seed|fact, W0 from the cbo json). MAIN = R1 matched binding at 128, seed-paired: CLZ-ramp - C0x2, NAME-lure - NAME-nolure, CLZ-ramp - CLZ-fix, CLZ-ramp - C0+ISO-mask, NAME-lure - C0+ISO-mask, every arm - ISOx2; reference P1s (one C0 write, job 1344t0) beside C0x2. Ladder 8..128 for the three measures, R2 at 128, in-window R2 / R2m / R2a (kvwin dir, if given), diagnostics: write 1 / write 2 gradient norm, clip share, CE, update norm per interval, write-2 mask pattern (hidden share of K, place first token hidden share, name tokens hidden share, opened / closed pairs), divergence flag (nothing excluded).
  python e59_cbo_analyze.py --dir CBO_DIR [--base KVMULTI_DIR] [--win KVWIN_CBO_DIR]"""
import sys,json,glob,random,os,numpy as np
from scipy import stats
a=sys.argv;g=lambda k:a[a.index(k)+1] if k in a else None
D=g('--dir');BD=g('--base');WD=g('--win')
SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/cbo_s*.json'));n=len(SEEDS)
RC={s:json.load(open(f'{D}/cbo_s{s}.json')) for s in SEEDS};R0={s:json.load(open(f'{BD}/kvmulti_s{s}.json')) for s in SEEDS if BD and os.path.exists(f'{BD}/kvmulti_s{s}.json')}
EXPO=[8,16,32,48,64,80,96,112,128];LAST=128;ARMS=('C0x2','C0+ISO-mask','ISOx2','CLZ-ramp','CLZ-fix','NAME-lure','NAME-nolure');REF='P1s'
def src(arm,s):
    if arm==REF:return R0[s]['arms']['P1s'] if s in R0 else None
    return RC[s]['arms'].get(arm)
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
rng=np.random.default_rng(0)
def bsi(d):
    d=np.asarray(d,float);d=d[~np.isnan(d)]
    if len(d)<2:return ''
    b=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)];return f'bootstrap [{np.percentile(b,2.5):6.3f},{np.percentile(b,97.5):6.3f}]'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def per_fact(dS,W0,seed):
    """copy of e59_kvmask_analyze.per_fact: [:,0] matched binding, [:,1] familiarity (exposed wrong - never exposed), [:,2] margin change"""
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);mar=lambda S:S[:,0]-S[:,1:12].max(1);out=np.full((16,3),np.nan)
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];out[f,0]=dS[f,0]-np.mean(oth) if oth else np.nan
    out[:,1]=dS[:,1:12].mean(1)-dS[:,12:16].mean(1);out[:,2]=mar(W0+dS)-mar(W0);return out
MET=('matched binding','familiarity','margin change')
V={};V2={}
for s in SEEDS:
    for arm in ARMS+(REF,):
        rec=src(arm,s)
        if rec is None:continue
        for e in EXPO:
            d=rec['reads'].get(str(e))
            if d is None:continue
            V[(arm,e,s)]=np.nanmean(per_fact(d['dS1'],RC[s]['W0_R1'],s),0)
        d=rec['reads'].get(str(LAST),{})
        if 'dS2' in d:V2[(arm,s)]=np.nanmean(per_fact(d['dS2'],RC[s]['W0_R2'],s),0)
def col(arm,e,mi):return np.array([V[(arm,e,s)][mi] if (arm,e,s) in V else np.nan for s in SEEDS])
print(f'1357 cbo: seeds {SEEDS} (n={n}); Delta over W0; mean over seeds [95% t] same-direction seeds; P1s = one C0 write per chunk (job 1344t0)')
print('== MAIN (R1 matched binding at 128, seed-paired):')
C=[('CLZ-ramp','C0x2'),('NAME-lure','NAME-nolure'),('CLZ-ramp','CLZ-fix'),('CLZ-ramp','C0+ISO-mask'),('NAME-lure','C0+ISO-mask')]+[(x,'ISOx2') for x in ARMS if x!='ISOx2']+[('C0x2',REF)]
for x,y in C:d=col(x,128,0)-col(y,128,0);print(f'  {x+" - "+y:28s} {tci(d)} {bsi(d)}')
for mi,mn in enumerate(MET):
    print(f'== R1 ladder [{mn}]')
    for arm in ARMS+(REF,):print(f'  {arm:12s}: '+' | '.join(f'e{e} {tci(col(arm,e,mi))}' for e in EXPO if np.isfinite(col(arm,e,mi)).any()))
print('== R2 at 128 (just-written chunk 127 as context) [matched binding | familiarity | margin change]')
for arm in ARMS+(REF,):
    vs=[V2[(arm,s)] for s in SEEDS if (arm,s) in V2]
    if vs:vs=np.asarray(vs);print(f'  {arm:12s}: '+' | '.join(tci(vs[:,i]) for i in range(3)))
if WD:
    print('== in-window reads (kvwin on the saved final weights) [matched binding]')
    for s in SEEDS:
        f=f'{WD}/kvwin_cbo_s{s}.json'
        if not os.path.exists(f):continue
        r=json.load(open(f))
        for arm,vv in r['arms'].items():
            print(f'  seed {s} {arm:12s}: '+' | '.join(f'{v} {np.nanmean(per_fact(np.asarray(vv[v])-np.asarray(r["W0"][v]),r["W0"][v],s)[:,0]):7.3f}' for v in vv if v in r.get('W0',{})))
print('== diagnostics per 32-chunk interval: write 1 |g| / clip share / CE ; write 2 |g| / clip share / CE ; write-2 mask: K hidden share, place-first hidden share, name hidden share')
for arm in ARMS:
    for s in SEEDS:
        rec=src(arm,s)
        if rec is None:continue
        st=rec['steps'];line=[]
        for q in range(0,len(st),32):
            b=st[q:q+32]
            if arm=='ISOx2':ss=[x for e in b for x in e['steps']];line.append(f'[{q}:{q+32}] |g| {np.mean([x["gnorm"] for x in ss]):.3f} clip {np.mean([x["coef"]<1 for x in ss]):.2f} CE {np.mean([x["ce"] for x in ss]):.3f}');continue
            ms=[rec['mask_stat'][str(x['k'])] for x in b];K=sum(m['K'] for m in ms);pf=sum(m['pf'] for m in ms);nm=sum(m['name'] for m in ms)
            line.append(f'[{q}:{q+32}] w1 {np.mean([x["gnorm1"] for x in b]):.3f}/{np.mean([x["coef1"]<1 for x in b]):.2f}/{np.mean([x["ce1"] for x in b]):.3f} w2 {np.mean([x["gnorm2"] for x in b]):.3f}/{np.mean([x["coef2"]<1 for x in b]):.2f}/{np.mean([x["ce2"] for x in b]):.3f} K {sum(m["K_hidden"] for m in ms)/K if K else float("nan"):.2f} pf {sum(m["pf_hidden"] for m in ms)/pf if pf else float("nan"):.2f} nm {sum(m["name_hidden"] for m in ms)/nm if nm else float("nan"):.2f}')
        print(f'  {arm:12s} s{s} diverged {rec["diverged"]}: '+' ; '.join(line))
