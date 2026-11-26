"""Analysis of job 1358b 'xstep' (e59_xstep.py; CPU; new file; implementation thread 11/26; descriptive: values and intervals over seeds, same-direction seeds, no verdict wording, no threshold). Matched binding / familiarity / margin as e59_kvmask_analyze.py (candidate rule kvpos2|seed|fact). Per step: change over W_t of the per-fact values of the facts with a sentence in the chunk (in U12: all 16 read facts; the 'not appearing' group is empty and printed as such).
MAIN = per state (C0 stream, SAMEVIS stream): one-step change of R1 matched binding, C0 natural - SAMEVIS natural, mean over t in {32, 64, 96} x 8 chunks inside a seed, then seed-paired over seeds (t interval, bootstrap, same-direction seeds). Same table: C0 natural - SAMEVIS size-matched, interval runs (8 consecutive steps) C0 - SAMEVIS, W0 (t = 0) separately; per t; familiarity and margin; update norms, gradient norms, clip shares, cosine C0 vs SAMEVIS update; copy side: R2 / R2m change and place-token NLL change under both keeps (C0 natural, SAMEVIS natural).
  python e59_xstep_analyze.py --dir XSTEP_DIR"""
import sys,json,glob,random,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1]
SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/xstep_s*.json'));RS={s:json.load(open(f'{D}/xstep_s{s}.json')) for s in SEEDS};n=len(SEEDS)
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):8.4f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:8.4f} [{m-h:7.4f},{m+h:7.4f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
rng=np.random.default_rng(0)
def bsi(d):
    d=np.asarray(d,float);d=d[~np.isnan(d)]
    if len(d)<2:return ''
    b=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)];return f'bootstrap [{np.percentile(b,2.5):7.4f},{np.percentile(b,97.5):7.4f}]'
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
MET=('matched binding','familiarity','margin')
def step_change(s,rec,ent,kind,mi):
    """per-fact change over W_t of measure mi, mean over the facts of the chunk"""
    W0=RS[s]['W0_R1'];b=per_fact(rec['R1_Wt'],W0,s)[:,mi];x=per_fact(ent['steps'][kind]['R1'],W0,s)[:,mi];f=ent['facts'];return float(np.nanmean((x-b)[f])) if f else np.nan
def seedvals(state,ts,fn):
    out=[]
    for s in SEEDS:
        v=[]
        for t in ts:
            rec=RS[s]['states'].get(state,{}).get(str(t))
            if rec is None:continue
            v+=[fn(s,rec,e) for e in rec['chunks']] if fn.__code__.co_argcount==3 else [fn(s,rec)]
        out.append(np.nanmean(v) if v else np.nan)
    return np.asarray(out)
print(f'1358b xstep: seeds {SEEDS} (n={n}); stage-1 identical to 1348: {[RS[s]["stage1_same_as_1348"] for s in SEEDS]}; checks one step = stream: {[all(c["hash_equal"] for c in RS[s]["checks"].values()) for s in SEEDS]}')
print('facts per chunk: all 16 read facts appear in every chunk of the U12 world (the "not appearing" group is empty)')
for mi,mn in enumerate(MET):
    print(f'== one-step change of R1 {mn} (appearing facts), seed-paired; mean over t x 8 chunks inside a seed')
    for state,ts in (('C0',(32,64,96)),('SAMEVIS',(32,64,96)),('W0',(0,))):
        for nm,A,B in (('C0 - SAMEVIS natural','C0','SAMEVIS'),('C0 - SAMEVIS size-matched','C0','SAMEVIS_sized')):
            d=seedvals(state,ts,lambda s,rec,e:step_change(s,rec,e,A,mi)-step_change(s,rec,e,B,mi));print(f'  state {state:8s} {nm:28s} {tci(d)} {bsi(d)}')
        for kind in ('C0','SAMEVIS','SAMEVIS_sized'):print(f'  state {state:8s} {kind:28s} {tci(seedvals(state,ts,lambda s,rec,e:step_change(s,rec,e,kind,mi)))}')
        di=seedvals(state,ts,lambda s,rec:float(np.nanmean(per_fact(rec['interval']['C0']['R1'],RS[s]['W0_R1'],s)[:,mi]-per_fact(rec['interval']['SAMEVIS']['R1'],RS[s]['W0_R1'],s)[:,mi])));print(f'  state {state:8s} {"interval 8 steps C0 - SAMEVIS":28s} {tci(di)}')
        if state!='W0':
            for t in ts:print(f'    t {t:3d}: C0 - SAMEVIS natural {tci(seedvals(state,(t,),lambda s,rec,e:step_change(s,rec,e,"C0",mi)-step_change(s,rec,e,"SAMEVIS",mi)))}')
print('== step diagnostics (mean over t x chunks inside a seed): gradient norm, clip share, update norm; cosine of the C0 and SAMEVIS updates (total)')
for state,ts in (('C0',(32,64,96)),('SAMEVIS',(32,64,96)),('W0',(0,))):
    for kind in ('C0','SAMEVIS'):
        print(f'  state {state:8s} step {kind:8s} |g| {tci(seedvals(state,ts,lambda s,rec,e:e["steps"][kind]["gnorm"]))} clip {tci(seedvals(state,ts,lambda s,rec,e:float(e["steps"][kind]["coef"]<1)))} |dW| {tci(seedvals(state,ts,lambda s,rec,e:e["steps"][kind]["update_norm"]))}')
    print(f'  state {state:8s} cosine total {tci(seedvals(state,ts,lambda s,rec,e:e["cos_C0_SAMEVIS"]["total"]))}')
print('== copy side (appearing facts): change over W_t of R2 / R2m gold-candidate score, and place-token NLL (mean over place tokens) under the C0 keep / SAMEVIS keep')
def r2c(s,rec,e,kind,v):
    if 'R2_Wt' not in e:return np.nan
    W=np.asarray(e['R2_Wt'][v]);x=np.asarray(e['steps'][kind]['R2'][v]);return float(np.nanmean(per_fact(x-W,W,s)[:,0]))
def nllc(s,rec,e,kind,mk):
    b=e['place_nll_Wt'][mk];x=e['steps'][kind]['place_nll'][mk];return float(np.mean([np.mean(x[str(f)])-np.mean(b[str(f)]) for f in e['facts']]))
for state,ts in (('C0',(32,64,96)),('SAMEVIS',(32,64,96)),('W0',(0,))):
    for kind in ('C0','SAMEVIS'):
        print(f'  state {state:8s} step {kind:8s} R2 matched {tci(seedvals(state,ts,lambda s,rec,e:r2c(s,rec,e,kind,"R2")))} R2m matched {tci(seedvals(state,ts,lambda s,rec,e:r2c(s,rec,e,kind,"R2m")))} | place NLL C0 keep {tci(seedvals(state,ts,lambda s,rec,e:nllc(s,rec,e,kind,"C0")))} SAMEVIS keep {tci(seedvals(state,ts,lambda s,rec,e:nllc(s,rec,e,kind,"SAMEVIS")))}')
