"""Analysis of job 1344w2 (e59_kvwin.py; CPU; new file; thread-server 11/25; descriptive: values and intervals, no verdict wording, no threshold). Per seed, arm (P0s, P1s) and read (R2 no mask, R2m own sentence hidden, R2a all 16 read-fact sentences hidden): Delta of the 16x16 candidate scores over the W0 scores of the SAME read; per fact matched binding (gold of fact f minus the same word as exposed decoy in the other questions), familiarity (exposed wrong minus never-exposed), margin change (gold minus best exposed wrong, Delta) and the W0 margin level; mean over the 16 facts, then mean [95% t] over seeds with same-direction seeds; P1s - P0s seed-paired (t and seed bootstrap); by the sentence order of the context chunk (127) of each fact (orders of 1348).
  python e59_kvwin_analyze.py --dir D --kvdir KVMULTI_DIR"""
import sys,json,glob,random,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];KD=a[a.index('--kvdir')+1];R=[json.load(open(f)) for f in sorted(glob.glob(D+'/kvwin_s*.json'))];n=len(R);SEEDS=[r['seed'] for r in R];VAR=R[0]['variants'];ARMS=('P0s','P1s')
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[g for g in range(16) if g!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def per_fact(S,W0,seed):
    dS=np.asarray(S,float)-np.asarray(W0,float);W0=np.asarray(W0,float);cf=_cands(seed);mar=lambda X:X[:,0]-X[:,1:12].max(1);out=np.zeros((16,4))
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];out[f,0]=dS[f,0]-np.mean(oth) if oth else np.nan
    out[:,1]=dS[:,1:12].mean(1)-dS[:,12:16].mean(1);out[:,2]=mar(W0+dS)-mar(W0);out[:,3]=mar(W0);return out
MET=('matched binding','familiarity','margin change','W0 margin level')
V={};VO={}
for r in R:
    kv=json.load(open(f"{KD}/kvmulti_s{r['seed']}.json"))
    for arm in ARMS:
        od=kv['arms'][arm]['orders']['128']
        for v in VAR:
            pf=per_fact(r['arms'][arm][v],r['W0'][v],r['seed']);V[(arm,v,r['seed'])]=np.nanmean(pf,0)
            for o in (0,1):
                mk=np.array([x==o for x in od])
                if mk.any():VO[(arm,v,r['seed'],o)]=np.nanmean(pf[mk],0)
col=lambda arm,v,mi,o=None:np.array([(V[(arm,v,s)] if o is None else VO.get((arm,v,s,o),np.full(4,np.nan)))[mi] for s in SEEDS])
print(f'1344w2: seeds {SEEDS} (n={n}); Delta over W0 of the same read; mean over seeds [95% t] same-direction seeds')
print('== checks (max over seeds): '+', '.join(f"{k} {max(r['checks'][k] for r in R):.4f}" for k in R[0]['checks'])+'; R2 recomputed vs saved 1348 dS2: '+', '.join(f"{arm} {max(r['arms'][arm]['R2_vs_saved_1348'] for r in R):.4f}" for arm in ARMS))
for mi,mn in enumerate(MET):
    print(f'== [{mn}]')
    for v in VAR:print(f'  {v}: '+' | '.join(f'{arm} {tci(col(arm,v,mi))}' for arm in ARMS)+(' | P1s-P0s '+tci(col('P1s',v,mi)-col('P0s',v,mi)) if mi<3 else ''))
print('== MAIN: matched binding P1s - P0s per read, seed-paired, seed bootstrap')
rng=np.random.default_rng(0)
for v in VAR:
    d=col('P1s',v,0)-col('P0s',v,0);bs=[d[rng.integers(0,n,n)].mean() for _ in range(5000)];print(f'  {v}: P1s-P0s {tci(d)} | seed bootstrap [{np.percentile(bs,2.5):6.3f},{np.percentile(bs,97.5):6.3f}]')
print('== matched binding: change relative to R2 (read minus R2), per arm and P1s-P0s of it')
for v in VAR[1:]:
    print(f'  {v} - R2: '+' | '.join(f'{arm} {tci(col(arm,v,0)-col(arm,"R2",0))}' for arm in ARMS)+' | (P1s-P0s) '+tci((col('P1s',v,0)-col('P1s','R2',0))-(col('P0s',v,0)-col('P0s','R2',0))))
for o,on in ((0,'place first'),(1,'name first')):
    print(f'== sentence order of the context chunk: {on}')
    for mi,mn in enumerate(MET[:3]):
        for v in VAR:print(f'  [{mn}] {v}: '+' | '.join(f'{arm} {tci(col(arm,v,mi,o))}' for arm in ARMS))
print('== W0 reads (absolute score level per read, mean over the 16 questions): gold, best exposed wrong, never exposed')
for v in VAR:
    g=[];e=[];u=[]
    for r in R:
        S=np.asarray(r['W0'][v]);g.append(S[:,0].mean());e.append(S[:,1:12].max(1).mean());u.append(S[:,12:16].mean())
    print(f'  {v}: gold {tci(g)} | best exposed wrong {tci(e)} | never exposed {tci(u)}')
