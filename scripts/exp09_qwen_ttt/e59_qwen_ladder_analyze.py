"""Analysis of job 1367 'qwen_ladder' (CPU; new file; implementation thread 11/28; descriptive, no threshold). Qwen U12 ladder P0s -> P1s -> P1d -> L2b -> ISO1: R1 (neutral 8192 SQuAD tail, 'out') and the 8-chunk in-window read ('in') of e59_qwen_run, arm minus W0 (cband_W0_s<seed>.json, same contexts), 16 x 16 candidates = 12 exposed + 4 never exposed (e59_kvpos2.items2 order). Matched binding / top-1 accuracy / familiarity as the E2E analyzers; MAIN = R1 matched binding at 128, neighbour differences P1s-P0s, P1d-P1s, L2b-P1d, ISO1-L2b (+ P1d-P0s), seed-paired; curves at the read points; step records (fact CE, applied norm).
  python e59_qwen_ladder_analyze.py --dir DIR"""
import sys,json,glob,random,os,numpy as np
try:
    from scipy import stats
    tq=lambda df:stats.t.ppf(.975,df)
except ImportError:          # the Qwen venv has no scipy (1367a analysis failure 11/28): two-sided 95% t quantiles
    _T=[12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131]
    tq=lambda df:_T[df-1] if df<=len(_T) else 1.96
D=sys.argv[sys.argv.index('--dir')+1];ARMS=('P0s','P1s','P1d','L2b','ISO1')
SEEDS=sorted({int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/e59q_*_s*.json')})
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=tq(len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def meas(dS,W0,seed):
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);mb=[]
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];mb.append(dS[f,0]-np.mean(oth))
    S=W0+dS;return dict(bind=float(np.mean(mb)),acc=float(np.mean(S.argmax(1)==0)),fam=float(np.mean(dS[:,1:12].mean(1)-dS[:,12:16].mean(1))))
R={(a,s):json.load(open(f'{D}/e59q_{a}_s{s}.json')) for a in ARMS for s in SEEDS if os.path.exists(f'{D}/e59q_{a}_s{s}.json')};W={s:json.load(open(f'{D}/cband_W0_s{s}.json'))['reads'] for s in SEEDS if os.path.exists(f'{D}/cband_W0_s{s}.json')}
def col(arm,e,ctx,key):
    out=[]
    for s in SEEDS:
        r=R.get((arm,s));d=(r or {}).get('reads',{}).get(str(e))
        if not d or s not in W:out.append(np.nan);continue
        W0=np.asarray(W[s][str(e)][ctx]);out.append(meas(np.asarray(d[ctx])-W0,W0,s)[key])
    return np.array(out)
EX=sorted({int(e) for r in R.values() for e in r['reads']})
print(f'1367 Qwen U12 ladder: seeds {SEEDS}; arms per seed '+' '.join(f'{a}:{sum((a,s) in R for s in SEEDS)}' for a in ARMS)+'; diverged '+str([f'{a}{s}' for (a,s),r in R.items() if r.get('diverged')]))
PAIRS=[('P1s','P0s'),('P1d','P1s'),('L2b','P1d'),('ISO1','L2b'),('P1d','P0s')]
for ctx,cn in (('out','R1 (neutral 8192 tail)'),('in','in-window (last 8 chunks)')):
    for key,nm in (('bind','matched binding'),('acc','top-1 accuracy'),('fam','familiarity')):
        print(f'== {cn} {nm}: levels')
        for e in EX:print(f'  e{e:3d}: '+' | '.join(f'{a} {tci(col(a,e,ctx,key))}' for a in ARMS))
        print(f'== {cn} {nm}: neighbour differences (seed-paired)'+(' [MAIN at e128]' if ctx=='out' and key=='bind' else ''))
        for e in EX:print(f'  e{e:3d}: '+' | '.join(f'{x}-{y} {tci(col(x,e,ctx,key)-col(y,e,ctx,key))}' for x,y in PAIRS))
print('== steps: mean fact CE first 8 / last 8 chunks, applied norm')
for a in ARMS:
    v=[(np.mean([x['fact_ce'] for x in R[(a,s)]['steps'][:8]]),np.mean([x['fact_ce'] for x in R[(a,s)]['steps'][-8:]]),np.mean([x['applied_norm'] for x in R[(a,s)]['steps']])) for s in SEEDS if (a,s) in R]
    if v:v=np.asarray(v);print(f'  {a:4s}: fact CE {v[:,0].mean():.3f} -> {v[:,1].mean():.3f}, applied norm {v[:,2].mean():.4f}')
