"""Analysis of job 1344m (one-step key / error swap and matched reads; CPU; new file; EXPLORATORY descriptive; per seed the mean over the measured chunks, then mean [95% t] over seeds and the number of seeds in the sign direction). Quantity: change of the out-of-window margin (gold - best wrong, summed log p, neutral tail, mean over the 16 read facts) of the W0 weights after ONE step built from the combination, minus the W0 margin ('dmargin'; positive = better recall). Blocks: validation (G[C0,C0] vs the autodiff gradient); real steps ACT[c]; swap table G[A,B] and the exact decompositions of SAMEVIS - C0 (key effect, error effect, interaction), also with the error SIZE restored to C0 and with the COMMON key component (along the mean read key) restored to C0 (fact-specific address vs shared component); position split (own answer positions / other facts' answer positions / rest, linear response with the clip coefficient of the full step; sum of the three vs the full step); H: 4 weights x 3 read contexts (neutral / same-fact prefix with hidden answer / other-fact prefix), per fact margin change over W0 of the same context. python3 e59_kvstep_analyze.py --dir D [--seeds 231-238]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);lo,hi=(231,238)
if '--seeds' in a:lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')]
SEEDS=[s for s in range(lo,hi+1) if (D/f'kvstep_s{s}.json').exists()];n=len(SEEDS);T=stats.t.ppf(.975,max(n-1,1))
def ci(d):d=np.asarray(d,float);m=d.mean();h=T*d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan');return f'{m:8.3f} [{m-h:8.3f},{m+h:8.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
R=[json.load(open(D/f'kvstep_s{s}.json')) for s in SEEDS]
def per_seed(name):return np.array([np.mean([r['combos'][name]['dmargin'] for r in X['rows']]) for X in R])
names=list(R[0]['rows'][0]['combos'])
print(f'1344m: one-step key / error swap, seeds {SEEDS} (n={n}); dmargin = out-of-window margin change over W0 (mean of the 16 facts, chunk mean per seed)')
v=np.array([[x[nm] for nm in ('w1','w2','w3')] for X in R for x in X['validation']]);print(f'validation G[C0,C0] vs autodiff gradient, relative error: median w1 {np.median(v[:,0]):.2e} w2 {np.median(v[:,1]):.2e} w3 {np.median(v[:,2]):.2e}, max {v.max():.2e}')
print('-- combinations');[print(f'   {nm:44s} {ci(per_seed(nm))}') for nm in names]
def d(a_,b_):return per_seed(a_)-per_seed(b_)
print('-- real steps: SAMEVIS - C0',ci(d('ACT[SAMEVIS]','ACT[C0]')),'| NATVIS - C0',ci(d('ACT[NATVIS]','ACT[C0]')),'| reproduction of ACT by G[A,A]: C0',ci(d('G[C0,C0]','ACT[C0]')),'SAMEVIS',ci(d('G[SAMEVIS,SAMEVIS]','ACT[SAMEVIS]')),'NATVIS',ci(d('G[NATVIS,NATVIS]','ACT[NATVIS]')))
print('-- exact decomposition of G[SAMEVIS,SAMEVIS] - G[C0,C0]')
tot=d('G[SAMEVIS,SAMEVIS]','G[C0,C0]');print('   total',ci(tot));print('   key effect with the C0 error  (G[S,C0]-G[C0,C0])',ci(d('G[SAMEVIS,C0]','G[C0,C0]')),'| control cell: C0 error + SAMEVIS key');print('   error effect with the C0 key   (G[C0,S]-G[C0,C0])',ci(d('G[C0,SAMEVIS]','G[C0,C0]')))
print('   error effect with the SAMEVIS key (G[S,S]-G[S,C0])',ci(d('G[SAMEVIS,SAMEVIS]','G[SAMEVIS,C0]')),'| key effect with the SAMEVIS error (G[S,S]-G[C0,S])',ci(d('G[SAMEVIS,SAMEVIS]','G[C0,SAMEVIS]')),'| interaction',ci(d('G[SAMEVIS,SAMEVIS]','G[SAMEVIS,C0]')-d('G[C0,SAMEVIS]','G[C0,C0]')))
print('-- error SIZE restored to C0 (direction from B, per-position per-layer size of C0)')
for nm in ('G[C0,SAMEVIS|size=C0]','G[SAMEVIS,SAMEVIS|size=C0]','G[C0,NATVIS|size=C0]','G[NATVIS,NATVIS|size=C0]'):print(f'   {nm:36s} - G[C0,C0] {ci(d(nm,"G[C0,C0]"))}')
print('   size effect with the SAMEVIS key (G[S,S] - G[S,S|size=C0])',ci(d('G[SAMEVIS,SAMEVIS]','G[SAMEVIS,SAMEVIS|size=C0]')))
print('-- COMMON key component restored (the part of x2 along the mean read key replaced by the C0 value)')
for nm,ref in (('G[SAMEVIS,SAMEVIS|common=C0]','G[SAMEVIS,SAMEVIS]'),('G[SAMEVIS,C0|common=C0]','G[SAMEVIS,C0]'),('G[SAMEVIS,SAMEVIS|size=C0,common=C0]','G[SAMEVIS,SAMEVIS|size=C0]'),('G[C0,C0|common=SAMEVIS]','G[C0,C0]'),('G[NATVIS,NATVIS|common=C0]','G[NATVIS,NATVIS]')):print(f'   {nm:40s} - {ref:34s} {ci(d(nm,ref))}')
print('-- position split of G[c,c] (linear response, clip of the full step): own answer positions | other facts answer positions | rest | sum of the three | full step')
for c in ('C0','SAMEVIS','NATVIS'):
    o,t,r=(per_seed(f'G[{c},{c}]@{p}') for p in ('own','other','rest'));f=per_seed(f'G[{c},{c}]');print(f'   {c:8s} own {ci(o)} | other {ci(t)} | rest {ci(r)} | sum {ci(o+t+r)} | full {ci(f)}')
for p in ('own','other','rest'):print(f'   SAMEVIS - C0, {p:5s} positions',ci(per_seed(f'G[SAMEVIS,SAMEVIS]@{p}')-per_seed(f'G[C0,C0]@{p}')))
print('-- H: per fact margin change over the W0 margin of the SAME context; weights W_C0 / W_SAMEVIS / W_NATVIS (one step from every measured chunk), mean over chunks and facts per seed')
for ctx in ('neutral','same','other'):
    ps={}
    for wn in ('W_C0','W_SAMEVIS','W_NATVIS'):
        v=[];
        for X in R:
            v.append(np.mean([np.mean(np.asarray(r['H'][ctx][wn])-np.asarray(r['H'][ctx]['W0'])) for r in X['rows']]))
        ps[wn]=np.array(v);print(f'   context {ctx:8s} {wn:10s} {ci(ps[wn])}')
    print(f'   context {ctx:8s} SAMEVIS - C0 {ci(ps["W_SAMEVIS"]-ps["W_C0"])} | NATVIS - C0 {ci(ps["W_NATVIS"]-ps["W_C0"])}')
print('   W0 margin level per context (mean over chunks, facts, seeds): '+', '.join(f'{ctx} {np.mean([np.mean(r["H"][ctx]["W0"]) for X in R for r in X["rows"]]):.2f}' for ctx in ('neutral','same','other')))
