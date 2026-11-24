"""Analysis of job 1344p (real one-step effect by position class; CPU; new file; EXPLORATORY descriptive; per seed the mean over the measured chunks, then mean [95% t] over seeds and the number of seeds in the sign direction). dmargin = out-of-window margin change over W0 (mean of the 16 read facts) of the step built from the positions of one class and applied with the clip coefficient of the FULL step. Blocks: validation; levels per class and condition with shares of the class sum and the full G / real step (additivity); SAMEVIS - C0 per class; fact level (facts 0..3): matrix M[f,g] = change of the margin of read fact g after the step built from the answer tokens (name tokens) of fact f only: own = mean of the diagonal, others = mean of the off-diagonal, per condition and SAMEVIS - C0, and the sum of the own diagonal over all 16 facts extrapolated by 4 facts vs the class 'ans' ('name'). python3 e59_kvpos_analyze.py --dir D [--seeds 231-238]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);lo,hi=(231,238)
if '--seeds' in a:lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')]
SEEDS=[s for s in range(lo,hi+1) if (D/f'kvpos_s{s}.json').exists()];n=len(SEEDS);T=stats.t.ppf(.975,max(n-1,1))
def ci(d):d=np.asarray(d,float);m=d.mean();h=T*d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan');return f'{m:8.3f} [{m-h:8.3f},{m+h:8.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
R=[json.load(open(D/f'kvpos_s{s}.json')) for s in SEEDS];CL=('ans','name','frame','extra','filler');CO=('C0','SAMEVIS')
def ps(nm):return np.array([np.mean([r['combos'][nm]['dmargin'] for r in X['rows']]) for X in R])
def fact_mat(X,c,kind):
    out=[]
    for r in X['rows']:
        out.append(np.array([r['combos'][f'G[{c},{c}]@f{f}_{kind}']['dmargin_facts'] for f in range(4)]))                # [4 step facts, 16 read facts]
    return np.mean(out,0)
print(f'1344p: real one-step effect by position class, seeds {SEEDS} (n={n}), chunks {R[0]["chunks"]}; dmargin = margin change over W0 (16 read facts)')
v=np.array([[x[k] for k in ('w1','w2','w3')] for X in R for x in X['validation']]);print(f'validation G[C0,C0] vs autodiff gradient: relative error median w1 {np.median(v[:,0]):.2e} w2 {np.median(v[:,1]):.2e} w3 {np.median(v[:,2]):.2e}, max {v.max():.2e}')
tok=np.mean([[r['posn'][c] for c in CL] for X in R for r in X['rows']],0);print('mean token counts per class and chunk: '+', '.join(f'{c} {t:.0f}' for c,t in zip(CL,tok)))
print('-- levels (class steps with the clip coefficient of the full step)')
for c in CO:
    cls={k:ps(f'G[{c},{c}]@{k}') for k in CL};sm=sum(cls.values());full=ps(f'G[{c},{c}]');act=ps(f'ACT[{c}]')
    print(f'  {c:8s} '+' | '.join(f'{k} {ci(v_)} ({v_.mean()/sm.mean()*100:4.0f}%)' for k,v_ in cls.items())+f' | sum {ci(sm)} | full G {ci(full)} | real step {ci(act)} | sum/full {sm.mean()/full.mean():.2f}')
print('-- SAMEVIS - C0 per class');[print(f'  {k:7s} {ci(ps(f"G[SAMEVIS,SAMEVIS]@{k}")-ps(f"G[C0,C0]@{k}"))}') for k in CL]
print('  sum of classes SAMEVIS - C0',ci(sum(ps(f'G[SAMEVIS,SAMEVIS]@{k}') for k in CL)-sum(ps(f'G[C0,C0]@{k}') for k in CL)),'| full G SAMEVIS - C0',ci(ps('G[SAMEVIS,SAMEVIS]')-ps('G[C0,C0]')))
print('-- fact level (facts 0..3): step from the answer / name tokens of ONE fact, margin change of the same fact (diagonal) and of the other 15 read facts (mean of the off-diagonal)')
for kind in ('ans','name'):
    res={}
    for c in CO:
        diag=[];off=[]
        for X in R:
            M=fact_mat(X,c,kind);diag.append(np.mean([M[f,f] for f in range(4)]));off.append(np.mean([np.mean([M[f,g] for g in range(16) if g!=f]) for f in range(4)]))
        res[c]=(np.array(diag),np.array(off))
        cls=ps(f'G[{c},{c}]@{kind}');print(f'  {kind:5s} {c:8s} own fact {ci(res[c][0])} | other read facts (mean per fact) {ci(res[c][1])} | own x 16 facts {res[c][0].mean()*16:.3f} + others x 240 pairs/16 {res[c][1].mean()*15:.3f} vs class step {cls.mean():.3f}')
    print(f'  {kind:5s} SAMEVIS - C0: own fact {ci(res["SAMEVIS"][0]-res["C0"][0])} | other read facts {ci(res["SAMEVIS"][1]-res["C0"][1])}')
