"""Analysis of job 1344p2 (e59_kvpos2.py; CPU; new file; thread-server 11/25; descriptive, seed-level t 95% intervals, n = seeds; per seed the mean over the measured chunks and the 16 read facts): the stored dS = candidate scores of the one-step state minus W0 (16 facts x 16 candidates: 0 = gold, 1..11 exposed other places, 12..15 never exposed). Per fact: BINDING = dS[gold] - mean dS[exposed decoys]; FAMILIARITY = mean dS[exposed decoys] - mean dS[unexposed]; headline margin = stored dmargin (gold - max exposed wrong). Rows: real steps ACT, full G, position classes (ans/name/frame/extra/filler: step of the class with the clip coefficient of the full step), fact level (answer / name tokens of facts 0..3: own fact vs the mean of the other 15 facts). Contrasts SAMEVIS - C0. python e59_kvpos2_analyze.py --dir D"""
import sys,json,glob,re,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];R=[json.load(open(f)) for f in sorted(glob.glob(D+'/kvpos2_s*.json'))]
def ci(d):
    d=np.asarray(d,float);m=d.mean()
    if len(d)<2:return f'{m:7.3f}'
    h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def parts(dS):
    dS=np.asarray(dS);g=dS[:,0];ex=dS[:,1:12].mean(1);un=dS[:,12:16].mean(1);return np.array([g-dS[:,1:12].max(1),g-ex,ex-un])         # [3,16]: headline(gold-max exposed), binding, familiarity
def get(r,name,f=None):
    out=[]
    for X in r['rows']:
        c=X['combos'][name];p=parts(c['dS']);out.append(p if f is None else p[:,f])
    return np.mean(out,0)
print(f'1344p2: seeds {[r["seed"] for r in R]} (n={len(R)}), chunks {R[0]["chunks"]}; columns headline(gold-max exposed) | BINDING gold-exposed decoys | FAMILIARITY exposed-unexposed (change over W0, summed log p)')
names=[k for k in R[0]['rows'][0]['combos'] if '@' not in k]
for nm in names+[k for k in R[0]['rows'][0]['combos'] if '@' in k and not re.search(r'@f\d',k)]:
    v=np.array([get(r,nm).mean(1) for r in R]);print(f'  {nm:24s} head {ci(v[:,0])} | bind {ci(v[:,1])} | fam {ci(v[:,2])}')
print('-- SAMEVIS - C0 (same naming with the other condition)')
for nm in [k for k in R[0]['rows'][0]['combos'] if k.startswith('G[C0,C0]') and not re.search(r'@f\d',k)]:
    n2=nm.replace('C0','SAMEVIS')
    if n2 in R[0]['rows'][0]['combos']:
        v=np.array([get(r,n2).mean(1)-get(r,nm).mean(1) for r in R]);print(f'  {nm:24s} head {ci(v[:,0])} | bind {ci(v[:,1])} | fam {ci(v[:,2])}')
print('-- fact level (facts 0..3): step from the answer / name tokens of fact j; OWN fact j vs mean of the OTHER 15 facts')
for tok in ('ans','name'):
    for cond in ('C0','SAMEVIS'):
        own=[];oth=[]
        for r in R:
            ow=[];ot=[]
            for X in r['rows']:
                for j in range(4):
                    p=parts(X['combos'][f'G[{cond},{cond}]@f{j}_{tok}']['dS']);ow.append(p[:,j]);ot.append(np.delete(p,j,axis=1).mean(1))
            own.append(np.mean(ow,0));oth.append(np.mean(ot,0))
        own=np.array(own);oth=np.array(oth);print(f'  {tok:4s} {cond:8s} own  head {ci(own[:,0])} | bind {ci(own[:,1])} | fam {ci(own[:,2])}');print(f'  {tok:4s} {cond:8s} other head {ci(oth[:,0])} | bind {ci(oth[:,1])} | fam {ci(oth[:,2])}')
print('NOTE on terms: the classes called ans / name in this report are the LATER / EARLIER mentioned entity of the sentence (2 tokens each; the U12 templates put the place before the person name in ~57% of the sentences), NOT always the place / the name. In name-first sentences ans = place (the answer), in place-first sentences ans = the person name.')
