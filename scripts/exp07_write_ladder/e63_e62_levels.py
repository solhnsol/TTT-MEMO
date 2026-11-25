"""E62 main: ABSOLUTE levels (change over W0, last mention) of BINDING (gold - exposed same-book same-type decoys) and FAMILIARITY (exposed - unexposed words) per arm (NAT / SAME), per cell (n, gap kind) and pooled (CPU; new file; thread-server 11/25). Same parsing as e62m_analyze.py. Intervals: t interval over facts (unit of analysis = fact) and, in the 'by book' line, over the 12 per-book means. Usage: python e63_e62_levels.py --json e62m.json"""
import sys,json,numpy as np,collections
from scipy import stats
a=sys.argv;R=json.load(open(a[a.index('--json')+1]))
def parts(S,f):
    S=np.asarray(S)[:,0,:];out=[]
    for i in range(S.shape[0]):
        mm=len(f['facts'][i]['exposed']);g=S[i,0];ex=S[i,1:1+mm];un=S[i,1+mm:]
        out.append((g-S[i,1:].max(),g-ex.mean(),ex.mean()-un.mean()))
    return np.array(out)
def ci(d):
    d=np.asarray(d,float);m=d.mean()
    if len(d)<2:return f'{m:7.2f}'
    h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:7.2f} [{m-h:6.2f},{m+h:6.2f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
rows=[]
for ui,B in enumerate(R['books']):
    W0=parts(B['W0'],B);man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    for ci_,f in enumerate(B['facts']):
        last=max(man[ci_],key=lambda r:r['mention'])['chunk']
        for ar,A in B['arms'].items():
            if str(last) not in A['reads']:continue
            v=parts(A['reads'][str(last)],B)[ci_]-W0[ci_];rows.append(dict(unit=ui,n=f['n'],kind=f['kind'],arm=ar,v=v))
MN=('headline','BINDING','FAMILIARITY')
print('E62 MAIN absolute levels at the last mention (change over W0, summed log p). columns: BINDING gold-exposed decoy | FAMILIARITY exposed-unexposed | headline gold-max; "k/n" = facts with the majority sign')
for arm in ('NAT','SAME'):
    print(f'\n== arm {arm}')
    cells=sorted({(r['n'],r['kind']) for r in rows})
    for n,k in cells:
        x=np.array([r['v'] for r in rows if r['arm']==arm and r['n']==n and r['kind']==k])
        if len(x):print(f'   n={n:2d} {k:6s} bind {ci(x[:,1])} | fam {ci(x[:,2])} | head {ci(x[:,0])}')
    for lo,hi,nm in ((2,8,'n=2..8'),(16,32,'n=16..32'),(2,32,'all n')):
        sel=[r for r in rows if r['arm']==arm and lo<=r['n']<=hi];x=np.array([r['v'] for r in sel])
        bu=collections.defaultdict(list)
        for r in sel:bu[r['unit']].append(r['v'])
        bm=np.array([np.mean(v,0) for v in bu.values()])
        print(f'   pooled {nm:9s} bind {ci(x[:,1])} | fam {ci(x[:,2])} | head {ci(x[:,0])}  (facts {len(x)})')
        print(f'   pooled {nm:9s} by book (12 means) bind {ci(bm[:,1])} | fam {ci(bm[:,2])} | head {ci(bm[:,0])}')
    for kind in ('close','edge','far','mixed'):
        sel=[r for r in rows if r['arm']==arm and r['kind']==kind];x=np.array([r['v'] for r in sel])
        if len(x):print(f'   pooled kind={kind:6s} bind {ci(x[:,1])} | fam {ci(x[:,2])} | head {ci(x[:,0])}  (facts {len(x)})')
