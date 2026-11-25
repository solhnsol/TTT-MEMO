"""E62 main: PG-19 TEST books (7) vs PG-19 TRAIN books (5) (CPU; new file; thread-server 11/25; Hans 'CPU 비교부터'): per group and per book, from the stored e62m.json. Per-book value = mean over the facts of the cell group of that book (the unit of analysis is the book); group value = mean over the books of the group with a 95% bootstrap interval (books resampled with replacement within the group, 5000 resamples); difference TRAIN - TEST with a bootstrap interval (both groups resampled) and a Welch t interval. Metrics are changes over W0 at the last mention (summed log p): BINDING = gold - exposed same-type decoys, FAMILIARITY = exposed - unexposed. Background / update diagnostics from the stored steps: mean non-planted chunk CE of the NAT arm (planted sentences excluded), mean pre-clip chunk gradient norm, mean applied update norm (coef x norm). No thresholds. Usage: python e63_e62_split.py --json e62m.json"""
import sys,json,collections,numpy as np
from scipy import stats
a=sys.argv;R=json.load(open(a[a.index('--json')+1]));BK=R['books'];rng=np.random.default_rng(0)
def parts(S,B):
    S=np.asarray(S)[:,0,:];out=[]
    for i in range(S.shape[0]):
        mm=len(B['facts'][i]['exposed']);g=S[i,0];ex=S[i,1:1+mm];un=S[i,1+mm:];out.append((g-S[i,1:].max(),g-ex.mean(),ex.mean()-un.mean()))
    return np.array(out)
grp=lambda B:'TEST' if B['book'].startswith('pg19_') else 'TRAIN'
F=[]                                           # per fact rows
for ui,B in enumerate(BK):
    W0=parts(B['W0'],B);man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    for ci,f in enumerate(B['facts']):
        last=max(man[ci],key=lambda r:r['mention'])['chunk'];row=dict(book=ui,grp=grp(B),n=f['n'],kind=f['kind'])
        for ar in ('NAT','SAME'):row[ar]=parts(B['arms'][ar]['reads'][str(last)],B)[ci]-W0[ci]
        F.append(row)
def bookvals(fn):
    """fn(row)->value or None; -> dict book -> (mean, count, positive count)"""
    d=collections.defaultdict(list)
    for r in F:
        v=fn(r)
        if v is not None:d[r['book']].append(v)
    return {b:(float(np.mean(v)),len(v),int(np.sum(np.asarray(v)>0))) for b,v in d.items()}
def gboot(vals,books,B=5000):
    x=np.array([vals[b][0] for b in books if b in vals]);idx=rng.integers(0,len(x),size=(B,len(x)));bs=x[idx].mean(1);return x.mean(),np.percentile(bs,[2.5,97.5]),x,bs
def report(name,vals):
    """vals: book -> (mean,count,pos); prints group means with bootstrap intervals, the difference TRAIN - TEST and the per-book table"""
    te=[i for i,B in enumerate(BK) if grp(B)=='TEST'];tr=[i for i,B in enumerate(BK) if grp(B)=='TRAIN']
    mt,lt,xt,bt=gboot(vals,te);mr,lr,xr,br=gboot(vals,tr);d=mr-mt;dd=br-bt[:len(br)] if False else np.array([rng.choice(xr,len(xr)).mean()-rng.choice(xt,len(xt)).mean() for _ in range(5000)])
    nf=lambda bs:sum(vals[b][1] for b in bs if b in vals);npos=lambda bs:sum(vals[b][2] for b in bs if b in vals)
    tw=stats.ttest_ind(xr,xt,equal_var=False);se=np.sqrt(xr.var(ddof=1)/len(xr)+xt.var(ddof=1)/len(xt)) if len(xr)>1 and len(xt)>1 else np.nan
    dfw=(xr.var(ddof=1)/len(xr)+xt.var(ddof=1)/len(xt))**2/((xr.var(ddof=1)/len(xr))**2/(len(xr)-1)+(xt.var(ddof=1)/len(xt))**2/(len(xt)-1)) if len(xr)>1 and len(xt)>1 else np.nan
    h=stats.t.ppf(.975,dfw)*se
    print(f'  {name:44s} TEST {mt:7.2f} [{lt[0]:6.2f},{lt[1]:6.2f}] facts {nf(te):3d} pos {npos(te):3d} | TRAIN {mr:7.2f} [{lr[0]:6.2f},{lr[1]:6.2f}] facts {nf(tr):3d} pos {npos(tr):3d} | TRAIN-TEST {d:7.2f} boot [{np.percentile(dd,2.5):6.2f},{np.percentile(dd,97.5):6.2f}] Welch [{d-h:6.2f},{d+h:6.2f}]')
    return vals
TAB={}                                         # name -> vals (for the per-book table)
def add(name,vals):TAB[name]=report(name,vals)
print('E62 MAIN, PG-19 TEST (7 books) vs PG-19 TRAIN (5 books); change over W0 at the last mention; per-book means, group mean [95% bootstrap over books], facts, positive facts')
print('== 1. NAT BINDING level by gap kind')
for k in ('close','edge','far','mixed'):add(f'NAT bind kind={k}',bookvals(lambda r,k=k:r['NAT'][1] if r['kind']==k else None))
add('NAT bind all kinds',bookvals(lambda r:r['NAT'][1]))
print('== 2. SAME binding close, SAME-NAT binding / familiarity')
add('SAME bind kind=close',bookvals(lambda r:r['SAME'][1] if r['kind']=='close' else None))
for nm,(lo,hi) in (('n=2..8',(2,8)),('n=16..32',(16,32))):
    add(f'SAME-NAT bind {nm}',bookvals(lambda r,lo=lo,hi=hi:(r['SAME'][1]-r['NAT'][1]) if lo<=r['n']<=hi else None))
    add(f'SAME-NAT fam {nm}',bookvals(lambda r,lo=lo,hi=hi:(r['SAME'][2]-r['NAT'][2]) if lo<=r['n']<=hi else None))
print('== 3. NAT FAMILIARITY level')
for nm,(lo,hi) in (('n=2..8',(2,8)),('n=16..32',(16,32))):add(f'NAT fam {nm}',bookvals(lambda r,lo=lo,hi=hi:r['NAT'][2] if lo<=r['n']<=hi else None))
print('== 4. background / update diagnostics per book (stored steps)')
def stepvals(fn):return {i:(float(np.mean([fn(s) for s in B['arms']['NAT']['steps']])),len(B['arms']['NAT']['steps']),0) for i,B in enumerate(BK)}
add('NAT non-planted chunk CE (mean over chunks)',stepvals(lambda s:s['nonplanted_ce_nat']))
add('NAT pre-clip gradient norm (mean)',stepvals(lambda s:s['gnorm']))
add('NAT applied update norm coef x norm (mean)',stepvals(lambda s:s['coef']*s['gnorm']))
add('NAT clip coefficient (mean)',stepvals(lambda s:s['coef']))
print('== 5. per-book table (the 12 books): value per book')
names=list(TAB);print('  '+f'{"book":34s} '+' '.join(f'{i:>5d}' for i in range(len(names))))
for i,B in enumerate(BK):print(f'  {grp(B):5s} {B["book"][:28]:28s} '+' '.join(f'{TAB[n][i][0]:5.2f}' if i in TAB[n] else '  n/a' for n in names))
print('  columns: '+'; '.join(f'{i}={n}' for i,n in enumerate(names)))
