"""Summary of the stage-0 natural-document check (CPU; new file; EXPLORATORY descriptive): per event g = nllB - nllA (summed m2-token NLL, band-1024 minus natural 8192 window; positive = the long context helps), per-token g/ntok too; by k class (k = 0 control, k >= 1), by prev distance class, with 95% intervals clustered by document (24 books: t-interval over per-book means) and the pooled bootstrap over events. python3 e59_stage0_analyze.py [--json F]"""
import sys,json,numpy as np
from scipy import stats
F=sys.argv[sys.argv.index('--json')+1] if '--json' in sys.argv else 'experiments/exp07_write_ladder/stage0/stage0_e2e_events.json'
r=json.load(open(F));ev=r['events'];print('events',len(ev),'dropped',r['dropped'],'books',len(r['books']))
g=np.array([e['nllB']-e['nllA'] for e in ev]);nt=np.array([e['ntok'] for e in ev]);k=np.array([e['k'] for e in ev]);pv=np.array([e['prev'] if e['prev'] is not None else -1 for e in ev]);doc=np.array([e['doc'] for e in ev]);pos=np.array([e['pos'] for e in ev])
def clus(mask,name,x=g):
    m=mask&np.isfinite(x);ds=sorted(set(doc[m]));pm=np.array([x[m&(doc==d)].mean() for d in ds]);n=len(ds)
    if n>=3:h=stats.t.ppf(.975,n-1)*pm.std(ddof=1)/np.sqrt(n);ci=f'[{pm.mean()-h:6.3f},{pm.mean()+h:6.3f}] ({n} books)'
    else:ci=''
    print(f'  {name:34s} n={int(m.sum()):5d} mean g {x[m].mean():7.3f}  per-book mean {pm.mean():7.3f} {ci}  median {np.median(x[m]):6.3f}  nllA {np.mean([e["nllA"] for e,mm in zip(ev,m) if mm]):6.2f} ntok {nt[m].mean():.2f}')
print('g = nllB - nllA (summed over m2 tokens); positive = long context helps');long_=pos>=8192
clus(k==0,'k = 0 (control)');clus(k>=1,'k >= 1');clus((k==0)&long_,'k = 0, pos >= 8192');clus((k>=1)&long_,'k >= 1, pos >= 8192')
for lo,hi,nm in ((0,1024,'prev <= 1024'),(1024,4096,'1024 < prev <= 4096'),(4096,8192,'4096 < prev <= 8192'),(8192,10**9,'prev > 8192')):clus((k>=1)&(pv>lo)&(pv<=hi)&long_,nm+' (pos >= 8192)')
print('per-token g (g / ntok)');clus((k==0)&long_,'k = 0, pos >= 8192',g/nt);clus((k>=1)&long_,'k >= 1, pos >= 8192',g/nt)
for kk in (1,2,3,5):clus((k==kk)&long_,f'k = {kk}, pos >= 8192')
clus((k>=6)&long_,'k >= 6, pos >= 8192')
# k >= 1 minus k = 0 difference, book-clustered (books having both classes)
ds=[d for d in sorted(set(doc)) if ((doc==d)&(k>=1)&long_).sum()>0 and ((doc==d)&(k==0)&long_).sum()>0];dm=np.array([g[(doc==d)&(k>=1)&long_].mean()-g[(doc==d)&(k==0)&long_].mean() for d in ds])
h=stats.t.ppf(.975,len(ds)-1)*dm.std(ddof=1)/np.sqrt(len(ds));print(f'  (k>=1) - (k=0) per-book difference, pos >= 8192: {dm.mean():.3f} [{dm.mean()-h:.3f},{dm.mean()+h:.3f}] ({len(ds)} books, {int((dm>0).sum())} positive)')
