"""Analysis of the natural-document write experiment (CPU; new file; EXPLORATORY descriptive): per event the summed natural-window NLL of the m2 tokens under the weights written by each arm (prequential, before the write of the event chunk). Event classes (derived from the stored events): FAR = k >= 1 events with prev > 8192 (the earlier mention is outside the natural window). For HIDE_PAIR the write of the previous occurrence was hidden only when that occurrence itself had an earlier occurrence inside the window, so FAR events split into TARGETED (k >= 2 and the previous occurrence had prev <= 8192) and PLACEBO-LIKE (k = 1: the previous occurrence is the first mention, written unmasked in every arm); for HIDE_EV the previous occurrence is masked whenever k >= 2 (TARGETED = k >= 2, PLACEBO = k = 1); MASK1024 masks every row (all FAR events are targeted). Control = k = 0 events (first mentions). Reported per arm: mean (arm - NAT) per class, book-clustered mean [95% t-CI over books with events in the class], and the difference to the k = 0 control (per book where both exist). python3 e59_nat_write_analyze.py --json F"""
import sys,json,collections,numpy as np
from scipy import stats
F=sys.argv[sys.argv.index('--json')+1] if '--json' in sys.argv else 'experiments/exp07_write_ladder/nat_write/nat_write.json'
r=json.load(open(F));ev=r['events'];arms=[a for a in next(iter(ev))['nll'] if a!='NAT'];print('events',len(ev),'books',len(r['books']),'arms',arms)
by={}
for e in ev:by.setdefault((e['doc'],tuple(e['pair'])),[]).append(e)
for L in by.values():
    L.sort(key=lambda x:x['k'])
    for j,e in enumerate(L):e['prevprev']=L[j-1]['prev'] if j>=1 and L[j-1]['k']>=1 and L[j-1]['k']==e['k']-1 else None
def cls(e,arm):
    if e['k']==0:return 'k0'
    if e['prev'] is None or e['prev']<=8192:return 'near'
    if e['k']==1:return 'far_placebo'
    if arm=='HIDE_PAIR':return 'far_targeted' if (e['prevprev'] is not None and e['prevprev']<=8192) else 'far_untargeted'
    return 'far_targeted'
def clus(vals_by_book):
    pm=np.array([np.mean(v) for v in vals_by_book.values()]);n=len(pm)
    if n<3:return f'{pm.mean():7.3f} (books {n})'
    h=stats.t.ppf(.975,n-1)*pm.std(ddof=1)/np.sqrt(n);return f'{pm.mean():7.3f} [{pm.mean()-h:7.3f},{pm.mean()+h:7.3f}] ({n} books, {int((pm>0).sum())} positive)'
for arm in arms:
    print(f'== {arm} - NAT (summed m2-token NLL, natural window, nat; negative = the arm remembers the pair better)')
    D={};
    for e in ev:
        c=cls(e,arm);D.setdefault(c,collections.defaultdict(list))[e['doc']].append(e['nll'][arm]-e['nll']['NAT'])
    for c in ('far_targeted','far_untargeted','far_placebo','near','k0'):
        if c in D:print(f'  {c:15s} n={sum(len(v) for v in D[c].values()):5d}  {clus(D[c])}')
    if 'k0' in D:
        for c in ('far_targeted','far_placebo'):
            if c in D:
                bk=[d for d in D[c] if d in D['k0']]
                if len(bk)>=3:
                    dd=np.array([np.mean(D[c][d])-np.mean(D['k0'][d]) for d in bk]);h=stats.t.ppf(.975,len(bk)-1)*dd.std(ddof=1)/np.sqrt(len(bk));print(f'  {c} minus k0 control (per book): {dd.mean():7.3f} [{dd.mean()-h:7.3f},{dd.mean()+h:7.3f}] ({len(bk)} books, {int((dd<0).sum())} negative)')
    nb=[(b['doc'],b['arms'][arm]['mean_nat_ce']-b['arms']['NAT']['mean_nat_ce']) for b in r['books']];x=np.array([v for _,v in nb]);h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));print(f'  all-token mean natural-window NLL of the whole book, arm - NAT: {x.mean():.4f} [{x.mean()-h:.4f},{x.mean()+h:.4f}] ({len(x)} books)')
