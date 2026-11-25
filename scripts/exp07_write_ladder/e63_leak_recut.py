"""Leak probe (job 1335b; the '1340 answer NLL: NAT 3.28, SHOWSAME 3.59, SHOWOTHER 16.45' = the SECOND = LATER mentioned entity NLL of e59_leak_analyze.py) re-cut by ENTITY ROLE x SENTENCE ORDER (CPU; new file; thread-server 11/25): from leak_probe_s<seed>.json rows (per read-fact sentence of the chunks >= 16: order, place_<cond>, name_<cond>, second_<cond>, first_<cond> = summed NLL of the entity tokens). Cells: place tokens / name tokens x place-first / name-first sentences. Per seed the mean over the sentences of the cell; mean over seeds [t] same-direction seeds. Usage: python e63_leak_recut.py --dir leak_probe [--first 16]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);first=int(a[a.index('--first')+1]) if '--first' in a else 16;CONDS=('NAT','ALLHID','SHOWSAME','SHOWOTHER')
S=[json.load(open(D/f'leak_probe_s{s}.json')) for s in range(231,239)];n=len(S);t=stats.t.ppf(.975,n-1)
def ci(d):d=np.asarray(d,float);m=np.nanmean(d);h=t*np.nanstd(d,ddof=1)/np.sqrt(n);return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{n}'
def per(r,nm,cond,order):
    x=[q[f'{nm}_{cond}'] for q in r['rows'] if q['k']>=first and (order is None or q['order']==order)];return float(np.mean(x)) if x else float('nan')
cnt={o:sum(1 for r in S for q in r['rows'] if q['k']>=first and q['order']==o) for o in ('place_first','name_first')}
print(f'leak probe re-cut: seeds 231-238, chunks >= {first}; sentences (all seeds): place first {cnt["place_first"]}, name first {cnt["name_first"]}; NLL summed over the 2 tokens of the entity; conditions: NAT natural window, ALLHID all earlier sentences hidden, SHOWSAME only the same fact visible, SHOWOTHER only other facts visible')
for nm,nmn in (('place','PLACE tokens (the answer of the question name -> place)'),('name','NAME tokens'),('second','SECOND = later mentioned entity (what e59_leak_analyze.py calls the answer)'),('first','FIRST = earlier mentioned entity')):
    for order in ('place_first','name_first',None):
        V={c:np.array([per(r,nm,c,order) for r in S]) for c in CONDS};on={'place_first':'place-first sentences','name_first':'name-first sentences',None:'all sentences'}[order]
        print(f'[{nmn} | {on}] means '+str({c:round(float(np.nanmean(V[c])),3) for c in CONDS}))
        for x,y in (('SHOWSAME','ALLHID'),('SHOWOTHER','ALLHID'),('NAT','ALLHID'),('SHOWOTHER','SHOWSAME')):print(f'    {x:9s} - {y:9s} {ci(V[x]-V[y])}')
    d1=np.array([per(r,nm,'SHOWSAME',"name_first")-per(r,nm,'ALLHID',"name_first")-(per(r,nm,'SHOWSAME',"place_first")-per(r,nm,'ALLHID',"place_first")) for r in S]);print(f'    (SHOWSAME-ALLHID)[name first] - (SHOWSAME-ALLHID)[place first], seed-paired: {ci(d1)}')
