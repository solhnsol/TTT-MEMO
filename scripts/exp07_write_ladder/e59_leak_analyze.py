"""Analysis of the leak probe (CPU; new file; EXPLORATORY): per seed the mean over sentences of chunks >= --first of the second-entity (answer), first-entity (control), place and name NLL per condition (NAT / ALLHID / SHOWSAME / SHOWOTHER), then paired differences over seeds, mean [95% t-CI], same-direction seeds; also split by template order (place_first / name_first) for the second entity. python3 e59_leak_analyze.py --dir D [--first 16] [--seeds 231-238]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);first=int(a[a.index('--first')+1]) if '--first' in a else 16;lo,hi=(231,238)
if '--seeds' in a:lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')]
CONDS=('NAT','ALLHID','SHOWSAME','SHOWOTHER');S=[json.load(open(D/f'leak_probe_s{s}.json')) for s in range(lo,hi+1)];n=len(S);t=stats.t.ppf(.975,n-1)
def per(r,nm,cond,order=None):
    x=[q[f'{nm}_{cond}'] for q in r['rows'] if q['k']>=first and (order is None or q['order']==order)];return float(np.mean(x))
def ci(d):d=np.asarray(d);m=d.mean();h=t*d.std(ddof=1)/np.sqrt(n);return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{n}'
print(f'leak probe: seeds {lo}-{hi}, chunks >= {first}; NLL summed over the tokens of the entity (nat per sentence), mean over sentences, then over seeds')
for nm,desc in (('second','SECOND-mentioned entity (answer, binding-dependent)'),('first','FIRST-mentioned entity (vocabulary / format control)'),('place','place tokens'),('name','person-name tokens')):
    for order in (None,'place_first','name_first') if nm=='second' else (None,):
        V={c:np.array([per(r,nm,c,order) for r in S]) for c in CONDS};print(f'[{desc}{"" if order is None else " | "+order}]  means',{c:round(float(V[c].mean()),3) for c in CONDS})
        for x,y in (('SHOWOTHER','ALLHID'),('SHOWSAME','ALLHID'),('NAT','ALLHID'),('SHOWOTHER','SHOWSAME'),('SHOWOTHER','NAT')):print(f'    {x:9s} - {y:9s} {ci(V[x]-V[y])}')
