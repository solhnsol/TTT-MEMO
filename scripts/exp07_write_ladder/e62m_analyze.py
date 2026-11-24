"""E62 MAIN summary (CPU; new file; thread-server 11/24; descriptive, no threshold): from e62m.json (12 units, arms NAT / SAME, one frame, candidates = gold, 1-2 EXPOSED decoys (same book, same type, same number of exposures), unexposed generated words). Per fact the change over W0 at the last mention (and 1st / 2nd / 4th / end) of: HEADLINE margin = gold - max(all 15 wrong); BINDING = gold - mean(exposed decoys); FAMILIARITY = mean(exposed decoys) - mean(unexposed); gold - mean(unexposed). Arm difference SAME - NAT per cell (n, gap kind) over the 12 units (t 95%, df = units - 1) and clustered by distinct book (the two Carmen Ariza units averaged), pooled n = 2..8 and n = 16..32, slope over log2 n, SD s of the per-fact differences, type effects, pre-write answer-token NLL axis (natural window; SAME - NAT vs the running count of later mentions with the previous mention inside the window), W0 margins and the positive control, masking checks. python3 e62m_analyze.py --json J"""
import sys,json,numpy as np,collections
from scipy import stats
a=sys.argv;R=json.load(open(a[a.index('--json')+1]))
def parts(S,f):
    S=np.asarray(S)[:,0,:];m=len(f['exposed']);out=[]
    for i in range(S.shape[0]):
        mm=len(f['_all'][i]['exposed']);g=S[i,0];ex=S[i,1:1+mm];un=S[i,1+mm:]
        out.append((g-S[i,1:].max(),g-ex.mean(),ex.mean()-un.mean(),g-un.mean(),g))
    return np.array(out)                                                        # [F,5]: headline, binding, familiarity, gold-unexposed mean, gold
METRICS=('headline gold-max','BINDING gold-exposed decoy','FAMILIARITY exposed-unexposed','gold-unexposed mean','gold score')
rows=[]
for ui,B in enumerate(R['books']):
    for f in B['facts']:f['_all']=B['facts']
    W0=parts(B['W0'],B);man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    cluster=B['book'].split('|')[0]
    for ci,f in enumerate(B['facts']):
        ms=sorted(man[ci],key=lambda r:r['mention']);pts={'m1':ms[0]['chunk'],'m2':ms[1]['chunk'],'m4':ms[3]['chunk'] if len(ms)>3 else None,'last':ms[-1]['chunk'],'end':B['nch']-1}
        for ar,A in B['arms'].items():
            for p,ch in pts.items():
                if ch is None or str(ch) not in A['reads']:continue
                v=parts(A['reads'][str(ch)],B)[ci]-W0[ci]
                rows.append(dict(unit=ui,cluster=cluster,cell=ci,n=f['n'],kind=f['kind'],type=f['type'],arm=ar,pt=p,v=v,nexp=len(f['exposed'])))
def ci(d):
    d=np.asarray(d,float);m=d.mean()
    if len(d)<2:return f'{m:8.3f}'
    h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:8.3f} [{m-h:8.3f},{m+h:8.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
V={(r['unit'],r['cell'],r['arm'],r['pt']):r for r in rows}
print(f'E62 MAIN: {len(R["books"])} units, margin changes over W0 (summed log p); exposed decoys per fact: '+str(dict(collections.Counter(r['nexp'] for r in rows if r['arm']=='NAT' and r['pt']=='last'))))
for mi,mn in enumerate(METRICS[:4]):
    print(f'== {mn}: SAME - NAT at the last mention per cell (12 units) | clustered by distinct book')
    cells=sorted({(r['n'],r['kind']) for r in rows})
    for n,k in cells:
        d=[];dc=collections.defaultdict(list)
        for (u,c,ar,p),r in V.items():
            if ar=='SAME' and p=='last' and r['n']==n and r['kind']==k and (u,c,'NAT','last') in V:
                x=r['v'][mi]-V[(u,c,'NAT','last')]['v'][mi];d.append(x);dc[r['cluster']].append(x)
        if d:print(f'   n={n:2d} {k:6s} {ci(d)} | clusters {ci([np.mean(v) for v in dc.values()])}')
    for lo,hi,nm in ((2,8,'n=2..8'),(16,32,'n=16..32')):
        d=[V[(u,c,'SAME','last')]['v'][mi]-V[(u,c,'NAT','last')]['v'][mi] for (u,c,ar,p),r in V.items() if ar=='SAME' and p=='last' and lo<=r['n']<=hi and (u,c,'NAT','last') in V];print(f'   pooled {nm}: {ci(d)}  (SD per fact {np.std(d,ddof=1):.3f})')
    x=[];y=[]
    for (u,c,ar,p),r in V.items():
        if ar=='SAME' and p=='last' and (u,c,'NAT','last') in V:x.append(np.log2(r['n']));y.append(r['v'][mi]-V[(u,c,'NAT','last')]['v'][mi])
    print(f'   slope over log2 n: {np.polyfit(x,y,1)[0]:.3f}; all-cell SD of the per-fact difference {np.std(y,ddof=1):.3f} over {len(y)} facts')
print('== NAT change over W0 by n (last mention), headline / binding / familiarity')
for n in sorted({r['n'] for r in rows}):
    sel=[r['v'] for r in rows if r['arm']=='NAT' and r['pt']=='last' and r['n']==n];print(f'   n={n:2d} headline {ci([v[0] for v in sel])} | binding {ci([v[1] for v in sel])} | familiarity {ci([v[2] for v in sel])}')
print('== types (SAME - NAT last mention, headline / binding), mean over facts')
for t in sorted({r['type'] for r in rows}):
    d=[(V[(u,c,'SAME','last')]['v'][0]-V[(u,c,'NAT','last')]['v'][0],V[(u,c,'SAME','last')]['v'][1]-V[(u,c,'NAT','last')]['v'][1]) for (u,c,ar,p),r in V.items() if ar=='SAME' and p=='last' and r['type']==t and (u,c,'NAT','last') in V];print(f'   {t:7s} {len(d):2d} facts: {np.mean([x[0] for x in d]):7.2f} / {np.mean([x[1] for x in d]):7.2f}')
print('== pre-write answer-token NLL (natural window), SAME - NAT vs the running count x of later mentions whose previous mention is inside the window')
bins=[(1,1),(2,2),(3,4),(5,8),(9,16),(17,99)]
for key in ('nll_nat','nll_arm'):
    acc=collections.defaultdict(list)
    for B in R['books']:
        if 'SAME' not in B['arms']:continue
        A={(d['cell'],d['mention']):d for d in B['arms']['SAME']['mention_nll']};N={(d['cell'],d['mention']):d for d in B['arms']['NAT']['mention_nll']}
        for cx in range(len(B['facts'])):
            ms=sorted([r for r in B['manifest'] if r['cell']==cx],key=lambda r:r['mention']);cnt=0;per=collections.defaultdict(list)
            for j in range(1,len(ms)):
                if ms[j]['tok'][0]-ms[j-1]['tok'][1]<=8192 and (cx,j) in A and (cx,j) in N and A[(cx,j)][key] and N[(cx,j)]['nll_nat']:
                    cnt+=1;d=np.mean(A[(cx,j)][key])-np.mean(N[(cx,j)]['nll_nat'] if key=='nll_nat' else N[(cx,j)]['nll_arm'])
                    for lo,hi in bins:
                        if lo<=cnt<=hi:per[(lo,hi)].append(d)
            for b_,v in per.items():acc[b_].append(float(np.mean(v)))
    print(f'   SAME - NAT, {key}: '+' | '.join(f'x={lo}-{hi}: {ci(acc[(lo,hi)]) if acc[(lo,hi)] else "-"}' for lo,hi in bins))
print('== W0 margins (headline) / positive control, masking checks')
for B in R['books']:
    for f in B['facts']:f['_all']=B['facts']
    w=parts(B['W0'],B)[:,0];pc=parts(B['W0_positive_control'],B)[:,0];print(f'   {B["book"][:34]:34s} W0 {w.mean():7.2f} positive control {pc.mean():7.2f}',{ar:(sum(s['masked_pairs'] for s in A['steps']),A.get('pre_first_mask_max_dgnorm_vs_NAT')) for ar,A in B['arms'].items() if ar!='NAT'})
