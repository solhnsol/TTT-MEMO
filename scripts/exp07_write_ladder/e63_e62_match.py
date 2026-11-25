"""E62 main: ANSWER-MATCHED binding, decomposition of negative SAME - NAT cells, and accuracy (CPU; new file; thread-server 11/25; from the stored e62m.json; descriptive; no thresholds). Storage: per book 14 questions (one per planted fact), per read point (hook chunk) the score of 16 candidates per question = [gold, 1-2 EXPOSED decoys = the answers of the other facts of the same type group (same n, other gap kinds), unexposed generated words]. Per fact the read at the chunk of its LAST mention is used; change over W0 (dS). OLD binding = dS[gold] - mean dS[exposed decoys of the SAME question]. MATCHED binding of fact f with answer a_f = dS[f, a_f] - mean over the OTHER questions q of the group in which a_f is an exposed decoy of dS[q, a_f] (the same word, scored as gold and as decoy). Decomposition: gold dS and decoy dS by the gap kind of the decoy's own fact. Accuracy: top-1 and mean rank of the gold among the exposed candidates only and among all 16 candidates (W0, NAT, SAME), chance level. Unit of analysis = book (per-book means; t 95% interval over books). python e63_e62_match.py --json e62m.json"""
import sys,json,collections,numpy as np
from scipy import stats
a=sys.argv;BK=json.load(open(a[a.index('--json')+1]))['books'];NBK=len(BK)
def tci(x):
    x=np.asarray(x,float);m=x.mean()
    if len(x)<2:return f'{m:7.2f}'
    h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.2f} [{m-h:6.2f},{m+h:6.2f}]'
rows=[]                                                   # per fact: dict(book, cell, n, kind, type, arm, values...)
for bi,B in enumerate(BK):
    W0=np.asarray(B['W0'])[:,0,:];man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    F=B['facts'];C=[B['cands'][str(i)] for i in range(len(F))];ans=[f['answer'] for f in F];mm=[len(f['exposed']) for f in F]
    owner={w:i for i,w in enumerate(ans)}
    for i,f in enumerate(F):
        last=max(man[i],key=lambda r:r['mention'])['chunk'];d=dict(book=bi,cell=i,n=f['n'],kind=f['kind'],type=f['type'],mm=mm[i])
        deco=[q for q in range(len(F)) if q!=i and ans[i] in C[q][1:1+mm[q]]]                      # questions in which a_f is an exposed decoy
        d['nq']=len(deco)
        for ar in ('W0','NAT','SAME'):
            S=W0 if ar=='W0' else np.asarray(B['arms'][ar]['reads'][str(last)])[:,0,:]
            if ar!='W0':S=S                                                                          # raw scores; dS computed below
            d[ar+'_S']=S[i]                                                                         # own question scores [16]
            d[ar+'_q']=[S[q,C[q].index(ans[i])] for q in deco]                                      # a_f as decoy in the other questions
            d[ar+'_old_decoy_owner']=[(owner.get(C[i][c]),S[i,c]) for c in range(1,1+mm[i])]
        d['W0_S_all']=W0
        rows.append(d)
def dS(r,ar):return r[ar+'_S']-r['W0_S']
def vals(r,ar):
    """-> gold_d, decoy_old_d, bind_old, decoy_match_d, bind_match (change over W0)"""
    g=r[ar+'_S'][0]-r['W0_S'][0];do=np.mean(r[ar+'_S'][1:1+r['mm']]-r['W0_S'][1:1+r['mm']]);dm=np.mean(np.asarray(r[ar+'_q'])-np.asarray(r['W0_q']))
    return np.array([g,do,g-do,dm,g-dm])
for r in rows:
    for ar in ('NAT','SAME'):r[ar+'_v']=vals(r,ar)
def book_table(sel,fn):
    """per-book mean of fn(row) over the rows selected -> array over books"""
    d=collections.defaultdict(list)
    for r in rows:
        if sel(r):d[r['book']].append(fn(r))
    return np.array([np.mean(v,0) for k,v in sorted(d.items())]),sum(len(v) for v in d.values())
CELLS=sorted({(r['n'],r['kind']) for r in rows});GR=[('n=2..8',lambda r:r['n']<=8),('n=16..32',lambda r:r['n']>=16)]+[(f'kind {k}',lambda r,k=k:r['kind']==k) for k in ('close','edge','far','mixed')]+[('all',lambda r:True)]
print(f'E62 MAIN matched binding: {NBK} books; change over W0 at the last mention (summed log p); mean [95% t over books]; columns: OLD bind (gold - decoys of the same question) | MATCHED bind (a_f as gold - a_f as decoy in the other questions of its group) | their gold dS | decoy dS (old) | decoy dS (matched)')
for title,items in (('cells (n x kind)',[(f'n={n:2d} {k}',(lambda r,n=n,k=k:r['n']==n and r['kind']==k)) for n,k in CELLS]),('pooled groups',GR)):
    print(f'== {title}: NAT | SAME | SAME - NAT, OLD bind and MATCHED bind')
    for nm,sel in items:
        X={ar:book_table(sel,lambda r,ar=ar:r[ar+'_v'])[0] for ar in ('NAT','SAME')};nf=book_table(sel,lambda r:1)[1]
        print(f'  {nm:14s} facts {nf:3d} | NAT old {tci(X["NAT"][:,2])} matched {tci(X["NAT"][:,4])} | SAME old {tci(X["SAME"][:,2])} matched {tci(X["SAME"][:,4])} | SAME-NAT old {tci(X["SAME"][:,2]-X["NAT"][:,2])} matched {tci(X["SAME"][:,4]-X["NAT"][:,4])}')
print('== components (pooled groups and the two negative cells): gold dS, old decoy dS, matched decoy dS; NAT | SAME | SAME - NAT')
for nm,sel in [('n=32 mixed',lambda r:r['n']==32 and r['kind']=='mixed'),('n=8 edge',lambda r:r['n']==8 and r['kind']=='edge'),('n=2..8',GR[0][1]),('n=16..32',GR[1][1]),('all',GR[-1][1])]:
    X={ar:book_table(sel,lambda r,ar=ar:r[ar+'_v'])[0] for ar in ('NAT','SAME')}
    print(f'  {nm:12s} gold {tci(X["NAT"][:,0])} | {tci(X["SAME"][:,0])} | {tci(X["SAME"][:,0]-X["NAT"][:,0])}')
    print(f'  {"":12s} decoy(old) {tci(X["NAT"][:,1])} | {tci(X["SAME"][:,1])} | {tci(X["SAME"][:,1]-X["NAT"][:,1])}')
    print(f'  {"":12s} decoy(matched) {tci(X["NAT"][:,3])} | {tci(X["SAME"][:,3])} | {tci(X["SAME"][:,3]-X["NAT"][:,3])}')
print('== decomposition of the OLD decoys by the gap kind of the decoy fact own (decoy candidate score change in the question of fact f): cells n=32 mixed, n=8 edge, and all cells by decoy kind')
for r in rows:
    r['W0_old_dec']=[(o,s) for o,s in r['W0_old_decoy_owner']]
for nm,sel in [('n=32 mixed',lambda r:r['n']==32 and r['kind']=='mixed'),('n=8 edge',lambda r:r['n']==8 and r['kind']=='edge'),('all cells',lambda r:True)]:
    res={}
    for r in rows:
        if not sel(r):continue
        kinds=[BK[r['book']]['facts'][o]['kind'] if o is not None else '?' for o,_ in r['W0_old_dec']]
        for k,(o,s0),(o1,sn),(o2,ss) in zip(kinds,r['W0_old_dec'],r['NAT_old_decoy_owner'],r['SAME_old_decoy_owner']):
            res.setdefault(k,{}).setdefault(r['book'],[]).append((sn-s0,ss-s0))
    print(f' [{nm}] decoy dS by the decoy fact gap kind: NAT | SAME | SAME - NAT (per-book means; number of books; decoy-question pairs)')
    for k in sorted(res):
        bk=res[k];nb=np.array([np.mean([x[0] for x in v]) for v in bk.values()]);sb=np.array([np.mean([x[1] for x in v]) for v in bk.values()]);npairs=sum(len(v) for v in bk.values())
        print(f'   decoy kind {k:6s} books {len(bk):2d} pairs {npairs:3d} | NAT {tci(nb)} | SAME {tci(sb)} | SAME-NAT {tci(sb-nb)}')
print('== accuracy at the last mention (top-1 and mean rank of the gold; ties counted as not above); exposed candidates only (1 + #exposed) and all 16 candidates; W0 | NAT | SAME; chance = 1/K for top-1 and (K+1)/2 for the rank; mean [95% t over books]')
def acc(r,ar,full):
    S=(r['W0_S_all'][r['cell']] if ar=='W0' else r[ar+'_S']);K=16 if full else 1+r['mm'];x=S[:K];rank=1+int((x[1:]>x[0]).sum());return np.array([rank==1,rank])
for r in rows:
    for ar in ('W0','NAT','SAME'):
        r[ar+'_a_exp']=acc(r,ar,False);r[ar+'_a_all']=acc(r,ar,True)
    r['chance_exp']=np.array([1/(1+r['mm']),(2+r['mm'])/2]);r['chance_all']=np.array([1/16,8.5])
for full,nmf in ((False,'exposed candidates only'),(True,'all 16 candidates (incl. never-exposed)')):
    print(f' [{nmf}]')
    key='_a_all' if full else '_a_exp';ck='chance_all' if full else 'chance_exp'
    for title,items in (('by n',[(f'n={n}',(lambda r,n=n:r['n']==n)) for n in (2,4,8,16,32)]),('by gap kind',[(f'kind {k}',(lambda r,k=k:r['kind']==k)) for k in ('close','edge','far','mixed')]),('by cell',[(f'n={n:2d} {k}',(lambda r,n=n,k=k:r['n']==n and r['kind']==k)) for n,k in CELLS]),('all',[('all',lambda r:True)])):
        print(f'  -- {title}: top-1 (W0 | NAT | SAME | chance) ; mean rank (W0 | NAT | SAME | chance)')
        for nm,sel in items:
            T={ar:book_table(sel,lambda r,ar=ar:r[ar+key])[0] for ar in ('W0','NAT','SAME')};ch=book_table(sel,lambda r:r[ck])[0]
            print(f'    {nm:14s} top-1 {tci(T["W0"][:,0])} | {tci(T["NAT"][:,0])} | {tci(T["SAME"][:,0])} | {ch[:,0].mean():5.2f}   rank {tci(T["W0"][:,1])} | {tci(T["NAT"][:,1])} | {tci(T["SAME"][:,1])} | {ch[:,1].mean():5.2f}')
