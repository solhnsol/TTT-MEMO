"""E62 main, book-cluster bootstrap on one or several e62m*.json files (CPU; new file; thread-server 11/25): the books of all given files are pooled (24 books for the two batches) and the pooled estimates are resampled BY BOOK (5000 resamples, books drawn with replacement, the pooled mean = sum over the drawn books / count; fixed rng). Rows: SAME - NAT at the last mention (per fact: v(SAME) - v(NAT)) and the absolute levels of NAT and SAME, for headline / BINDING (gold - exposed decoys) / FAMILIARITY (exposed - unexposed) / gold - unexposed; groups: pooled n = 2..8, n = 16..32, all n, by gap kind (close / edge / far / mixed), and every (n, kind) cell. Also the plain per-fact t interval and the number of books with the majority sign. Descriptive; no thresholds.
  python e63_e62_bootstrap.py --json a.json[,b.json,...]"""
import sys,json,collections,numpy as np
from scipy import stats
a=sys.argv;FILES=a[a.index('--json')+1].split(',');BOOKS=[]
for f in FILES:BOOKS+=json.load(open(f))['books']
def parts(S,B):
    S=np.asarray(S)[:,0,:];out=[]
    for i in range(S.shape[0]):
        mm=len(B['facts'][i]['exposed']);g=S[i,0];ex=S[i,1:1+mm];un=S[i,1+mm:];out.append((g-S[i,1:].max(),g-ex.mean(),ex.mean()-un.mean(),g-un.mean()))
    return np.array(out)
MN=('headline gold-max','BINDING gold-exposed decoy','FAMILIARITY exposed-unexposed','gold-unexposed mean')
rows=[]                                     # dict(book, n, kind, arm, v[4])
for ui,B in enumerate(BOOKS):
    W0=parts(B['W0'],B);man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    for ci,f in enumerate(B['facts']):
        last=max(man[ci],key=lambda r:r['mention'])['chunk']
        for ar,A in B['arms'].items():
            if str(last) in A['reads']:rows.append(dict(book=ui,n=f['n'],kind=f['kind'],arm=ar,cell=ci,v=parts(A['reads'][str(last)],B)[ci]-W0[ci]))
V={(r['book'],r['cell'],r['arm']):r for r in rows}
pairs=[(V[(b,c,'SAME')],V[(b,c,'NAT')]) for (b,c,ar) in V if ar=='SAME' and (b,c,'NAT') in V]
rng=np.random.default_rng(0);NB=len(BOOKS)
def groupsel(g):
    lo,hi,kind=g;return lambda r:lo<=r['n']<=hi and (kind is None or r['kind']==kind)
def boot(per_book,B=5000):
    """per_book: dict book -> (sum[4], count); -> mean and 95% bootstrap interval of the pooled mean over the books"""
    ks=sorted(per_book);S=np.array([per_book[k][0] for k in ks]);C=np.array([per_book[k][1] for k in ks],float);tot=S.sum(0)/max(C.sum(),1)
    idx=rng.integers(0,len(ks),size=(B,len(ks)));num=S[idx].sum(1);den=C[idx].sum(1)[:,None];bs=num/np.maximum(den,1)
    return tot,np.percentile(bs,[2.5,97.5],axis=0),S,C
def fmt(m,lohi,j):return f'{m[j]:7.2f} [{lohi[0][j]:6.2f},{lohi[1][j]:6.2f}]'
def table(title,items,sel):
    print(f'== {title}');
    for name,g in items:
        pb=collections.defaultdict(lambda:[np.zeros(4),0])
        for x in sel(g):
            pb[x[0]][0]+=x[1];pb[x[0]][1]+=1
        if not pb:continue
        m,lohi,S,C=boot(pb);npos=int(((S.sum(1)if S.ndim==1 else S[:,1]/np.maximum(C,1))>0).sum())
        print(f'   {name:16s} facts {int(C.sum()):4d} books {len(pb):2d} | headline {fmt(m,lohi,0)} | BIND {fmt(m,lohi,1)} | FAM {fmt(m,lohi,2)} | gold-unexp {fmt(m,lohi,3)} | books with bind>0: {int((S[:,1]/np.maximum(C,1)>0).sum())}/{len(pb)}')
GRP=[('n=2..8',(2,8,None)),('n=16..32',(16,32,None)),('all n',(2,32,None)),('kind close',(2,32,'close')),('kind edge',(2,32,'edge')),('kind far',(2,32,'far')),('kind mixed',(2,32,'mixed'))]
CELLS=sorted({(r['n'],r['kind']) for r in rows});CG=[(f'n={n} {k}',(n,n,k)) for n,k in CELLS]
print(f'E62 MAIN book-cluster bootstrap: files {FILES}, {NB} books, {len(pairs)} paired facts; change over W0 at the last mention, summed log p')
for title,sel in (('SAME - NAT (per fact paired)',lambda g:[(s['book'],s['v']-n['v']) for s,n in pairs if groupsel(g)(s)]),('NAT absolute level',lambda g:[(r['book'],r['v']) for r in rows if r['arm']=='NAT' and groupsel(g)(r)]),('SAME absolute level',lambda g:[(r['book'],r['v']) for r in rows if r['arm']=='SAME' and groupsel(g)(r)])):
    table(title+' pooled groups',GRP,sel);table(title+' cells',CG,sel)
