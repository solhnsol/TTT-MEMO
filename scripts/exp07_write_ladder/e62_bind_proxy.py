"""E62 P0a: what can be computed now about familiarity vs binding (CPU; new file; thread-server 11/24; descriptive): in P0a every wrong candidate is UNEXPOSED in the run of its book (other books' planted answers and generated unplanted words; each book restarts from W0), so gold - candidate mixes familiarity (the answer string was written n times) and binding (name -> answer). Computed from the stored 16-candidate score vectors at the last-mention read: (1) SAME - NAT change over W0 of the gold score and of the mean / max of the 15 candidates (per fact, 3-frame mean), (2) the SD across facts of the SAME - NAT change of 'gold - one single candidate' (proxy for a design with 1 or 2 decoys instead of the max over 15), (3) the SAME - NAT change of gold - mean(candidates)."""
import sys,json,numpy as np,collections
from scipy import stats
R=json.load(open(sys.argv[sys.argv.index('--json')+1]))
rows=[]
for B in R['books']:
    W0=np.asarray(B['W0']);man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    for ci,f in enumerate(B['facts']):
        last=max(man[ci],key=lambda r:r['mention'])['chunk'];N=np.asarray(B['arms']['NAT']['reads'][str(last)])[ci];S=np.asarray(B['arms']['SAME']['reads'][str(last)])[ci];w=W0[ci]   # [3 frames,16]
        rows.append(dict(n=f['n'],kind=f['kind'],dN=(N-w).mean(0),dS=(S-w).mean(0)))                     # [16] change over W0, mean over frames
def ci(x):x=np.asarray(x,float);m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
dN=np.array([r['dN'] for r in rows]);dS=np.array([r['dS'] for r in rows]);n=np.array([r['n'] for r in rows]);kind=np.array([r['kind'] for r in rows])
gold=dS[:,0]-dN[:,0];cm=dS[:,1:].mean(1)-dN[:,1:].mean(1);cx=dS[:,1:].max(1)-dN[:,1:].max(1)
print(f'SAME - NAT change over W0 at the last mention, {len(rows)} facts (summed log p):')
print('  gold score        ',ci(gold));print('  mean of 15 cands  ',ci(cm));print('  max of 15 cands   ',ci(cx));print('  gold - mean cands ',ci(gold-cm));print('  gold - max cands  ',ci(gold-cx))
print('  SD across facts of: gold - max %.3f | gold - mean %.3f | gold alone %.3f | mean cands alone %.3f'%((gold-cx).std(ddof=1),(gold-cm).std(ddof=1),gold.std(ddof=1),cm.std(ddof=1)))
sd1=[(dS[:,0]-dN[:,0]-(dS[:,c]-dN[:,c])).std(ddof=1) for c in range(1,16)];print('  SD across facts of gold - ONE single candidate (mean over the 15 candidates): %.3f [min %.3f max %.3f]'%(np.mean(sd1),min(sd1),max(sd1)))
for m in (1,2,3):
    sdm=np.mean([(dS[:,0]-dN[:,0]-(dS[:,cs]-dN[:,cs]).mean(1)).std(ddof=1) for cs in [list(np.random.default_rng(i).choice(np.arange(1,16),m,replace=False)) for i in range(30)]]);print(f'  SD of gold - mean of {m} random candidates: {sdm:.3f}')
for nm,sel in (('close n=8',(n==8)&(kind=='close')),('close n=16',(n==16)&(kind=='close')),('close n=32',(n==32)&(kind=='close')),('all close',kind=='close')):
    print(f'  {nm:10s} ({sel.sum():2d} facts) gold {gold[sel].mean():6.2f} | mean cands {cm[sel].mean():6.2f} | gold-mean {np.mean(gold[sel]-cm[sel]):6.2f} | gold-max {np.mean(gold[sel]-cx[sel]):6.2f}')
