"""E62 P0a: question frames 3 vs 1 (CPU; new file; thread-server 11/24; Hans 'CPU check first'; descriptive): per fact and frame the margin (gold - max wrong, summed log p) at the last-mention read minus W0, SAME - NAT per fact and frame, correlation of the three frames, pooled mean [t 95%] per frame / frame mean, SD of the per-fact differences with 3 frames averaged vs one frame, and the number of books (= facts per cell) needed for a 95% interval that excludes 0 with 80% power for the close n=8 / n=16 cells (N ~ ((t_{.975}+t_{.8}) s / delta)^2, iterated with the t quantile). python3 e62_frames.py --json J"""
import sys,json,numpy as np,itertools,collections
from scipy import stats
R=json.load(open(sys.argv[sys.argv.index('--json')+1]))
def mg(S):S=np.asarray(S);return S[:,:,0]-S[:,:,1:].max(2)               # [facts, frames]
rows=[]
for B in R['books']:
    W0=mg(B['W0']);man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    for ci,f in enumerate(B['facts']):
        last=max(man[ci],key=lambda r:r['mention'])['chunk'];N=mg(B['arms']['NAT']['reads'][str(last)])[ci]-W0[ci];S=mg(B['arms']['SAME']['reads'][str(last)])[ci]-W0[ci]
        rows.append(dict(book=B['book'][:12],cell=ci,n=f['n'],kind=f['kind'],type=f['type'],d=(S-N)))   # d: 3 frames of SAME - NAT
D=np.array([r['d'] for r in rows]);n=np.array([r['n'] for r in rows]);kind=np.array([r['kind'] for r in rows]);typ=np.array([r['type'] for r in rows])
def ci(x):x=np.asarray(x,float);m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
print(f'{len(rows)} facts; SAME - NAT at the last mention (margin change over W0), per frame and frame mean')
for nm,sel in (('all facts',np.ones(len(rows),bool)),('n=2..8',n<=8),('n=16..32',n>=16),('close cells',kind=='close')):
    print(f' {nm:12s} n_facts={sel.sum():2d} '+' | '.join(f'frame{j}: {ci(D[sel][:,j])}' for j in range(3))+f' | mean of 3: {ci(D[sel].mean(1))}')
print('frames correlation of the per-fact SAME - NAT (Pearson / Spearman):',{f'{a}{b}':(round(float(np.corrcoef(D[:,a],D[:,b])[0,1]),3),round(float(stats.spearmanr(D[:,a],D[:,b])[0]),3)) for a,b in itertools.combinations(range(3),2)})
print('same sign of the three frames per fact:',int(np.mean(np.sign(D)==np.sign(D.mean(1,keepdims=True))[:, :].repeat(3,1)).round(3)*100),'% of fact-frames agree with the frame-mean sign;',f'{np.mean((D>0).all(1)|(D<0).all(1)):.2f} of the facts have all three frames in one direction')
sd3=D.mean(1).std(ddof=1);print(f'SD of the per-fact difference: 3-frame mean {sd3:.3f}; single frames '+', '.join(f'frame{j} {D[:,j].std(ddof=1):.3f}' for j in range(3))+f'; ratio single/3-mean {np.mean([D[:,j].std(ddof=1) for j in range(3)])/sd3:.3f}')
for t_ in sorted(set(typ)):
    s=typ==t_;print(f'   type {t_:7s} n_facts {s.sum():2d}: frame means '+' '.join(f'{D[s][:,j].mean():7.2f}' for j in range(3)))
def need(delta,s,power=.8):
    N=4
    for _ in range(60):
        t1=stats.t.ppf(.975,N-1);t2=stats.t.ppf(power,N-1);N=int(np.ceil(((t1+t2)*s/delta)**2));N=max(N,3)
    return N
print('books needed per cell (facts per cell = books), 95% interval excluding 0 with 80% power, delta = P0a point estimate and half of it; s = per-fact SD of the pooled all-cell difference (3-frame mean vs single frame; also the cell-specific SD)')
for nm,sel in (('n=8 close',(n==8)&(kind=='close')),('n=16 close',(n==16)&(kind=='close')),('n=32 close',(n==32)&(kind=='close')),('n=4 close',(n==4)&(kind=='close'))):
    d3=D[sel].mean(1);d1=D[sel][:,0]
    for lab,s_all,s_cell in (('3-frame mean',sd3,d3.std(ddof=1)),('single frame (mean of 3 frames SDs)',np.mean([D[:,j].std(ddof=1) for j in range(3)]),np.mean([D[sel][:,j].std(ddof=1) for j in range(3)]))):
        dl=d3.mean();print(f'  {nm:11s} {lab:36s} delta {dl:6.2f} (3 facts): books for delta / delta/2 with the all-cell s {s_all:.2f}: {need(dl,s_all)} / {need(dl/2,s_all)}; with the cell SD {s_cell:.2f}: {need(dl,s_cell)} / {need(dl/2,s_cell)}')
