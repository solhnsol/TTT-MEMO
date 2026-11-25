"""Block-stream (job 1296 stored OUT reads; later also 1345w / 1345x read files via --dir/--tag) familiarity / binding split (CPU; new file; thread-server 11/25; descriptive, t 95% intervals over seeds). Candidates of an item are the places of its own block (gen_facts block_items), the read is the neutral out-of-window tail read of all 128 facts (+4 ctrl) at every read point a = 16 (k+1); blocks <= k are written, blocks > k are NOT YET written (never exposed) -> lag = k - block < 0. Per item and read, dS = candidate scores minus W0. BINDING = dS[gold] - mean dS[decoys of the same block]; DECOY = mean dS[decoys] (an exposed block reads up, an unwritten block should stay ~0); FAMILIARITY proxy = DECOY(written-block items at lag L) - DECOY(items of unwritten blocks) in the same read (cross-question control; needs k < 7). Usage: python e63_bs8_famsplit.py [--dir D] [--seeds 82,83,84,85]"""
import sys,json,random,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1] if '--dir' in a else 'experiments/exp07_write_ladder/e57_blockstream8';S=[int(x) for x in (a[a.index('--seeds')+1] if '--seeds' in a else '82,83,84,85').split(',')]
TAG=a[a.index('--tag')+1] if '--tag' in a else 'bs8'
def perm(seed,k):
    p=list(range(k));random.Random(f'E57bs|{seed}|blockperm').shuffle(p);return p
def ci(d):
    d=np.asarray(d,float);m=d.mean();h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}]'
def per_seed(arm,seed):
    d=json.load(open(f'{D}/E55_{TAG}_{arm}_s{seed}.read.json'));m=d['meta'];W0=np.asarray(d['W0']['out']);pm=perm(seed,128);g=np.asarray(m['gold_idx'])
    res={}
    for pt in sorted(d['read'],key=int):
        k=int(pt)//16-1;dS=np.asarray(d['read'][pt]['out'])-W0;blk=np.asarray([pm[i]//16 for i in range(128)]);lag=k-blk
        bind=np.array([dS[i,g[i]]-np.delete(dS[i],g[i]).mean() for i in range(128)]);dec=np.array([np.delete(dS[i],g[i]).mean() for i in range(128)])
        res[k]=(lag,bind,dec)
    return res
def groups(lag):return {'lag0':lag==0,'lag1-3':(lag>=1)&(lag<=3),'lag4-7':lag>=4,'written':lag>=0,'unwritten':lag<0}
print(f'block stream {TAG} out reads, seeds {S}: per arm, per read point k (blocks 0..k written); BIND = gold - decoys (change over W0), DECOY = mean decoy change; FAM-proxy = DECOY written - DECOY unwritten')
R={arm:[per_seed(arm,s) for s in S] for arm in ('NATS','NATMASK')}
for arm in R:
    print(f'== {arm}')
    for lagname in ('lag0','lag1-3','lag4-7','written'):
        for k in (3,5,6,7):
            vb=[];vd=[];vf=[]
            for r in R[arm]:
                lag,bind,dec=r[k];G=groups(lag)[lagname]
                if G.sum()==0:continue
                vb.append(bind[G].mean());vd.append(dec[G].mean())
                if (lag<0).sum()>0:vf.append(dec[G].mean()-dec[lag<0].mean())
            if vb:print(f'   {lagname:8s} k={k}: bind {ci(vb)} | decoy {ci(vd)} | fam-proxy {ci(vf) if len(vf)>1 else "n/a (no unwritten block)"}')
print('== NATMASK - NATS (seed-paired)')
for lagname in ('lag0','lag1-3','lag4-7','written'):
    for k in (3,5,6,7):
        vb=[];vd=[];vf=[]
        for rm,rn in zip(R['NATMASK'],R['NATS']):
            row=[]
            for r in (rm,rn):
                lag,bind,dec=r[k];G=groups(lag)[lagname]
                row.append((bind[G].mean() if G.sum() else np.nan,dec[G].mean() if G.sum() else np.nan,(dec[G].mean()-dec[lag<0].mean()) if G.sum() and (lag<0).sum() else np.nan))
            vb.append(row[0][0]-row[1][0]);vd.append(row[0][1]-row[1][1]);vf.append(row[0][2]-row[1][2])
        if not np.isnan(vb).any():print(f'   {lagname:8s} k={k}: bind {ci(vb)} | decoy {ci(vd)} | fam-proxy {ci(vf) if not np.isnan(vf).any() else "n/a"}')
