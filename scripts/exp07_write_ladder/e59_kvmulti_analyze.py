"""Analysis of job 1344t0 (e59_kvmulti.py; CPU; new file; thread-server 11/25; descriptive: values, intervals over seeds, positive seeds; NO verdict wording, no threshold). Per seed and arm P0s (SAMEVIS mask in every chunk) / P1s (C0 mask in every chunk) and read after exposure e in (8, 16, 32, 64, 96, 128): dS = candidate scores minus W0 of the same read (16 facts x 16 candidates: 0 gold, 1..11 exposed decoys, 12..15 never exposed; the candidate lists are recreated from the seed as in e59_kvpos2_analyze). MATCHED binding of fact f = dS[f, gold] - mean over the other questions that contain the word of f as an exposed decoy of dS[q, that word]; FAMILIARITY = mean over facts of (mean exposed decoy dS - mean never-exposed dS) of the own question; MARGIN change = gold - max exposed wrong. Per seed the mean over the 16 facts (and over the facts of one sentence order of the chunk just written). Main: R1 matched binding at exposure 128, P1s - P0s, seed-paired (t interval and a seed bootstrap interval). The familiarity-matched comparison uses only observed exposures (no interpolation).
  python e59_kvmulti_analyze.py --dir D"""
import sys,json,glob,random,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];R=[json.load(open(f)) for f in sorted(glob.glob(D+'/kvmulti_s*.json'))];n=len(R);SEEDS=[r['seed'] for r in R];EXPO=R[0]['expo'];ARMS=('P0s','P1s')
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[g for g in range(16) if g!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def per_fact(dS,W0,seed):
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);S=W0+dS;mar=lambda X:X[:,0]-X[:,1:12].max(1);out=np.zeros((16,3))
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]]
        out[f,0]=dS[f,0]-np.mean(oth) if oth else np.nan
    out[:,1]=dS[:,1:12].mean(1)-dS[:,12:16].mean(1);out[:,2]=mar(S)-mar(W0);return out        # [16,3]: matched binding, familiarity, margin change
V={};VG={}                                                  # (arm,read,e,seed) -> [3]; (arm,read,e,seed,order) -> [3]
for r in R:
    for arm in ARMS:
        rec=r['arms'][arm]
        for e in EXPO:
            d=rec['reads'].get(str(e))
            if d is None:continue
            for rd_,key,W in (('R1','dS1',r['W0_R1']),('R2','dS2',r['W0_R2'])):
                if key not in d:continue
                pf=per_fact(d[key],W,r['seed']);V[(arm,rd_,e,r['seed'])]=np.nanmean(pf,0);od=rec['orders'].get(str(e))
                if od:
                    for o in (0,1):
                        mk=np.array([x==o for x in od])
                        if mk.any():VG[(arm,rd_,e,r['seed'],o)]=np.nanmean(pf[mk],0)
def col(arm,rd_,e,mi,o=None):
    if o is None:return np.array([V[(arm,rd_,e,s)][mi] for s in SEEDS if (arm,rd_,e,s) in V])
    return np.array([VG[(arm,rd_,e,s,o)][mi] for s in SEEDS if (arm,rd_,e,s,o) in VG])
MET=('matched binding','familiarity','margin change')
print(f'1344t0: seeds {SEEDS} (n={n}), exposures {EXPO}; Delta over W0; mean over seeds [95% t] same-direction seeds')
for rd_ in ('R1','R2'):
    print(f'== read {rd_}: arm x exposure')
    for mi,mn in enumerate(MET):
        print(f'  [{mn}]')
        for arm in ARMS:
            print(f'    {arm}: '+' | '.join(f'e{e} {tci(col(arm,rd_,e,mi))}' for e in EXPO if len(col(arm,rd_,e,mi))))
    for o,on in ((0,'place first'),(1,'name first')):
        print(f'  -- sentence order of the chunk just written: {on}')
        for mi,mn in enumerate(MET):
            for arm in ARMS:
                print(f'    [{mn}] {arm}: '+' | '.join(f'e{e} {tci(col(arm,rd_,e,mi,o))}' for e in EXPO if len(col(arm,rd_,e,mi,o))))
print('== MAIN: R1 matched binding, P1s - P0s, seed-paired, by exposure (curve); main = exposure 128')
rng=np.random.default_rng(0)
for e in EXPO:
    a1=col('P1s','R1',e,0);a0=col('P0s','R1',e,0)
    if len(a1)==n and len(a0)==n:
        d=a1-a0;bs=[d[rng.integers(0,n,n)].mean() for _ in range(5000)];print(f'  e={e:3d}: P1s {tci(a1)} | P0s {tci(a0)} | P1s-P0s {tci(d)} | seed bootstrap [{np.percentile(bs,2.5):6.3f},{np.percentile(bs,97.5):6.3f}]'+('   <== main' if e==max(EXPO) else ''))
for rd_ in ('R2',):
    e=max(EXPO)
    if len(col('P1s',rd_,e,0))==n:d=col('P1s',rd_,e,0)-col('P0s',rd_,e,0);print(f'  R2 e={e}: P1s - P0s {tci(d)}')
print('== familiarity-matched comparison: per seed (familiarity, matched binding) pairs of both arms; P0s at its last exposure vs the P1s observed exposure with the closest familiarity (no interpolation)')
E=max(EXPO);sel=[]
for s in SEEDS:
    f0=V[('P0s','R1',E,s)][1];b0=V[('P0s','R1',E,s)][0];best=min((abs(V[('P1s','R1',e,s)][1]-f0),e) for e in EXPO if ('P1s','R1',e,s) in V)[1];f1=V[('P1s','R1',best,s)][1];b1=V[('P1s','R1',best,s)][0];sel.append((best,b1,b0,f1,f0))
    print(f'  seed {s}: P0s e={E}: fam {f0:7.3f} bind {b0:7.3f} | P1s e={best}: fam {f1:7.3f} bind {b1:7.3f} | P1s all: '+' '.join(f'e{e}:({V[("P1s","R1",e,s)][1]:.2f},{V[("P1s","R1",e,s)][0]:.2f})' for e in EXPO if ('P1s','R1',e,s) in V))
x=np.array(sel);print(f'  P1s binding at the closest-familiarity exposure - P0s binding at {E}: {tci(x[:,1]-x[:,2])}; chosen exposures {x[:,0].astype(int).tolist()}; familiarity P1s {tci(x[:,3])} vs P0s {tci(x[:,4])}')
print('== per-chunk diagnostics (mean over chunks and seeds): pre-clip gradient norm, clip coefficient, chunk CE; divergence flag (not finite or last-8 CE > 4.0; reported, nothing excluded)')
for arm in ARMS:
    st=[x for r in R for x in r['arms'][arm]['steps']];print(f'  {arm}: gnorm {np.mean([x["gnorm"] for x in st]):.4f} coef {np.mean([x["coef"] for x in st]):.4f} CE {np.mean([x["ce"] for x in st]):.3f} | per seed last-8 CE {[round(float(np.mean([x["ce"] for x in r["arms"][arm]["steps"][-8:]])),2) for r in R]} | diverged {[r["arms"][arm]["diverged"] for r in R]}')
for r in R:
    if 'check_chunk0_dgnorm' in r:print(f'  seed {r["seed"]}: chunk 0 |d gnorm| P0s vs P1s {r["check_chunk0_dgnorm"]:.2e} (no earlier sentence: must be ~0)')
print('== error-size diagnostics at chunks 8, 16, ..., 128 (place tokens of the 16 read facts, weights of the arm at that moment): ratio ||e_C0|| / ||e_SV||, share of tokens with ratio > 100, place-token NLL under SAMEVIS and C0 masks; median [quartiles]')
for arm in ARMS:
    for ce in sorted({int(k) for r in R for k in r['arms'][arm]['diag']}):
        rs=[];hit=[];nsv=[];nc0=[];cs=[];msv=[];mc0=[]
        for r in R:
            d=r['arms'][arm]['diag'].get(str(ce))
            if d is None:continue
            own,oth=np.array(d['en_own']),np.array(d['en_other']);nown,noth=np.array(d['nll_own']),np.array(d['nll_other'])
            if d['cond_own']=='SAMEVIS':esv,ec0,nl_sv,nl_c0=own,oth,nown,noth
            else:esv,ec0,nl_sv,nl_c0=oth,own,noth,nown
            q=ec0/np.maximum(esv,1e-12);rs+=q.tolist();hit+=(q>100).tolist();nsv+=nl_sv.tolist();nc0+=nl_c0.tolist()
            if 'cos' in d:
                cs+=d['cos'];mo,mt=np.array(d['mass_own']),np.array(d['mass_other']);msv+=(mo if d['cond_own']=='SAMEVIS' else mt).tolist();mc0+=(mt if d['cond_own']=='SAMEVIS' else mo).tolist()
        if rs:print(f'  {arm} chunk {ce:3d}: ratio {np.median(rs):8.3g} [{np.percentile(rs,25):.3g},{np.percentile(rs,75):.3g}] share>100 {np.mean(hit):.2f} | NLL SAMEVIS {np.median(nsv):.2f} C0 {np.median(nc0):.2f} (n tokens {len(rs)})'+(f' | cos(e_SV,e_C0) {np.median(cs):.3f} [{np.percentile(cs,25):.3f},{np.percentile(cs,75):.3f}] | candidate first-token share of |e|: SAMEVIS {np.median(msv):.3f} C0 {np.median(mc0):.3f}' if cs else ''))
