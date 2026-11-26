"""Analysis of job 1344t2 'kvmask' (e59_kvmask.py + e59_kvwin.py --srcs; CPU; new file; thread-server 11/26; descriptive: values and intervals over seeds, same-direction seeds, no verdict wording, no threshold). Arms P0s, P1s from job 1344t0 (kvmulti dir), Mrand, M2As-q, M2As-s from kvmask/kvmask_s<seed>.json, E2As (reference) from kvmulti2. Per seed, arm and exposure: R1 matched binding / familiarity (exposed wrong minus never exposed) / margin change (Delta over W0), also by the sentence order of the chunk just written; MAIN = R1 matched binding at 128: M2As-q - P0s beside P1s - P0s (seed-paired, t interval and seed bootstrap, same-direction seeds), with P1s - M2As-q, M2As-s - M2As-q, Mrand - P0s; ladder per exposure; (familiarity, matched binding) pairs per arm, exposure and seed; R2 at 128; in-window reads R2 / R2m / R2a (job 1349 for P0s, P1s; job 1353 for E2As; job 1344t2 for the new arms); diagnostics: clip coefficient, gradient norm, original chunk CE per interval, hidden pairs and query rows per chunk, every 8 chunks NLL and ||e|| of the place and name tokens under C0, SAMEVIS and the arm mask and the correct logit (SAMEVIS), divergence flag (not finite or last-8 CE > 4.0; nothing excluded).
  python e59_kvmask_analyze.py --dir KVMASK_DIR --base KVMULTI_DIR --kv2 KVMULTI2_DIR --win KVWIN_DIR --win2 KVWIN2_DIR --win3 KVWIN3_DIR"""
import sys,json,glob,random,os,numpy as np
from scipy import stats
a=sys.argv;g=lambda k:a[a.index(k)+1] if k in a else None
D=g('--dir');BD=g('--base');KD2=g('--kv2');WD=g('--win');WD2=g('--win2');WD3=g('--win3')
SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/kvmask_s*.json'));n=len(SEEDS)
R0={s:json.load(open(f'{BD}/kvmulti_s{s}.json')) for s in SEEDS};RM={s:json.load(open(f'{D}/kvmask_s{s}.json')) for s in SEEDS};R2={s:json.load(open(f'{KD2}/kvmulti2_s{s}.json')) for s in SEEDS if os.path.exists(f'{KD2}/kvmulti2_s{s}.json')}
EXPO=[8,16,32,48,64,80,96,112,128];LAST=128;ARMS=('P0s','Mrand','M2As-q','M2As-s','P1s');REF='E2As'
def src(arm,s):
    if arm in ('P0s','P1s'):return R0[s]['arms'][arm]
    if arm==REF:return R2[s]['arms'][arm]
    return RM[s]['arms'][arm]
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
rng=np.random.default_rng(0)
def bsi(d):
    d=np.asarray(d,float);d=d[~np.isnan(d)]
    if len(d)<2:return ''
    b=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)];return f'bootstrap [{np.percentile(b,2.5):6.3f},{np.percentile(b,97.5):6.3f}]'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def per_fact(dS,W0,seed):
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);mar=lambda X:X[:,0]-X[:,1:12].max(1);out=np.zeros((16,3))
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];out[f,0]=dS[f,0]-np.mean(oth) if oth else np.nan
    out[:,1]=dS[:,1:12].mean(1)-dS[:,12:16].mean(1);out[:,2]=mar(W0+dS)-mar(W0);return out
MET=('matched binding','familiarity','margin change')
V={};VO={};V2={}
for s in SEEDS:
    for arm in ARMS+((REF,) if s in R2 else ()):
        rec=src(arm,s)
        for e in EXPO:
            d=rec['reads'].get(str(e))
            if d is None:continue
            pf=per_fact(d['dS1'],R0[s]['W0_R1'],s);V[(arm,e,s)]=np.nanmean(pf,0);od=rec['orders'].get(str(e))
            if od:
                for o in (0,1):
                    mk=np.array([x==o for x in od])
                    if mk.any():VO[(arm,e,s,o)]=np.nanmean(pf[mk],0)
        d=rec['reads'][str(LAST)]
        if 'dS2' in d:V2[(arm,s)]=np.nanmean(per_fact(d['dS2'],R0[s]['W0_R2'],s),0)
def col(arm,e,mi,o=None):
    out=[]
    for s in SEEDS:
        v=V.get((arm,e,s)) if o is None else VO.get((arm,e,s,o));out.append(np.nan if v is None else v[mi])
    return np.array(out)
print(f'1344t2 kvmask: seeds {SEEDS} (n={n}); Delta over W0; mean over seeds [95% t] same-direction seeds; P0s, P1s from job 1344t0 (empty where not read); E2As = reference from job 1344t1')
print('== MAIN (R1 matched binding at 128, seed-paired):')
for nm,d in (('M2As-q - P0s',col('M2As-q',128,0)-col('P0s',128,0)),('P1s - P0s',col('P1s',128,0)-col('P0s',128,0)),('P1s - M2As-q',col('P1s',128,0)-col('M2As-q',128,0)),('M2As-s - M2As-q',col('M2As-s',128,0)-col('M2As-q',128,0)),('Mrand - P0s',col('Mrand',128,0)-col('P0s',128,0)),('M2As-s - P0s',col('M2As-s',128,0)-col('P0s',128,0)),('E2As - P0s',col(REF,128,0)-col('P0s',128,0))):print(f'  {nm:16s} {tci(d)} {bsi(d)}')
for o,on in ((None,'all'),(0,'place first'),(1,'name first')):
    print(f'== R1, sentence order of the chunk just written: {on}')
    for mi,mn in enumerate(MET):
        print(f'  [{mn}]')
        for arm in ARMS+(REF,):print(f'    {arm:7s}: '+' | '.join(f'e{e} {tci(col(arm,e,mi,o))}' for e in EXPO if np.isfinite(col(arm,e,mi,o)).any()))
print('== LADDER (same seeds, paired): Mrand - P0s | M2As-q - Mrand | M2As-s - M2As-q | P1s - M2As-q, per exposure')
for mi,mn in enumerate(MET):
    print(f'  [{mn}]')
    for e in EXPO:
        c={ar:col(ar,e,mi) for ar in ARMS}
        if not all(np.isfinite(c[ar]).any() for ar in ARMS):continue
        parts=[('Mrand - P0s',c['Mrand']-c['P0s']),('M2As-q - Mrand',c['M2As-q']-c['Mrand']),('M2As-s - M2As-q',c['M2As-s']-c['M2As-q']),('P1s - M2As-q',c['P1s']-c['M2As-q'])]
        print(f'    e={e:3d}: '+' | '.join(f'{nm} {tci(d)}'+(f' {bsi(d)}' if (mi==0 and e==LAST) else '') for nm,d in parts))
print('== (familiarity, matched binding) per arm, exposure and seed')
for arm in ARMS:
    print(f'  {arm}:')
    for s in SEEDS:print(f'    seed {s}: '+' '.join(f'e{e}:({V[(arm,e,s)][1]:.2f},{V[(arm,e,s)][0]:.2f})' for e in EXPO if (arm,e,s) in V))
print(f'== R2 (changed read) at exposure {LAST}: matched binding / familiarity / margin change')
for mi,mn in enumerate(MET):print(f'  [{mn}] '+' | '.join(f'{arm} {tci([V2[(arm,s)][mi] for s in SEEDS if (arm,s) in V2])}' for arm in ARMS+(REF,)))
def load(dr,pre):return {s:json.load(open(f'{dr}/{pre}_s{s}.json')) for s in SEEDS if dr and os.path.exists(f'{dr}/{pre}_s{s}.json')}
W1=load(WD,'kvwin');W2=load(WD2,'kvwin2');W3=load(WD3,'kvwin3')
def wrec(arm,s):
    r=W1.get(s) if arm in ('P0s','P1s') else (W2.get(s) if arm==REF else W3.get(s));return r if (r is not None and arm in r['arms']) else None
if W3:
    print('== in-window reads at 128 (R2 no mask, R2m own sentence hidden, R2a all 16 read-fact sentences hidden), Delta over the W0 score of the same read')
    for mi,mn in enumerate(MET):
        print(f'  [{mn}]')
        for v in ('R2','R2m','R2a'):
            row=[]
            for arm in ARMS+(REF,):
                x=[per_fact(np.asarray(r['arms'][arm][v])-np.asarray(r['W0'][v]),r['W0'][v],s).mean(0)[mi] for s in SEEDS for r in [wrec(arm,s)] if r is not None];row.append(f'{arm} {tci(x)}')
            print(f'    {v:4s}: '+' | '.join(row))
    print('  matched binding of the in-window reads, differences (seed-paired):')
    for v in ('R2','R2m','R2a'):
        def cc(arm):return np.array([per_fact(np.asarray(r['arms'][arm][v])-np.asarray(r['W0'][v]),r['W0'][v],s).mean(0)[0] if (r:=wrec(arm,s)) is not None else np.nan for s in SEEDS])
        print(f'    {v:4s}: '+' | '.join(f'{nm} {tci(cc(x)-cc(y))}' for nm,x,y in (('M2As-q - P0s','M2As-q','P0s'),('P1s - P0s','P1s','P0s'),('P1s - M2As-q','P1s','M2As-q'),('M2As-s - M2As-q','M2As-s','M2As-q'),('Mrand - P0s','Mrand','P0s'))))
INT=[(1,16),(17,32),(33,64),(65,96),(97,128)]
print('== per-chunk values, interval means over chunks, mean over seeds [per-seed min..max]')
for key,nm in (('coef','clip coefficient'),('gnorm','pre-clip gradient norm'),('ce','original chunk CE (the arm mask loss)')):
    print(f'  [{nm}]')
    for arm in ARMS:
        row=[]
        for lo,hi in INT:
            v=np.array([np.mean([x[key] for x in src(arm,s)['steps'][lo-1:hi]]) for s in SEEDS]);row.append(f'c{lo}-{hi}: {v.mean():.3f} [{v.min():.3f}..{v.max():.3f}]')
        print(f'    {arm:7s}: '+' | '.join(row))
print('== hidden (query row, key) pairs and hidden query rows per chunk (mean over chunks 1-127 and seeds)')
for arm in ('Mrand','M2As-q','M2As-s'):
    st=[src(arm,s)['steps'][1:] for s in SEEDS];print(f'  {arm}: pairs {np.mean([x["extra_pairs"] for t in st for x in t]):.0f}, rows {np.mean([x["q_rows"] for t in st for x in t]):.1f}')
print('== divergence (not finite or last-8 CE > 4.0; nothing excluded):')
for arm in ARMS:
    st=[src(arm,s)['steps'] for s in SEEDS];print(f'  {arm}: last-8 CE {[round(float(np.mean([x["ce"] for x in t[-8:]])),2) for t in st]} diverged {[bool((not all(np.isfinite([x["gnorm"] for x in t]))) or np.mean([x["ce"] for x in t[-8:]])>4.0) for t in st]}')
def dg(arm,ck,s,mask,key,tok,o=None):
    if arm in ('P0s','P1s'):return np.nan
    d=src(arm,s)['diag'].get(str(ck))
    if d is None:return np.nan
    v=np.array(d[mask][f'{key}_{tok}'])
    if o is not None:
        od=d['order'];ix=np.array(d['idx_place'] if tok=='place' else d['idx_name']);return np.nan
    return np.median(v)
print('== every 8 chunks (before the write step, weights of the arm), median over tokens of a seed, mean over seeds: place tokens | name tokens; NLL under C0 | SAMEVIS | arm mask; ||e|| under the same; correct logit (SAMEVIS)')
for arm in ('Mrand','M2As-q','M2As-s'):
    print(f'  {arm}:')
    for ck in range(8,129,8):
        row=[]
        for tok in ('place','name'):
            nl=[np.nanmean([dg(arm,ck,s,m_,'nll',tok) for s in SEEDS]) for m_ in ('C0','SAMEVIS','ARM')];en=[np.nanmean([dg(arm,ck,s,m_,'en',tok) for s in SEEDS]) for m_ in ('C0','SAMEVIS','ARM')];gl=np.nanmean([dg(arm,ck,s,'SAMEVIS','gold',tok) for s in SEEDS])
            row.append(f'{tok}: NLL {nl[0]:.2f}|{nl[1]:.4f}|{nl[2]:.4f} |e| {en[0]:.3f}|{en[1]:.4f}|{en[2]:.4f} logit {gl:.1f}')
        print(f'    chunk {ck:3d}: '+' || '.join(row))
print('== checks of the smoke runs:')
for f in sorted(glob.glob(D+'/smoke/kvmask_s231.json')):print('  ',json.load(open(f)).get('checks'))
