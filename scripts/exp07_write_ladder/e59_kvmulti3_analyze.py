"""Analysis of the 8-seed bundle (jobs 1344t1b = 1351, 1344t2 = 1352 E2All, 1344w3 = 1353; CPU; new file; thread-server 11/25; descriptive: values and intervals over seeds, same-direction seeds, no verdict wording, no threshold). Arms P0s, P1s from job 1344t0 (kvmulti dir; exposures 8/16/32/64/96/128), S2As-u and E2As from kvmulti2/kvmulti2_s<seed>.json (jobs 1350 + 1351), E2All from kvmulti2/kvmulti2_e2all_s<seed>.json (job 1352). Per seed, arm and exposure: R1 matched binding / familiarity / margin change (Delta over W0), also by the sentence order of the chunk just written; main = R1 matched binding at 128: P1s - E2All (seed-paired: t interval and seed bootstrap) with E2All - P0s, E2All - E2As, P1s - E2As beside it; ladder S2As-u - P0s, E2As - P0s, E2All - E2As, P1s - E2All per exposure; (familiarity, matched binding) pairs per arm, exposure and seed; R2 at 128; in-window reads R2 / R2m / R2a (job 1349 for P0s, P1s; job 1353 for the others); diagnostics: clip coefficient, gradient norm, original CE per interval, place-token NLL (C0 / SAMEVIS), ||e_C0||, correct logit every 8 chunks, E2All token-kind norms, divergence flag.
  python e59_kvmulti3_analyze.py --dir KVMULTI2_DIR --base KVMULTI_DIR --win KVWIN_DIR --win2 KVWIN2_DIR"""
import sys,json,glob,random,os,numpy as np
from scipy import stats
a=sys.argv;g=lambda k:a[a.index(k)+1] if k in a else None
D=g('--dir');BD=g('--base');WD=g('--win');WD2=g('--win2')
SEEDS=sorted(set(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/kvmulti2_s*.json')) & set(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/kvmulti2_e2all_s*.json')))
n=len(SEEDS);R0={s:json.load(open(f'{BD}/kvmulti_s{s}.json')) for s in SEEDS};R2={s:json.load(open(f'{D}/kvmulti2_s{s}.json')) for s in SEEDS};RA={s:json.load(open(f'{D}/kvmulti2_e2all_s{s}.json')) for s in SEEDS}
EXPO=[8,16,32,48,64,80,96,112,128];LAST=128;ARMS=('P0s','S2As-u','E2As','E2All','P1s')
def src(arm,s):return R0[s]['arms'][arm] if arm in ('P0s','P1s') else (RA[s]['arms'][arm] if arm=='E2All' else R2[s]['arms'][arm])
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
    for arm in ARMS:
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
print(f'8-seed bundle: seeds {SEEDS} (n={n}); Delta over W0; mean over seeds [95% t] same-direction seeds; P0s, P1s from job 1344t0 (empty where not read)')
print('== MAIN (R1 matched binding at 128, seed-paired): '+' | '.join(f'{nm} {tci(d)} {bsi(d)}' for nm,d in (('P1s - E2All',col('P1s',128,0)-col('E2All',128,0)),('E2All - P0s',col('E2All',128,0)-col('P0s',128,0)),('E2All - E2As',col('E2All',128,0)-col('E2As',128,0)),('P1s - E2As',col('P1s',128,0)-col('E2As',128,0)))))
for o,on in ((None,'all'),(0,'place first'),(1,'name first')):
    print(f'== R1, sentence order of the chunk just written: {on}')
    for mi,mn in enumerate(MET):
        print(f'  [{mn}]')
        for arm in ARMS:print(f'    {arm:7s}: '+' | '.join(f'e{e} {tci(col(arm,e,mi,o))}' for e in EXPO if np.isfinite(col(arm,e,mi,o)).any()))
print('== LADDER (same seeds, paired): S2As-u - P0s | E2As - P0s | E2All - E2As | P1s - E2All, per exposure')
for mi,mn in enumerate(MET):
    print(f'  [{mn}]')
    for e in EXPO:
        c={ar:col(ar,e,mi) for ar in ARMS}
        if not all(np.isfinite(c[ar]).any() for ar in ARMS):continue
        parts=[('S2As-u - P0s',c['S2As-u']-c['P0s']),('E2As - P0s',c['E2As']-c['P0s']),('E2All - E2As',c['E2All']-c['E2As']),('P1s - E2All',c['P1s']-c['E2All'])]
        print(f'    e={e:3d}: '+' | '.join(f'{nm} {tci(d)}'+(f' {bsi(d)}' if (mi==0 and e==LAST) else '') for nm,d in parts))
print('== (familiarity, matched binding) per arm, exposure and seed')
for arm in ARMS:
    print(f'  {arm}:')
    for s in SEEDS:print(f'    seed {s}: '+' '.join(f'e{e}:({V[(arm,e,s)][1]:.2f},{V[(arm,e,s)][0]:.2f})' for e in EXPO if (arm,e,s) in V))
print(f'== R2 (changed read) at exposure {LAST}: matched binding / familiarity / margin change')
for mi,mn in enumerate(MET):print(f'  [{mn}] '+' | '.join(f'{arm} {tci([V2[(arm,s)][mi] for s in SEEDS if (arm,s) in V2])}' for arm in ARMS))
# in-window reads
W1={s:json.load(open(f'{WD}/kvwin_s{s}.json')) for s in SEEDS if os.path.exists(f'{WD}/kvwin_s{s}.json')};W2={s:json.load(open(f'{WD2}/kvwin2_s{s}.json')) for s in SEEDS if os.path.exists(f'{WD2}/kvwin2_s{s}.json')}
def wrec(arm,s):
    r=W1.get(s) if arm in ('P0s','P1s') else W2.get(s);return None if r is None else r
if W1 or W2:
    print('== in-window reads at 128 (R2 no mask, R2m own sentence hidden, R2a all 16 read-fact sentences hidden), Delta over the W0 score of the same read')
    for mi,mn in enumerate(MET):
        print(f'  [{mn}]')
        for v in ('R2','R2m','R2a'):
            row=[]
            for arm in ARMS:
                x=[per_fact(np.asarray(r['arms'][arm][v])-np.asarray(r['W0'][v]),r['W0'][v],s).mean(0)[mi] for s in SEEDS for r in [wrec(arm,s)] if r is not None and arm in r['arms']];row.append(f'{arm} {tci(x)}')
            print(f'    {v:4s}: '+' | '.join(row))
    print('  matched binding of the in-window reads, differences (seed-paired):')
    for v in ('R2','R2m','R2a'):
        def cc(arm):return np.array([per_fact(np.asarray(r['arms'][arm][v])-np.asarray(r['W0'][v]),r['W0'][v],s).mean(0)[0] if (r:=wrec(arm,s)) is not None and arm in r['arms'] else np.nan for s in SEEDS])
        print(f'    {v:4s}: '+' | '.join(f'{nm} {tci(cc(x)-cc(y))}' for nm,x,y in (('P1s - E2All','P1s','E2All'),('E2All - P0s','E2All','P0s'),('E2All - E2As','E2All','E2As'),('P1s - P0s','P1s','P0s'))))
INT=[(1,16),(17,32),(33,64),(65,96),(97,128)]
print('== per-chunk values, interval means over chunks, mean over seeds [per-seed min..max]')
for key,nm in (('coef','clip coefficient'),('gnorm','pre-clip gradient norm'),('ce','original chunk CE')):
    print(f'  [{nm}]')
    for arm in ARMS:
        row=[]
        for lo,hi in INT:
            v=np.array([np.mean([x[key] for x in src(arm,s)['steps'][lo-1:hi]]) for s in SEEDS]);row.append(f'c{lo}-{hi}: {v.mean():.3f} [{v.min():.3f}..{v.max():.3f}]')
        print(f'    {arm:7s}: '+' | '.join(row))
print('== divergence (not finite or last-8 CE > 4.0; nothing excluded):')
for arm in ARMS:
    st=[src(arm,s)['steps'] for s in SEEDS];print(f'  {arm}: last-8 CE {[round(float(np.mean([x["ce"] for x in t[-8:]])),2) for t in st]} diverged {[bool((not all(np.isfinite([x["gnorm"] for x in t]))) or np.mean([x["ce"] for x in t[-8:]])>4.0) for t in st]}')
def dg(arm,ck,s,key):
    d=src(arm,s)['diag'].get(str(ck))
    if d is None:return np.nan
    own=d['cond_own']
    if key=='nll_c0':return np.median(d['nll_own' if own=='C0' else 'nll_other'])
    if key=='nll_sv':return np.median(d['nll_own' if own=='SAMEVIS' else 'nll_other'])
    if key=='en_c0':return np.median(d['en_own' if own=='C0' else 'en_other'])
    if key=='logit':return np.median(d['gold_logit_sv']) if 'gold_logit_sv' in d else np.nan
print('== every 8 chunks, place tokens, median over tokens of a seed, mean over seeds: NLL C0 | NLL SAMEVIS | ||e_C0|| | correct logit (SAMEVIS; "-" where not stored)')
for arm in ARMS:
    print(f'  {arm}:')
    for ck in range(8,129,8):
        v={k:np.array([dg(arm,ck,s,k) for s in SEEDS]) for k in ('nll_c0','nll_sv','en_c0','logit')}
        if not np.isfinite(v['nll_c0']).any():continue
        lg='-' if not np.isfinite(v['logit']).any() else f"{np.nanmean(v['logit']):6.2f}"
        print(f'    chunk {ck:3d}: NLL C0 {np.nanmean(v["nll_c0"]):6.2f} | NLL SAMEVIS {np.nanmean(v["nll_sv"]):7.4f} | ||e_C0|| {np.nanmean(v["en_c0"]):6.3f} | logit {lg}')
print('== E2All: mean ||e_C0|| and ||e_SV|| by token kind every 8 chunks (mean over seeds of the chunk means; n = tokens of one chunk)')
for ck in range(8,129,8):
    row=[]
    for kd in ('place','name','frame','extra','filler'):
        a_=[RA[s]['arms']['E2All']['diag'][str(ck)]['kinds'][kd] for s in SEEDS];m0=[x['en_c0'] for x in a_ if x['en_c0'] is not None];m1=[x['en_sv'] for x in a_ if x['en_sv'] is not None]
        row.append(f'{kd}: {np.mean(m0):.3f}/{np.mean(m1):.4f} (n {int(np.mean([x["n"] for x in a_]))})' if m0 else f'{kd}: -')
    print(f'  chunk {ck:3d}: '+' | '.join(row))
print('== E2All correct logit (SAMEVIS) trend, chunks 1/32/64/96/128: '+' | '.join(f'c{c}: {np.mean([RA[s]["arms"]["E2All"]["steps"][c-1]["gold_logit"] for s in SEEDS]):.2f}' for c in (1,32,64,96,128)))
print('== validation of the first chunk and chunk 8 (smoke runs):')
for f in sorted(glob.glob(D+'/smoke1351/kvmulti2_s231.json')+glob.glob(D+'/smoke1352/kvmulti2_e2all_s231.json')):
    r=json.load(open(f));print('  ',f.split('/')[-2],r.get('validation'),r.get('validation_chunk8'))
