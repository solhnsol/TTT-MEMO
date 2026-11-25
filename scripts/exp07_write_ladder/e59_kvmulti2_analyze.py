"""Analysis of job 1344t1 (e59_kvmulti2.py; CPU; new file; thread-server 11/25; descriptive: values and intervals over seeds, same-direction seeds, no verdict wording, no threshold). Arms P0s and P1s are taken from job 1344t0 (kvmulti dir, exposures 8/16/32/64/96/128; 48/80/112 empty), S2As-u and E2As from job 1344t1. Per seed, arm and exposure: R1 matched binding / familiarity (exposed wrong minus never exposed) / margin change (Delta over W0) of the 16 read facts, also by the sentence order of the chunk just written; the three ladder differences S2As-u - P0s, E2As - S2As-u, P1s - E2As (same seeds, paired: mean [95% t] and seed bootstrap); per arm the (familiarity, matched binding) pairs per exposure and seed; diagnostics: per chunk pre-clip gradient norm, clip coefficient, original chunk CE, mean norm of the inserted cotangent; every 8 chunks NLL (SAMEVIS, C0), ||e_C0||, ||e_SV||, cos, correct logit and correct minus largest wrong logit (SAMEVIS) of the A tokens; divergence flag (not finite or last-8 CE > 4.0; nothing excluded); R2 at the last exposure.
  python e59_kvmulti2_analyze.py --dir D --base KVMULTI_DIR"""
import sys,json,glob,random,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];BD=a[a.index('--base')+1]
R2=[json.load(open(f)) for f in sorted(glob.glob(D+'/kvmulti2_s*.json'))];SEEDS=[r['seed'] for r in R2];n=len(SEEDS);R0={r['seed']:r for r in (json.load(open(f'{BD}/kvmulti_s{s}.json')) for s in SEEDS)}
EXPO=R2[0]['expo'];ARMS=('P0s','S2As-u','E2As','P1s');LAST=max(EXPO)
def src(arm,s):return R0[s]['arms'][arm] if arm in ('P0s','P1s') else [r for r in R2 if r['seed']==s][0]['arms'][arm]
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
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);mar=lambda X:X[:,0]-X[:,1:12].max(1);out=np.zeros((16,3))
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];out[f,0]=dS[f,0]-np.mean(oth) if oth else np.nan
    out[:,1]=dS[:,1:12].mean(1)-dS[:,12:16].mean(1);out[:,2]=mar(W0+dS)-mar(W0);return out
MET=('matched binding','familiarity','margin change')
V={};VO={}
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
def col(arm,e,mi,o=None):
    out=[]
    for s in SEEDS:
        v=V.get((arm,e,s)) if o is None else VO.get((arm,e,s,o));out.append(np.nan if v is None else v[mi])
    return np.array(out)
print(f'1344t1: seeds {SEEDS} (n={n}), exposures {EXPO}; Delta over W0; mean over seeds [95% t] same-direction seeds; P0s, P1s from job 1344t0 (empty where not read)')
for o,on in ((None,'all'),(0,'place first'),(1,'name first')):
    print(f'== R1, sentence order of the chunk just written: {on}')
    for mi,mn in enumerate(MET):
        print(f'  [{mn}]')
        for arm in ARMS:print(f'    {arm:7s}: '+' | '.join(f'e{e} {tci(col(arm,e,mi,o))}' for e in EXPO if np.isfinite(col(arm,e,mi,o)).any()))
print('== LADDER (matched binding, same seeds paired): S2As-u - P0s | E2As - S2As-u | P1s - E2As, per exposure; seed bootstrap in brackets')
rng=np.random.default_rng(0)
def bsi(d):
    d=d[~np.isnan(d)];return '' if len(d)<2 else f'bootstrap [{np.percentile([d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)],2.5):6.3f},{np.percentile([d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)],97.5):6.3f}]'
for mi,mn in enumerate(MET):
    print(f'  [{mn}]')
    for e in EXPO:
        c={ar:col(ar,e,mi) for ar in ARMS}
        if not all(np.isfinite(c[ar]).any() for ar in ARMS):continue
        parts=[('S2As-u - P0s',c['S2As-u']-c['P0s']),('E2As - S2As-u',c['E2As']-c['S2As-u']),('P1s - E2As',c['P1s']-c['E2As']),('(P1s - P0s)',c['P1s']-c['P0s']),('(E2As - P0s)',c['E2As']-c['P0s'])]
        print(f'    e={e:3d}: '+' | '.join(f'{nm} {tci(d)} {bsi(d)}' if (mi==0 and e==LAST) else f'{nm} {tci(d)}' for nm,d in parts))
print('== (familiarity, matched binding) per arm, exposure and seed')
for arm in ARMS:
    print(f'  {arm}:')
    for s in SEEDS:print(f'    seed {s}: '+' '.join(f'e{e}:({V[(arm,e,s)][1]:.2f},{V[(arm,e,s)][0]:.2f})' for e in EXPO if (arm,e,s) in V))
print(f'== R2 (changed read) at exposure {LAST}: matched binding / familiarity / margin change')
for mi,mn in enumerate(MET):
    row=[]
    for arm in ARMS:
        x=[]
        for s in SEEDS:
            rec=src(arm,s);d=rec['reads'][str(LAST)]
            if 'dS2' in d:x.append(per_fact(d['dS2'],R0[s]['W0_R2'],s).mean(0)[mi])
        row.append(f'{arm} {tci(x)}')
    print(f'  [{mn}] '+' | '.join(row))
print('== per-chunk diagnostics (mean over chunks and seeds): pre-clip gradient norm, clip coefficient, original chunk CE, inserted cotangent norm (S2As-u / E2As only), last-8 CE per seed, divergence flag')
for arm in ARMS:
    st=[src(arm,s)['steps'] for s in SEEDS];ce8=[round(float(np.mean([x['ce'] for x in t[-8:]])),2) for t in st];dv=[bool((not all(np.isfinite([x['gnorm'] for x in t]))) or np.mean([x['ce'] for x in t[-8:]])>4.0) for t in st]
    cot=f" cot norm {np.mean([x['cot_norm'] for t in st for x in t]):.3f} bad-ratio tokens {sum(x['nbad'] for t in st for x in t)}" if arm in ('S2As-u','E2As') else ''
    print(f'  {arm}: gnorm {np.mean([x["gnorm"] for t in st for x in t]):.4f} coef {np.mean([x["coef"] for t in st for x in t]):.4f} CE {np.mean([x["ce"] for t in st for x in t]):.3f}{cot} | last-8 CE {ce8} | diverged {dv}')
print('== A-token trajectory (new arms; every 8 chunks, mean over seeds of per-seed medians): correct logit (SAMEVIS), correct minus largest wrong logit (SAMEVIS), NLL (SAMEVIS | C0), ||e_C0||, ||e_SV||, cos')
for arm in ('S2As-u','E2As'):
    print(f'  {arm}:')
    for c8 in [str(k) for k in range(8,LAST+1,8)]:
        rows=[src(arm,s)['diag'].get(c8) for s in SEEDS];rows=[r for r in rows if r]
        if not rows:continue
        md=lambda key:np.mean([np.median(r[key]) for r in rows])
        print(f'    chunk {c8:>3s}: gold logit {md("gold_logit_sv"):7.2f} | margin {md("margin_sv"):7.2f} | NLL {md("nll_own"):6.2f} | {md("nll_other"):6.2f} | |e_C0| {md("en_other"):6.3f} |e_SV| {md("en_own"):8.5f} | cos {md("cos"):.3f}')
print('== chunk-mean correct logit (SAMEVIS) trend, all arms with step records, mean over seeds, chunks 1/32/64/96/128')
for arm in ('S2As-u','E2As'):
    print(f'  {arm}: '+' | '.join(f'c{c}: {np.mean([src(arm,s)["steps"][c-1]["gold_logit"] for s in SEEDS]):.2f}' for c in (1,32,64,96,128) if all(len(src(arm,s)['steps'])>=c for s in SEEDS)))
V0=[r['validation'] for r in R2 if r.get('validation')]
print('== validation on the real model at the first chunk:',V0[:1] if V0 else 'only the smoke runs it (--validate)')
