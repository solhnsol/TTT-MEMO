"""Analysis of job 1344s (e59_kvlogit.py; CPU; new file; thread-server 11/25; descriptive: values, intervals, sign-consistent seeds; NO verdict wording; thresholds are not defined). Input kvlogit_s<seed>.json (stored per read: dS = candidate scores of the arm's one-step state minus W0 of the SAME context; 16 facts x 16 candidates: 0 gold, 1..11 exposed decoys, 12..15 never-exposed places). Per arm and read (R1 neutral out-of-window, R2 just-written chunk as context): MARGIN = change of (gold - max exposed wrong), BINDING = gold change - mean exposed decoy change, FAMILIARITY = mean exposed decoy change - mean never-exposed change; per seed the mean over the measured chunks and the 16 facts. Main: R(arm) = (D(arm) - D(P0)) / (D(P1) - D(P0)) with a seed bootstrap interval (5000 resamples of the seeds, ratio of means) and the paired t interval of D(arm) - D(P0); X2A: reproduction ratio (D(P1) - D(X2A)) / Gap. If the t interval of Gap = D(P1) - D(P0) contains 0 no R is given for that metric (differences only).
  python e59_kvlogit_analyze.py --dir D [--ref experiments/exp07_write_ladder/kvpos]"""
import sys,json,glob,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];REF=a[a.index('--ref')+1] if '--ref' in a else None
R=[json.load(open(f)) for f in sorted(glob.glob(D+'/kvlogit_s*.json'))];SEEDS=[r['seed'] for r in R];n=len(R)
ARMS=['P0','P1','S2A','S2F','S2O','S2A-nat','X2A'];MET=('margin','binding','familiarity');READS=('R1','R2')
def tci(d):
    d=np.asarray(d,float);m=d.mean()
    if len(d)<2:return f'{m:8.3f}'
    h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:8.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def vals(dS,W0):
    dS=np.asarray(dS);W0=np.asarray(W0);S=W0+dS
    mar=lambda X:X[:,0]-X[:,1:12].max(1)
    return np.array([(mar(S)-mar(W0)).mean(),(dS[:,0]-dS[:,1:12].mean(1)).mean(),(dS[:,1:12].mean(1)-dS[:,12:16].mean(1)).mean()])
V={}                                                                   # (arm,read,seed) -> [margin,binding,familiarity] mean over chunks
for r in R:
    for arm in ARMS:
        for rd_ in READS:
            xs=[]
            for row in r['rows']:
                if arm in row['arms']:
                    A=row['arms'][arm];W=r['W0_R1'] if rd_=='R1' else A['W0_R2'];xs.append(vals(A['dS1' if rd_=='R1' else 'dS2'],W))
            if xs:V[(arm,rd_,r['seed'])]=np.mean(xs,0)
def col(arm,rd_,mi):return np.array([V[(arm,rd_,s)][mi] for s in SEEDS if (arm,rd_,s) in V])
print(f'1344s: seeds {SEEDS} (n={n}), chunks {R[0]["chunks"]}, cap {R[0]["cap"]}; D = change over W0 of the same context (summed log p); columns: mean [95% t interval] sign-consistent seeds')
for rd_ in READS:
    print(f'== raw values, read {rd_}')
    for mi,mn in enumerate(MET):
        print(f'  [{mn}]');[print(f'    {arm:8s} {tci(col(arm,rd_,mi))}') for arm in ARMS if len(col(arm,rd_,mi))]
rng=np.random.default_rng(0)
def boot_ratio(num,den,B=5000):
    idx=rng.integers(0,len(num),size=(B,len(num)));m=num[idx].mean(1);d=den[idx].mean(1);r=m/np.where(np.abs(d)<1e-12,np.nan,d);return np.nanpercentile(r,[2.5,97.5])
for rd_ in READS:
    print(f'== recovery ratios, read {rd_}: R = (D(arm) - D(P0)) / (D(P1) - D(P0)); X2A: (D(P1) - D(X2A)) / Gap')
    for mi,mn in enumerate(MET):
        g=col('P1',rd_,mi)-col('P0',rd_,mi);h=stats.t.ppf(.975,n-1)*g.std(ddof=1)/np.sqrt(n);lo,hi=g.mean()-h,g.mean()+h;ok=not(lo<=0<=hi)
        print(f'  [{mn}] Gap = D(P1)-D(P0) {tci(g)}'+('' if ok else '   (interval contains 0: no R, differences only)'))
        for arm in ARMS[2:]:
            if len(col(arm,rd_,mi))<n:continue
            d=col(arm,rd_,mi)-col('P0',rd_,mi) if arm!='X2A' else col('P1',rd_,mi)-col(arm,rd_,mi)
            s=f'    {arm:8s} '+('D(arm)-D(P0) ' if arm!='X2A' else 'D(P1)-D(X2A) ')+tci(d)
            if ok:b=boot_ratio(d,g);s+=f' | R {d.mean()/g.mean():7.3f} [{b[0]:7.3f},{b[1]:7.3f}]'
            print(s)
print('== step diagnostics (mean over seeds and chunks): pre-clip global norm, applied clip coefficient, update norm = coef x norm')
for arm in ARMS:
    xs=[r_['arms'][arm] for r in R for r_ in r['rows'] if arm in r_['arms']]
    if xs:print(f'  {arm:8s} gn {np.mean([x["gn"] for x in xs]):9.4f} coef {np.mean([x["coef"] for x in xs]):7.4f} upd {np.mean([x["upd"] for x in xs]):9.4f}')
print('== set-only gradient norm (weights outside the set = 0): arm / C0 plain and arm / SAMEVIS plain (mean over seeds and chunks); cap hit fraction and realized/target ||e|| (median of the per-seed medians) by token kind')
for arm in ('S2A','S2F','S2O','X2A'):
    st=[r_['stats'][arm] for r in R for r_ in r['rows'] if arm in r_['stats']]
    if not st:continue
    s=f'  {arm:5s} setgn/C0 {np.mean([x["setgn"]/x["setgn_C0"] for x in st]):7.3f}'+(f' setgn/SV {np.mean([x["setgn"]/x["setgn_SV"] for x in st]):7.3f}' if 'setgn_SV' in st[0] else '')
    for kd in ('ans','name','frame','extra'):
        c=[x[f'caphit_{kd}'] for x in st if f'caphit_{kd}' in x];q=[x[f'ratio_{kd}'][1] for x in st if f'ratio_{kd}' in x]
        if c:s+=f' | {kd}: cap {np.mean(c):.3f} ratio med {np.median(q):.3f}'
    print(s)
print('== validation / checks')
for r in R:
    print(f'  s{r["seed"]}: '+'; '.join(str({k:v for k,v in x.items()}) for x in r['validation']))
    for row in r['rows']:print(f'     k={row["k"]}: '+str(row['check']))
dg=[(r['seed'],row['k'],row['diag']) for r in R for row in r['rows'] if row['diag']]
if dg:
    print('== token diagnostics (A/F/O tokens pooled over seeds and chunks): median [quartiles]')
    for kd in ('ans','name','frame','extra'):
        cols={c:[] for c in ('en_SV','en_C0','nll_SV','nll_C0','cos','mass_SV','mass_C0')}
        for _,_,d in dg:
            for i,k_ in enumerate(d['kind']):
                if k_==kd:
                    for c in cols:cols[c].append(d[c][i])
        if cols['en_SV']:print(f'  {kd:6s} n={len(cols["en_SV"]):5d} '+' | '.join(f'{c} {np.median(v):.3g} [{np.percentile(v,25):.3g},{np.percentile(v,75):.3g}]' for c,v in cols.items()))
if REF:
    print('== reproduction of the real steps (P0 = ACT[SAMEVIS], P1 = ACT[C0]) against job 1344p (same seed / chunk): max relative difference of pre-clip norm and clip coefficient')
    for arm,nm in (('P0','ACT[SAMEVIS]'),('P1','ACT[C0]')):
        dd=[];dc=[]
        for r in R:
            try:rf=json.load(open(f'{REF}/kvpos_s{r["seed"]}.json'))
            except Exception:continue
            ref={x['k']:x['combos'][nm] for x in rf['rows']}
            for row in r['rows']:
                if row['k'] in ref and arm in row['arms']:dd.append(abs(row['arms'][arm]['gn']-ref[row['k']]['gnorm'])/ref[row['k']]['gnorm']);dc.append(abs(row['arms'][arm]['coef']-ref[row['k']]['coef'])/ref[row['k']]['coef'])
        if dd:print(f'  {arm}: n={len(dd)} gn {max(dd):.2e} coef {max(dc):.2e}')
