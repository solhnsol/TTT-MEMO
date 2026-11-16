"""E57 job 1280-cache (job 1302) identity check and analysis (CPU). PRE-REGISTRATION (rules fixed by the principal investigator 11/16 via the core-claims thread; the sha256 of this file is frozen in a signed tag BEFORE the first seed is run). The equivalence margin is fixed IN ADVANCE as an absolute value derived from the 1280c data, NOT from this run: delta = 0.21 nat / fact on the primary (logsumexp) margin (0.25 x 0.830, 0.830 = the 1280c gain G = M_NATMASK - M_NATS, 30 seeds) and delta = 0.21 nat / fact on the secondary (largest-wrong) margin (0.25 x 0.855 = 0.214, rounded to 0.21; 0.855 = the 1280c secondary G). No threshold is changed after a result has been seen.
PURPOSE: job 1280c measured the one-step margins of NATS, NATMASK (ALLQ256 on the cached K,V), RECOMP and RECOMP_PL, but the CACHE path (the mask applied to the cached window K,V without recomputing the prefix) was measured only as NLL (E_cache = a - nc, placebo a_P - nc), never as a margin. This job reads the one-step margin of the cached oracle (CACHE_A: OracleUV keep, condition A of 1280) and of the cached placebo (CACHE_AP: PlaceboOracleUV keep, condition A_P) at the SAME branch chunks of the SAME seeds, together with NATS and NATMASK, so that the cached selective hiding can be compared with the band mask on margins.
DESIGN: seeds 101..108 (the first 8 seeds of 1280c; their natural trajectories and branch chunks are reproduced bit for bit, which the identity check enforces), cell v3_32_4_a16, K0 = 0 irrelevant (NATS trajectory), branch chunks = those of results_s<seed>.json of 1280c, one-step margins from W_t before the write of chunk k, 8 facts per chunk read with the 1280 reader (16 candidates, neutral 8192 tail). Seed value = mean over the facts of a chunk, then mean over the seed's branch chunks (e57_1280_margin.aggregate). Primary margin = logsumexp (log p(gold) - logsumexp(log p wrong)); secondary = largest wrong (the 1283 convention), reported with the same contrasts.
IDENTITY CHECK (aborts the job, exit 13, nothing is analysed): for every seed and branch chunk (i) the isolation gnorm / coef of the new NATS step equal the logged trajectory values (|d gnorm| <= 1e-6 max(1,|gnorm|), |d coef| <= 1e-6, as 1280); (ii) the NATS and NATMASK primary and secondary margins equal the stored 1280c values and (iii) the NC, A, A_P NLL arrays on the F tokens equal the stored 1280c arrays, in each case EXACTLY (deterministic XLA ops, same code path; the maximum absolute difference is reported). A mismatch means the run is not tied to the 1280c trajectories: the result is not interpreted as a 1280c extension and a rerun needs a new sentence of the principal investigator.
CONTRASTS (per seed values, unit = seed, n = 8, primary margin, nat / fact): C1 = M_CACHE_A - M_NATS, C2 = M_CACHE_AP - M_NATS, C3 = M_CACHE_A - M_CACHE_AP, C4 = M_CACHE_A - M_NATMASK; G = M_NATMASK - M_NATS (reference). For each: mean, two-sided 95% t interval, one-sided 95% t bounds, exact sign-flip p (min 1/256 at n = 8). Ratios of means C1 / G and (C1 - C2) / G are descriptive.
RULE (the only pass / fail rule): TOST of EQUIVALENCE of the cached selective hiding and the band mask on the per-seed paired differences C4 = M_CACHE_A - M_NATMASK, for the primary margin and, separately, for the secondary margin: equivalent iff the 90% two-sided t interval (df = 7) of C4 lies inside [-0.21, +0.21]; if it lies completely outside on one side the arms DIFFER; otherwise INCONCLUSIVE. The two-sided 95% interval of C4 is reported as well. 'Not equivalent' or 'inconclusive' is not evidence of no effect. The contrasts C1 (A - NATS), C2 (A_P - NATS), C3 (A - A_P) and G are REPORTED with their intervals and exact sign-flip p-values; they carry no threshold and no pass / fail rule.
  python3 e57_1280c_cache_analyze.py --check --dir D --ref R --seeds 101,... | --analyze --dir D --seeds 101,... --json J | --selftest"""
import sys,json,argparse,itertools,math,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_1280_margin as MG
DELTA_EQ=dict(margin_primary=0.21,margin_secondary=0.21)          # fixed in advance (0.25 x the 1280c gain G: 0.830 primary, 0.855 secondary)
TQ2={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131}
TQ1={1:6.314,2:2.920,3:2.353,4:2.132,5:2.015,6:1.943,7:1.895,8:1.860,9:1.833,10:1.812,11:1.796,12:1.782,13:1.771,14:1.761,15:1.753}
TQ90={1:6.314,2:2.920,3:2.353,4:2.132,5:2.015,6:1.943,7:1.895,8:1.860,9:1.833,10:1.812,11:1.796,12:1.782,13:1.771,14:1.761,15:1.753}   # two-sided 90% = one-sided 95%
def load(D,s):return json.load(open(pathlib.Path(D)/f'results_cache_s{s}.json'))
def check(D,R,seeds):
    bad=[];rep={}
    for s in seeds:
        n=load(D,s);r=json.load(open(pathlib.Path(R)/f'results_s{s}.json'));ks=sorted(n['results'],key=int);worst=0.0
        if not n['isolation_ok']:bad.append((s,'isolation gnorm / coef differ from the trajectory log'))
        assert [int(k) for k in ks]==[int(k) for k in r['branches']] or set(map(int,ks))<=set(map(int,r['branches'])),('branch chunks differ',s)
        for k in ks:
            a=n['results'][k];b=r['results'][k] if k in r['results'] else r['results'][str(k)]
            for an in ('NATS','NATMASK'):
                for key in ('margin_primary','margin_secondary'):
                    d=float(np.max(np.abs(np.asarray(a[key][an])-np.asarray(b[key][an]))));worst=max(worst,d)
                    if d!=0.0:bad.append((s,k,an,key,d))
            for cn in ('NC','A','A_P'):
                d=float(np.max(np.abs(np.asarray(a['nll'][cn])-np.asarray(b['nll'][cn])))) if len(a['nll'][cn])==len(b['nll'][cn]) else float('inf');worst=max(worst,d)
                if d!=0.0:bad.append((s,k,'nll',cn,d))
        rep[s]=worst
    return dict(ok=not bad,max_abs_diff_per_seed=rep,mismatches=bad[:20])
def seed_values(D,s,key):
    n=load(D,s);ks=sorted(n['results'],key=int);v={}
    for an in ('NATS','NATMASK','CACHE_A','CACHE_AP'):v[an]=MG.aggregate([n['results'][k][key][an] for k in ks])
    return v
def signflip_p(v):
    d=np.asarray(v,float);m=len(d);obs=d.mean();cnt=0
    for s in itertools.product((1,-1),repeat=m):cnt+=(np.dot(s,d)/m)>=obs-1e-15
    return cnt/2**m
def contrast(x):
    x=np.asarray(x,float);n=len(x);m=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(n));t2=TQ2[n-1];t1=TQ1[n-1]
    return dict(mean=m,se=se,ci95_two_sided=[m-t2*se,m+t2*se],lower_onesided_95=m-t1*se,upper_onesided_95=m+t1*se,ci90_two_sided=[m-TQ90[n-1]*se,m+TQ90[n-1]*se],p_signflip_onesided_gt0=signflip_p(x),per_seed=[float(v) for v in x])
def analyze(D,seeds):
    out={'seeds':list(seeds),'delta_eq':DELTA_EQ,'status':'rules fixed in advance'}
    for key,name in (('margin_primary','primary'),('margin_secondary','secondary')):
        V=[seed_values(D,s,key) for s in seeds];A=lambda an:np.array([v[an] for v in V])
        C=dict(C1_A_minus_NATS=A('CACHE_A')-A('NATS'),C2_AP_minus_NATS=A('CACHE_AP')-A('NATS'),C3_A_minus_AP=A('CACHE_A')-A('CACHE_AP'),C4_A_minus_NATMASK=A('CACHE_A')-A('NATMASK'),G_NATMASK_minus_NATS=A('NATMASK')-A('NATS'))
        res={k:contrast(v) for k,v in C.items()};res['means']={an:float(A(an).mean()) for an in ('NATS','NATMASK','CACHE_A','CACHE_AP')}
        res['ratio_C1_over_G']=float(C['C1_A_minus_NATS'].mean()/C['G_NATMASK_minus_NATS'].mean());res['ratio_C1_minus_C2_over_G']=float((C['C1_A_minus_NATS']-C['C2_AP_minus_NATS']).mean()/C['G_NATMASK_minus_NATS'].mean())
        ci=res['C4_A_minus_NATMASK']['ci90_two_sided'];de=DELTA_EQ[key]
        res['TOST_A_vs_NATMASK']=dict(delta=de,ci90=ci,ci95=res['C4_A_minus_NATMASK']['ci95_two_sided'],verdict=('EQUIVALENT' if (ci[0]>=-de and ci[1]<=de) else ('DIFFER' if (ci[0]>de or ci[1]<-de) else 'INCONCLUSIVE')))
        out[name]=res
    return out
def selftest():
    import tempfile
    rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        R=pathlib.Path(td)/'ref';N=pathlib.Path(td)/'new';R.mkdir();N.mkdir();seeds=list(range(101,109))
        for s in seeds:
            ref={'branches':[10,12],'results':{}};new={'seed':s,'branches':[10,12],'isolation_ok':True,'results':{}}
            for k in (10,12):
                mk=lambda base:{an:(base+rng.normal(0,.1,8)).tolist() for an in ('NATS',)}
                P={'NATS':(-4+rng.normal(0,.2,8)).tolist()};P['NATMASK']=(np.array(P['NATS'])+0.8).tolist();P['CACHE_A']=(np.array(P['NATS'])+0.78).tolist();P['CACHE_AP']=(np.array(P['NATS'])+0.02).tolist()
                nll={cn:(2+rng.normal(0,.1,30)).tolist() for cn in ('NC','A','A_P')}
                ref['results'][str(k)]=dict(margin_primary={a:P[a] for a in ('NATS','NATMASK')},margin_secondary={a:P[a] for a in ('NATS','NATMASK')},nll=nll)
                new['results'][str(k)]=dict(margin_primary=P,margin_secondary=P,nll=nll)
            json.dump(ref,open(R/f'results_s{s}.json','w'));json.dump(new,open(N/f'results_cache_s{s}.json','w'))
        c=check(N,R,seeds);assert c['ok'] and all(v==0 for v in c['max_abs_diff_per_seed'].values()),c
        a=analyze(N,seeds);p=a['primary'];assert abs(p['C1_A_minus_NATS']['mean']-0.78)<0.05 and abs(p['C4_A_minus_NATMASK']['mean']+0.02)<0.05 and p['TOST_A_vs_NATMASK']['verdict']=='EQUIVALENT' and 'R1_selective_beyond_placebo' not in p
        assert abs(p['C3_A_minus_AP']['mean']-0.76)<0.05
        d=json.load(open(N/'results_cache_s101.json'));d['results']['10']['nll']['A'][0]+=1e-9;json.dump(d,open(N/'results_cache_s101.json','w'));c2=check(N,R,seeds);assert not c2['ok'],'a 1e-9 difference must fail the exact check'
    print('e57_1280c_cache_analyze selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--analyze',action='store_true');ap.add_argument('--dir');ap.add_argument('--ref');ap.add_argument('--seeds');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    seeds=[int(x) for x in a.seeds.split(',')]
    if a.check:
        c=check(a.dir,a.ref,seeds);print(json.dumps(c,indent=1))
        if not c['ok']:print('IDENTITY CHECK FAILED: not tied to the 1280c trajectories');sys.exit(13)
    if a.analyze:
        o=analyze(a.dir,seeds);s=json.dumps(o,indent=1,default=float);print(s)
        if a.json:pathlib.Path(a.json).write_text(s)
if __name__=='__main__':main()
