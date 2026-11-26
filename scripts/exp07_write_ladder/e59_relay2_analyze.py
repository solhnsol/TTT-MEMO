"""Analysis of job 1360 'relay2' (e59_relay2.py; CPU; new file; implementation thread 11/26; descriptive: values and intervals over seeds, same-direction seeds, no verdict wording, no threshold). Matched binding / familiarity / margin as e59_kvmask_analyze.py. MAIN (S2626-RELAY2) = per state (C0 stream, SAMEVIS stream): one-step change of R1 matched binding of the facts of the chunk (all 16 in U12), D step - C0 step, mean over t in {32, 64, 96} x 8 chunks inside a seed, seed-paired. Same table: D - C0x (numerics-matched control), C0x - C0 (numerics + W_t chain), D_sized - C0, per t; familiarity, margin; |g|, update norms, clip, cosines; place-token NLL at W_t (C0 vs D condition = the [S2626-RELAY] A - D) and its change after the steps.
  python e59_relay2_analyze.py --dir RELAY2_DIR"""
import sys,json,glob,random,numpy as np
from scipy import stats
D=sys.argv[sys.argv.index('--dir')+1];SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/relay2_s*.json'));RS={s:json.load(open(f'{D}/relay2_s{s}.json')) for s in SEEDS}
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):8.4f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:8.4f} [{m-h:7.4f},{m+h:7.4f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def per_fact(dS,W0,seed):
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);mar=lambda S:S[:,0]-S[:,1:12].max(1);out=np.full((16,3),np.nan)
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];out[f,0]=dS[f,0]-np.mean(oth) if oth else np.nan
    out[:,1]=dS[:,1:12].mean(1)-dS[:,12:16].mean(1);out[:,2]=mar(W0+dS)-mar(W0);return out
MET=('matched binding','familiarity','margin')
def ch_(s,rec,e,kind,mi):
    W0=RS[s]['W0_R1'];b=per_fact(rec['R1_Wt'],W0,s)[:,mi];x=per_fact(e['steps'][kind]['R1'],W0,s)[:,mi];return float(np.nanmean((x-b)[e['facts']]))
def sv(state,ts,fn):
    out=[]
    for s in SEEDS:
        v=[fn(s,rec,e) for t in ts for rec in [RS[s]['states'].get(state,{}).get(str(t))] if rec for e in rec['chunks']];out.append(np.nanmean(v) if v else np.nan)
    return np.asarray(out)
print(f'1360 relay2: seeds {SEEDS}; stage-1 identical to 1348 {[RS[s]["stage1_same_as_1348"] for s in SEEDS]}; C0 step = stream {[all(v.get("C0_step_hash_equal_stream",True) for k,v in RS[s]["checks"].items() if k!="recorder") for s in SEEDS]}')
for s in SEEDS[:1]:print('recorder check (seed %d):'%s,json.dumps(RS[s]['checks'].get('recorder'))[:600])
TS=(32,64,96)
for mi,mn in enumerate(MET):
    print(f'== one-step change of R1 {mn} (facts of the chunk), seed-paired; mean over t x 8 chunks inside a seed')
    for state in ('C0','SAMEVIS'):
        for A,B in (('D','C0'),('D','C0x'),('C0x','C0'),('D_sized','C0')):print(f'  state {state:8s} {A+" - "+B:14s} {tci(sv(state,TS,lambda s,r,e:ch_(s,r,e,A,mi)-ch_(s,r,e,B,mi)))}')
        for kd in ('C0','C0x','D','D_sized'):print(f'  state {state:8s} {kd:14s} {tci(sv(state,TS,lambda s,r,e:ch_(s,r,e,kd,mi)))}')
        if mi==0:
            for t in TS:print(f'    t {t}: D - C0 {tci(sv(state,(t,),lambda s,r,e:ch_(s,r,e,"D",0)-ch_(s,r,e,"C0",0)))} | D - C0x {tci(sv(state,(t,),lambda s,r,e:ch_(s,r,e,"D",0)-ch_(s,r,e,"C0x",0)))}')
print('== steps: |g|, update norm, clip share; cosines of the updates (total)')
for state in ('C0','SAMEVIS'):
    for kd in ('C0','C0x','D'):print(f'  state {state:8s} {kd:4s} |g| {tci(sv(state,TS,lambda s,r,e:e["steps"][kd]["gnorm"]))} |dW| {tci(sv(state,TS,lambda s,r,e:e["steps"][kd]["update_norm"]))} clip {tci(sv(state,TS,lambda s,r,e:float(e["steps"][kd]["coef"]<1)))}')
    for p in ('C0-D','C0x-D','C0-C0x'):print(f'  state {state:8s} cos {p:7s} {tci(sv(state,TS,lambda s,r,e:e["cos"][p]["total"]))}')
print('== place-token NLL at W_t (mean over facts and place tokens): C0 condition - D condition (= [S2626-RELAY] A - D), C0x - D; change after the step under its own condition')
nl=lambda d:np.mean([np.mean(v) for v in d.values()])
for state in ('C0','SAMEVIS'):
    print(f'  state {state:8s} C0 - D {tci(sv(state,TS,lambda s,r,e:nl(e["place_nll_Wt"]["C0"])-nl(e["place_nll_Wt"]["D"])))} | C0x - D {tci(sv(state,TS,lambda s,r,e:nl(e["place_nll_Wt"]["C0x"])-nl(e["place_nll_Wt"]["D"])))}')
    for kd in ('C0','D'):print(f'  state {state:8s} after {kd} step, NLL change under C0 cond {tci(sv(state,TS,lambda s,r,e:nl(e["steps"][kd]["place_nll"]["C0"])-nl(e["place_nll_Wt"]["C0"])))} under D cond {tci(sv(state,TS,lambda s,r,e:nl(e["steps"][kd]["place_nll"]["D"])-nl(e["place_nll_Wt"]["D"])))}')
