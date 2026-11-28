"""Analysis of job 1366 'ladder128' (S2821-LADDER128; CPU; new file; implementation thread 11/28; descriptive, no threshold). Ladder at 128 exposures, seeds 231-238: P0s, P1s (1348), P1d (1362), L2, L2b, L2c (this job), ISO1 (1363 for 231-234, this job for 235-238). R1 (neutral 8192-token tail) matched binding / top-1 accuracy (16 candidates) / familiarity at e32 / e64 / e96 / e128; MAIN = the six neighbour differences at 128 (P1s-P0s, P1d-P1s, L2-P1d, L2b-L2, L2c-L2b, ISO1-L2c), seed-paired; the same at e32 / e64 / e96; the in-window reads R8 / R8m / R8a (8-chunk context 120..127, job 1366b) per arm; "late catch-up": exposure at which each arm reaches ISO1's e32 and e64 binding (linear interpolation over its read points).
  python e59_ladder128_analyze.py --dir LADDER_DIR --p1d P1D --base KVMULTI --iso REDO1_ISO1 [--win WIN8_DIR]"""
import sys,json,glob,random,os,numpy as np
from scipy import stats
a=sys.argv;g=lambda k,d=None:a[a.index(k)+1] if k in a else d;D=g('--dir');PD=g('--p1d');BA=g('--base');IS=g('--iso');WD=g('--win')
SEEDS=list(range(231,239));ARMS=('P0s','P1s','P1d','L2','L2b','L2c','ISO1')
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def meas(dS,W0,seed):
    dS=np.asarray(dS,float);W0=np.asarray(W0,float);cf=_cands(seed);mb=[]
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];mb.append(dS[f,0]-np.mean(oth))
    S=W0+dS;return dict(bind=float(np.mean(mb)),acc=float(np.mean(S.argmax(1)==0)),fam=float(np.mean(dS[:,1:12].mean(1)-dS[:,12:16].mean(1))))
def src(arm,s):
    if arm in ('P0s','P1s'):p=f'{BA}/kvmulti_s{s}.json'
    elif arm=='P1d':p=f'{PD}/kvmulti_s{s}.json'
    elif arm=='ISO1':p=f'{IS}/cbo_s{s}.json' if os.path.exists(f'{IS}/cbo_s{s}.json') else f'{D}/iso1/cbo_s{s}.json'
    else:p=f'{D}/kvmulti_s{s}.json'
    if not os.path.exists(p):return None,None
    r=json.load(open(p));return r['arms'].get(arm),r['W0_R1']
C={(arm,s):src(arm,s) for arm in ARMS for s in SEEDS}
def col(arm,e,key):
    out=[]
    for s in SEEDS:
        r,W0=C[(arm,s)];d=(r or {}).get('reads',{}).get(str(e));out.append(meas(d['dS1'],W0,s)[key] if d else np.nan)
    return np.array(out)
PAIRS=[('P1s','P0s'),('P1d','P1s'),('L2','P1d'),('L2b','L2'),('L2c','L2b'),('ISO1','L2c')]
print(f'1366 ladder at 128: seeds {SEEDS}; arms present per seed: '+' '.join(f'{arm}:{sum(C[(arm,s)][0] is not None for s in SEEDS)}' for arm in ARMS))
for key,nm in (('bind','R1 matched binding'),('acc','R1 top-1 accuracy'),('fam','familiarity')):
    print(f'== {nm}: levels')
    for e in (32,64,96,128):print(f'  e{e:3d}: '+' | '.join(f'{arm} {tci(col(arm,e,key))}' for arm in ARMS))
    print(f'== {nm}: neighbour differences (seed-paired)'+(' [MAIN at e128]' if key=='bind' else ''))
    for e in (32,64,96,128):print(f'  e{e:3d}: '+' | '.join(f'{x}-{y} {tci(col(x,e,key)-col(y,e,key))}' for x,y in PAIRS))
print('== late catch-up: exposure at which each arm reaches ISO1 e32 / e64 binding (per seed, linear interpolation; nan = not reached by 128)')
for arm in ARMS[:-1]:
    v=[]
    for s in SEEDS:
        r,W0=C[(arm,s)];ri,Wi=C[('ISO1',s)]
        if r is None or ri is None:v.append((np.nan,np.nan));continue
        es=[0]+sorted(int(e) for e in r['reads']);ys=[0.0]+[meas(r['reads'][str(e)]['dS1'],W0,s)['bind'] for e in es[1:]];out=[]
        for t in (32,64):
            tgt=meas(ri['reads'][str(t)]['dS1'],Wi,s)['bind'];hit=[i for i in range(1,len(es)) if ys[i]>=tgt]
            out.append(np.nan if not hit else float(np.interp(tgt,[ys[hit[0]-1],ys[hit[0]]],[es[hit[0]-1],es[hit[0]]])))
        v.append(out)
    v=np.asarray(v,float);print(f'  {arm:4s}: reaches ISO1-e32 at {tci(v[:,0])} (reached {int(np.isfinite(v[:,0]).sum())}/8) | ISO1-e64 at {tci(v[:,1])} (reached {int(np.isfinite(v[:,1]).sum())}/8)')
if WD:
    print('== in-window reads (8-chunk context 120..127), matched binding of (arm - W0) scores: R8 | R8m | R8a')
    W={s:json.load(open(f'{WD}/win8_s{s}.json')) for s in SEEDS if os.path.exists(f'{WD}/win8_s{s}.json')}
    for arm in ARMS:
        cols=[]
        for v in ('R8','R8m','R8a'):cols.append(tci([meas(np.asarray(W[s]['arms'][arm][v])-np.asarray(W[s]['W0'][v]),W[s]['W0'][v],s)['bind'] if arm in W[s]['arms'] else np.nan for s in W]))
        print(f'  {arm:4s}: '+' | '.join(cols))
