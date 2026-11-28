"""Analysis of job 1364 'early1' (S2628-EARLY step 1; CPU; new file; implementation thread 11/28; descriptive, no threshold). Ladder P1d (job 1362) -> L1 -> L2 -> L2b -> ISO1 (job 1363 for seeds 231-234, this job's 32-chunk ISO1 for 235-238; the first 32 chunks of ISO1 do not depend on the run length). R1 (neutral 8192-token tail) matched binding and top-1 accuracy among the 16 candidates at e8 / e16 / e32, seed-paired differences of neighbouring rungs and to P1d / ISO1, plus familiarity.
  python e59_early1_analyze.py --dir EARLY1 --p1d P1D --iso REDO1_ISO1"""
import sys,json,glob,random,os,numpy as np
from scipy import stats
a=sys.argv;g=lambda k:a[a.index(k)+1];D=g('--dir');PD=g('--p1d');IS=g('--iso')
SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/kvmulti_s*.json'))
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
    S=W0+dS;return dict(bind=float(np.mean(mb)),fam=float(np.mean(dS[:,1:12].mean(1)-dS[:,12:16].mean(1))),acc=float(np.mean(S.argmax(1)==0)))
def src(arm,s):
    if arm=='P1d':r=json.load(open(f'{PD}/kvmulti_s{s}.json'));return r['arms']['P1d'],r['W0_R1']
    if arm=='ISO1':
        p=f'{IS}/cbo_s{s}.json' if os.path.exists(f'{IS}/cbo_s{s}.json') else f'{D}/iso1/cbo_s{s}.json'
        if not os.path.exists(p):return None,None
        r=json.load(open(p));return r['arms']['ISO1'],r['W0_R1']
    r=json.load(open(f'{D}/kvmulti_s{s}.json'));return r['arms'].get(arm),r['W0_R1']
ARMS=('P1d','L1','L2','L2b','ISO1')
V={(arm,s,e):meas(src(arm,s)[0]['reads'][str(e)]['dS1'],src(arm,s)[1],s) for arm in ARMS for s in SEEDS for e in (8,16,32) if src(arm,s)[0]}
col=lambda arm,e,key:np.array([V[(arm,s,e)][key] if (arm,s,e) in V else np.nan for s in SEEDS])
print(f'1364 early ladder: seeds {SEEDS}')
for key,nm in (('bind','R1 matched binding'),('acc','R1 top-1 accuracy (16 candidates)'),('fam','familiarity')):
    print(f'== {nm}')
    for e in (8,16,32):print(f'  e{e:2d}: '+' | '.join(f'{arm} {tci(col(arm,e,key))}' for arm in ARMS))
    print(f'  rung differences (seed-paired):')
    for e in (8,16,32):print(f'  e{e:2d}: '+' | '.join(f'{x}-{y} {tci(col(x,e,key)-col(y,e,key))}' for x,y in (('L1','P1d'),('L2','L1'),('L2b','L2'),('ISO1','L2b'),('ISO1','P1d'))))
