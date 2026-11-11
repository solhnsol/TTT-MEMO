"""E57 job 1280 (recompute oracle) analysis, decision table v3 (CPU; DRAFT for Council re-check; thresholds are PLACEHOLDERS until confirmed; written before any 1280 data exists; the SESOI anchor is a placeholder because it came from K0 = 8 results, pending Council).
INPUT: one record per seed, every quantity already aggregated WITHIN the seed (mean over the measured chunks / facts): NLL-type quantities in nat/token over the fixed set of later-exposure fact tokens of the measured chunks (nc cached natural, a cached oracle, c recomputed natural, b recomputed oracle, a_P cached placebo, b_P recomputed placebo, X_MASK = recomputed band NLL - recomputed natural NLL at the SAME snapshot and update target); margin-type quantities in nat (margin = log p(gold) - logsumexp(log p wrong), mean over facts): branch margins after one step from the same snapshot (M_NATS, M_MASK, M_RECOMP, M_RECOMP_PL), gain_noprior (NATMASK - NATS margin on the filler-replaced facts), G (NATMASK - NATS margin of the trained trajectories at the read exposure), plus booleans t1_ok (c ~ nc), t2_ok (all-keys-visible mask bit identical to no mask), t3_ok (token-substitution leak test passes for the recompute oracle), p16_ok (p = 16 visibility check). NLL-type and margin-type quantities are never compared with each other.
TESTS (seed-paired one-sided t, df = n-1, alpha = .05): d_R = (a-nc) - 0.8 (b-c); d_Q = (b-c) - 0.7 X_MASK; d_F = (M_RECOMP - M_NATS) - 0.7 (M_MASK - M_NATS); d_Flow = (M_RECOMP - M_NATS) - 0.3 (M_MASK - M_NATS); d_noprior = gain_noprior - 0.5 G; TOST of M_RECOMP_PL - M_NATS within +-delta (delta = 0.25 nat primary, 0.10 nat exploratory, power disclosed).
PRECONDITIONS: t1, t2, t3, p16 true in every seed; lower 95% bound of (b-c) >= 0.25 nat/token; lower bound of (M_MASK - M_NATS) >= 0.5 nat; lower bound of X_MASK >= 0.1 nat/token. Failure -> 'MEASUREMENT INVALID / interpretation stopped'.
ORDER: preconditions -> H-gen flag (separate, independent of the rest: d_noprior lower bound > 0) -> LEAK CONFIRMED = (d_R upper bound < 0) AND (d_Q lower > 0) AND (d_F lower > 0) AND TOST(RECOMP-PL ~ NATS) -> CACHE SUFFICIENT = d_R lower > 0 -> LOW BRANCH RECOVERY = (d_Q lower > 0) AND (d_Flow upper < 0) -> otherwise INCONCLUSIVE (default). Wording stays low ('consistent with', 'does not exclude').
LEAK AMOUNT: primary R = (a-nc)/(b-c) (point estimate + the linear test); L_specific = (b-a) - (b_P-a_P) descriptive only.
SEQUENTIAL (aggregation unit = seed, 4 seeds = instrumentation only, no confirmatory inference): looks at n = 10, 15, 20 (3-look) or n = 10, 20 (2-look), Lan-DeMets O'Brien-Fleming spending alpha(t) = 2 (1 - Phi(z_.975 / sqrt(t))) (cumulative .00557, .02363, .05; 2-look .00557, .05); t-statistic boundaries by Monte Carlo (2M normal data sets, variance unknown): 3-look 3.179 / 2.232 / 1.833; 2-look 3.183 / 1.757; type-I error checked with independent draws (.0499 normal, .0482 heavy-tailed t5, 2-look .0499). Non-binding futility at n = 10, 15: upper 95% bound of the monitored effect < SESOI or conditional power < 10%.
SEEDS: 1280 seed IDs 101..120 in this order; a failed or colliding ID is replaced by the next ID (121, ...); 1284 uses 201..220 (disjoint).
  python3 e57_1280_analyze.py --selftest | --json records.json [--design 3look|2look]"""
import sys,json,argparse
import numpy as np
from scipy import stats
SEEDS_1280=list(range(101,121));SEEDS_1284=list(range(201,221))
BOUNDS={'3look':dict(looks=[10,15,20],t=[3.179,2.232,1.833]),'2look':dict(looks=[10,20],t=[3.183,1.757])}
P=dict(R_k=0.8,Q_k=0.7,F_k=0.7,Flow_k=0.3,noprior_k=0.5,bc_min=0.25,gain_min=0.5,xmask_min=0.1,tost_delta=0.25,tost_delta_exploratory=0.10,sesoi='PLACEHOLDER (0.5 nat pending Council)')
def t1(d,alpha=0.05):
    d=np.asarray(d,float);n=len(d);m=d.mean();se=d.std(ddof=1)/np.sqrt(n);t=stats.t.ppf(1-alpha,n-1);return dict(mean=float(m),lower=float(m-t*se),upper=float(m+t*se),n=int(n),se=float(se))
def tost(d,delta,alpha=0.05):
    r=t1(d,alpha);return dict(mean=r['mean'],ci90=[r['lower'],r['upper']],delta=delta,equivalent=bool(r['lower']>-delta and r['upper']<delta))
def tost_power(sd,n,delta,alpha=0.05,true=0.0):
    """approximate TOST power (known-variance normal approximation of the t-based test): P(|mean - true| small enough that the 90% interval lies inside +-delta) with true effect `true`"""
    se=sd/np.sqrt(n);tc=stats.t.ppf(1-alpha,n-1);lo=(-delta+tc*se-true)/se;hi=(delta-tc*se-true)/se
    return float(max(0.0,stats.norm.cdf(hi)-stats.norm.cdf(lo))) if hi>lo else 0.0
def evaluate(recs,p=P):
    A=lambda k:np.array([r[k] for r in recs],float);nc,a,c,b,aP,bP,X=[A(k) for k in ('nc','a','c','b','a_P','b_P','X_MASK')]
    MN,MM,MR,MP=[A(k) for k in ('M_NATS','M_MASK','M_RECOMP','M_RECOMP_PL')];gnp=A('gain_noprior');G=A('G');out={'n_seeds':len(recs)}
    flags={k:bool(all(r[k] for r in recs)) for k in ('t1_ok','t2_ok','t3_ok','p16_ok')}
    bc=t1(b-c);gm=t1(MM-MN);xm=t1(X);pre=dict(flags=flags,bc_lower=bc['lower'],bc_ok=bool(bc['lower']>=p['bc_min']),gain_lower=gm['lower'],gain_ok=bool(gm['lower']>=p['gain_min']),xmask_lower=xm['lower'],xmask_ok=bool(xm['lower']>=p['xmask_min']))
    out['preconditions']=pre;pre['all_ok']=bool(all(flags.values()) and pre['bc_ok'] and pre['gain_ok'] and pre['xmask_ok'])
    hg=t1(gnp-p['noprior_k']*G);out['H_gen_flag']=dict(stats=hg,flag=bool(hg['lower']>0))
    if not pre['all_ok']:out['label']='MEASUREMENT INVALID / interpretation stopped';return out
    dR=t1((a-nc)-p['R_k']*(b-c));dQ=t1((b-c)-p['Q_k']*X);dF=t1((MR-MN)-p['F_k']*(MM-MN));dFl=t1((MR-MN)-p['Flow_k']*(MM-MN));tp=tost(MP-MN,p['tost_delta']);tp2=tost(MP-MN,p['tost_delta_exploratory'])
    out['tests']=dict(d_R=dR,d_Q=dQ,d_F=dF,d_Flow=dFl,tost_RECOMP_PL_vs_NATS=tp,tost_exploratory=tp2,R_point=float(np.mean(a-nc)/np.mean(b-c)),L_specific_descriptive=float(np.mean((b-a)-(bP-aP))))
    leak=bool(dR['upper']<0 and dQ['lower']>0 and dF['lower']>0 and tp['equivalent']);cache=bool(dR['lower']>0);low=bool(dQ['lower']>0 and dFl['upper']<0)
    out['label']='LEAK CONFIRMED (consistent with)' if leak else ('CACHE SUFFICIENT (consistent with)' if cache else ('LOW BRANCH RECOVERY (consistent with)' if low else 'INCONCLUSIVE'))
    return out
def gsd_check(n,tstat,design='3look'):
    d=BOUNDS[design];assert n in d['looks'];c=d['t'][d['looks'].index(n)];return dict(n=n,t=float(tstat),boundary=c,stop_for_efficacy=bool(tstat>c))
def conditional_power(n,N,z_n,c_final):
    t=n/N;B=z_n*np.sqrt(t);return float(1-stats.norm.cdf((c_final-B/t)/np.sqrt(1-t)))
def futility(n,N,upper_bound,sesoi,z_n,c_final):
    cp=conditional_power(n,N,z_n,c_final);return dict(upper_lt_sesoi=bool(upper_bound<sesoi),conditional_power=cp,recommend_stop=bool(upper_bound<sesoi or cp<0.10),binding=False)
def selftest():
    rng=np.random.default_rng(0);S=10
    def mk(nc=2.0,a=2.4,c=2.0,b=2.5,aP=2.0,bP=2.0,X=0.4,MN=-3.8,MM=-1.8,MR=-1.8,MP=-3.8,gnp=1.1,G=2.0,sd=0.05,ok=True):
        return [dict(nc=nc+rng.normal(0,sd),a=a+rng.normal(0,sd),c=c+rng.normal(0,sd),b=b+rng.normal(0,sd),a_P=aP+rng.normal(0,sd),b_P=bP+rng.normal(0,sd),X_MASK=X+rng.normal(0,sd),M_NATS=MN+rng.normal(0,sd),M_MASK=MM+rng.normal(0,sd),M_RECOMP=MR+rng.normal(0,sd),M_RECOMP_PL=MP+rng.normal(0,sd),gain_noprior=gnp+rng.normal(0,sd),G=G+rng.normal(0,sd),t1_ok=ok,t2_ok=ok,t3_ok=ok,p16_ok=ok) for _ in range(S)]
    # cache sufficient: a-nc = 0.4 vs b-c = 0.5 -> R = 0.8+ ; here a - nc = .4, 0.8*(b-c) = .4 -> use a slightly larger
    assert evaluate(mk(a=2.5,b=2.5))['label'].startswith('CACHE SUFFICIENT')
    # leak confirmed: a-nc small (0.1), b-c large (0.5), Q = .5/.4 high, F high, placebo ~ NATS
    assert evaluate(mk(a=2.1,b=2.5,X=0.5))['label'].startswith('LEAK CONFIRMED')
    # low branch recovery: b-c large vs X (Q ok), a-nc between (R undecided: a-nc = .38 ~ .76 (b-c)) and F small
    r=evaluate(mk(a=2.38,b=2.5,MR=-3.5));assert r['label'].startswith('LOW BRANCH RECOVERY'),r['label']
    # inconclusive: effects too noisy
    assert evaluate(mk(sd=0.6))['label'] in ('INCONCLUSIVE','MEASUREMENT INVALID / interpretation stopped')
    # precondition failure (b-c small)
    assert evaluate(mk(b=2.1))['label'].startswith('MEASUREMENT INVALID')
    # a failed unit test
    assert evaluate(mk(ok=False))['label'].startswith('MEASUREMENT INVALID')
    # H-gen flag independent of the rest
    assert evaluate(mk(gnp=1.5))['H_gen_flag']['flag'] and not evaluate(mk(gnp=0.2))['H_gen_flag']['flag']
    assert gsd_check(10,3.3)['stop_for_efficacy'] and not gsd_check(10,2.5)['stop_for_efficacy'] and gsd_check(20,1.9)['stop_for_efficacy']
    f=futility(10,20,0.3,0.5,0.2,1.833);assert f['recommend_stop'] and not f['binding']
    assert 0<=tost_power(0.45,10,0.25)<=1 and tost_power(0.45,10,0.10)<=tost_power(0.45,10,0.25) and tost_power(0.45,20,0.25)>tost_power(0.45,10,0.25)
    assert SEEDS_1280[0]==101 and SEEDS_1280[-1]==120 and not set(SEEDS_1280)&set(SEEDS_1284)
    print('e57_1280_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.json:print(json.dumps(evaluate(json.load(open(a.json))),indent=1,default=float))
