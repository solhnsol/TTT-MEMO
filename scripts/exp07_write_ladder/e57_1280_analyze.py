"""E57 job 1280 (recompute oracle) analysis, decision table v4 (CPU; DRAFT for Council re-check; thresholds and the two Hans decisions (placebo delta .5 vs .25, SESOI anchor) are PLACEHOLDERS; written before any 1280 data exists).
SIGN CONVENTION (explicit): effect sizes are POSITIVE when blocking the prior exposures RAISES the NLL: E_cache = a - nc, E_rec = b - c, E_mask = X_MASK = NLL(recomputed band) - NLL(recomputed natural) (nat/token over the fixed later-exposure fact tokens). The 'improvement-sign' quantities of the request are D_cache = nc - a = -E_cache, D_rec = c - b = -E_rec, D_mask = NLL_natural - NLL_band = -E_mask; ratios are unchanged (R = D_cache/D_rec = E_cache/E_rec) but the linear forms flip: T_R(D) = D_cache - .8 D_rec = -(E_cache - .8 E_rec), T_Q(D) = D_rec - .7 D_mask = -(E_rec - .7 E_mask). The code works in E-form with the semantics 'cache sufficient <=> E_cache - .8 E_rec has lower bound > 0' (= upper bound of T_R(D) < 0); this mapping must be confirmed (open question to the core-claims thread). NLL-type quantities (R, Q) and margin-type quantities (G, noprior) are NEVER mixed.
QUANTITIES (one record per seed, aggregated WITHIN the seed): nc, a, c, b, a_P, b_P, X_MASK (nat/token); margins (nat; margin = log p(gold) - logsumexp(log p wrong), mean over facts) after one branch step from the SAME snapshot: M_NATS, M_MASK, M_RECOMP, M_RECOMP_PL; G = M_MASK - M_NATS (ONE name; the branch measurement at the measured chunks; no separate trajectory G); gain_noprior (NATMASK - NATS margin on the filler-replaced facts); half_pl_unmatched_frac; booleans t1_ok (c ~ nc, |c - nc| < 1e-4 on identical input), t2_ok (all-keys-visible mask bit identical to no mask), t3_ok (token-substitution leak test), p16_ok (p = 16 visibility).
TESTS (seed-paired one-sided t, df = n-1; critical value = t(.95, n-1) unless a sequential boundary is passed): T_R = E_cache - .8 E_rec; T_Q = E_rec - .7 E_mask; T_F = H - .7 G with H = M_RECOMP - M_NATS (H0: H <= .7 G; leak needs lower bound > 0); T_Flow = H - .3 G (H0: H >= .3 G; needs upper bound < 0); T_noprior = gain_noprior - .5 G (lower > 0 -> H-gen flag); PL non-superiority: upper bound of M_RECOMP_PL - M_NATS < delta_PL (placeholder .5 or .25, pending Hans); the original TOST +-.25 is reported as a SENSITIVITY analysis, not as a gate. Interpretation thresholds (E_rec >= .25, E_mask >= .1) are applied to POINT estimates; gate validity of a denominator = one-sided 95% lower bound > 0.
DECISION TABLE (ordered): 1 PROTOCOL INVALID (any unit test false) ; 2 DENOMINATOR NOT CERTIFIED (lower bound of E_rec <= 0, or lower bound of G <= 0: 'F undefined') ; 2b Q UNDEFINED (lower bound of E_mask < .1; R rows stay allowed) ; 3 LEAK CONFIRMED [confirmatory] = (T_R upper < 0) and (T_Q lower > 0) and (T_F lower > 0) and PL non-superiority ; 4 CACHE SUFFICIENT [confirmatory] = T_R lower > 0 ; 5 LOW BRANCH RECOVERY [secondary] = T_Flow upper < 0 only ; 6 INCONCLUSIVE (default). The H-gen flag is separate and independent. Wording stays 'consistent with'. The three labels 'protocol invalid / denominator not certified / result inconclusive' are distinct.
SEQUENTIAL: all confirmatory tests (rows 3 and 4) use the SAME boundaries: looks fixed at n = 10 and n = 20 (2-look), t boundaries 3.183 / 1.757 (critical values replace t(.95, n-1) at the look). Interim actions (stop for efficacy at n = 10, non-binding futility recommendation: upper bound < SESOI or conditional power < 10%) are separated from the final verdict, which is only issued at a boundary crossing or at n = 20. Monte Carlo: 2,000,000 normal data sets (numpy default_rng, seed 7 for the boundary search, stage-wise spending alpha(t) = 2(1 - Phi(z_.975/sqrt(t))): cumulative .00557 at n = 10, .05 at n = 20), t statistics with estimated variance; validation with independent draws (default_rng seed 12345, 2,000,000 sets): type-I error .0499 for normal data; a heavy-tailed t(5) run (.0482 for the 3-look design) is a robustness simulation only, not part of the validity claim.
SEEDS: instrumentation (variance components, bench; no confirmatory inference) 91..94; confirmatory 1280 = 101..120 in this order; 1284 = 201..220; disjoint (asserted). A confirmatory ID is replaced by the next ID (121, ...) ONLY for pre-specified technical failures (world generation raises, non-finite values, a unit-test gate failing for numerical reasons |delta| > 1e-4), never on outcome; 32 facts per seed; sigma_b^2 from the 3 pilot seeds has 2 degrees of freedom (very uncertain); all statistics are aggregated within the seed first, then seed-paired t.
HALF-PL: facts without a deterministic match are excluded; if more than 10% of the facts are unmatched the HALF-PL secondary is declared invalid. BRANCH CHUNKS: 3 per trajectory (measured chunks t >= 8 so that the window is full for ALL measurements; the recomputation uses the complete window, there is no tail truncation of the recomputed context, so b is not pulled toward a; the stale-weights effect is handled by the (c - nc) correction).
  python3 e57_1280_analyze.py --selftest | --json records.json [--look 10|20]"""
import sys,json,argparse
import numpy as np
from scipy import stats
SEEDS_INSTR=list(range(91,95));SEEDS_1280=list(range(101,121));SEEDS_1284=list(range(201,221))
BOUNDS={'2look':dict(looks=[10,20],t=[3.183,1.757])}
P=dict(R_k=0.8,Q_k=0.7,F_k=0.7,Flow_k=0.3,noprior_k=0.5,erec_min=0.25,emask_min=0.1,delta_PL='PLACEHOLDER .5 or .25 (Hans)',delta_PL_value=0.25,sesoi='PLACEHOLDER (.5 nat pending anchor decision)',unmatched_max=0.10)
def t1(d,crit=None,alpha=0.05):
    d=np.asarray(d,float);n=len(d);m=d.mean();se=d.std(ddof=1)/np.sqrt(n);t=stats.t.ppf(1-alpha,n-1) if crit is None else crit;return dict(mean=float(m),lower=float(m-t*se),upper=float(m+t*se),n=int(n),se=float(se),crit=float(t))
def evaluate(recs,crit=None,p=P):
    A=lambda k:np.array([r[k] for r in recs],float);nc,a,c,b,aP,bP,X=[A(k) for k in ('nc','a','c','b','a_P','b_P','X_MASK')];MN,MM,MR,MP=[A(k) for k in ('M_NATS','M_MASK','M_RECOMP','M_RECOMP_PL')];G=MM-MN;gnp=A('gain_noprior')
    Ec,Er,Em=a-nc,b-c,X;H=MR-MN;out={'n_seeds':len(recs),'sign_convention':'E-form (positive = blocking raises NLL); D = -E'}
    flags={k:bool(all(r[k] for r in recs)) for k in ('t1_ok','t2_ok','t3_ok','p16_ok')};out['unit_tests']=flags
    hg=t1(gnp-p['noprior_k']*G,crit);out['H_gen_flag']=dict(stats=hg,flag=bool(hg['lower']>0))
    if not all(flags.values()):out['label']='1 PROTOCOL INVALID';return out
    er=t1(Er,crit);em=t1(Em,crit);gg=t1(G,crit);out['denominators']=dict(E_rec=er,E_mask=em,G=gg,E_rec_point_ge_0_25=bool(er['mean']>=p['erec_min']),E_mask_point_ge_0_1=bool(em['mean']>=p['emask_min']))
    if er['lower']<=0 or not out['denominators']['E_rec_point_ge_0_25']:out['label']='2 DENOMINATOR NOT CERTIFIED (E_rec)';return out
    if gg['lower']<=0:out['F_undefined']=True
    out['Q_undefined']=bool(em['lower']<p['emask_min'])
    TR=t1(Ec-p['R_k']*Er,crit);TQ=t1(Er-p['Q_k']*Em,crit);TF=t1(H-p['F_k']*G,crit);TFl=t1(H-p['Flow_k']*G,crit);PL=t1(MP-MN,crit)
    nonsup=bool(PL['upper']<p['delta_PL_value']);tost=dict(ci90=[PL['lower'],PL['upper']],within_pm_0_25=bool(PL['lower']>-0.25 and PL['upper']<0.25))
    out['tests']=dict(T_R=TR,T_Q=TQ,T_F=TF,T_Flow=TFl,PL_minus_NATS=PL,PL_nonsuperior=nonsup,TOST_sensitivity=tost,R_point=float(Ec.mean()/Er.mean()),L_specific_descriptive=float(np.mean((b-a)-(bP-aP))))
    D=lambda t:dict(T_R_D_form_upper=-t['lower'],T_R_D_form_lower=-t['upper']);out['T_R_D_form']=D(TR)
    leak=bool(TR['upper']<0 and (not out['Q_undefined']) and TQ['lower']>0 and (not out.get('F_undefined',False)) and TF['lower']>0 and nonsup);cache=bool(TR['lower']>0);low=bool((not out.get('F_undefined',False)) and TFl['upper']<0)
    if leak:out['label']='3 LEAK CONFIRMED [confirmatory] (consistent with)'
    elif cache:out['label']='4 CACHE SUFFICIENT [confirmatory] (consistent with)'
    elif out['Q_undefined']:out['label']='2b Q UNDEFINED (X_MASK lower bound < 0.1); R rows evaluated: R not decisive'
    elif low:out['label']='5 LOW BRANCH RECOVERY [secondary] (consistent with)'
    else:out['label']='6 INCONCLUSIVE'
    return out
def interim_action(n,tstat_primary,upper_bound,z_n,sesoi=0.5):
    """interim action only (no verdict): stop for efficacy if the t statistic exceeds the boundary of the look; non-binding futility recommendation; the verdict is issued only at a boundary crossing or at n = 20"""
    d=BOUNDS['2look'];assert n in d['looks'];c=d['t'][d['looks'].index(n)];cp=conditional_power(n,20,z_n,d['t'][-1])
    return dict(n=n,boundary=c,stop_for_efficacy=bool(tstat_primary>c),futility_recommend=bool(upper_bound<sesoi or cp<0.10),conditional_power=cp,binding=False,verdict_issued=bool(tstat_primary>c or n==20))
def conditional_power(n,N,z_n,c_final):
    t=n/N;B=z_n*np.sqrt(t);return float(1-stats.norm.cdf((c_final-B/t)/np.sqrt(1-t)))
def mde(sd,n,alpha=0.05,power=0.8):return float((stats.t.ppf(1-alpha,n-1)+stats.t.ppf(power,n-1))*sd/np.sqrt(n))
def selftest():
    rng=np.random.default_rng(0);S=10
    def mk(nc=2.0,a=2.5,c=2.0,b=2.5,aP=2.0,bP=2.0,X=0.4,MN=-3.8,MM=-1.8,MR=-1.8,MP=-3.8,gnp=1.1,sd=0.05,ok=True):
        return [dict(nc=nc+rng.normal(0,sd),a=a+rng.normal(0,sd),c=c+rng.normal(0,sd),b=b+rng.normal(0,sd),a_P=aP+rng.normal(0,sd),b_P=bP+rng.normal(0,sd),X_MASK=X+rng.normal(0,sd),M_NATS=MN+rng.normal(0,sd),M_MASK=MM+rng.normal(0,sd),M_RECOMP=MR+rng.normal(0,sd),M_RECOMP_PL=MP+rng.normal(0,sd),gain_noprior=gnp+rng.normal(0,sd),t1_ok=ok,t2_ok=ok,t3_ok=ok,p16_ok=ok) for _ in range(S)]
    assert evaluate(mk(a=2.5,b=2.5))['label'].startswith('4 CACHE SUFFICIENT')
    assert evaluate(mk(a=2.1,b=2.5,X=0.5))['label'].startswith('3 LEAK CONFIRMED')
    assert evaluate(mk(a=2.1,b=2.5,X=0.5,MP=-1.8))['label']!='3 LEAK CONFIRMED [confirmatory] (consistent with)'      # placebo as good as MASK: not non-superior
    r=evaluate(mk(a=2.38,b=2.5,MR=-3.5));assert r['label'].startswith('5 LOW BRANCH RECOVERY'),r['label']
    assert not evaluate(mk(sd=0.6))['label'].startswith(('3 ','4 '))                      # pure noise must not confirm anything
    assert evaluate(mk(b=2.1))['label'].startswith('2 DENOMINATOR NOT CERTIFIED')
    assert evaluate(mk(ok=False))['label'].startswith('1 PROTOCOL INVALID')
    assert evaluate(mk(X=0.05,a=2.5,b=2.5))['Q_undefined']
    assert evaluate(mk(gnp=1.5))['H_gen_flag']['flag'] and not evaluate(mk(gnp=0.2))['H_gen_flag']['flag']
    assert evaluate(mk(a=2.5,b=2.5),crit=1.757)['label'].startswith('4 CACHE SUFFICIENT')
    ia=interim_action(10,3.3,0.8,3.3);assert ia['stop_for_efficacy'] and ia['verdict_issued'] and not ia['binding']
    ia=interim_action(10,2.0,0.3,0.2);assert not ia['stop_for_efficacy'] and ia['futility_recommend'] and not ia['verdict_issued']
    assert evaluate(mk())['sign_convention'].startswith('E-form') and 'T_R_D_form' in evaluate(mk())
    assert not (set(SEEDS_1280)|set(SEEDS_1284))&set(SEEDS_INSTR) and not set(SEEDS_1280)&set(SEEDS_1284) and SEEDS_1280[0]==101
    assert mde(0.02,20)<mde(0.02,10)
    print('e57_1280_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--look',type=int);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.json:
        crit=None if a.look is None else BOUNDS['2look']['t'][BOUNDS['2look']['looks'].index(a.look)];print(json.dumps(evaluate(json.load(open(a.json)),crit),indent=1,default=float))
