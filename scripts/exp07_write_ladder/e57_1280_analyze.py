"""E57 job 1280 (recompute oracle) analysis, decision table v5 (CPU; DRAFT for Council re-check; 3 placeholders remain (delta_PL, SESOI anchor, 1284 primary: Hans) and MUST be filled before the registration hash; written before any 1280 data exists).
DEFINITIONS. E-form is the PRIMARY definition (positive = blocking the prior exposures RAISES the NLL): E_cache = a - nc, E_rec = b - c, E_mask = X_MASK = NLL(recomputed band) - NLL(recomputed natural), all in nat/token over the fixed later-exposure fact tokens of the measured chunk. The D-form (D = -E: D_cache = nc - a, D_rec = c - b, D_mask = NLL_natural - NLL_band) is only RECORDED (R = E_cache/E_rec is the same in both forms; T_R(D) = -T_R(E)). The v3 text of 11/11 22:05 KST already used the E form ((a-nc) - .8 (b-c)); the 'sign mismatch' came from the D-form wording of the v4 request. NLL-type quantities (R, Q) and margin-type quantities (G, H, noprior) are NEVER mixed. TIMING: nc, a, c, b, a_P, b_P, X_MASK are losses at the PRE-UPDATE SNAPSHOT W_t (the weights before the write of chunk t, same snapshot, same update target for every one of them); margin = log p(gold) - logsumexp(log p wrong), mean over facts (nat).
RECORD per seed (aggregated within the seed over its measured chunks / facts): nc, a, c, b, a_P, b_P, X_MASK; branch margins after ONE step from the same snapshot M_NATS, M_MASK, M_RECOMP, M_RECOMP_PL (G = M_MASK - M_NATS and H = M_RECOMP - M_NATS are measured in the SAME branch; the only name for MASK - NATS is G); gain_noprior; c_minus_nc_abs_max (max over the seed's measured chunks of |c - nc|); t2_ok, t3_ok, p16_ok; half_pl_unmatched (count out of 32 facts); finite (bool).
BRANCH CHUNKS (what '3' means): per seed and per trajectory 3 measured branch chunks t >= 8 (window full), stratified: one in each of the chunk ranges [8,24), [24,44), [44,64), drawn by a fixed RNG (numpy default_rng((seed, 20261112))), never chosen by outcome; both trajectories (NATS-trained, NATMASK-trained) are used; the seed's G and H are the means over its branch chunks (6 values per seed). The variance-based choice among k in {3, 6, 8} branch chunks is made ONCE from the instrumentation seeds 91..94 alone (variance components only): the smallest k whose projected MDE of L_F at n = 10 is <= 20% of the instrumentation mean of |G|; none -> k = 8 (function choose_k).
GATES (separate): 2a E_rec: lower 95% bound > 0 and point >= .25; 2b E_mask: lower bound > 0 and point >= .10 (only then Q is defined); 2c G: lower bound > 0 (only then F is defined). Unit/instrument gates: per-seed |c - nc| <= 1e-4 (c_minus_nc_abs_max), t2, t3, p16, finite, determinism (automated: the same measurement repeated at the same W_t must agree within 1e-4). All bounds use the critical value c_k of the look (one-sided alpha = .05) except the two directions of T_R, which use the alpha = .025 critical value each.
TESTS (seed-paired one-sided t; for every gate and every reported limit the SAME critical value c_k of the current look): T_R = E_cache - .8 E_rec (leak direction: upper < 0; cache direction: lower > 0; each direction at alpha = .025 with its own MC boundaries); T_Q = E_rec - .7 E_mask (lower > 0); per-seed linear contrasts measured in the same branch: L_F = H - .7 G (lower > 0), L_Flow = H - .3 G (upper < 0); placebo non-superiority: upper bound of (M_RECOMP_PL - M_NATS) < delta_PL, UNIT = ABSOLUTE nat (margin units, not a ratio to G), delta_PL = PLACEHOLDER (.5 or .25; Hans). The original TOST +-.25 is a sensitivity analysis only.
PRIMARY VERDICT (exclusive, one label): 1 PROTOCOL / INSTRUMENT INVALID ; 2a E_rec NOT CERTIFIED (R undefined) ; 4 CACHE SUFFICIENT = 2a passes and T_R cache direction lower > 0 [confirmatory; reaches only if E_cache >= .8 E_rec: practically unreachable, see disclosure] ; 3 LEAK hierarchy (requires 2a), fixed order R-leak -> Q -> F -> placebo, the verdict is the deepest step reached: 3a = R-leak only (T_R upper < 0) ; 3b = 3a and 2b and T_Q lower > 0 ; 3c = 3b and 2c and L_F lower > 0 and placebo non-superiority [confirmatory] ; 6 INCONCLUSIVE otherwise. Independent FLAGS reported next to the verdict (never replacing it): F_LOW (2c passes and L_Flow upper < 0; secondary), H_GEN (noprior - .5 G lower > 0), HALF_PL_VALID (a seed is excluded from HALF-PL when more than 3 of 32 facts are unmatched), ANOMALY (E_cache > 1.2 E_rec point estimate). Wording: 'consistent with', never 'proves'.
SEQUENTIAL (2-look, n = 10 and 20): rejection = the FIRST look at which the boundary is crossed; the same c_k for all gates and limits; efficacy stop only if the target verdict (3a/3b/3c/4) and every gate it needs hold at that look; interim futility advice is non-binding (upper bound < SESOI or conditional power < 10%) and is separate from the final verdict, which is issued at the first crossing or at n = 20. Boundaries (t, estimated variance) by Monte Carlo: alpha = .05: 3.179 / 1.755 ; alpha = .025: 4.011 / 2.109 (N = 2,000,000 normal data sets, numpy default_rng seed 7, cumulative spending alpha(t) = 2 (1 - Phi(z_(1-alpha/2) / sqrt(t))): .00557 at n = 10 for alpha = .05 and .00153 for alpha = .025). These are ESTIMATES under the stated model (normal data, estimated variance); MC standard error of an error rate: sqrt(alpha(1-alpha)/N) = .000154 (alpha = .05), .000110 (alpha = .025), the boundaries themselves carry about +-.004 (an earlier independent run of the alpha = .05 search gave 3.183 / 1.757). Two RNG streams reported SEPARATELY: calibration sample (seed 7; error rate .0500 and .0250 by construction) and independent draws (seed 12345, N = 2,000,000): alpha = .05 design .049933 (normal), .048903 (t5); alpha = .025 design .024675 (normal), .023288 (t5); the t5 values are a robustness run, not a validity claim. Code: sim_bounds / check_alpha below.
SEEDS: instrumentation 91..94 (variance only); confirmatory 1280 = 101..120 in this order; replacement list 121..130, at most 3 replacements, ONLY for pre-specified technical failures (world generation raises, non-finite values, automated determinism check failure) and NaN -> that seed is replaced; EVERY attempted seed is preserved (attempt log); 1284 = 201..220; the lists are disjoint (asserted).
DISCLOSURES: (i) row 4 is practically unreachable: the pilot (3 seeds, p = 4) gave X_ORACLE .076/.065/.079 vs X_MASK .337/.367/.383, i.e. the cached oracle captured only about one fifth; the thresholds .8 / .7 / .7 / .3 and the .25 / .1 gates were fixed AFTER seeing the pilot. (ii) the pilot placebo difference M_PLACEBO - M_NATS was +.096 / +.036 / -.003 (mean +.043, SD .050) and RAND - NATS +.048 / -.010 / -.022 (mean +.005, SD .037); the power calculation was the trigger for replacing the TOST by non-superiority. (iii) combined (joint) power of the confirmatory conditions of 3c is reported by joint_power_sim (assumption grid) and is lower than any single test's power. (iv) sigma_b^2 from the 3 pilot seeds has 2 degrees of freedom: very uncertain; MDE grid below uses SD multipliers {.5, 1, 2, 3, 4.4} x correlations {-.5, 0, .5, .9}.
  python3 e57_1280_analyze.py --selftest | --json records.json [--look 10|20]"""
import sys,json,argparse
import numpy as np
from scipy import stats
SEEDS_INSTR=list(range(91,95));SEEDS_1280=list(range(101,121));SEEDS_REPL=list(range(121,131));SEEDS_1284=list(range(201,221));MAX_REPL=3
CRIT={'05':{10:3.179,20:1.755},'025':{10:4.011,20:2.109}}
P=dict(R_k=0.8,Q_k=0.7,F_k=0.7,Flow_k=0.3,noprior_k=0.5,erec_min=0.25,emask_min=0.10,anomaly_ratio=1.2,unmatched_max=3,delta_PL_value=0.25,delta_PL='PLACEHOLDER (.5 or .25, Hans)',sesoi='PLACEHOLDER (anchor decision, Hans)',primary_1284='PLACEHOLDER (Hans)',det_tol=1e-4,c_nc_tol=1e-4)
def bounds_t(d,c,alpha_level='05',n=None):
    d=np.asarray(d,float);m=d.mean();se=d.std(ddof=1)/np.sqrt(len(d));return dict(mean=float(m),lower=float(m-c*se),upper=float(m+c*se),n=int(len(d)),se=float(se),crit=float(c))
def crit_for(look,n,level):
    if look is None:return float(stats.t.ppf(1-(0.05 if level=='05' else 0.025),n-1))
    return CRIT[level][look]
def evaluate(recs,look=None,p=P):
    n=len(recs);A=lambda k:np.array([r[k] for r in recs],float);nc,a,c,b,aP,bP,X=[A(k) for k in ('nc','a','c','b','a_P','b_P','X_MASK')];MN,MM,MR,MP=[A(k) for k in ('M_NATS','M_MASK','M_RECOMP','M_RECOMP_PL')];G=MM-MN;H=MR-MN;gnp=A('gain_noprior')
    c05=crit_for(look,n,'05');c025=crit_for(look,n,'025');Ec,Er,Em=a-nc,b-c,X;out={'n_seeds':n,'look':look,'crit_05':c05,'crit_025':c025,'definition':'E-form primary; D = -E recorded only'}
    inst=dict(c_nc_ok=bool(all(r['c_minus_nc_abs_max']<=p['c_nc_tol'] for r in recs)),t2_ok=bool(all(r['t2_ok'] for r in recs)),t3_ok=bool(all(r['t3_ok'] for r in recs)),p16_ok=bool(all(r['p16_ok'] for r in recs)),finite=bool(all(r['finite'] for r in recs) and all(np.isfinite(x).all() for x in (nc,a,c,b,X,MN,MM,MR,MP))),determinism_ok=bool(all(r.get('determinism_abs',0.0)<=p['det_tol'] for r in recs)))
    out['instrument_gates']=inst;hp=int(sum(1 for r in recs if r['half_pl_unmatched']>p['unmatched_max']))
    flags=dict(HALF_PL_VALID_seeds=int(n-hp),HALF_PL_excluded_seeds=hp)
    if not all(inst.values()):out['label']='1 PROTOCOL / INSTRUMENT INVALID';out['flags']=flags;return out
    er=bounds_t(Er,c05);em=bounds_t(Em,c05);gg=bounds_t(G,c05);g2a=bool(er['lower']>0 and er['mean']>=p['erec_min']);g2b=bool(em['lower']>0 and em['mean']>=p['emask_min']);g2c=bool(gg['lower']>0)
    out['gates']=dict(g2a_E_rec=dict(stats=er,ok=g2a),g2b_E_mask=dict(stats=em,ok=g2b,Q_defined=g2b),g2c_G=dict(stats=gg,ok=g2c,F_defined=g2c))
    TR=bounds_t(Ec-p['R_k']*Er,c025);TQ=bounds_t(Er-p['Q_k']*Em,c05);LF=bounds_t(H-p['F_k']*G,c05);LFl=bounds_t(H-p['Flow_k']*G,c05);PL=bounds_t(MP-MN,c05);hg=bounds_t(gnp-p['noprior_k']*G,c05)
    out['tests']=dict(T_R=TR,T_R_D_form=dict(lower=-TR['upper'],upper=-TR['lower']),T_Q=TQ,L_F=LF,L_Flow=LFl,PL_minus_NATS=PL,PL_nonsuperior_abs_nat=bool(PL['upper']<p['delta_PL_value']),TOST_sensitivity_pm_0_25=bool(PL['lower']>-0.25 and PL['upper']<0.25),R_point=float(Ec.mean()/Er.mean()),L_specific_descriptive=float(np.mean((b-a)-(bP-aP))))
    out['flags']=dict(F_LOW=bool(g2c and LFl['upper']<0),H_GEN=bool(hg['lower']>0),ANOMALY_E_cache_gt_1_2_E_rec=bool(Ec.mean()>p['anomaly_ratio']*Er.mean()),**flags)
    if not g2a:out['label']='2a E_rec NOT CERTIFIED (R undefined)';return out
    if TR['lower']>0:out['label']='4 CACHE SUFFICIENT [confirmatory] (consistent with)';return out
    lab='6 INCONCLUSIVE'
    if TR['upper']<0:
        lab='3a R-LEAK ONLY (consistent with)'
        if g2b and TQ['lower']>0:
            lab='3b R-LEAK AND Q (consistent with)'
            if g2c and LF['lower']>0 and out['tests']['PL_nonsuperior_abs_nat']:lab='3c LEAK CONFIRMED [confirmatory] (consistent with)'
    out['label']=lab;return out
def interim_action(n,verdict_label,gates_ok,upper_bound,z_n,sesoi=0.5):
    """interim action only: efficacy stop if the target verdict and every gate it needs hold at this look; non-binding futility advice; the final verdict is issued at the first crossing or at n = 20"""
    cp=conditional_power(n,20,z_n,CRIT['05'][20]);target=verdict_label.startswith(('3a','3b','3c','4 '))
    return dict(n=n,stop_for_efficacy=bool(target and gates_ok),futility_advice=bool(upper_bound<sesoi or cp<0.10),conditional_power=cp,binding=False)
def conditional_power(n,N,z_n,c_final):
    t=n/N;B=z_n*np.sqrt(t);return float(1-stats.norm.cdf((c_final-B/t)/np.sqrt(1-t)))
def sim_bounds(alpha,looks=(10,20),N=2_000_000,seed=7,batch=250_000):
    rng=np.random.default_rng(seed);z=stats.norm.ppf(1-alpha/2);tf=np.array(looks)/looks[-1];spend=2*(1-stats.norm.cdf(z/np.sqrt(tf)));spend[-1]=alpha;T=[]
    for _ in range(N//batch):
        x=rng.normal(size=(batch,looks[-1]));T.append(np.stack([x[:,:k].mean(1)/(x[:,:k].std(1,ddof=1)/np.sqrt(k)) for k in looks],1))
    T=np.concatenate(T);alive=np.ones(len(T),bool);inc=np.diff(np.concatenate([[0],spend]));out=[]
    for k in range(len(looks)):
        tk=T[:,k];cand=np.sort(tk[alive])[::-1];c=cand[int(inc[k]*len(T))];out.append(float(c));alive&=~(tk>c)
    return out
def check_alpha(bounds,looks=(10,20),N=2_000_000,seed=12345,dist='normal',batch=250_000):
    rng=np.random.default_rng(seed);rej=0
    for _ in range(N//batch):
        x=rng.normal(size=(batch,looks[-1])) if dist=='normal' else rng.standard_t(5,size=(batch,looks[-1]));alive=np.ones(batch,bool)
        for k,c in zip(looks,bounds):
            t=x[:,:k].mean(1)/(x[:,:k].std(1,ddof=1)/np.sqrt(k));hit=alive&(t>c);rej+=hit.sum();alive&=~hit
    return rej/N
def mde(sd,n,alpha=0.05,power=0.8):return float((stats.t.ppf(1-alpha,n-1)+stats.t.ppf(power,n-1))*sd/np.sqrt(n))
def mde_grid(n,sd_a=0.0072,sd_rec=0.0236,sd_m=0.0236,mults=(0.5,1,2,3,4.4),rhos=(-0.5,0,0.5,0.9)):
    """MDE (80% power, one-sided .05, nat/token) of T_R and T_Q over SD multipliers x correlations (corr(E_cache, E_rec) for T_R, corr(E_rec, E_mask) for T_Q)"""
    g={}
    for m in mults:
        for r in rhos:
            vR=(m*sd_a)**2+0.64*(m*sd_rec)**2-2*0.8*r*(m*sd_a)*(m*sd_rec);vQ=(m*sd_rec)**2+0.49*(m*sd_m)**2-2*0.7*r*(m*sd_rec)*(m*sd_m)
            g[f'x{m}_rho{r}']=dict(T_R=mde(np.sqrt(max(vR,1e-12)),n),T_Q=mde(np.sqrt(max(vQ,1e-12)),n))
    return g
def joint_power_sim(n,eff,sd,N=20000,crit=None,seed=0):
    """joint power of the 3c conditions by simulation: eff/sd are dicts with keys E_cache_minus_08Erec (true T_R mean, negative = leak), TQ, LF, PL(M_RECOMP_PL - M_NATS) and their seed-level SDs (independent); each test at its one-sided critical value (default t(.95, n-1), T_R leak direction t(.975, n-1))"""
    rng=np.random.default_rng(seed);c05=stats.t.ppf(0.95,n-1) if crit is None else crit['05'];c025=stats.t.ppf(0.975,n-1) if crit is None else crit['025'];ok=0
    for _ in range(N):
        tr=rng.normal(eff['TR'],sd['TR'],n);tq=rng.normal(eff['TQ'],sd['TQ'],n);lf=rng.normal(eff['LF'],sd['LF'],n);pl=rng.normal(eff['PL'],sd['PL'],n)
        f=lambda x:(x.mean(),x.std(ddof=1)/np.sqrt(n))
        m,s=f(tr);c1=m+c025*s<0;m,s=f(tq);c2=m-c05*s>0;m,s=f(lf);c3=m-c05*s>0;m,s=f(pl);c4=m+c05*s<P['delta_PL_value'];ok+=bool(c1 and c2 and c3 and c4)
    return ok/N
def choose_k(sd_seed,sd_chunk,mean_abs_G,n=10,ks=(3,6,8)):
    """pre-rule for the branch-chunk count k (variance components from the instrumentation seeds 91..94 ONLY): smallest k whose projected MDE of L_F at n seeds is <= 20% of mean |G|; none -> 8"""
    for k in ks:
        if mde(np.sqrt(sd_seed**2+sd_chunk**2/k),n)<=0.2*mean_abs_G:return k
    return ks[-1]
def selftest():
    rng=np.random.default_rng(0);S=10
    def mk(nc=2.0,a=2.5,c=2.0,b=2.5,aP=2.0,bP=2.0,X=0.4,MN=-3.8,MM=-1.8,MR=-1.8,MP=-3.8,gnp=1.1,sd=0.05,ok=True,um=0):
        return [dict(nc=nc+rng.normal(0,sd),a=a+rng.normal(0,sd),c=c+rng.normal(0,sd),b=b+rng.normal(0,sd),a_P=aP+rng.normal(0,sd),b_P=bP+rng.normal(0,sd),X_MASK=X+rng.normal(0,sd),M_NATS=MN+rng.normal(0,sd),M_MASK=MM+rng.normal(0,sd),M_RECOMP=MR+rng.normal(0,sd),M_RECOMP_PL=MP+rng.normal(0,sd),gain_noprior=gnp+rng.normal(0,sd),c_minus_nc_abs_max=(5e-5 if ok else 1e-3),t2_ok=ok,t3_ok=ok,p16_ok=ok,finite=True,half_pl_unmatched=um,determinism_abs=1e-6) for _ in range(S)]
    L=lambda **k:evaluate(mk(**k))['label']
    assert L(a=2.5,b=2.5).startswith('4 CACHE SUFFICIENT')
    assert L(a=2.1,b=2.5,X=0.5).startswith('3c LEAK CONFIRMED')
    assert L(a=2.1,b=2.5,X=0.5,MP=-1.8).startswith('3b')                                  # placebo as good as MASK: hierarchy stops at 3b
    assert L(a=2.1,b=2.5,X=0.5,MR=-3.5).startswith('3b')                                  # F not satisfied (H small): stops at 3b
    assert L(a=2.1,b=2.5,X=0.05).startswith('3a')                                         # E_mask gate fails -> Q undefined -> 3a only
    assert L(b=2.1).startswith('2a')
    assert L(ok=False).startswith('1 PROTOCOL')
    assert not L(sd=0.6).startswith(('3c','4 '))
    r=evaluate(mk(a=2.1,b=2.5,X=0.5,MR=-3.5,MN=-3.8));assert r['flags']['F_LOW'] in (True,False) and 'T_R_D_form' in r['tests']
    assert evaluate(mk(gnp=1.5))['flags']['H_GEN'] and not evaluate(mk(gnp=0.2))['flags']['H_GEN']
    assert evaluate(mk(um=4))['flags']['HALF_PL_excluded_seeds']==S and evaluate(mk(um=3))['flags']['HALF_PL_excluded_seeds']==0
    assert evaluate(mk(a=3.2,b=2.5))['flags']['ANOMALY_E_cache_gt_1_2_E_rec']
    assert evaluate(mk(a=2.5,b=2.5),look=20)['crit_05']==1.755 and evaluate(mk(a=2.5,b=2.5),look=20)['crit_025']==2.109
    ia=interim_action(10,'3c LEAK CONFIRMED',True,0.8,3.3);assert ia['stop_for_efficacy'] and not ia['binding']
    assert not interim_action(10,'6 INCONCLUSIVE',True,0.8,3.3)['stop_for_efficacy']
    g=mde_grid(10);assert g['x1_rho0']['T_R']>0 and len(g)==20 and g['x3_rho0.9']['T_R']>0
    assert choose_k(0.05,0.3,2.0)==3 and choose_k(0.5,2.0,0.5)==8
    jp=joint_power_sim(10,dict(TR=-0.1,TQ=0.1,LF=0.5,PL=0.0),dict(TR=0.02,TQ=0.03,LF=0.45,PL=0.45),N=2000);assert 0<=jp<=1
    assert not (set(SEEDS_1280)|set(SEEDS_1284)|set(SEEDS_REPL))&set(SEEDS_INSTR) and not set(SEEDS_1280)&set(SEEDS_1284) and not set(SEEDS_REPL)&set(SEEDS_1280) and not set(SEEDS_REPL)&set(SEEDS_1284) and MAX_REPL==3
    print('e57_1280_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--look',type=int);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.json:print(json.dumps(evaluate(json.load(open(a.json)),a.look),indent=1,default=float))
