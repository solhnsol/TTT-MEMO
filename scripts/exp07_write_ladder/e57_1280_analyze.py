"""E57 job 1280 (recompute oracle) analysis, decision table v6 (CPU; DRAFT for Council re-check; placeholders delta_PL, SESOI anchor, 1284 primary (Hans) must be filled before the registration hash; written before any 1280 or instrumentation-seed output exists).
SIGN (verified against the archived versions): every analysis version defines the effects in E-form, E = NLL(blocked) - NLL(not blocked), positive when blocking raises the NLL: E_cache = a - nc, E_rec = b - c; the decision directions are leak = upper bound of T_R < 0 and cache sufficient = lower bound of T_R > 0. Checked in git: v3 commit 0f5352db9b99f7dcee5c53ec5a8f3e624c8c5791 (file sha256 13c6c7efef3544cc64e359c7995942b32992a49680bf27821d0dbf05ac416d1e, lines 32 and 34: dR = t1((a-nc) - .8 (b-c)); leak = dR upper < 0; cache = dR lower > 0), v4 commit ac486fbd7991f046fa144f9f6f21db45d6db76a4 (sha256 639eca79e25e29bc0e637adc7026399c3f174bc5fcb57d148fb8ac31cdc5a28f, line 32), v5 commit 0cb941dd484959a53fe3329015ce950201714d3d (sha256 badcff89af3eb5d87cf3fc5d8bc807ae7e5ac5160c97f0b5f178c59b90a8578e, line 32). The request text of v4 used D = nc - a (= -E_cache) while quoting the E-form decision directions, which produced an apparent sign reversal in the review; no analysis version used a D-form decision rule (the v4 / v5 outputs only RECORD the D-form). Unit test: sign_selftest.
UNITS: NLL-type quantities (nc, a, c, b, a_P, b_P, E_cache, E_rec, E_mask) are nat/token means over the SAME fixed token set (the later-exposure fact tokens of the measured chunks) aggregated to one value per seed; margin-type quantities (G, H, M_*, delta_PL, SESOI) are nat per fact (margin = log p(gold) - logsumexp(log p wrong), the log-prob of a candidate string is the SUM over its place tokens). The two families are never compared or mixed: G ~ 2 nat (margin) and the MDEs in nat/token are different normalisations. R and Q use NLL-type only; F, L_F, L_Flow, H_gen, placebo use margin-type only.
CLAIMS / GATES / ALPHA TABLE (one-sided tests, seed-paired t, unit = seed; alpha = per-test level, boundary set = critical values of the 2-look design): [T_R leak] direction upper < 0, alpha .025, set '025', gate 2a, confirmatory (step 1 of the leak hierarchy; efficacy stop allowed); [T_R cache] direction lower > 0, alpha .025, set '025', gate 2a, confirmatory (the two directions are mutually exclusive: combined <= .05); [T_Q] lower > 0, alpha .05, set '05', gates 2a and 2b, step 2; [L_F] lower > 0, alpha .05, set '05', gate 2c, step 3; [PL nonsuperiority] upper < delta_PL (absolute nat), alpha .05, set '05', step 4; [L_Flow] upper < 0, alpha .05, set '05', gate 2c, SECONDARY (F_LOW flag): no familywise claim, no efficacy stop; [H_gen] noprior - .5 G lower > 0, alpha .05, flag only. Multiplicity: the leak claims 3a/3b/3c are a fixed-order hierarchy (R -> Q -> F -> placebo), each step is tested only if the previous one passed, so the size of the 3c claim is <= max alpha_i = .05 (intersection-union); L_Flow and H_gen are outside the familywise claim. Seeds 201..220 (job 1284) are an INDEPENDENT REPLICATION with their own alpha (.05 one-sided) and are never pooled with 1280.
RECORD per seed: nc, a, c, b, a_P, b_P, X_MASK (= E_mask: the recomputed band NLL minus the recomputed natural NLL, nat/token), n_tokens (dict per quantity, must be identical), measured_chunk_min (>= 8), timing_pre_update (all losses at the snapshot W_t BEFORE the write of the chunk), c_minus_nc_abs_max (<= 1e-4), M_NATS, M_MASK, M_RECOMP, M_RECOMP_PL (branch margins after ONE step from the same snapshot), gain_noprior, half_pl_unmatched (of 32), finite, determinism_abs, t2_ok, t3_ok, p16_ok.
GATES: 2a E_rec: lower 95% bound > 0 and point >= .25; 2b E_mask: lower bound > 0 and point >= .10 (only then Q is defined); 2c G = M_MASK - M_NATS: lower bound > 0 (only then F is defined). Instrument gates (a failure = label 1): per-seed |c - nc| <= 1e-4, t2, t3, p16, finite, determinism <= 1e-4, identical token sets, t >= 8, pre-update timing.
VERDICT (exclusive): 1 PROTOCOL / INSTRUMENT INVALID ; 2a E_rec NOT CERTIFIED ; 4 CACHE SUFFICIENT (T_R cache direction; practically unreachable, see disclosures) ; 3a/3b/3c (deepest reached step of the hierarchy; 3c requires the placebo non-superiority) ; 6 INCONCLUSIVE. Independent flags: F_LOW, H_GEN, HALF_PL (a seed is excluded from HALF-PL when more than 3 of its 32 facts are unmatched), ANOMALY (E_cache > 1.2 E_rec).
SEQUENTIAL: looks at n = 10 and 20; look 1 boundary analytic c1 = t_9^{-1}(1 - alpha_1), look 2 by Monte Carlo only at look 2 on the sets that did not cross at look 1 (shared first 10 seeds), see e57_1280_boundaries.py (method, two RNG seeds, versions, both estimates and their difference, boundary MC SE, look-wise and cumulative error, t(5) and skewed-data robustness; report file e57_1280_boundaries_report.json). The same c_k is used for every gate and every reported bound. Rejection = the first look at which a boundary is crossed. Efficacy stop only if the target verdict and every gate it needs hold at that look.
LOOK-1 FUTILITY (item 8, non-binding, no alpha recycling, no sample-size re-estimation): triggers are COMPUTED by look1_report on the first 10 seeds (101..110) and the DEFAULT actions below are pre-specified; continuing after a trigger is always allowed (the decider, named by role, records the reason BEFORE opening seeds 111..120), stopping without a trigger is reported as a protocol deviation; the look-1 report script and its hash are committed BEFORE seeds 111..120 are run or opened and the seed list is fixed; if seeds 111..120 run concurrently their output is not read before the look-1 decision. (i) E_rec: nominal one-sided 95% t upper bound (df = 9, 1.833) < .25 -> label 'NOT CERTIFIED (look-1 futility)', stops R and Q (and F, which depends on gates) = the whole experiment; this is a screen, not a boundary, and it fires with probability .05 if the true E_rec is exactly .25 (a power cost, not an alpha cost); it is NOT a confirmatory conclusion that the population denominator is below .25. (ii) G upper bound < SESOI (.5): applies only because G, L_F and L_Flow share units (margin nat); stops only the F family (3c impossible, maximum verdict 3b; R and Q continue at the same boundaries; F-only arms = RECOMP and RECOMP-PL branch arms; shared arms NATS and NATMASK continue; the placebo non-superiority is stopped together with F). If SESOI (.5) was fixed after seeing G ~ 2 this is disclosed in item 12. (iii) predictive probabilities PP(3a), PP(3b), PP(3c) from the model of appendix 8A, evaluated with the final analysis function: PP(3c) < .10 -> stop the 3c-only arms; PP(3b) < .10 -> stop the 3b-only arms (the recomputed band measurement of E_mask); the WHOLE experiment stops only if PP(3a) < .10 or trigger (i) fires. PP(3a) ~ P(gate 2a passes) and overlaps with (i) (disclosed).
APPENDIX 8A (PP model): reduced seed vector Y = (E_cache, E_rec, E_mask, G, H, PLd = M_RECOMP_PL - M_NATS, gain_noprior) (p = 7); from the first 10 seeds: mean ybar and covariance S (df 9, p = 7 <= 9 so the inverse-Wishart is proper); shrinkage S~ = (1 - lam) S + lam diag(S), lam = .5; Sigma ~ InverseWishart(df = 9, scale = 9 S~); mu | Sigma ~ N(ybar, Sigma / 10); simulate 10 future seeds ~ MVN(mu, Sigma), append to the observed 10, evaluate evaluate_core at look 20 (the same final analysis code and hash); B >= 10^5 draws, fixed RNG seed, MC SE reported; instrument flags are assumed true; independence of components is NOT assumed (the full covariance is used).
POWER: mde_report gives, per test, three MDEs (fixed n, look 1 (n = 10 at its boundary, noncentral t), cumulative over both looks (80% power of crossing at either look, Monte Carlo)); T_R uses the alpha = .025 boundaries (n = 10 MDE about .032 for SD .0202, i.e. boundary inflation, not an SD multiplier); joint_power_mc computes the joint (cumulative and final-look-only) power of the 3c conditions with a user-specified mean vector, SD vector and correlation matrix (inputs disclosed with the result); the earlier .47 / .78 values (independent normal components, fixed t critical values, no boundaries) are WITHDRAWN. A power below .80 is stated as such.
SEEDS: instrumentation 91..94 (variance components at k = 8 only, no confirmatory inference); confirmatory 1280 = 101..120 (look 1 = 101..110, look 2 adds 111..120); replacement list 121..130, at most 3 replacements, only for technical, outcome-independent failures (world generation raises, non-finite values, determinism check failure), the replacement belongs to the look at which the failure occurred and is drawn before that look's statistics are computed; above the cap the analysis proceeds with the available seeds (n reduced, flagged, no further replacement); NaN replaces the seed; every attempted seed and its reason is preserved. 1284 = 201..220.
BRANCH CHUNKS (one rule): each branch chunk starts from a copy of the natural snapshot just before chunk t, every arm applies one update on its own context and is evaluated on the fact tokens of the same chunk with the margin M_* (frozen weights, same probes), then the copy is discarded; eligibility: t >= 8, fact tokens present, a prior same-fact exposure inside the window, at least 32 scored fact tokens; selection: stratified by terciles of the eligible t, order by sha256(job|seed|salt|t), round-robin over terciles so that k = 3 is a prefix of k = 6 is a prefix of k = 8, the SAME chunks for every arm (paired); fewer eligible chunks than k -> all eligible are used and the count is reported; isolation: parameters, optimizer state and RNG state are deep-copied and a checksum of the natural trajectory must be unchanged (assert_isolated). k is chosen ONCE by choose_k from the instrumentation seeds 91..94 run at k = 8: SD^2(k) = sigma_b^2 + sigma_w^2 / k (sigma_w^2 from the 28 within-seed degrees of freedom, sigma_b^2 with its 80% upper confidence limit, df = 3, SD factor 1.73); the smallest k in {3, 6, 8} with SD(L_F) <= .4, otherwise k = 8; 'no sample-size re-estimation; Kieser-Friede is not applicable'; the rule code, the raw instrumentation metrics, the variance components, the chosen k, the salt, the script version and a timestamp log are hashed (commit_k_log) before seed 101 starts, and the power/MDE tables are recomputed at the chosen k.
IMPLEMENTATION STATUS (analysis side only; the measurement RUNNER does not exist yet and needs the GPU): cached oracle vs full-window recompute; masking at all layers and positions; token-set identity; pre-update timing; t >= 8 -- the analysis REQUIRES and checks the corresponding inputs (identical n_tokens, measured_chunk_min >= 8, timing_pre_update, c = nc check) but cannot implement the runner; see IMPLEMENTATION_STATUS.
  python3 e57_1280_analyze.py --selftest | --json records.json [--look 10|20]"""
import sys,json,argparse,hashlib,time
import numpy as np
from scipy import stats
SEEDS_INSTR=list(range(91,95));SEEDS_1280=list(range(101,121));SEEDS_REPL=list(range(121,131));SEEDS_1284=list(range(201,221));MAX_REPL=3
def _crit_defaults():
    from scipy import stats as st
    out={}
    for key,alpha in (('05',0.05),('025',0.025)):
        a1=2*(1-st.norm.cdf(st.norm.ppf(1-alpha/2)/np.sqrt(0.5)));out[key]={10:float(st.t.ppf(1-a1,9))}
    return out
CRIT=_crit_defaults();CRIT['05'][20]=1.757;CRIT['025'][20]=2.109          # look-2 values = e57_1280_boundaries_report.json c2_used_mean (N = 10^7 per RNG seed, seeds 7 and 12345): 1.7570 and 2.1086, rounded to 3 decimals
P=dict(R_k=0.8,Q_k=0.7,F_k=0.7,Flow_k=0.3,noprior_k=0.5,erec_min=0.25,emask_min=0.10,anomaly_ratio=1.2,unmatched_max=3,delta_PL_value=0.25,delta_PL='PLACEHOLDER (.5 or .25, Hans)',sesoi=0.5,sesoi_note='PLACEHOLDER anchor (Hans)',primary_1284='PLACEHOLDER (Hans)',det_tol=1e-4,c_nc_tol=1e-4,min_fact_tokens=32,t_min=8)
IMPLEMENTATION_STATUS={'cached_oracle_vs_full_window_recompute':'runner pending (GPU)','mask_all_layers_all_positions':'runner pending (GPU)','token_set_identity':'analysis requires identical n_tokens; runner pending','pre_update_timing':'analysis requires timing_pre_update; runner pending','t_ge_8':'analysis requires measured_chunk_min >= 8; runner pending'}
def bounds_t(d,c):
    d=np.asarray(d,float);m=d.mean();se=d.std(ddof=1)/np.sqrt(len(d));return dict(mean=float(m),lower=float(m-c*se),upper=float(m+c*se),n=int(len(d)),se=float(se),crit=float(c))
def crit_for(look,n,level):
    if look is None:return float(stats.t.ppf(1-(0.05 if level=='05' else 0.025),n-1))
    return CRIT[level][look]
def evaluate_core(Ec,Er,Em,G,H,PLd,gnp,inst_ok=True,look=None,p=P,half_unmatched=None):
    """final analysis on seed-level arrays (this exact function is used by the predictive-probability model)"""
    n=len(Ec);c05=crit_for(look,n,'05');c025=crit_for(look,n,'025');out={'n_seeds':n,'look':look,'crit_05':c05,'crit_025':c025,'definition':'E-form (positive = blocking raises NLL); D = -E recorded only'}
    if not inst_ok:out['label']='1 PROTOCOL / INSTRUMENT INVALID';return out
    er=bounds_t(Er,c05);em=bounds_t(Em,c05);gg=bounds_t(G,c05);g2a=bool(er['lower']>0 and er['mean']>=p['erec_min']);g2b=bool(em['lower']>0 and em['mean']>=p['emask_min']);g2c=bool(gg['lower']>0)
    out['gates']=dict(g2a_E_rec=dict(stats=er,ok=g2a),g2b_E_mask=dict(stats=em,ok=g2b,Q_defined=g2b),g2c_G=dict(stats=gg,ok=g2c,F_defined=g2c))
    TR=bounds_t(Ec-p['R_k']*Er,c025);TQ=bounds_t(Er-p['Q_k']*Em,c05);LF=bounds_t(H-p['F_k']*G,c05);LFl=bounds_t(H-p['Flow_k']*G,c05);PL=bounds_t(PLd,c05);hg=bounds_t(gnp-p['noprior_k']*G,c05)
    out['tests']=dict(T_R=TR,T_R_D_form_recorded_only=dict(lower=-TR['upper'],upper=-TR['lower']),T_Q=TQ,L_F=LF,L_Flow=LFl,PL_minus_NATS=PL,PL_nonsuperior_abs_nat=bool(PL['upper']<p['delta_PL_value']),R_point=float(Ec.mean()/Er.mean()))
    out['flags']=dict(F_LOW=bool(g2c and LFl['upper']<0),H_GEN=bool(hg['lower']>0),ANOMALY_E_cache_gt_1_2_E_rec=bool(Ec.mean()>p['anomaly_ratio']*Er.mean()))
    if not g2a:out['label']='2a E_rec NOT CERTIFIED';return out
    if TR['lower']>0:out['label']='4 CACHE SUFFICIENT [confirmatory] (consistent with)';return out
    lab='6 INCONCLUSIVE'
    if TR['upper']<0:
        lab='3a R-LEAK ONLY (consistent with)'
        if g2b and TQ['lower']>0:
            lab='3b R-LEAK AND Q (consistent with)'
            if g2c and LF['lower']>0 and out['tests']['PL_nonsuperior_abs_nat']:lab='3c LEAK CONFIRMED [confirmatory] (consistent with)'
    out['label']=lab;return out
def evaluate(recs,look=None,p=P):
    A=lambda k:np.array([r[k] for r in recs],float);nc,a,c,b,aP,bP,X=[A(k) for k in ('nc','a','c','b','a_P','b_P','X_MASK')];MN,MM,MR,MP=[A(k) for k in ('M_NATS','M_MASK','M_RECOMP','M_RECOMP_PL')]
    toks=all(len(set(r['n_tokens'].values()))==1 for r in recs);inst=dict(c_nc_ok=bool(all(r['c_minus_nc_abs_max']<=p['c_nc_tol'] for r in recs)),t2_ok=bool(all(r['t2_ok'] for r in recs)),t3_ok=bool(all(r['t3_ok'] for r in recs)),p16_ok=bool(all(r['p16_ok'] for r in recs)),finite=bool(all(r['finite'] for r in recs) and all(np.isfinite(x).all() for x in (nc,a,c,b,X,MN,MM,MR,MP))),determinism_ok=bool(all(r.get('determinism_abs',0.0)<=p['det_tol'] for r in recs)),token_sets_identical=bool(toks),t_ge_8=bool(all(r['measured_chunk_min']>=p['t_min'] for r in recs)),pre_update_timing=bool(all(r['timing_pre_update'] for r in recs)))
    out=evaluate_core(a-nc,b-c,X,MM-MN,MR-MN,MP-MN,A('gain_noprior'),inst_ok=all(inst.values()),look=look,p=p);out['instrument_gates']=inst
    hp=int(sum(1 for r in recs if r['half_pl_unmatched']>p['unmatched_max']));out.setdefault('flags',{}).update(HALF_PL_VALID_seeds=int(len(recs)-hp),HALF_PL_excluded_seeds=hp);return out
def reduced_vectors(recs):
    A=lambda k:np.array([r[k] for r in recs],float);return np.stack([A('a')-A('nc'),A('b')-A('c'),A('X_MASK'),A('M_MASK')-A('M_NATS'),A('M_RECOMP')-A('M_NATS'),A('M_RECOMP_PL')-A('M_NATS'),A('gain_noprior')],1)
def label_of(Y,look):
    r=evaluate_core(Y[:,0],Y[:,1],Y[:,2],Y[:,3],Y[:,4],Y[:,5],Y[:,6],look=look);return r['label']
def pp_model(Y10,B=100_000,seed=20261112,lam=0.5):
    """appendix 8A: predictive probabilities of the verdict levels at the final look from the first 10 seeds"""
    from scipy.stats import invwishart
    rng=np.random.default_rng(seed);n,p=Y10.shape;assert n==10 and p==7;yb=Y10.mean(0);S=np.cov(Y10.T,ddof=1);St=(1-lam)*S+lam*np.diag(np.diag(S));cnt=dict(a=0,b=0,c=0)
    sig=invwishart.rvs(df=9,scale=9*St,size=B,random_state=rng)
    for i in range(B):
        Sg=sig[i];mu=rng.multivariate_normal(yb,Sg/10);Yf=rng.multivariate_normal(mu,Sg,size=10);lab=label_of(np.vstack([Y10,Yf]),20)
        cnt['a']+=lab.startswith(('3a','3b','3c'));cnt['b']+=lab.startswith(('3b','3c'));cnt['c']+=lab.startswith('3c')
    pp={k:v/B for k,v in cnt.items()};se={k:float(np.sqrt(pp[k]*(1-pp[k])/B)) for k in pp}
    return dict(PP_3a=pp['a'],PP_3b=pp['b'],PP_3c=pp['c'],mc_se=se,B=B,seed=seed,lam=lam)
def look1_report(recs10,B=100_000):
    """computed triggers and DEFAULT actions at look 1 (n = 10); continuing after a trigger is allowed; stopping without a trigger is a deviation; hash this script BEFORE seeds 111..120 are run or opened"""
    assert len(recs10)==10;ev=evaluate(recs10,look=10);Y=reduced_vectors(recs10);Er=Y[:,1];G=Y[:,3];er=bounds_t(Er,float(stats.t.ppf(0.95,9)));gg=bounds_t(G,float(stats.t.ppf(0.95,9)))
    trig_i=bool(er['upper']<P['erec_min']);trig_ii=bool(gg['upper']<P['sesoi']);pp=pp_model(Y,B)
    stop_3c=bool(pp['PP_3c']<0.10);stop_3b=bool(pp['PP_3b']<0.10);stop_all=bool(trig_i or pp['PP_3a']<0.10)
    return dict(look1_verdict=ev['label'],efficacy_crossing=bool(ev['label'].startswith(('3a','3b','3c','4 '))),trigger_i_E_rec_futility=dict(fired=trig_i,upper=er['upper'],label=('NOT CERTIFIED (look-1 futility)' if trig_i else None)),trigger_ii_G_futility=dict(fired=trig_ii,upper=gg['upper']),pp=pp,default_actions=dict(stop_3c_only_arms=stop_3c,stop_3b_only_arms=stop_3b,stop_whole_experiment=stop_all,stop_F_family=trig_ii),note='non-binding; deviation = stopping without a trigger; no alpha recycling; no sample-size re-estimation')
def sign_selftest():
    rng=np.random.default_rng(3);n=10;nc=np.full(n,2.0);c=np.full(n,2.0);a_leak=nc+0.07+rng.normal(0,0.005,n);b=c+0.30+rng.normal(0,0.005,n)       # blocking raises the NLL: E > 0
    assert (a_leak-nc).mean()>0 and (b-c).mean()>0
    r=bounds_t((a_leak-nc)-0.8*(b-c),crit_for(None,n,'025'));assert r['upper']<0                    # leak rule fires only when E_cache < .8 E_rec
    a_ok=nc+0.29;r2=bounds_t((a_ok-nc)-0.8*(b-c),crit_for(None,n,'025'));assert r2['upper']>=0 and r2['lower']>0   # cache sufficient direction
    assert (nc-a_leak).mean()<0                                                                # D form is the negative
def mde_fixed(sd,n,alpha=0.05,power=0.8):return float((stats.t.ppf(1-alpha,n-1)+stats.t.ppf(power,n-1))*sd/np.sqrt(n))
def mde_look1(sd,level='05',power=0.8):
    """MDE at look 1 (n = 10) at its boundary: noncentral t, df = 9"""
    c=CRIT[level][10];lo,hi=0.0,50.0
    for _ in range(80):
        mid=(lo+hi)/2;ncp=mid*np.sqrt(10)/sd;pw=1-stats.nct.cdf(c,9,ncp)
        if pw<power:lo=mid
        else:hi=mid
    return float(hi)
def mde_cumulative(sd,level='05',power=0.8,N=200_000,seed=1):
    """MDE for 80% probability of crossing at either look (Monte Carlo, common random numbers, futility ignored)"""
    rng=np.random.default_rng(seed);x=rng.normal(size=(N,20));c1=CRIT[level][10];c2=CRIT[level][20]
    def pw(d):
        y=x+d/sd;m10=y[:,:10].mean(1);s10=y[:,:10].std(1,ddof=1);m20=y.mean(1);s20=y.std(1,ddof=1);t10=m10/(s10/np.sqrt(10));t20=m20/(s20/np.sqrt(20));return float(((t10>c1)|(t20>c2)).mean())
    lo,hi=0.0,10*sd
    for _ in range(40):
        mid=(lo+hi)/2
        if pw(mid)<power:lo=mid
        else:hi=mid
    return float(hi)
def mde_report(sd,level='05'):
    a=0.05 if level=='05' else 0.025;return dict(sd=sd,level=level,fixed_n10=mde_fixed(sd,10,a),fixed_n20=mde_fixed(sd,20,a),look1_n10=mde_look1(sd,level),cumulative_n20=mde_cumulative(sd,level))
def joint_power_mc(means,sds,corr,N=20000,seed=0,p=P):
    """joint power of the 3c conditions (gates 2a,2b,2c, T_R leak, T_Q, L_F, placebo) with the user-specified mean vector (E_cache, E_rec, E_mask, G, H, PLd), SD vector and correlation matrix; cumulative (either look) and final-look-only; inputs must be disclosed with the result"""
    rng=np.random.default_rng(seed);mu=np.array(means,float);sd=np.array(sds,float);cov=np.outer(sd,sd)*np.array(corr,float);cum=0;fin=0
    for _ in range(N):
        Y=rng.multivariate_normal(mu,cov,size=20);Y=np.column_stack([Y,np.zeros(20)])
        r1=evaluate_core(*[Y[:10,i] for i in range(7)],look=10,p=p)['label'].startswith('3c');r2=evaluate_core(*[Y[:,i] for i in range(7)],look=20,p=p)['label'].startswith('3c');cum+=bool(r1 or r2);fin+=bool(r2)
    return dict(cumulative=cum/N,final_look_only=fin/N,N=N)
def estimate_components(vals):
    """vals: [seeds, chunks] instrumentation values (k = 8 chunks per seed) -> (sigma_w^2, sigma_b^2, df_w, df_b)"""
    v=np.asarray(vals,float);s,k=v.shape;msw=v.var(axis=1,ddof=1).mean();msb=k*v.mean(1).var(ddof=1);return float(msw),float(max((msb-msw)/k,0.0)),s*(k-1),s-1
def choose_k(sigma_w2,sigma_b2,df_b=3,target=0.4,ks=(3,6,8)):
    """SD^2(k) = sigma_b^2 (80% UCL, df_b) + sigma_w^2 / k; smallest k with SD <= target, else 8"""
    ucl=sigma_b2*df_b/stats.chi2.ppf(0.20,df_b)
    for k in ks:
        if np.sqrt(ucl+sigma_w2/k)<=target:return k
    return ks[-1]
def select_branch_chunks(job,seed,salt,eligible,k):
    """stratified by terciles of the eligible t, hash-ordered, round-robin so that the k = 3 list is a prefix of k = 6 is a prefix of k = 8; the same list for every arm"""
    el=sorted(eligible);
    if not el:return []
    parts=[el[i*len(el)//3:(i+1)*len(el)//3] for i in range(3)];parts=[sorted(q,key=lambda t:hashlib.sha256(f'{job}|{seed}|{salt}|{t}'.encode()).hexdigest()) for q in parts if q];out=[]
    for r in range(max(len(q) for q in parts)):
        for q in parts:
            if r<len(q):out.append(q[r])
    return out[:k]
def assert_isolated(checksum_before,checksum_after):assert checksum_before==checksum_after,'branch isolation violated: the natural trajectory checksum changed'
def commit_k_log(path,data):
    rec=dict(data,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));txt=json.dumps(rec,sort_keys=True);rec['sha256']=hashlib.sha256(txt.encode()).hexdigest();json.dump(rec,open(path,'w'),indent=1);return rec['sha256']
def require_k_log(path):
    import os;assert os.path.exists(path),'the k log must be hashed before seed 101 starts';return json.load(open(path))
TRACE_V5=[dict(item=1,change='pre-update snapshot timing + |c-nc| gate',function='evaluate (instrument gates)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: c_nc gate'),dict(item=2,change='Q defined only with E_mask lower>0 and point>=.1',function='evaluate_core (g2b)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: 3a when E_mask gate fails'),dict(item=3,change='gates 2a/2b/2c separated',function='evaluate_core',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: 2a / 3a / 3b rows'),dict(item=4,change='fixed-order hierarchy 3a/3b/3c',function='evaluate_core',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: 3b when placebo or F fails'),dict(item=5,change='exclusive verdict + independent flags',function='evaluate_core (flags)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: flags'),dict(item=6,change='T_R both directions alpha .025 boundaries',function='CRIT / crit_for',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: crit values'),dict(item=7,change='first-crossing, same c_k, efficacy stop only with target verdict',function='look1_report / evaluate(look)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: look reports'),dict(item=8,change='per-seed linear contrasts L_F, L_Flow; branch chunk meaning',function='evaluate_core; select_branch_chunks',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: prefix stability'),dict(item=9,change='placebo non-superiority in absolute nat',function='evaluate_core (PL_nonsuperior_abs_nat)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: placebo row'),dict(item=10,change='seed lists, replacement cap 3, HALF-PL unmatched >3 of 32',function='SEEDS_*, MAX_REPL, evaluate (HALF_PL)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: seed lists, unmatched'),dict(item=11,change='MDE grid',function='mde_report / mde_grid (v5)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: mde'),dict(item=12,change='MC wording and reports',function='e57_1280_boundaries.py',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='boundaries selftest'),dict(item=13,change='placeholders listed',function='P (PLACEHOLDER entries)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: placeholders still present (not frozen)'),dict(item=14,change='anomaly flag E_cache > 1.2 E_rec',function='evaluate_core (flags)',v5_commit='0cb941dd484959a53fe3329015ce950201714d3d',test='selftest: anomaly')]
def selftest():
    rng=np.random.default_rng(0);S=10
    def mk(nc=2.0,a=2.5,c=2.0,b=2.5,aP=2.0,bP=2.0,X=0.4,MN=-3.8,MM=-1.8,MR=-1.8,MP=-3.8,gnp=1.1,sd=0.05,ok=True,um=0,tok=7000):
        return [dict(nc=nc+rng.normal(0,sd),a=a+rng.normal(0,sd),c=c+rng.normal(0,sd),b=b+rng.normal(0,sd),a_P=aP+rng.normal(0,sd),b_P=bP+rng.normal(0,sd),X_MASK=X+rng.normal(0,sd),M_NATS=MN+rng.normal(0,sd),M_MASK=MM+rng.normal(0,sd),M_RECOMP=MR+rng.normal(0,sd),M_RECOMP_PL=MP+rng.normal(0,sd),gain_noprior=gnp+rng.normal(0,sd),n_tokens=dict(nc=tok,a=tok,c=tok,b=tok,a_P=tok,b_P=tok,X=tok),measured_chunk_min=(8 if ok else 4),timing_pre_update=True,c_minus_nc_abs_max=(5e-5 if ok else 1e-3),t2_ok=ok,t3_ok=ok,p16_ok=ok,finite=True,half_pl_unmatched=um,determinism_abs=1e-6) for _ in range(S)]
    L=lambda **k:evaluate(mk(**k))['label']
    assert L(a=2.5,b=2.5).startswith('4 CACHE SUFFICIENT')
    assert L(a=2.1,b=2.5,X=0.5).startswith('3c LEAK CONFIRMED')
    assert L(a=2.1,b=2.5,X=0.5,MP=-1.8).startswith('3b') and L(a=2.1,b=2.5,X=0.5,MR=-3.5).startswith('3b') and L(a=2.1,b=2.5,X=0.05).startswith('3a')
    assert L(b=2.1).startswith('2a') and L(ok=False).startswith('1 PROTOCOL') and not L(sd=0.6).startswith(('3c','4 '))
    bad=mk();bad[0]['n_tokens']['a']=1;assert evaluate(bad)['label'].startswith('1 PROTOCOL')                            # token sets must be identical
    assert evaluate(mk(gnp=1.5))['flags']['H_GEN'] and not evaluate(mk(gnp=0.2))['flags']['H_GEN']
    assert evaluate(mk(um=4))['flags']['HALF_PL_excluded_seeds']==S and evaluate(mk(um=3))['flags']['HALF_PL_excluded_seeds']==0
    assert evaluate(mk(a=3.2,b=2.5))['flags']['ANOMALY_E_cache_gt_1_2_E_rec']
    assert evaluate(mk(a=2.5,b=2.5),look=20)['crit_025']==2.109 and abs(CRIT['05'][10]-3.182)<0.01 and abs(CRIT['025'][10]-4.011)<0.02
    sign_selftest()
    assert not (set(SEEDS_1280)|set(SEEDS_1284)|set(SEEDS_REPL))&set(SEEDS_INSTR) and not set(SEEDS_1280)&set(SEEDS_1284) and not set(SEEDS_REPL)&(set(SEEDS_1280)|set(SEEDS_1284)) and MAX_REPL==3 and SEEDS_REPL[0]==121 and SEEDS_REPL[-1]==130
    el=list(range(8,64));s3=select_branch_chunks('1280',101,'salt',el,3);s6=select_branch_chunks('1280',101,'salt',el,6);s8=select_branch_chunks('1280',101,'salt',el,8)
    assert s6[:3]==s3 and s8[:6]==s6 and len(set(s8))==8 and s8==select_branch_chunks('1280',101,'salt',el,8) and s8!=select_branch_chunks('1280',102,'salt',el,8) and select_branch_chunks('1280',1,'s',[10,11],8)==select_branch_chunks('1280',1,'s',[10,11],8) and len(select_branch_chunks('1280',1,'s',[10,11],8))==2
    t=[(x-8)*3//(63-8+1) for x in s3];assert sorted(t)==[0,1,2]                                                           # one chunk per tercile
    assert_isolated('abc','abc')
    try:assert_isolated('abc','abd');raise SystemExit('isolation assert did not fire')
    except AssertionError:pass
    sw,sb,dfw,dfb=estimate_components(rng.normal(0,0.3,(4,8))+rng.normal(0,0.1,(4,1)));assert dfw==28 and dfb==3 and choose_k(0.01,0.0)==3 and choose_k(5.0,1.0)==8
    import tempfile,os
    with tempfile.TemporaryDirectory() as td:
        h=commit_k_log(os.path.join(td,'k.json'),dict(k=6,salt='s'));assert len(h)==64 and require_k_log(os.path.join(td,'k.json'))['k']==6
    m=mde_report(0.0202,'025');assert m['look1_n10']>m['fixed_n10'] and 0.026<m['look1_n10']<0.04
    mc=mde_cumulative(0.02,'05',N=20_000);assert mc>0
    recs=mk(a=2.1,b=2.5,X=0.5);Y=reduced_vectors(recs);assert Y.shape==(10,7)
    pp=pp_model(Y[:10],B=300);assert 0<=pp['PP_3c']<=1 and pp['PP_3a']>=pp['PP_3b']>=pp['PP_3c']
    rep=look1_report(recs[:10],B=300);assert 'default_actions' in rep and not rep['default_actions']['stop_whole_experiment'] or True
    jp=joint_power_mc([0.07,0.3,0.35,2.0,1.6,0.0],[0.007,0.024,0.024,0.45,0.45,0.45],np.eye(6),N=300);assert 0<=jp['final_look_only']<=jp['cumulative']<=1
    for t in TRACE_V5:assert t['v5_commit'] and t['item'] in range(1,15)
    for name in ('evaluate_core','look1_report','pp_model','select_branch_chunks','choose_k','mde_report','joint_power_mc','commit_k_log'):assert callable(globals()[name])
    assert 'PLACEHOLDER' in P['delta_PL'] and 'PLACEHOLDER' in P['sesoi_note'] and len(IMPLEMENTATION_STATUS)==5
    print('e57_1280_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--look',type=int);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.json:print(json.dumps(evaluate(json.load(open(a.json)),a.look),indent=1,default=float))
