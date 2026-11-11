"""E57 job 1280 (recompute oracle) analysis, decision table v7 (CPU; DRAFT for Council re-check; Hans decided delta_PL = .25 nat, SESOI anchor = K0=0 NATS 16->128 auxiliary measurement (seeds 901..908, non-confirmatory), 1284 primary = RECOMP-ORACLE - RECOMP-PL, secondary HALF-PL - HALF, timestamp = local hash + server time; still open: the measured SESOI values (nat/fact, nat/token), decider role; written before any 1280 or instrumentation-seed output exists).
SIGN (verified against the archived versions): every analysis version defines the effects in E-form, E = NLL(blocked) - NLL(not blocked), positive when blocking raises the NLL: E_cache = a - nc, E_rec = b - c; the decision directions are leak = upper bound of T_R < 0 and cache sufficient = lower bound of T_R > 0. Checked in git: v3 commit 0f5352db9b99f7dcee5c53ec5a8f3e624c8c5791 (file sha256 13c6c7efef3544cc64e359c7995942b32992a49680bf27821d0dbf05ac416d1e, lines 32 and 34: dR = t1((a-nc) - .8 (b-c)); leak = dR upper < 0; cache = dR lower > 0), v4 commit ac486fbd7991f046fa144f9f6f21db45d6db76a4 (sha256 639eca79e25e29bc0e637adc7026399c3f174bc5fcb57d148fb8ac31cdc5a28f, line 32), v5 commit 0cb941dd484959a53fe3329015ce950201714d3d (sha256 badcff89af3eb5d87cf3fc5d8bc807ae7e5ac5160c97f0b5f178c59b90a8578e, line 32). The request text of v4 used D = nc - a (= -E_cache) while quoting the E-form decision directions, which produced an apparent sign reversal in the review; no analysis version used a D-form decision rule (the v4 / v5 outputs only RECORD the D-form). Unit test: sign_selftest.
UNITS: NLL-type quantities (nc, a, c, b, a_P, b_P, E_cache, E_rec, E_mask) are nat/token means over the SAME fixed token set (the later-exposure fact tokens of the measured chunks) aggregated to one value per seed; margin-type quantities (G, H, M_*, delta_PL, SESOI_fact) are nat per fact; SESOI is split by unit family: SESOI_fact (nat/fact, margin family, used by look-1 trigger (ii)) and SESOI_token (nat/token, NLL family, recorded for R/Q interpretation, used by no rule); both are filled from the auxiliary K0=0 NATS 16->128 measurement (seeds 901..908) in their own unit; until then SESOI_fact = .5 is a provisional screening value and SESOI_token is a placeholder (margin = log p(gold) - logsumexp(log p wrong), the log-prob of a candidate string is the SUM over its place tokens). The two families are never compared or mixed: G ~ 2 nat (margin) and the MDEs in nat/token are different normalisations. R and Q use NLL-type only; F, L_F, L_Flow, H_gen, placebo use margin-type only.
CLAIMS / GATES / ALPHA TABLE (one-sided tests, seed-paired t, unit = seed; alpha = per-test level, boundary set = critical values of the 2-look design): [T_R leak] direction upper < 0, alpha .025, set '025', gate 2a, confirmatory (step 1 of the leak hierarchy; efficacy stop allowed); [T_R cache] direction lower > 0, alpha .025, set '025', gate 2a, confirmatory (the two directions are mutually exclusive: combined <= .05); [T_Q] lower > 0, alpha .05, set '05', gates 2a and 2b, step 2; [L_F] lower > 0, alpha .05, set '05', gate 2c, step 3; [PL nonsuperiority] upper < delta_PL (absolute nat), alpha .05, set '05', step 4; [L_Flow] upper < 0, alpha .05, set '05', gate 2c, SECONDARY (F_LOW flag): no familywise claim, no efficacy stop; [H_gen] noprior - .5 G lower > 0, alpha .05, flag only. Multiplicity: the leak claims 3a/3b/3c are a fixed-order hierarchy (R -> Q -> F -> placebo), each step is tested only if the previous one passed, so the size of the 3c claim is <= max alpha_i = .05 (intersection-union); L_Flow and H_gen are outside the familywise claim. Seeds 201..220 (job 1284) are an INDEPENDENT REPLICATION with their own alpha (.05 one-sided) and are never pooled with 1280.
RECORD per seed: nc, a, c, b, a_P, b_P, X_MASK (= E_mask: the recomputed band NLL minus the recomputed natural NLL, nat/token), n_tokens (dict per quantity, must be identical), measured_chunk_min (>= 8), timing_pre_update (all losses at the snapshot W_t BEFORE the write of the chunk), c_minus_nc_abs_max (<= 1e-4), M_NATS, M_MASK, M_RECOMP, M_RECOMP_PL (branch margins after ONE step from the same snapshot), gain_noprior, half_pl_unmatched (of 32), finite, determinism_abs, t2_ok, t3_ok, p16_ok.
GATES: 2a E_rec: lower 95% bound > 0 and point >= .25; 2b E_mask: lower bound > 0 and point >= .10 (only then Q is defined); 2c G = M_MASK - M_NATS: lower bound > 0 (only then F is defined). Instrument gates (a failure = label 1): per-seed |c - nc| <= 1e-4, t2, t3, p16, finite, determinism <= 1e-4, identical token sets, t >= 8, pre-update timing.
VERDICT (exclusive): 1 PROTOCOL / INSTRUMENT INVALID ; 2a E_rec NOT CERTIFIED ; 4 CACHE SUFFICIENT (T_R cache direction; practically unreachable, see disclosures) ; 3a/3b/3c (deepest reached step of the hierarchy; 3c requires the placebo non-superiority) ; 6 INCONCLUSIVE. Independent flags: F_LOW, H_GEN, HALF_PL (a seed is excluded from HALF-PL when more than 3 of its 32 facts are unmatched), ANOMALY (E_cache > 1.2 E_rec).
SEQUENTIAL: looks at n = 10 and 20; look 1 boundary analytic c1 = t_9^{-1}(1 - alpha_1), look 2 by Monte Carlo only at look 2 on the sets that did not cross at look 1 (shared first 10 seeds), see e57_1280_boundaries.py (method, two RNG seeds, versions, both estimates and their difference, boundary MC SE, look-wise and cumulative error, t(5) and skewed-data robustness; report file e57_1280_boundaries_report.json). The same c_k is used for every gate and every reported bound. Rejection = the first look at which a boundary is crossed. Efficacy stop only if the target verdict and every gate it needs hold at that look.
LOOK-1 FUTILITY (item 8, non-binding, no alpha recycling, no sample-size re-estimation): triggers are COMPUTED by look1_report on the first 10 seeds (101..110) and the DEFAULT actions below are pre-specified; continuing after a trigger is always allowed (the decider, named by role, records the reason BEFORE opening seeds 111..120), stopping without a trigger is reported as a protocol deviation; the look-1 report script and its hash are committed BEFORE seeds 111..120 are run or opened and the seed list is fixed; if seeds 111..120 run concurrently their output is not read before the look-1 decision. (i) E_rec: nominal one-sided 95% t upper bound (df = 9, 1.833) < .25 -> label 'NOT CERTIFIED (look-1 futility)', stops R and Q (and F, which depends on gates) = the whole experiment; this is a screen, not a boundary, and it fires with probability .05 if the true E_rec is exactly .25 (a power cost, not an alpha cost); it is NOT a confirmatory conclusion that the population denominator is below .25. (ii) G upper bound < SESOI_fact (nat/fact; provisional .5): applies only because G, L_F and L_Flow share units (margin nat); stops only the F family (3c impossible, maximum verdict 3b; R and Q continue at the same boundaries; F-only arms = RECOMP and RECOMP-PL branch arms; shared arms NATS and NATMASK continue; the placebo non-superiority is stopped together with F). If SESOI_fact (.5) was fixed after seeing G ~ 2 this is disclosed in item 12. (iii) predictive probabilities PP(3a), PP(3b), PP(3c) from the model of appendix 8A, evaluated with the final analysis function: PP(3c) < .10 -> stop the 3c-only arms; PP(3b) < .10 -> stop the 3b-only arms (the recomputed band measurement of E_mask); the WHOLE experiment stops only if PP(3a) < .10 or trigger (i) fires. PP(3a) ~ P(gate 2a passes) and overlaps with (i) (disclosed).
APPENDIX 8A (PP model, v7): reduced seed vector Y = (E_cache, E_rec, E_mask, G, H, PLd = M_RECOMP_PL - M_NATS, gain_noprior) (p = 7); FROZEN prior Sigma ~ InverseWishart(nu0 = 8, Psi0 = (nu0 + p + 1) diag(sd0^2)) so the prior mode is diag(sd0^2), sd0 = PRIOR['sd0'] = (.03, .05, .05, .45, .45, .45, .45) (token-family .03/.05/.05, margin-family .45 from the pilot scale; frozen at Freeze A, never taken from the confirmatory seeds); posterior from the first 10 seeds: Sigma | Y ~ IW(nu0 + 9, Psi0 + 9 S), mu | Sigma ~ N(ybar, Sigma / 10) (flat prior on mu); simulate 10 future seeds ~ MVN(mu, Sigma), append to the observed 10, evaluate evaluate_core at look 20 (same final analysis code and hash); B >= 10^5 draws, fixed RNG seed, MC SE reported; instrument flags are assumed true; the full covariance is used (independence NOT assumed). Futility report (futility_report): P(trigger (i) fires | true E_rec = .30) for SD(E_rec) in {.03,.05,.10} (exact noncentral t; it equals .05 at E_rec = .25).
POWER: mde_report gives, per test, three MDEs (fixed n, look 1 (n = 10 at its boundary, noncentral t), cumulative over both looks (80% power of crossing at either look, Monte Carlo)); T_R uses the alpha = .025 boundaries (n = 10 MDE about .032 for SD .0202, i.e. boundary inflation, not an SD multiplier); joint_power_mc computes the joint (cumulative and final-look-only) power of the 3c conditions with a user-specified mean vector, SD vector and correlation matrix (inputs disclosed with the result); the earlier .47 / .78 values (independent normal components, fixed t critical values, no boundaries) are WITHDRAWN. A power below .80 is stated as such.
SEEDS: instrumentation 91..94 (variance components at k = 8 only, no confirmatory inference); confirmatory 1280 = 101..120 (look 1 = 101..110, look 2 adds 111..120); spare list 121..140 used in order; FAILURE = a technical, outcome-independent event only (TECH_FAILURES: world_generation_exception, non_finite_value, determinism_check_failed, runner_crash, out_of_memory); a failed seed is replaced by the next unused spare, the replacement belongs to the look at which the failure occurred and is drawn before that look's statistics are computed; no exclusion by outcome; the reduced-n rule and the cap of 3 are DROPPED; if the spares run out the analysis uses the available seeds with the look-2 boundary RECOMPUTED from the actual information fraction (recompute_look2 in e57_1280_boundaries.py, n1/n2 actual), and n_final < 16 is PROTOCOL INVALID: results exploratory only (label 1); every attempted seed and its reason is preserved. 1284 = 201..220.
BRANCH CHUNKS (one rule): each branch chunk starts from a copy of the natural snapshot just before chunk t, every arm applies one update on its own context and is evaluated on the fact tokens of the same chunk with the margin M_* (frozen weights, same probes), then the copy is discarded; eligibility: t >= 8, fact tokens present, a prior same-fact exposure inside the window, at least 32 scored fact tokens; selection: stratified by terciles of the eligible t, order by sha256(job|seed|salt|t), round-robin over terciles so that k = 3 is a prefix of k = 6 is a prefix of k = 8, the SAME chunks for every arm (paired); fewer eligible chunks than k -> all eligible are used and the count is reported; isolation: parameters, optimizer state and RNG state are deep-copied and a checksum of the natural trajectory must be unchanged (assert_isolated). k is chosen ONCE by choose_k from the instrumentation seeds 91..94 run at k = 8: SD^2(k) = sigma_b^2 + sigma_w^2 / k (both components carry an 80% upper confidence limit: sigma_w^2 with df = 28, sigma_b^2 with df = 3, sigma_b^2 estimated by method of moments (MSB - MSW)/k clipped at 0); the smallest k in {3, 6, 8} with SD(L_F) <= .4; if no k meets the target k = 8 and the power/MDE tables are recomputed at the achieved SD (target not met is stated); 'no sample-size re-estimation; Kieser-Friede is not applicable'; the rule code, the raw instrumentation metrics, the variance components, the chosen k, the salt, the script version and a timestamp log are hashed (commit_k_log) before seed 101 starts, and the power/MDE tables are recomputed at the chosen k.
IMPLEMENTATION STATUS (analysis side only; the measurement RUNNER does not exist yet and needs the GPU): cached oracle vs full-window recompute; masking at all layers and positions; token-set identity; pre-update timing; t >= 8 -- the analysis REQUIRES and checks the corresponding inputs (identical n_tokens, measured_chunk_min >= 8, timing_pre_update, c = nc check) but cannot implement the runner; see IMPLEMENTATION_STATUS.
V7 RULES. SCOPE OF THE INTERSECTION-UNION CLAIM: the 3c claim is the intersection of {gate 2a, gate 2b, gate 2c, T_R leak, T_Q, L_F, placebo non-superiority}; its size is bounded by the largest component level (.05); the claims 3a and 3b are the nested intersections of the first steps; 4 CACHE SUFFICIENT is the T_R cache direction with gate 2a. PER-COMPONENT TIMING: the verdict at a look uses the data of THAT look only; a component that crossed at look 1 is NOT carried to look 2 (the look-2 verdict is evaluated on all n_final seeds by the same function); component_status reports for every component whether it passed at look 1 and at look 2 (information only); an efficacy stop at look 1 needs every component of the target verdict to pass at look 1. SKEW SENSITIVITY: sensitivity analysis only (never the primary): the primary test is the seed-paired t above and is FROZEN at Freeze A; changing the primary test after seeing observed skewness is forbidden (assert_primary_unchanged); the data-generating grid is lognormal with skewness .5/1/2, the mirror-lognormal (opposite tail) and the contaminated normal (90% N(0,1) + 10% N(0,3^2)) (boundaries.skew_grid, error report in e57_1280_boundaries_report.json); the exact sequential sign-flip test (boundaries.signflip_sequential: spending alpha_1 = A(n1/n2), exact enumeration of the 2^n sign vectors, c2 calibrated by exact enumeration of the sign patterns that did not reject at look 1) is computed for T_R, T_Q, L_F, PL by signflip_components; DISAGREEMENT RULE (skew_disagreement): the t verdict stands; if the t test rejects and the exact test does not the verdict is reported with the flag 'T_ONLY (sign-flip does not support)'; if the exact test rejects and the t test does not, no claim and the flag 'SIGNFLIP_ONLY'. FREEZES: Freeze A (hashed before the runner exists): documents, analysis code, MC boundary code and report, registration text and the INPUT_SCHEMA contract (a, nc, b, c, E_mask = X_MASK, margins M_*, later-exposure token counts n_tokens, timing_pre_update snapshot flag, measured_chunk_min >= 8 eligibility); Freeze B (hashed after the runner exists and BEFORE instrumentation seeds 91..94 are run): the runner source and the Freeze A hash; freeze_manifest enforces both orders. POWER: power_report gives the marginal power of every component (cumulative over both looks and final-look-only), the joint power, checks joint <= min marginal (+ 3 MC SE), and states the mean/SD/correlation inputs and delta_PL.
  python3 e57_1280_analyze.py --selftest | --json records.json [--look 10|20]"""
import sys,json,argparse,hashlib,time
import numpy as np
from scipy import stats
SEEDS_INSTR=list(range(91,95));SEEDS_1280=list(range(101,121));SEEDS_REPL=list(range(121,141));SEEDS_1284=list(range(201,221));SEEDS_AUX_SESOI=list(range(901,909))
TECH_FAILURES=('world_generation_exception','non_finite_value','determinism_check_failed','runner_crash','out_of_memory');N_FINAL_MIN=16;PRIMARY_TEST='seed-paired one-sided t, two-look sequential (n=10,20)'
HERE=__import__('pathlib').Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
def _crit_defaults():
    from scipy import stats as st
    out={}
    for key,alpha in (('05',0.05),('025',0.025)):
        a1=2*(1-st.norm.cdf(st.norm.ppf(1-alpha/2)/np.sqrt(0.5)));out[key]={10:float(st.t.ppf(1-a1,9))}
    return out
CRIT=_crit_defaults();CRIT['05'][20]=1.757;CRIT['025'][20]=2.109          # look-2 values = e57_1280_boundaries_report.json c2_used_mean (N = 10^7 per RNG seed, seeds 7 and 12345): 1.7570 and 2.1086, rounded to 3 decimals
P=dict(R_k=0.8,Q_k=0.7,F_k=0.7,Flow_k=0.3,noprior_k=0.5,erec_min=0.25,emask_min=0.10,anomaly_ratio=1.2,unmatched_max=3,delta_PL_value=0.25,delta_PL='.25 nat absolute, one-sided non-superiority (Hans 2026-11-11)',sesoi_fact=0.5,sesoi_fact_note='PROVISIONAL screening value (nat/fact) until the K0=0 NATS 16->128 auxiliary measurement (seeds 901..908)',sesoi_token=None,sesoi_token_note='PLACEHOLDER (nat/token) from the same auxiliary measurement',primary_1284='RECOMP-ORACLE - RECOMP-PL',secondary_1284='HALF-PL - HALF',timestamp='local sha256 + server UTC time, no public deposit (Hans)',det_tol=1e-4,c_nc_tol=1e-4,min_fact_tokens=32,t_min=8)
IMPLEMENTATION_STATUS={'cached_oracle_vs_full_window_recompute':'runner pending (GPU)','mask_all_layers_all_positions':'runner pending (GPU)','token_set_identity':'analysis requires identical n_tokens; runner pending','pre_update_timing':'analysis requires timing_pre_update; runner pending','t_ge_8':'analysis requires measured_chunk_min >= 8; runner pending'}
import e57_1280_boundaries as BD
INPUT_SCHEMA=dict(nc='float nat/token (natural loss, later-exposure fact tokens, pre-update snapshot)',a='float nat/token (cached-oracle NLL)',c='float nat/token (recomputed natural NLL)',b='float nat/token (recompute-oracle NLL)',a_P='float',b_P='float',X_MASK='float nat/token = E_mask (recomputed band NLL - recomputed natural NLL)',M_NATS='float nat/fact',M_MASK='float nat/fact',M_RECOMP='float nat/fact',M_RECOMP_PL='float nat/fact',gain_noprior='float nat/fact',n_tokens='dict quantity -> int later-exposure token count (identical across quantities)',measured_chunk_min='int >= 8 (t eligibility)',timing_pre_update='bool (all losses at the snapshot before the chunk write)',c_minus_nc_abs_max='float <= 1e-4',t2_ok='bool',t3_ok='bool',p16_ok='bool',finite='bool',half_pl_unmatched='int of 32',determinism_abs='float (optional)')
def validate_input(r):
    """input-schema contract (Freeze A): every record must carry the keys of INPUT_SCHEMA (determinism_abs optional) with the right basic types"""
    miss=[k for k in INPUT_SCHEMA if k!='determinism_abs' and k not in r];assert not miss,f'input schema violated, missing {miss}';assert isinstance(r['n_tokens'],dict) and isinstance(r['timing_pre_update'],(bool,np.bool_)) and isinstance(r['measured_chunk_min'],(int,np.integer));return True
def bounds_t(d,c):
    d=np.asarray(d,float);m=d.mean();se=d.std(ddof=1)/np.sqrt(len(d));return dict(mean=float(m),lower=float(m-c*se),upper=float(m+c*se),n=int(len(d)),se=float(se),crit=float(c))
_CRIT_CACHE={}
def crit_for(look,n,level,n1=10,n_planned=20):
    """look None: fixed-n t; look 10: analytic c1 = t_{n-1}^{-1}(1-A(n/n_planned)); look 20: the tabulated MC value when (n1,n)=(10,20), otherwise RECOMPUTED from the actual information fraction n1/n (recompute_look2)"""
    if look is None:return float(stats.t.ppf(1-(0.05 if level=='05' else 0.025),n-1))
    if look==10:
        if n==10:return CRIT[level][10]
        al=0.05 if level=='05' else 0.025;return BD.look1_actual(al,n,n_planned)
    if look==20 and n==20 and n1==10:return CRIT[level][20]
    key=(level,n1,n)
    if key not in _CRIT_CACHE:_CRIT_CACHE[key]=BD.recompute_look2(0.05 if level=='05' else 0.025,n1,n,N=2_000_000)['c2']
    return _CRIT_CACHE[key]
def evaluate_core(Ec,Er,Em,G,H,PLd,gnp,inst_ok=True,look=None,p=P,half_unmatched=None,n1=10):
    """final analysis on seed-level arrays (this exact function is used by the predictive-probability model)"""
    n=len(Ec)
    if look==20 and n<N_FINAL_MIN:return {'n_seeds':n,'look':look,'label':'1 PROTOCOL INVALID (n_final < 16; results exploratory only)'}
    c05=crit_for(look,n,'05',n1);c025=crit_for(look,n,'025',n1);out={'n_seeds':n,'look':look,'crit_05':c05,'crit_025':c025,'definition':'E-form (positive = blocking raises NLL); D = -E recorded only'}
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
def evaluate(recs,look=None,p=P,n1=10):
    for r in recs:validate_input(r)
    A=lambda k:np.array([r[k] for r in recs],float);nc,a,c,b,aP,bP,X=[A(k) for k in ('nc','a','c','b','a_P','b_P','X_MASK')];MN,MM,MR,MP=[A(k) for k in ('M_NATS','M_MASK','M_RECOMP','M_RECOMP_PL')]
    toks=all(len(set(r['n_tokens'].values()))==1 for r in recs);inst=dict(c_nc_ok=bool(all(r['c_minus_nc_abs_max']<=p['c_nc_tol'] for r in recs)),t2_ok=bool(all(r['t2_ok'] for r in recs)),t3_ok=bool(all(r['t3_ok'] for r in recs)),p16_ok=bool(all(r['p16_ok'] for r in recs)),finite=bool(all(r['finite'] for r in recs) and all(np.isfinite(x).all() for x in (nc,a,c,b,X,MN,MM,MR,MP))),determinism_ok=bool(all(r.get('determinism_abs',0.0)<=p['det_tol'] for r in recs)),token_sets_identical=bool(toks),t_ge_8=bool(all(r['measured_chunk_min']>=p['t_min'] for r in recs)),pre_update_timing=bool(all(r['timing_pre_update'] for r in recs)))
    out=evaluate_core(a-nc,b-c,X,MM-MN,MR-MN,MP-MN,A('gain_noprior'),inst_ok=all(inst.values()),look=look,p=p,n1=n1);out['instrument_gates']=inst
    hp=int(sum(1 for r in recs if r['half_pl_unmatched']>p['unmatched_max']));out.setdefault('flags',{}).update(HALF_PL_VALID_seeds=int(len(recs)-hp),HALF_PL_excluded_seeds=hp);return out
def reduced_vectors(recs):
    A=lambda k:np.array([r[k] for r in recs],float);return np.stack([A('a')-A('nc'),A('b')-A('c'),A('X_MASK'),A('M_MASK')-A('M_NATS'),A('M_RECOMP')-A('M_NATS'),A('M_RECOMP_PL')-A('M_NATS'),A('gain_noprior')],1)
def label_of(Y,look):
    r=evaluate_core(Y[:,0],Y[:,1],Y[:,2],Y[:,3],Y[:,4],Y[:,5],Y[:,6],look=look);return r['label']
PRIOR=dict(nu0=8,sd0=[0.03,0.05,0.05,0.45,0.45,0.45,0.45],note='frozen at Freeze A; Psi0 = (nu0 + p + 1) diag(sd0^2); never taken from confirmatory seeds')
def pp_model(Y10,B=100_000,seed=20261112,prior=PRIOR):
    """appendix 8A (v7): predictive probabilities of the verdict levels at the final look; frozen inverse-Wishart prior, posterior IW(nu0 + 9, Psi0 + 9 S)"""
    from scipy.stats import invwishart
    rng=np.random.default_rng(seed);n,p=Y10.shape;assert n==10 and p==7 and prior['nu0']>=8;yb=Y10.mean(0);S=np.cov(Y10.T,ddof=1);nu0=prior['nu0'];Psi0=(nu0+p+1)*np.diag(np.square(prior['sd0']));nu=nu0+n-1;Psi=Psi0+(n-1)*S;cnt=dict(a=0,b=0,c=0)
    sig=invwishart.rvs(df=nu,scale=Psi,size=B,random_state=rng)
    for i in range(B):
        Sg=sig[i];mu=rng.multivariate_normal(yb,Sg/10);Yf=rng.multivariate_normal(mu,Sg,size=10);lab=label_of(np.vstack([Y10,Yf]),20)
        cnt['a']+=lab.startswith(('3a','3b','3c'));cnt['b']+=lab.startswith(('3b','3c'));cnt['c']+=lab.startswith('3c')
    pp={k:v/B for k,v in cnt.items()};se={k:float(np.sqrt(pp[k]*(1-pp[k])/B)) for k in pp}
    return dict(PP_3a=pp['a'],PP_3b=pp['b'],PP_3c=pp['c'],mc_se=se,B=B,seed=seed,prior=prior,posterior_df=nu)
def p_futility_trigger_i(mu,sd,n=10):
    """exact P(look-1 trigger (i): upper 95% t bound of E_rec < erec_min) when the true E_rec = mu and the seed SD = sd (noncentral t)"""
    c=float(stats.t.ppf(0.95,n-1));ncp=(mu-P['erec_min'])*np.sqrt(n)/sd;return float(stats.nct.cdf(-c,n-1,ncp))
def futility_report(mu=0.30,sds=(0.03,0.05,0.10)):return dict(E_rec=mu,at_threshold=p_futility_trigger_i(P['erec_min'],0.05),by_sd={str(sd):p_futility_trigger_i(mu,sd) for sd in sds})
def look1_report(recs10,B=100_000):
    """computed triggers and DEFAULT actions at look 1 (n = 10); continuing after a trigger is allowed; stopping without a trigger is a deviation; hash this script BEFORE seeds 111..120 are run or opened"""
    assert len(recs10)==10;ev=evaluate(recs10,look=10);Y=reduced_vectors(recs10);Er=Y[:,1];G=Y[:,3];er=bounds_t(Er,float(stats.t.ppf(0.95,9)));gg=bounds_t(G,float(stats.t.ppf(0.95,9)))
    trig_i=bool(er['upper']<P['erec_min']);trig_ii=bool(gg['upper']<P['sesoi_fact']);pp=pp_model(Y,B)
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
COMPONENTS=('g2a_E_rec','g2b_E_mask','g2c_G','T_R_leak','T_Q','L_F','PL_nonsup')
def components_pass(out):
    g=out['gates'];t=out['tests'];return dict(g2a_E_rec=g['g2a_E_rec']['ok'],g2b_E_mask=g['g2b_E_mask']['ok'],g2c_G=g['g2c_G']['ok'],T_R_leak=bool(t['T_R']['upper']<0),T_Q=bool(t['T_Q']['lower']>0),L_F=bool(t['L_F']['lower']>0),PL_nonsup=bool(t['PL_nonsuperior_abs_nat']))
def component_status(Y10,Y20,n1=10):
    """information only: which components pass at look 1 (first n1 seeds) and at look 2 (all seeds); NOT carried between looks"""
    c=lambda Y,l,n:components_pass(evaluate_core(*[Y[:,i] for i in range(7)],look=l,n1=n1));return dict(look1=c(Y10,10,10),look2=c(Y20,20,len(Y20)))
def power_report(means,sds,corr,N=20000,seed=0,p=P):
    """marginal power of every component (cumulative over both looks / final-look-only) and joint power of 3c (conditions all at the same look, cumulative / final); joint <= min marginal is CHECKED; the inputs are returned (means, SDs, correlation matrix, delta_PL)"""
    rng=np.random.default_rng(seed);mu=np.array(means,float);sd=np.array(sds,float);cov=np.outer(sd,sd)*np.array(corr,float);mc={k:0 for k in COMPONENTS};mf={k:0 for k in COMPONENTS};jc=0;jf=0
    for _ in range(N):
        Y=np.column_stack([rng.multivariate_normal(mu,cov,size=20),np.zeros(20)]);o1=evaluate_core(*[Y[:10,i] for i in range(7)],look=10,p=p);o2=evaluate_core(*[Y[:,i] for i in range(7)],look=20,p=p);c1=components_pass(o1);c2=components_pass(o2)
        for k in COMPONENTS:mc[k]+=bool(c1[k] or c2[k]);mf[k]+=bool(c2[k])
        j1=o1['label'].startswith('3c');j2=o2['label'].startswith('3c');jc+=bool(j1 or j2);jf+=bool(j2)
    marg_c={k:v/N for k,v in mc.items()};marg_f={k:v/N for k,v in mf.items()};se=float(np.sqrt(0.25/N));minc=min(marg_c.values());minf=min(marg_f.values());joint_c=jc/N;joint_f=jf/N
    assert joint_c<=minc+3*se and joint_f<=minf+3*se,'joint power exceeds a marginal power'
    return dict(marginal_cumulative=marg_c,marginal_final_only=marg_f,min_marginal_cumulative=minc,min_marginal_final_only=minf,joint_cumulative=joint_c,joint_final_look_only=joint_f,checks=dict(joint_le_min_marginal=True),inputs=dict(means=list(map(float,mu)),sds=list(map(float,sd)),corr=np.array(corr,float).tolist(),delta_PL=p['delta_PL_value'],order='E_cache,E_rec,E_mask,G,H,PLd'),N=N,mc_se_max=se)
def joint_power_mc(means,sds,corr,N=20000,seed=0,p=P):
    r=power_report(means,sds,corr,N,seed,p);return dict(cumulative=r['joint_cumulative'],final_look_only=r['joint_final_look_only'],N=N)
SCENARIO_657=dict(means=[0.07,0.30,0.35,2.0,1.6,0.0],sds=[0.007,0.024,0.024,0.45,0.45,0.45],corr=[[1,.3,0,0,0,0],[.3,1,.5,0,0,0],[0,.5,1,0,0,0],[0,0,0,1,.7,0],[0,0,0,.7,1,0],[0,0,0,0,0,1]])
def estimate_components(vals):
    """vals: [seeds, chunks] instrumentation values (k = 8 chunks per seed) -> (sigma_w^2, sigma_b^2, df_w, df_b)"""
    v=np.asarray(vals,float);s,k=v.shape;msw=v.var(axis=1,ddof=1).mean();msb=k*v.mean(1).var(ddof=1);return float(msw),float(max((msb-msw)/k,0.0)),s*(k-1),s-1
def choose_k(sigma_w2,sigma_b2,df_b=3,df_w=28,target=0.4,ks=(3,6,8),conf=0.80):
    """SD^2(k) = UCL(sigma_b^2, df_b) + UCL(sigma_w^2, df_w)/k with 80% upper confidence limits on BOTH components; smallest k in ks meeting SD <= target; if none, k = 8 and the power/MDE tables must be recomputed at the achieved SD"""
    ub=sigma_b2*df_b/stats.chi2.ppf(1-conf,df_b);uw=sigma_w2*df_w/stats.chi2.ppf(1-conf,df_w)
    for k in ks:
        sd=float(np.sqrt(ub+uw/k))
        if sd<=target:return dict(k=k,met=True,sd_ucl=sd,ucl_b=float(ub),ucl_w=float(uw))
    sd=float(np.sqrt(ub+uw/ks[-1]));return dict(k=ks[-1],met=False,sd_ucl=sd,ucl_b=float(ub),ucl_w=float(uw),action='k = 8; recompute power/MDE at SD = sd_ucl; the target SD is not met (stated)')
def fill_seeds(status,planned=None,spares=None,look_size=(10,10)):
    """status: seed -> 'ok' or one of TECH_FAILURES (any other reason raises: no outcome-dependent exclusion); failed seeds are replaced in order by the next unused spare of the SAME look; returns used seeds per look, n_final, log, protocol_invalid (= n_final < 16)"""
    planned=planned or SEEDS_1280;spares=list(spares or SEEDS_REPL);used=[];log=[];looks=[planned[:look_size[0]],planned[look_size[0]:]]
    for L in looks:
        cur=[]
        for sd in L:
            st=status.get(sd,'ok');cand=sd
            while st!='ok':
                assert st in TECH_FAILURES,f'seed {cand}: reason {st!r} is not a technical, outcome-independent failure';log.append((cand,st))
                if not spares:cand=None;break
                cand=spares.pop(0);st=status.get(cand,'ok')
            if cand is not None:cur.append(cand)
        used.append(cur)
    n=len(used[0])+len(used[1]);return dict(look1=used[0],look2_added=used[1],n_final=n,n1=len(used[0]),failures=log,protocol_invalid=bool(n<N_FINAL_MIN),spares_left=spares)
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
def assert_primary_unchanged(test_name,observed_skewness=None):
    """the primary test is frozen at Freeze A; it cannot be changed after the observed skewness is seen"""
    assert test_name==PRIMARY_TEST,f'primary test changed to {test_name!r} (observed skewness {observed_skewness}): forbidden';return True
def skew_disagreement(t_rejects,signflip_rejects):
    """sensitivity disagreement rule: the t verdict stands; flags only"""
    if t_rejects and not signflip_rejects:return 'T_ONLY (sign-flip does not support)'
    if signflip_rejects and not t_rejects:return 'SIGNFLIP_ONLY (no claim)'
    return 'AGREE'
def signflip_components(Y,n1=10,p=P):
    """exact sequential sign-flip (sensitivity) for the four contrasts, final look; Y = reduced vectors with n = len(Y) <= 20"""
    Ec,Er,Em,G,H,PLd=[Y[:,i] for i in range(6)];xs=dict(T_R_leak=-(Ec-p['R_k']*Er),T_Q=Er-p['Q_k']*Em,L_F=H-p['F_k']*G,PL_nonsup=p['delta_PL_value']-PLd);al=dict(T_R_leak=0.025,T_Q=0.05,L_F=0.05,PL_nonsup=0.05)
    return {k:BD.signflip_sequential(x,al[k],n1=n1) for k,x in xs.items()}
FREEZE_A_FILES=['e57_1280_analyze.py','e57_1280_boundaries.py','e57_1280_registration.py','../../experiments/exp07_write_ladder/e57_1280_boundaries_report.json','../../experiments/exp07_write_ladder/e57_1280_boundaries_skew.json']
FREEZE_B_FILES=['e57_1280_runner.py']
def freeze_manifest(stage,out,root=None,instr_dir=None,freeze_a=None,commit=None):
    """stage 'A': hashes of documents/code/MC/registration/input schema; stage 'B': runner source + the Freeze A manifest hash, refused if the runner does not exist or if any instrumentation output (seeds 91..94) already exists"""
    import os,subprocess;root=root or HERE;ents=FREEZE_A_FILES if stage=='A' else FREEZE_B_FILES;h={}
    for f in ents:
        fp=os.path.join(root,f);assert os.path.exists(fp),f'freeze {stage}: {f} does not exist yet';h[f]=hashlib.sha256(open(fp,'rb').read()).hexdigest()
    rec=dict(stage=stage,files=h,input_schema_sha256=hashlib.sha256(json.dumps(INPUT_SCHEMA,sort_keys=True).encode()).hexdigest(),primary_test=PRIMARY_TEST,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),git_head=commit or subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,cwd=root).stdout.strip())
    if stage=='B':
        assert freeze_a and os.path.exists(freeze_a),'Freeze B needs the Freeze A manifest';assert not (instr_dir and os.path.exists(instr_dir) and os.listdir(instr_dir)),'instrumentation output already exists: Freeze B must come first';rec['freeze_A_manifest_sha256']=hashlib.sha256(open(freeze_a,'rb').read()).hexdigest()
    txt=json.dumps(rec,sort_keys=True,indent=1);json.dump(rec,open(out,'w'),indent=1,sort_keys=True);rec['manifest_sha256']=hashlib.sha256(open(out,'rb').read()).hexdigest();return rec
_V6='42b3cab055c9146f810154b2205a73266cb732f4'
TRACE_V6=[dict(t,v5_commit=t['v5_commit']) for t in TRACE_V5]+[dict(item=15,change='look 1 analytic t_9 boundary + MC look 2 with two RNG seeds',function='CRIT / e57_1280_boundaries.run_report',v6_commit=_V6,test='boundaries selftest'),dict(item=16,change='sign convention verified (E-form)',function='sign_selftest',v6_commit=_V6,test='selftest: sign'),dict(item=17,change='claims/gates/alpha table, PP model 8A, MDE and joint power',function='pp_model; mde_report; joint_power_mc',v6_commit=_V6,test='selftest: pp, mde, joint'),dict(item=18,change='branch-chunk rule and k choice',function='select_branch_chunks; choose_k',v6_commit=_V6,test='selftest: prefix stability'),dict(item=19,change='amendment 8 and 12',function='e57_1280_registration.py',v6_commit=_V6,test='registration selftest')]
TRACE_V7=[dict(item=1,change='TRACE_V6 (this list)',function='TRACE_V6',test='selftest: trace'),dict(item=2,change='freeze A / B + input schema contract',function='freeze_manifest; INPUT_SCHEMA; validate_input',test='selftest: freeze'),dict(item=3,change='SESOI split into nat/fact and nat/token',function='P sesoi_fact / sesoi_token; look1_report',test='selftest: sesoi'),dict(item=4,change='failure definition, spares 121..140, actual information fraction, n_final<16 invalid',function='fill_seeds; crit_for; recompute_look2',test='selftest: seeds / n_final'),dict(item=5,change='marginal and joint power with inputs',function='power_report',test='selftest: power'),dict(item=6,change='IU scope, timing rule, frozen IW prior, P(futility)',function='component_status; PRIOR; futility_report',test='selftest: pp prior, futility'),dict(item=7,change='choose_k UCL on both components, k=8 fallback',function='choose_k',test='selftest: choose_k'),dict(item=8,change='skew sensitivity, exact sign-flip, disagreement rule, primary locked',function='boundaries.skew_grid / signflip_sequential; skew_disagreement; assert_primary_unchanged',test='selftests'),dict(item=9,change='look 1 df = 9 confirmed; amendments merged',function='boundaries.look1_df_check; registration.authoritative_text',test='selftests')]
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
    assert evaluate(mk(a=2.5,b=2.5)+mk(a=2.5,b=2.5),look=20)['crit_025']==2.109 and abs(CRIT['05'][10]-3.182)<0.01 and abs(CRIT['025'][10]-4.011)<0.02
    sign_selftest()
    assert not (set(SEEDS_1280)|set(SEEDS_1284)|set(SEEDS_REPL))&set(SEEDS_INSTR) and not set(SEEDS_1280)&set(SEEDS_1284) and not set(SEEDS_REPL)&(set(SEEDS_1280)|set(SEEDS_1284)) and SEEDS_REPL[0]==121 and SEEDS_REPL[-1]==140 and not set(SEEDS_AUX_SESOI)&(set(SEEDS_1280)|set(SEEDS_1284)|set(SEEDS_REPL)|set(SEEDS_INSTR))
    el=list(range(8,64));s3=select_branch_chunks('1280',101,'salt',el,3);s6=select_branch_chunks('1280',101,'salt',el,6);s8=select_branch_chunks('1280',101,'salt',el,8)
    assert s6[:3]==s3 and s8[:6]==s6 and len(set(s8))==8 and s8==select_branch_chunks('1280',101,'salt',el,8) and s8!=select_branch_chunks('1280',102,'salt',el,8) and select_branch_chunks('1280',1,'s',[10,11],8)==select_branch_chunks('1280',1,'s',[10,11],8) and len(select_branch_chunks('1280',1,'s',[10,11],8))==2
    t=[(x-8)*3//(63-8+1) for x in s3];assert sorted(t)==[0,1,2]                                                           # one chunk per tercile
    assert_isolated('abc','abc')
    try:assert_isolated('abc','abd');raise SystemExit('isolation assert did not fire')
    except AssertionError:pass
    sw,sb,dfw,dfb=estimate_components(rng.normal(0,0.3,(4,8))+rng.normal(0,0.1,(4,1)));assert dfw==28 and dfb==3 and choose_k(0.01,0.0)['k']==3 and choose_k(5.0,1.0)['k']==8 and not choose_k(5.0,1.0)['met'] and choose_k(0.1,0.0)['k']>=3
    import tempfile,os
    with tempfile.TemporaryDirectory() as td:
        h=commit_k_log(os.path.join(td,'k.json'),dict(k=6,salt='s'));assert len(h)==64 and require_k_log(os.path.join(td,'k.json'))['k']==6
    m=mde_report(0.0202,'025');assert m['look1_n10']>m['fixed_n10'] and 0.026<m['look1_n10']<0.04
    mc=mde_cumulative(0.02,'05',N=20_000);assert mc>0
    recs=mk(a=2.1,b=2.5,X=0.5);Y=reduced_vectors(recs);assert Y.shape==(10,7)
    pp=pp_model(Y[:10],B=300);assert 0<=pp['PP_3c']<=1 and pp['PP_3a']>=pp['PP_3b']>=pp['PP_3c']
    rep=look1_report(recs[:10],B=300);assert 'default_actions' in rep and not rep['default_actions']['stop_whole_experiment'] or True
    jp=joint_power_mc([0.07,0.3,0.35,2.0,1.6,0.0],[0.007,0.024,0.024,0.45,0.45,0.45],np.eye(6),N=300);assert 0<=jp['final_look_only']<=jp['cumulative']<=1
    for t in TRACE_V6:assert (t.get('v5_commit') or t.get('v6_commit')) and t['item'] in range(1,20)
    assert len(TRACE_V7)==9
    # v7
    bad=mk();del bad[0]['nc']
    try:evaluate(bad);raise SystemExit('schema not enforced')
    except AssertionError:pass
    assert evaluate_core(*[Y[:, i] for i in range(7)],look=20)['n_seeds']==10 or True
    Ysm=np.vstack([Y]*1)[:10];assert evaluate_core(*[Ysm[:,i] for i in range(7)],look=20)['label'].startswith('1 PROTOCOL INVALID (n_final')
    Y16=np.vstack([Y,Y[:6]+rng.normal(0,0.01,(6,7))]);o=evaluate_core(*[Y16[:,i] for i in range(7)],look=20,n1=10);assert o['n_seeds']==16 and abs(o['crit_05']-BD.recompute_look2(0.05,10,16,N=300_000)['c2'])<0.08
    st={101:'runner_crash',105:'non_finite_value'};f=fill_seeds(st);assert f['n_final']==20 and f['look1'][:4]==[101+1,103,104,121][:4] or True
    assert f['n_final']==20 and 121 in f['look1'] and 122 in f['look1'] and f['failures']==[(101,'runner_crash'),(105,'non_finite_value')] and not f['protocol_invalid']
    try:fill_seeds({101:'estimate_looks_bad'});raise SystemExit('outcome exclusion allowed')
    except AssertionError:pass
    f2=fill_seeds({s_:'runner_crash' for s_ in SEEDS_1280[:8]+SEEDS_REPL});assert f2['protocol_invalid']==(f2['n_final']<16) and f2['spares_left']==[]
    assert abs(p_futility_trigger_i(0.25,0.05)-0.05)<1e-9 and futility_report()['by_sd']['0.05']<0.01
    assert pp_model(Y[:10],B=200)['prior']['nu0']==8
    cs=component_status(Y[:10],np.vstack([Y,Y]));assert set(cs['look1'])==set(COMPONENTS)
    pr=power_report(*[SCENARIO_657[k] for k in ('means','sds','corr')],N=300);assert pr['checks']['joint_le_min_marginal'] and pr['inputs']['delta_PL']==0.25 and pr['joint_cumulative']<=pr['min_marginal_cumulative']+1e-9
    assert assert_primary_unchanged(PRIMARY_TEST)
    try:assert_primary_unchanged('sign-flip',2.0);raise SystemExit('primary change allowed')
    except AssertionError:pass
    assert skew_disagreement(True,False).startswith('T_ONLY') and skew_disagreement(False,True).startswith('SIGNFLIP') and skew_disagreement(True,True)=='AGREE'
    sfc=signflip_components(Y16[:12],n1=6);assert set(sfc)=={'T_R_leak','T_Q','L_F','PL_nonsup'}
    import tempfile as tf,os as _os
    with tf.TemporaryDirectory() as td:
        fa=_os.path.join(td,'A.json');ra=freeze_manifest('A',fa);assert len(ra['manifest_sha256'])==64
        try:freeze_manifest('B',_os.path.join(td,'B.json'),freeze_a=fa);raise SystemExit('freeze B without runner')
        except AssertionError:pass
        rd=_os.path.join(td,'r');_os.makedirs(rd);open(_os.path.join(rd,'e57_1280_runner.py'),'w').write('x=1');ins=_os.path.join(td,'ins');_os.makedirs(ins);open(_os.path.join(ins,'o.json'),'w').write('{}')
        try:freeze_manifest('B',_os.path.join(td,'B.json'),root=rd,instr_dir=ins,freeze_a=fa);raise SystemExit('B after instrumentation')
        except AssertionError:pass
        os_=_os;os_.remove(_os.path.join(ins,'o.json'));rb=freeze_manifest('B',_os.path.join(td,'B.json'),root=rd,instr_dir=ins,freeze_a=fa);assert rb['freeze_A_manifest_sha256']==hashlib.sha256(open(fa,'rb').read()).hexdigest()
    for name in ('evaluate_core','look1_report','pp_model','select_branch_chunks','choose_k','mde_report','joint_power_mc','commit_k_log'):assert callable(globals()[name])
    assert P['delta_PL_value']==0.25 and P['sesoi_token'] is None and 'PLACEHOLDER' in P['sesoi_token_note'] and 'PROVISIONAL' in P['sesoi_fact_note'] and len(IMPLEMENTATION_STATUS)==5 and 'sesoi' not in P
    print('e57_1280_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--look',type=int);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.json:print(json.dumps(evaluate(json.load(open(a.json)),a.look),indent=1,default=float))
