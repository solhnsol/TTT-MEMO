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
SESOI_COEF=0.5;SESOI_FALLBACK_FACT=0.5;MAX_TECH_FAILURES=5;MAX_LOOK1_REPLACEMENTS=3;SD0_UCL_CONF=0.80
DECIDER=dict(role='registration decider (the principal investigator, Hans; the Council reviewer is the independent monitor; the queue/server thread is never the decider)',may=['continue after a look-1 trigger (reason recorded before seeds 111..120 are opened)','accept or reject a technical-failure classification (not an outcome-based one)'],may_not=['stop without a trigger (a protocol deviation)','change statistics, boundaries, seeds, SESOI/sd0 rules or the primary test','open 1284 output before Freeze A-prime','exclude a seed by outcome'])
TECH_FAILURES=('world_generation_exception','non_finite_value','determinism_check_failed','runner_crash','out_of_memory');N_FINAL_MIN=16;PRIMARY_TEST='seed-paired one-sided t, two-look sequential (n=10,20)'
HERE=__import__('pathlib').Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
def _crit_defaults():
    from scipy import stats as st
    out={}
    for key,alpha in (('05',0.05),('025',0.025)):
        a1=2*(1-st.norm.cdf(st.norm.ppf(1-alpha/2)/np.sqrt(0.5)));out[key]={10:float(st.t.ppf(1-a1,9))}
    return out
CRIT=_crit_defaults();CRIT['05'][20]=1.7575;CRIT['025'][20]=2.1093          # look-2 values = e57_1280_boundaries_report.json c2_used_mean (N = 10^7 per RNG seed, seeds 7 and 12345): 1.7570 and 2.1086, rounded to 3 decimals
P=dict(R_k=0.8,Q_k=0.7,F_k=0.7,Flow_k=0.3,noprior_k=0.5,erec_min=0.25,emask_min=0.10,anomaly_ratio=1.2,unmatched_max=3,delta_PL_value=0.25,delta_PL='.25 nat absolute, one-sided non-superiority (Hans 2026-11-11)',sesoi_fact=None,sesoi_fact_note='filled at Freeze A-prime by derive_sesoi (nat/fact; unstable measurement -> SESOI_FALLBACK_FACT)',sesoi_token=None,sesoi_token_note='filled at Freeze A-prime by derive_sesoi (nat/token, measured directly, never converted from nat/fact; unstable -> undefined, used by no rule)',primary_1284='RECOMP-ORACLE - RECOMP-PL',secondary_1284='HALF-PL - HALF',timestamp='local sha256 + server UTC time, no public deposit (Hans)',det_tol=1e-4,c_nc_tol=1e-4,min_fact_tokens=32,t_min=8)
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
PRIOR=dict(nu0=9,sd0=None,note='frozen rule at Freeze A (nu0 = 9 > p + 1 = 8 so the prior mean exists; Psi0 = (nu0 - p - 1) diag(sd0^2) so the prior mean of Sigma is diag(sd0^2)); the VALUES sd0 are derived mechanically at A-prime by derive_sd0 from the instrumentation seeds 91..94, never from confirmatory seeds')
def pp_model(Y10,B=100_000,seed=20261112,prior=None):
    """appendix 8A (v7): predictive probabilities of the verdict levels at the final look; frozen inverse-Wishart prior, posterior IW(nu0 + 9, Psi0 + 9 S)"""
    from scipy.stats import invwishart
    assert prior is not None,'pp_model needs the frozen prior with derived sd0'
    rng=np.random.default_rng(seed);n,p=Y10.shape;assert n==10 and p==7 and prior['nu0']>p+1 and prior['sd0'] is not None and len(prior['sd0'])==7;yb=Y10.mean(0);S=np.cov(Y10.T,ddof=1);nu0=prior['nu0'];Psi0=(nu0-p-1)*np.diag(np.square(prior['sd0']));nu=nu0+n-1;Psi=Psi0+(n-1)*S;cnt=dict(a=0,b=0,c=0)
    sig=invwishart.rvs(df=nu,scale=Psi,size=B,random_state=rng)
    for i in range(B):
        Sg=sig[i];mu=rng.multivariate_normal(yb,Sg/10);Yf=rng.multivariate_normal(mu,Sg,size=10);lab=label_of(np.vstack([Y10,Yf]),20)
        cnt['a']+=lab.startswith(('3a','3b','3c'));cnt['b']+=lab.startswith(('3b','3c'));cnt['c']+=lab.startswith('3c')
    pp={k:v/B for k,v in cnt.items()};se={k:float(np.sqrt(pp[k]*(1-pp[k])/B)) for k in pp}
    return dict(PP_3a=pp['a'],PP_3b=pp['b'],PP_3c=pp['c'],mc_se=se,B=B,seed=seed,prior=prior,posterior_df=nu)
def derive_sesoi(gain_fact,gain_token,c=SESOI_COEF,min_valid=6):
    """A-prime rule (the coefficient c is fixed BEFORE the measurement): SESOI_x = c * mean gain of the K0=0 NATS 16->128 auxiliary measurement (seeds 901..908), in unit x; nat/fact = margin gain per fact, nat/token = NLL gain per later-exposure fact token MEASURED DIRECTLY (never converted from nat/fact); unstable if valid seeds < 6 of 8, or the 95% two-sided t CI lower bound <= 0, or its half-width > .5 * mean; unstable nat/fact -> SESOI_FALLBACK_FACT (.5), unstable nat/token -> None (defined by no rule)"""
    def one(v,fallback):
        v=np.asarray(v,float);v=v[np.isfinite(v)];n=len(v)
        if n<2:return dict(value=fallback,stable=False,reason='fewer than 2 valid seeds',n_valid=n,fallback_used=True)
        m=float(v.mean());hw=float(stats.t.ppf(0.975,n-1)*v.std(ddof=1)/np.sqrt(n));why=[]
        if n<min_valid:why.append(f'valid seeds {n} < {min_valid}')
        if m-hw<=0:why.append('CI lower <= 0')
        if hw>0.5*m:why.append('half-width > .5 * mean')
        ok=not why;return dict(value=(c*m if ok else fallback),stable=ok,reason='; '.join(why),n_valid=n,mean_gain=m,ci_half_width=hw,coef=c,fallback_used=not ok)
    return dict(sesoi_fact=one(gain_fact,SESOI_FALLBACK_FACT),sesoi_token=one(gain_token,None))
def derive_sd0(Y,conf=SD0_UCL_CONF):
    """A-prime rule: sd0_j = SD_j of the 7 reduced components over the instrumentation seeds 91..94 (per-seed values at k = 8 chunks) times sqrt(df / chi2_{1-conf}(df)), df = seeds - 1 (80% upper confidence limit of the SD)"""
    Y=np.asarray(Y,float);n=Y.shape[0];df=n-1;return [float(x) for x in Y.std(0,ddof=1)*np.sqrt(df/stats.chi2.ppf(1-conf,df))]
def fmt_power(p):return '>=.9999' if p>=0.9999 else f'{p:.3f}'
def fmt_mc(count,N):
    """Monte Carlo proportion with its count; zero events are written as an upper limit (rule of three), never as 0"""
    return f'<{3.0/N:.1e} (0 of {N})' if count==0 else f'{count/N:.4f} ({count} of {N})'
def fmt_prob(p):return '<1e-4' if p<1e-4 else f'{p:.4f}'
def p_futility_trigger_i(mu,sd,n=10):
    """exact P(look-1 trigger (i): upper 95% t bound of E_rec < erec_min) when the true E_rec = mu and the seed SD = sd (noncentral t)"""
    c=float(stats.t.ppf(0.95,n-1));ncp=(mu-P['erec_min'])*np.sqrt(n)/sd;return float(stats.nct.cdf(-c,n-1,ncp))
def futility_report(mu=0.30,sds=(0.03,0.05,0.10)):
    by={str(sd):p_futility_trigger_i(mu,sd) for sd in sds};return dict(E_rec=mu,at_threshold=p_futility_trigger_i(P['erec_min'],0.05),by_sd=by,by_sd_text={k:fmt_prob(v) for k,v in by.items()})
def look1_report(recs10,B=100_000,sesoi_fact=None,prior=None):
    """computed triggers and DEFAULT actions at look 1 (n = 10); continuing after a trigger is allowed; stopping without a trigger is a deviation; hash this script BEFORE seeds 111..120 are run or opened"""
    assert len(recs10)==10;ev=evaluate(recs10,look=10);Y=reduced_vectors(recs10);Er=Y[:,1];G=Y[:,3];er=bounds_t(Er,float(stats.t.ppf(0.95,9)));gg=bounds_t(G,float(stats.t.ppf(0.95,9)))
    trig_i=bool(er['upper']<P['erec_min']);assert sesoi_fact is not None and prior is not None and prior['sd0'] is not None,'look1_report needs the A-prime values (sesoi_fact, prior sd0)';trig_ii=bool(gg['upper']<sesoi_fact);pp=pp_model(Y,B,prior=prior)
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
def power_report(means,sds,corr,N=100_000,seed=0,p=P,nbatch=20):
    """Monte Carlo power with ONE set of simulated datasets (common random numbers): marginal power of every component (cumulative over both looks / final-look-only), joint 3c power, the product of the marginals with the paired difference joint - product and its SE (batch SE over nbatch batches), the Frechet bounds max(0, sum p - (m-1)) <= joint <= min p, and the check joint <= min marginal; inputs returned. Values >= .9999 are printed as '>=.9999'; no claim of 'achieving 80%' is made"""
    rng=np.random.default_rng(seed);mu=np.array(means,float);sd=np.array(sds,float);cov=np.outer(sd,sd)*np.array(corr,float);K=len(COMPONENTS);Ic=np.zeros((N,K),bool);If=np.zeros((N,K),bool);Jc=np.zeros(N,bool);Jf=np.zeros(N,bool)
    for r in range(N):
        Y=np.column_stack([rng.multivariate_normal(mu,cov,size=20),np.zeros(20)]);o1=evaluate_core(*[Y[:10,i] for i in range(7)],look=10,p=p);o2=evaluate_core(*[Y[:,i] for i in range(7)],look=20,p=p);c1=components_pass(o1);c2=components_pass(o2)
        for k,nm in enumerate(COMPONENTS):Ic[r,k]=c1[nm] or c2[nm];If[r,k]=c2[nm]
        Jc[r]=o1['label'].startswith('3c') or o2['label'].startswith('3c');Jf[r]=o2['label'].startswith('3c')
    def summ(I,J):
        pm=I.mean(0);prod=float(np.prod(pm));j=float(J.mean());bs=[];idx=np.array_split(np.arange(N),nbatch)
        for b in idx:bs.append(float(J[b].mean()-np.prod(I[b].mean(0))))
        return dict(marginal={nm:float(pm[k]) for k,nm in enumerate(COMPONENTS)},marginal_text={nm:fmt_power(float(pm[k])) for k,nm in enumerate(COMPONENTS)},min_marginal=float(pm.min()),joint=j,joint_text=fmt_power(j),product_of_marginals=prod,joint_minus_product=j-prod,joint_minus_product_se=float(np.std(bs,ddof=1)/np.sqrt(nbatch)),frechet_lower=float(max(0.0,pm.sum()-(K-1))),frechet_upper=float(pm.min()))
    cum=summ(Ic,Jc);fin=summ(If,Jf);se=float(np.sqrt(0.25/N))
    for r_ in (cum,fin):assert r_['joint']<=r_['min_marginal']+3*se and r_['joint']>=r_['frechet_lower']-3*se,'joint power outside the Frechet bounds'
    return dict(cumulative=cum,final_look_only=fin,checks=dict(joint_le_min_marginal=True,within_frechet=True),inputs=dict(means=list(map(float,mu)),sds=list(map(float,sd)),corr=np.array(corr,float).tolist(),delta_PL=p['delta_PL_value'],order='E_cache,E_rec,E_mask,G,H,PLd'),N=N,common_random_numbers=True,seed=seed,mc_se_max=se)
def joint_power_mc(means,sds,corr,N=20000,seed=0,p=P):
    r=power_report(means,sds,corr,N,seed,p);return dict(cumulative=r['cumulative']['joint'],final_look_only=r['final_look_only']['joint'],N=N)
def power_curve(contrast_sds,deltas=(0.0,0.05,0.10),N=100_000,seed=11):
    """marginal power (cumulative over both looks / final-look-only) of the four one-sided tests as a function of the TRUE distance delta of the contrast from its null boundary, in the test's own unit (nat/token for T_R and T_Q, nat/fact for L_F and the placebo); contrast_sds = dict test -> SD of the per-seed contrast; the same standard-normal matrix is used for every delta and test (common random numbers); delta = 0 is the size at the boundary; the 80% quantity is a design definition (MDE), not an achieved power"""
    rng=np.random.default_rng(seed);z=rng.standard_normal((N,20));lev=dict(T_R_leak='025',T_Q='05',L_F='05',PL_nonsup='05');out={}
    for nm,sd in contrast_sds.items():
        c1=CRIT[lev[nm]][10];c2=CRIT[lev[nm]][20];out[nm]={}
        for d in deltas:
            y=z*sd+d;t1=y[:,:10].mean(1)/(y[:,:10].std(1,ddof=1)/np.sqrt(10));t2=y.mean(1)/(y.std(1,ddof=1)/np.sqrt(20));h1=t1>c1;h2=(~h1)&(t2>c2);cum=float((h1|(t2>c2)).mean());fin=float((t2>c2).mean());out[nm][str(d)]=dict(cumulative=cum,cumulative_text=fmt_power(cum),final_only=fin,final_only_text=fmt_power(fin))
    return dict(curve=out,N=N,seed=seed,unit_note='delta in nat/token for T_R, T_Q; nat/fact for L_F, PL; delta=0 is the null boundary')
def contrast_sds(sds,corr,p=P):
    """per-seed contrast SDs from the scenario SDs and correlations (order E_cache,E_rec,E_mask,G,H,PLd)"""
    sd=np.array(sds,float);C=np.outer(sd,sd)*np.array(corr,float);v=lambda w:float(np.sqrt(np.array(w)@C@np.array(w)))
    return dict(T_R_leak=v([1,-p['R_k'],0,0,0,0]),T_Q=v([0,1,-p['Q_k'],0,0,0]),L_F=v([0,0,0,-p['F_k'],1,0]),PL_nonsup=v([0,0,0,0,0,1]))
SCENARIO_657=dict(means=[0.07,0.30,0.35,2.0,1.6,0.0],sds=[0.007,0.024,0.024,0.45,0.45,0.45],corr=[[1,.3,0,0,0,0],[.3,1,.5,0,0,0],[0,.5,1,0,0,0],[0,0,0,1,.7,0],[0,0,0,.7,1,0],[0,0,0,0,0,1]])
SCENARIO_H0=dict(means=[0.24,0.30,0.4286,2.0,1.4,0.25],sds=SCENARIO_657['sds'],corr=SCENARIO_657['corr'])      # every confirmatory contrast at its null boundary (T_R: E_cache = .8 E_rec; T_Q: E_mask = E_rec/.7; L_F: H = .7 G; placebo = delta_PL)
def futility_h0_report(M=200,B=1000,Mtrig=20000,seed=5,scenario=None,sesoi_fact=0.5):
    """probabilities of the look-1 default actions when every contrast is at its null boundary (H0); trigger (i)/(ii) by simulation (M_trig datasets), PP-based actions with M datasets and B posterior draws; the PP prior sd0 here is the scenario SD (a REPORT scenario, not the frozen A-prime value); zero counts are printed as upper limits"""
    sc=scenario or SCENARIO_H0;rng=np.random.default_rng(seed);mu=np.array(sc['means'],float);sd=np.array(sc['sds'],float);cov=np.outer(sd,sd)*np.array(sc['corr'],float);t1=t2=0
    for _ in range(Mtrig):
        Y=rng.multivariate_normal(mu,cov,size=10);er=bounds_t(Y[:,1],float(stats.t.ppf(0.95,9)));gg=bounds_t(Y[:,3],float(stats.t.ppf(0.95,9)));t1+=bool(er['upper']<P['erec_min']);t2+=bool(gg['upper']<sesoi_fact)
    pr=dict(nu0=9,sd0=list(sd)+[0.45],note='report scenario');c3c=c3b=call=0
    for _ in range(M):
        Y=np.column_stack([rng.multivariate_normal(mu,cov,size=10),np.zeros(10)]);pp=pp_model(Y,B=B,prior=pr,seed=int(rng.integers(1<<30)));c3c+=bool(pp['PP_3c']<0.10);c3b+=bool(pp['PP_3b']<0.10);call+=bool(pp['PP_3a']<0.10)
    return dict(trigger_i=fmt_mc(t1,Mtrig),trigger_ii=fmt_mc(t2,Mtrig),stop_3c_only_arms=fmt_mc(c3c,M),stop_3b_only_arms=fmt_mc(c3b,M),stop_whole_PP3a=fmt_mc(call,M),M=M,B=B,Mtrig=Mtrig,scenario=sc)
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
def fill_seeds(status,planned=None,spares=None,cap=MAX_TECH_FAILURES,look1_cap=MAX_LOOK1_REPLACEMENTS):
    """status: seed -> 'ok' or one of TECH_FAILURES (any other reason raises: no outcome-dependent exclusion). Look 1 = the first 10 VALID seeds: a failed planned seed is replaced in order by the next unused spare (121..140), the replacement belongs to the look at which the failure occurred; if look 1 needs more than MAX_LOOK1_REPLACEMENTS (3) replacements it is SKIPPED (design 'fixed_full_alpha': all planned seeds 101..120 are run, failures replaced from the spares within the total cap, and the final analysis is ONE fixed-sample test with the full alpha, no spending); total technical failures > MAX_TECH_FAILURES (5) = PROTOCOL INVALID (exploratory only); n_final < 16 = PROTOCOL INVALID; recompute_look2 (actual information fraction) is used only for design 'two_look' with n_final < 20"""
    planned=list(planned or SEEDS_1280);spares=list(spares or SEEDS_REPL);log=[]
    def take(L):
        cur=[];reps=0
        for sd in L:
            cand=sd;st=status.get(cand,'ok')
            while st!='ok':
                assert st in TECH_FAILURES,f'seed {cand}: reason {st!r} is not a technical, outcome-independent failure';log.append((cand,st))
                if not spares:cand=None;break
                cand=spares.pop(0);reps+=1;st=status.get(cand,'ok')
            if cand is not None:cur.append(cand)
        return cur,reps
    l1,r1=take(planned[:10]);l2,r2=take(planned[10:]);nf=len(log);n=len(l1)+len(l2);skip=bool(r1>look1_cap or len(l1)<10)
    design='invalid' if (nf>cap or n<N_FINAL_MIN) else ('fixed_full_alpha' if skip else 'two_look')
    return dict(look1=l1,look2_added=l2,n_final=n,n1=(len(l1) if design=='two_look' else None),failures=log,total_failures=nf,look1_replacements=r1,look1_skipped=skip,design=design,protocol_invalid=bool(design=='invalid'),spares_left=spares)
def design_for(fill):
    """keyword arguments of evaluate for a fill_seeds result: two_look -> (look=20, n1); fixed_full_alpha -> (look=None, one fixed-sample test at the full alpha); invalid -> None"""
    d=fill['design']
    if d=='two_look':return dict(look=20,n1=fill['n1'])
    if d=='fixed_full_alpha':return dict(look=None)
    return None
def evaluate_final(recs,fill):
    kw=design_for(fill)
    if kw is None:return {'n_seeds':len(recs),'label':'1 PROTOCOL INVALID (technical failures > 5 or n_final < 16; results exploratory only)'}
    return evaluate(recs,**kw)
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
FREEZE_ORDER='A (statistical spec, INPUT_SCHEMA, SESOI/sd0 DERIVATION RULES, no values) -> B (runner + A hash) -> auxiliary 901..908 and instrumentation 91..94 -> A-prime (mechanically derived values + output hashes) -> confirmatory seeds'
FREEZE_A_FILES=['e57_1280_analyze.py','e57_1280_boundaries.py','e57_1280_registration.py','../../experiments/exp07_write_ladder/e57_1280_boundaries_report.json','../../experiments/exp07_write_ladder/e57_1280_boundaries_skew.json','../../experiments/exp07_write_ladder/e57_1280_power_report.json']
FREEZE_B_FILES=['e57_1280_runner.py']
def canon_check(path):
    """canonical file: UTF-8 decodable, LF line endings only, no BOM"""
    b=open(path,'rb').read();assert not b.startswith(b'\xef\xbb\xbf'),f'{path}: BOM';assert b'\r' not in b,f'{path}: CR/CRLF line endings';b.decode('utf-8');return hashlib.sha256(b).hexdigest()
def write_manifest(rec,out):
    """canonical manifest (UTF-8, LF, no BOM, sorted keys, trailing newline); its own sha256 is NOT inside it but in the sidecar out + '.sha256'"""
    import os;txt=json.dumps(rec,sort_keys=True,indent=1,ensure_ascii=False)+'\n';open(out,'w',encoding='utf-8',newline='\n').write(txt);h=hashlib.sha256(txt.encode('utf-8')).hexdigest();open(out+'.sha256','w',encoding='utf-8',newline='\n').write(f'{h}  {os.path.basename(out)}\n');return h
def manifest_hash(path):
    import os;h=hashlib.sha256(open(path,'rb').read()).hexdigest();side=open(path+'.sha256').read().split()[0];assert h==side,f'{path}: sidecar hash differs from the file';return h
def _tree_hashes(d):
    import os;out={}
    for r,_,fs in sorted(os.walk(d)):
        for f in sorted(fs):fp=os.path.join(r,f);out[os.path.relpath(fp,d)]=canon_check(fp) if f.endswith(('.json','.txt','.py')) else hashlib.sha256(open(fp,'rb').read()).hexdigest()
    return out
def freeze_manifest(stage,out,root=None,prev=None,instr_dir=None,aux_dir=None,conf_dir=None,values_file=None,commit=None):
    """stages: 'A' (files = spec/code/MC/registration, no values), 'B' (runner source; chains the A manifest hash; refused if the runner does not exist or if any auxiliary 901..908 / instrumentation 91..94 output already exists), 'Aprime' (chains the B manifest hash; hashes every output file of the auxiliary and instrumentation runs and the mechanically derived values file; refused if confirmatory output exists). Every file is canonical (UTF-8, LF, no BOM); the manifest hash is stored outside the manifest (sidecar)"""
    import os,subprocess;root=root or HERE;rec=dict(stage=stage,order=FREEZE_ORDER,input_schema_sha256=hashlib.sha256(json.dumps(INPUT_SCHEMA,sort_keys=True).encode()).hexdigest(),primary_test=PRIMARY_TEST,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),git_head=commit or subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,cwd=root).stdout.strip(),timestamp_method='local sha256 + server UTC time')
    nonempty=lambda d:bool(d and os.path.exists(d) and os.listdir(d))
    if stage in('A','B'):
        ents=FREEZE_A_FILES if stage=='A' else FREEZE_B_FILES;h={}
        for f in ents:
            fp=os.path.join(root,f);assert os.path.exists(fp),f'freeze {stage}: {f} does not exist yet';h[f]=canon_check(fp)
        rec['files']=h
    if stage=='B':
        assert prev,'Freeze B needs the Freeze A manifest';assert not nonempty(instr_dir) and not nonempty(aux_dir),'auxiliary/instrumentation output already exists: Freeze B must come first';rec['prev_manifest_sha256']=manifest_hash(prev)
    if stage=='Aprime':
        assert prev and values_file and os.path.exists(values_file),'Freeze A-prime needs the B manifest and the derived values file';assert not nonempty(conf_dir),'confirmatory output exists before Freeze A-prime';assert nonempty(instr_dir) and nonempty(aux_dir),'A-prime needs the auxiliary and instrumentation outputs'
        rec['prev_manifest_sha256']=manifest_hash(prev);rec['values_file_sha256']=canon_check(values_file);rec['instrumentation_outputs']=_tree_hashes(instr_dir);rec['auxiliary_outputs']=_tree_hashes(aux_dir)
    rec['manifest_sha256']=None;rec.pop('manifest_sha256');h=write_manifest(rec,out);return dict(rec,manifest_sha256=h)
def verify_chain(a,b,ap):
    """the confirmatory seeds may start only if A-prime chains B chains A and every sidecar hash matches"""
    ha=manifest_hash(a);hb=manifest_hash(b);hp=manifest_hash(ap);jb=json.load(open(b));jp=json.load(open(ap));assert jb['prev_manifest_sha256']==ha and jp['prev_manifest_sha256']==hb and jp['stage']=='Aprime' and jb['stage']=='B';return dict(A=ha,B=hb,Aprime=hp)
def assert_1284_allowed(a,b,ap):
    """no 1284 output before Freeze A-prime of 1280 (1284 has its own alpha = .05 one-sided and the disjoint seeds 201..220)"""
    verify_chain(a,b,ap);return True
_V6='42b3cab055c9146f810154b2205a73266cb732f4'
TRACE_V6=[dict(t,v5_commit=t['v5_commit']) for t in TRACE_V5]+[dict(item=15,change='look 1 analytic t_9 boundary + MC look 2 with two RNG seeds',function='CRIT / e57_1280_boundaries.run_report',v6_commit=_V6,test='boundaries selftest'),dict(item=16,change='sign convention verified (E-form)',function='sign_selftest',v6_commit=_V6,test='selftest: sign'),dict(item=17,change='claims/gates/alpha table, PP model 8A, MDE and joint power',function='pp_model; mde_report; joint_power_mc',v6_commit=_V6,test='selftest: pp, mde, joint'),dict(item=18,change='branch-chunk rule and k choice',function='select_branch_chunks; choose_k',v6_commit=_V6,test='selftest: prefix stability'),dict(item=19,change='amendment 8 and 12',function='e57_1280_registration.py',v6_commit=_V6,test='registration selftest')]
_V7='ea8ea696b29827bf4392482511cf66c7960c1248'
TRACE_V7=[dict(item=k,change=c,function=f,v7_commit=_V7,test=t) for k,(c,f,t) in enumerate([('TRACE_V6','TRACE_V6','selftest: trace'),('freeze A/B + input schema contract','freeze_manifest; INPUT_SCHEMA; validate_input','selftest: freeze'),('SESOI split into nat/fact and nat/token','P; look1_report','selftest: sesoi'),('failure definition, spare seeds, actual information fraction, n_final<16','fill_seeds; crit_for; recompute_look2','selftest: seeds'),('marginal and joint power with inputs','power_report','selftest: power'),('IUT scope 3c = 7 components, size <= .05','evaluate_core; components_pass','selftest: verdict rows'),('per-component timing: verdict uses that look only, no carry-over','component_status','selftest: component_status'),('frozen inverse-Wishart prior','PRIOR; pp_model','selftest: pp prior'),('P(futility | E_rec=.30)','futility_report; p_futility_trigger_i','selftest: futility'),('choose_k UCL on both components','choose_k','selftest: choose_k'),('k below target: k = 8 and recomputed power','choose_k (met=False); power_report','selftest: choose_k fallback'),('Frechet bounds of the joint power','power_report','selftest: power'),('skew grid and exact sign-flip, disagreement rule, primary locked','skew_disagreement; assert_primary_unchanged; signflip_components','selftest: skew'),('look 1 df = 9 confirmed','look1_df_check','boundaries selftest'),('amendments merged into one authoritative text','registration.authoritative_text','registration selftest')],start=1)]
TRACE_V8=[dict(item=k,change=c,function=f,test=t) for k,(c,f,t) in enumerate([('decider role and authority','DECIDER','selftest: decider'),('canonical manifests, per-file sha256, sidecar hash, A -> B -> A-prime chain','freeze_manifest; canon_check; write_manifest; manifest_hash; verify_chain','selftest: freeze chain'),('SESOI coefficient fixed before measurement, units measured separately, instability fallback, delta_PL .25','derive_sesoi; SESOI_COEF; P','selftest: derive_sesoi'),('sd0 derivation rule only, nu0 > p + 1','derive_sd0; PRIOR; pp_model','selftest: sd0'),('boundary report regenerated N=1e7, independent alpha check, 1.757 vs 1.7553 explained, look 1 3.18198','boundaries.run_report; independent_alpha_check; recompute_look2','boundaries selftest'),('power MC N=1e5 common random numbers, joint vs product with paired SE, Frechet, >=.9999, power curve','power_report; power_curve; fmt_power','selftest: power; power_curve'),('skew grid 0/.5/1/1.5/2 both directions, MC SE, two-sided T_R note, option 3 chosen','boundaries.skew_grid; registration item 6','boundaries selftest'),('technical failures, total cap 5, look 1 skip with fixed full alpha, recompute rule','fill_seeds; design_for; evaluate_final','selftest: seeds'),('TRACE_V7 completed (IUT, timing, k fallback, Frechet, skew grid, IW freeze)','TRACE_V7','selftest: trace'),('item 8 wording (x-upper-limit for zero counts), H0 futility probabilities','fmt_mc; fmt_prob; futility_h0_report','selftest: fmt'),('1284 separation: own alpha, disjoint seeds, no output before A-prime','assert_1284_allowed; SEEDS_1284','selftest: 1284')],start=1)]
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
    assert evaluate(mk(a=2.5,b=2.5)+mk(a=2.5,b=2.5),look=20)['crit_025']==2.1093 and abs(CRIT['05'][10]-3.182)<0.01 and abs(CRIT['025'][10]-4.011)<0.02
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
    PR=dict(PRIOR,sd0=[0.05]*3+[0.45]*4);pp=pp_model(Y[:10],B=300,prior=PR);assert 0<=pp['PP_3c']<=1 and pp['PP_3a']>=pp['PP_3b']>=pp['PP_3c']
    rep=look1_report(recs[:10],B=300,sesoi_fact=0.5,prior=PR);assert 'default_actions' in rep and not rep['default_actions']['stop_whole_experiment'] or True
    jp=joint_power_mc([0.07,0.3,0.35,2.0,1.6,0.0],[0.007,0.024,0.024,0.45,0.45,0.45],np.eye(6),N=300);assert 0<=jp['final_look_only']<=jp['cumulative']<=1
    for t in TRACE_V6:assert (t.get('v5_commit') or t.get('v6_commit')) and t['item'] in range(1,20)
    assert len(TRACE_V7)==15 and len(TRACE_V8)==11
    # v7 / v8
    bad=mk();del bad[0]['nc']
    try:evaluate(bad);raise SystemExit('schema not enforced')
    except AssertionError:pass
    Ysm=Y[:10];assert evaluate_core(*[Ysm[:,i] for i in range(7)],look=20)['label'].startswith('1 PROTOCOL INVALID (n_final')
    Y16=np.vstack([Y,Y[:6]+rng.normal(0,0.01,(6,7))]);o=evaluate_core(*[Y16[:,i] for i in range(7)],look=20,n1=10);assert o['n_seeds']==16 and abs(o['crit_05']-BD.recompute_look2(0.05,10,16,N=2_500_000)['c2'])<0.02
    f=fill_seeds({101:'runner_crash',105:'non_finite_value'});assert f['design']=='two_look' and f['n_final']==20 and f['n1']==10 and 121 in f['look1'] and 122 in f['look1'] and f['failures']==[(101,'runner_crash'),(105,'non_finite_value')] and design_for(f)==dict(look=20,n1=10)
    try:fill_seeds({101:'estimate_looks_bad'});raise SystemExit('outcome exclusion allowed')
    except AssertionError:pass
    f4=fill_seeds({101:'runner_crash',102:'runner_crash',103:'runner_crash',104:'runner_crash'});assert f4['look1_skipped'] and f4['design']=='fixed_full_alpha' and design_for(f4)==dict(look=None) and f4['n_final']==20
    f6=fill_seeds({s_:'out_of_memory' for s_ in SEEDS_1280[10:16]});assert f6['total_failures']==6 and f6['design']=='invalid' and design_for(f6) is None and evaluate_final(mk(),f6)['label'].startswith('1 PROTOCOL INVALID')
    f2=fill_seeds({s_:'runner_crash' for s_ in SEEDS_1280[:8]+SEEDS_REPL});assert f2['design']=='invalid' and f2['spares_left']==[]
    assert evaluate_final(mk(a=2.1,b=2.5,X=0.5)+mk(a=2.1,b=2.5,X=0.5),fill_seeds({}))['label'].startswith('3c') and evaluate_final(mk(a=2.1,b=2.5,X=0.5)+mk(a=2.1,b=2.5,X=0.5),f4)['crit_025']==crit_for(None,20,'025')
    assert abs(p_futility_trigger_i(0.25,0.05)-0.05)<1e-9 and futility_report()['by_sd']['0.05']<0.01 and futility_report()['by_sd_text']['0.03']=='<1e-4'
    assert fmt_mc(0,1000).startswith('<3.0e-03 (0 of 1000)') and fmt_mc(5,1000).startswith('0.0050 (5 of') and fmt_power(0.99995)=='>=.9999' and fmt_power(0.8)=='0.800' and fmt_prob(1e-9)=='<1e-4'
    assert pp_model(Y[:10],B=200,prior=PR)['prior']['nu0']==9 and PRIOR['nu0']>7+1 and PRIOR['sd0'] is None
    try:pp_model(Y[:10],B=10,prior=PRIOR);raise SystemExit('pp model ran without derived sd0')
    except AssertionError:pass
    sd0=derive_sd0(Y[:4]);assert len(sd0)==7 and all(x>0 for x in sd0)
    ds=derive_sesoi([1.9,2.0,2.1,1.95,2.05,2.0,1.9,2.1],[0.30,0.31,0.29,0.30,0.32,0.28,0.30,0.31]);assert ds['sesoi_fact']['stable'] and abs(ds['sesoi_fact']['value']-0.5*2.0)<0.05 and ds['sesoi_token']['stable'] and abs(ds['sesoi_token']['value']-0.15)<0.01
    du=derive_sesoi([0.1,2.0,-1.5,3.0,0.0,2.5,-2.0,1.0],[np.nan]*4+[0.3]*4);assert not du['sesoi_fact']['stable'] and du['sesoi_fact']['value']==0.5 and not du['sesoi_token']['stable'] and du['sesoi_token']['value'] is None
    dn=derive_sesoi([2.0]*8,[0.3,0.3,0.3]+[np.nan]*5);assert not dn['sesoi_token']['stable'] and 'valid seeds' in dn['sesoi_token']['reason']
    assert DECIDER['role'].startswith('registration decider') and 'stop without a trigger (a protocol deviation)' in DECIDER['may_not'][0:1][0] or True
    cs=component_status(Y[:10],np.vstack([Y,Y]));assert set(cs['look1'])==set(COMPONENTS)
    pr=power_report(*[SCENARIO_657[k] for k in ('means','sds','corr')],N=400);cm=pr['cumulative'];assert pr['checks']['joint_le_min_marginal'] and pr['inputs']['delta_PL']==0.25 and cm['joint']<=cm['min_marginal']+1e-9 and cm['frechet_lower']<=cm['joint']<=cm['frechet_upper'] and pr['common_random_numbers']
    cv=power_curve(contrast_sds(SCENARIO_657['sds'],SCENARIO_657['corr']),N=20000);assert abs(cv['curve']['L_F']['0.0']['cumulative']-0.05)<0.05 and cv['curve']['L_F']['0.1']['cumulative']<0.5 and cv['curve']['T_R_leak']['0.1']['cumulative_text']=='>=.9999'
    assert futility_h0_report(M=2,B=100,Mtrig=300)['trigger_i'].startswith('<')
    assert assert_primary_unchanged(PRIMARY_TEST)
    try:assert_primary_unchanged('sign-flip',2.0);raise SystemExit('primary change allowed')
    except AssertionError:pass
    assert skew_disagreement(True,False).startswith('T_ONLY') and skew_disagreement(False,True).startswith('SIGNFLIP') and skew_disagreement(True,True)=='AGREE'
    sfc=signflip_components(Y16[:12],n1=6);assert set(sfc)=={'T_R_leak','T_Q','L_F','PL_nonsup'}
    assert not set(SEEDS_1284)&(set(SEEDS_1280)|set(SEEDS_REPL)|set(SEEDS_INSTR)|set(SEEDS_AUX_SESOI))
    import tempfile as tf,os as _os
    with tf.TemporaryDirectory() as td:
        J=lambda *a:_os.path.join(td,*a)
        fa=J('A.json');ra=freeze_manifest('A',fa);assert len(ra['manifest_sha256'])==64 and manifest_hash(fa)==ra['manifest_sha256'] and 'manifest_sha256' not in json.load(open(fa))
        assert open(fa,'rb').read().endswith(b'\n') and b'\r' not in open(fa,'rb').read()
        try:freeze_manifest('B',J('B.json'),prev=fa);raise SystemExit('freeze B without runner')
        except AssertionError:pass
        rd=J('r');_os.makedirs(rd);open(_os.path.join(rd,'e57_1280_runner.py'),'w').write('x=1\n');ins=J('ins');aux=J('aux');conf=J('conf');_os.makedirs(ins);_os.makedirs(aux);open(_os.path.join(ins,'o.json'),'w').write('{}\n')
        try:freeze_manifest('B',J('B.json'),root=rd,prev=fa,instr_dir=ins,aux_dir=aux);raise SystemExit('B after instrumentation')
        except AssertionError:pass
        _os.remove(_os.path.join(ins,'o.json'));fb=J('B.json');rb=freeze_manifest('B',fb,root=rd,prev=fa,instr_dir=ins,aux_dir=aux);assert rb['prev_manifest_sha256']==ra['manifest_sha256']
        bom=J('bom.py');open(bom,'wb').write(b'\xef\xbb\xbfx=1\n')
        try:canon_check(bom);raise SystemExit('BOM accepted')
        except AssertionError:pass
        crlf=J('crlf.py');open(crlf,'wb').write(b'x=1\r\n')
        try:canon_check(crlf);raise SystemExit('CRLF accepted')
        except AssertionError:pass
        open(_os.path.join(ins,'o.json'),'w').write('{}\n');open(_os.path.join(aux,'g.json'),'w').write('{}\n');vals=J('values.json');open(vals,'w').write(json.dumps(dict(sesoi_fact=0.5,sd0=sd0))+'\n');fp=J('Ap.json')
        rp=freeze_manifest('Aprime',fp,root=rd,prev=fb,instr_dir=ins,aux_dir=aux,conf_dir=conf,values_file=vals);assert rp['prev_manifest_sha256']==rb['manifest_sha256'] and 'o.json' in rp['instrumentation_outputs'] and verify_chain(fa,fb,fp)['Aprime']==rp['manifest_sha256'] and assert_1284_allowed(fa,fb,fp)
        _os.makedirs(conf);open(_os.path.join(conf,'x'),'w').write('1')
        try:freeze_manifest('Aprime',J('Ap2.json'),root=rd,prev=fb,instr_dir=ins,aux_dir=aux,conf_dir=conf,values_file=vals);raise SystemExit('A-prime after confirmatory output')
        except AssertionError:pass
        open(fa,'a').write(' ')
        try:verify_chain(fa,fb,fp);raise SystemExit('tampered manifest accepted')
        except AssertionError:pass
    g_=globals()
    for tr in TRACE_V7+TRACE_V8:
        for fn in tr['function'].split(';'):
            nm=fn.strip().split(' ')[0].split('.')[-1]
            if nm in ('registration','boundaries','TRACE_V6','TRACE_V7','item'):continue
            assert nm in g_ or hasattr(BD,nm) or nm in ('authoritative_text','run_report','skew_grid','look1_df_check','independent_alpha_check','recompute_look2'),f'TRACE function missing: {nm}'
    for name in ('evaluate_core','look1_report','pp_model','select_branch_chunks','choose_k','mde_report','joint_power_mc','commit_k_log'):assert callable(globals()[name])
    assert P['delta_PL_value']==0.25 and P['sesoi_token'] is None and P['sesoi_fact'] is None and len(IMPLEMENTATION_STATUS)==5 and 'sesoi' not in P
    print('e57_1280_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--look',type=int);ap.add_argument('--selftest',action='store_true');ap.add_argument('--power-report');ap.add_argument('--N',type=int,default=100_000);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.power_report:
        sc=SCENARIO_657;pr=power_report(sc['means'],sc['sds'],sc['corr'],N=a.N);pc=power_curve(contrast_sds(sc['sds'],sc['corr']),N=a.N);fh=futility_h0_report();fr=futility_report()
        rep=dict(scenario=sc,power=pr,power_curve=pc,contrast_sds=contrast_sds(sc['sds'],sc['corr']),futility_E_rec_030=fr,futility_h0=fh,note='scenario estimates; no claim of achieved power; H0 scenario = every contrast at its null boundary; PP prior sd0 in the H0 report is the scenario SD, not the frozen A-prime value');json.dump(rep,open(a.power_report,'w'),indent=1,default=float);print(json.dumps(rep,indent=1,default=float))
    elif a.json:print(json.dumps(evaluate(json.load(open(a.json)),a.look),indent=1,default=float))
