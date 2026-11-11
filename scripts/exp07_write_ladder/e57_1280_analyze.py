"""E57 job 1280 (recompute oracle) analysis, v10 LIGHT (CPU; DRAFT; pilot-informed analysis committed BEFORE any 1280 output exists).
DESIGN: n = 30 confirmatory seeds (101..130), ONE fixed-sample analysis, no interim look, no boundary, no futility rule, no hash freeze, no external timestamp. Timing evidence = the public commit hash of this script (and of the registration text) that precedes the first confirmatory output (check_commit_precedes_results).
SIGN (E-form, unchanged since v3): E = NLL(blocked) - NLL(not blocked), positive when blocking raises the NLL; E_cache = a - nc, E_rec = b - c, E_mask = X_MASK (recomputed band NLL - recomputed natural NLL); D = -E is recorded only. Leak direction = T_R = E_cache - .8 E_rec < 0.
UNITS: NLL-type quantities (nc, a, c, b, a_P, b_P, E_*) are nat/token over the SAME later-exposure fact tokens, one value per seed; margin-type quantities (G, H, M_*, delta_PL, SESOI_fact) are nat per fact (margin = log p(gold) - logsumexp(log p wrong)). The families are never mixed; SESOI_fact (nat/fact) and SESOI_token (nat/token) are measured separately.
TESTS (unit = seed, one-sided, alpha = .05 each, n = 30): the seed-paired t test (df = 29) AND the exact sign-flip test (all 2^30 sign vectors, meet-in-the-middle, exact under symmetry of the seed contrast about 0 under H0). WORDING: 'significant' = BOTH tests reject; t rejects but the exact test does not = 'supported, distribution-sensitive'; the exact test alone rejects = no claim (flag SIGNFLIP_ONLY); not significant is NOT evidence that there is no leak.
CLAIMS: leak hierarchy (fixed order, intersection-union, size <= .05): 3a = gate 2a and T_R leak; 3b = 3a and gate 2b and T_Q; 3c = 3b and gate 2c and L_F and placebo non-superiority; 4 CACHE SUFFICIENT = gate 2a and T_R cache direction (a separate claim at its own .05). Contrasts: T_R = E_cache - .8 E_rec, T_Q = E_rec - .7 E_mask, L_F = H - .7 G, placebo: PLd = M_RECOMP_PL - M_NATS has one-sided upper confidence bound < delta_PL = .25 nat (absolute). Gates: 2a E_rec > 0 and mean >= .25; 2b E_mask > 0 and mean >= .10; 2c G > 0. The intersection-union claim (3a/3b/3c) is NOT multiplicity-corrected (IUT). The four component tests (T_R leak, T_Q, L_F, placebo) are additionally reported Holm-adjusted over four tests at level .05 using, for each test, the LARGER of its two p-values (max(p_t, p_exact)); a component outside the intersection claim without a Holm-adjusted rejection is labelled EXPLORATORY. Flags: F_LOW (H < .3 G), H_GEN (gain_noprior > .5 G), HALF_PL (a seed is excluded when more than 3 of 32 facts are unmatched), ANOMALY (E_cache > 1.2 E_rec), G_PRACTICAL (one-sided 95% lower bound of G >= .5 nat/fact).
SEEDS: instrumentation 91..94 (variance components at k = 8 only) and auxiliary SESOI measurement 901..908 (K0 = 0, NATS 16->128) are separate from the confirmatory seeds and never enter them; 1284 (201..220) is a SEPARATE FAMILY with its own alpha = .05. FAILURE (one line): a seed whose run diverges or yields a non-finite value; at most 3 failed seeds are replaced in order by 131..133, every failure is reported, beyond the cap the analysis uses the remaining seeds (n reported).
SESOI: SESOI_fact = .5 nat/fact, FIXED by decision (Hans 2026-11-11) irrespective of the pilot uncertainty; no coefficient, no derivation from the auxiliary seeds and no replacement rule; the auxiliary NATS 16->128 measurement (901..908) is reported descriptively and defines nothing; nat/token has no SESOI. G_PRACTICAL = the one-sided 95% lower bound of G is >= .5 nat/fact.
RECORD per seed: see INPUT_SCHEMA (nc, a, c, b, a_P, b_P, X_MASK, n_tokens identical across quantities, measured_chunk_min >= 8, timing_pre_update, c_minus_nc_abs_max <= 1e-4, M_NATS, M_MASK, M_RECOMP, M_RECOMP_PL, gain_noprior, half_pl_unmatched, finite, determinism_abs, t2_ok, t3_ok, p16_ok).
BRANCH CHUNKS: each branch chunk starts from a copy of the natural snapshot just before chunk t; every arm applies one update on its own context and is evaluated on the fact tokens of the same chunk (frozen weights, same probes), then the copy is discarded; eligibility t >= 8, fact tokens present, a prior same-fact exposure inside the window, >= 32 scored fact tokens; selection stratified by terciles of the eligible t, hash-ordered, round-robin (k = 3 prefix of 6 prefix of 8), the same chunks for every arm; isolation asserted by a checksum; k is chosen once from the instrumentation seeds by choose_k.
  python3 e57_1280_analyze.py --selftest | --json records.json | --power-report OUT"""
import sys,json,argparse,hashlib,time,re,subprocess,os
import numpy as np
from scipy import stats
SEEDS_INSTR=list(range(91,95));SEEDS_1280=list(range(101,131));SEEDS_REPL=[131,132,133];SEEDS_1284=list(range(201,221));SEEDS_AUX_SESOI=list(range(901,909));MAX_REPL=3;N_FIXED=30;ALPHA=0.05
P=dict(R_k=0.8,Q_k=0.7,F_k=0.7,Flow_k=0.3,noprior_k=0.5,erec_min=0.25,emask_min=0.10,anomaly_ratio=1.2,unmatched_max=3,n_min=27,delta_PL_value=0.25,sesoi_fact=0.5,det_tol=1e-4,c_nc_tol=1e-4,min_fact_tokens=32,t_min=8)
IMPLEMENTATION_STATUS={'cached_oracle_vs_full_window_recompute':'runner pending (GPU)','mask_all_layers_all_positions':'runner pending (GPU)','token_set_identity':'analysis requires identical n_tokens; runner pending','pre_update_timing':'analysis requires timing_pre_update; runner pending','t_ge_8':'analysis requires measured_chunk_min >= 8; runner pending'}
INPUT_SCHEMA=dict(nc='float nat/token',a='float nat/token (cached-oracle NLL)',c='float nat/token (recomputed natural NLL)',b='float nat/token (recompute-oracle NLL)',a_P='float',b_P='float',X_MASK='float nat/token = E_mask',M_NATS='float nat/fact',M_MASK='float nat/fact',M_RECOMP='float nat/fact',M_RECOMP_PL='float nat/fact',gain_noprior='float nat/fact',n_tokens='dict quantity -> int (identical across quantities)',measured_chunk_min='int >= 8',timing_pre_update='bool',c_minus_nc_abs_max='float <= 1e-4',t2_ok='bool',t3_ok='bool',p16_ok='bool',finite='bool',half_pl_unmatched='int of 32',determinism_abs='float (optional)')
def validate_input(r):
    miss=[k for k in INPUT_SCHEMA if k!='determinism_abs' and k not in r];assert not miss,f'input schema violated, missing {miss}';assert isinstance(r['n_tokens'],dict) and isinstance(r['timing_pre_update'],(bool,np.bool_)) and isinstance(r['measured_chunk_min'],(int,np.integer));return True
# ---------------- tests ----------------
_SIGNS={}
def _signmat(m):
    if m not in _SIGNS:
        idx=np.arange(2**m,dtype=np.uint32);_SIGNS[m]=((idx[:,None]>>np.arange(m,dtype=np.uint32))&1).astype(np.float64)*2-1
    return _SIGNS[m]
def signflip_p_exact(x):
    """exact one-sided p = P(S >= S_obs), S = sum s_i x_i, s_i = +-1 uniform: all 2^n sign vectors, meet in the middle (halves of n // 2 and n - n // 2 signs, sorted sums, binary search)"""
    x=np.asarray(x,float);n=len(x);assert n<=32,'exact sign-flip is limited to n <= 32 seeds (2^n sign vectors, meet in the middle); use the Monte Carlo fallback'
    m=n//2;a=_signmat(m)@x[:m];b=np.sort(_signmat(n-m)@x[m:]);tol=1e-9*np.abs(x).sum()+1e-300;need=x.sum()-a-tol
    ge=len(b)-np.searchsorted(b,need,side='left');return float(ge.sum()/(len(a)*len(b)))
def t_p(x):
    x=np.asarray(x,float);n=len(x);sd=x.std(ddof=1)
    if sd==0:return 0.0 if x.mean()>0 else 1.0
    return float(stats.t.sf(x.mean()/(sd/np.sqrt(n)),n-1))
def holm(ps):
    ps=np.asarray(ps,float);o=np.argsort(ps);m=len(ps);adj=np.empty(m);run=0.0
    for r,i in enumerate(o):run=max(run,(m-r)*ps[i]);adj[i]=min(run,1.0)
    return adj
def _one(x,exact):
    x=np.asarray(x,float);pt=t_p(x);pe=signflip_p_exact(x) if exact else None;return dict(mean=float(x.mean()),n=int(len(x)),p_t=pt,p_exact=pe,rej_t=bool(pt<=ALPHA),rej_exact=(bool(pe<=ALPHA) if exact else None))
EXACT_TESTS=('g2a','g2b','g2c','leak','cache','Q','F','PL')
def evaluate_core(Ec,Er,Em,G,H,PLd,gnp,inst_ok=True,p=P,exact=True,pad=0):
    """final fixed-sample analysis on seed-level arrays; exact = also the exact sign-flip test; pad = number of missing seeds imputed as ZERO EFFECT (every contrast and gate quantity 0) for the worst-case missing-seed sensitivity"""
    n=len(Ec);out={'n_seeds':n,'n_imputed_zero_effect':int(pad),'definition':'E-form (positive = blocking raises NLL); D = -E recorded only'}
    if not inst_ok:out['label']='1 PROTOCOL / INSTRUMENT INVALID';return out
    TR=Ec-p['R_k']*Er;x=dict(g2a=Er,g2b=Em,g2c=G,leak=-TR,cache=TR,Q=Er-p['Q_k']*Em,F=H-p['F_k']*G,PL=p['delta_PL_value']-PLd,Flow=p['Flow_k']*G-H,hgen=gnp-p['noprior_k']*G)
    if pad:x={k:np.concatenate([v,np.zeros(pad)]) for k,v in x.items()}
    tests={k:_one(v,exact and k in EXACT_TESTS) for k,v in x.items()}
    def ok(k,m):
        r=tests[k]['rej_t'] if m=='t' else tests[k]['rej_exact']
        if not r:return False
        if k=='g2a':return tests[k]['mean']>=p['erec_min']
        if k=='g2b':return tests[k]['mean']>=p['emask_min']
        return True
    def level(m):
        if not ok('g2a',m):return ('none',0)
        if ok('cache',m):return ('cache',4)
        L=0
        if ok('leak',m):
            L=1
            if ok('g2b',m) and ok('Q',m):
                L=2
                if ok('g2c',m) and ok('F',m) and ok('PL',m):L=3
        return ('leak',L)
    lt=level('t');ls=level('sf') if exact else None;hk=('leak','Q','F','PL');pmax=[max(tests[k]['p_t'],tests[k]['p_exact']) if exact else tests[k]['p_t'] for k in hk];hp=holm(pmax)
    out['tests']=tests;out['holm']={k:dict(p_t=tests[k]['p_t'],p_exact=tests[k]['p_exact'],p_max=float(pmax[i]),p_holm=float(hp[i]),significant_individual=bool(hp[i]<=ALPHA) if exact else None,label=('' if exact and hp[i]<=ALPHA else 'EXPLORATORY')) for i,k in enumerate(hk)}
    Gp=x['g2c'];G_lo=float(Gp.mean()-stats.t.ppf(0.95,len(Gp)-1)*Gp.std(ddof=1)/np.sqrt(len(Gp)))
    out['flags']=dict(F_LOW=bool(tests['g2c']['rej_t'] and tests['Flow']['rej_t']),H_GEN=bool(tests['hgen']['rej_t']),ANOMALY_E_cache_gt_1_2_E_rec=bool(Ec.mean()>p['anomaly_ratio']*Er.mean()),G_PRACTICAL=bool(G_lo>=p['sesoi_fact']),R_point=float(Ec.mean()/Er.mean()))
    names={0:'6 INCONCLUSIVE',1:'3a R-LEAK ONLY',2:'3b R-LEAK AND Q',3:'3c LEAK CONFIRMED'};sens=None;flag=None;out['level_t']=lt
    if not exact:
        out['label']='4 CACHE SUFFICIENT (t only)' if lt[0]=='cache' else (names[lt[1]]+' (t only)' if lt[0]=='leak' and lt[1]>0 else ('2a E_rec NOT CERTIFIED (t only)' if lt[0]=='none' else '6 INCONCLUSIVE (t only)'));return out
    if lt[0]=='none':
        out['label']='2a E_rec NOT CERTIFIED'
        if ls[0]!='none' and ls[1]>0:flag='SIGNFLIP_ONLY (no claim)'
    elif lt[0]=='cache':
        out['label']='4 CACHE SUFFICIENT [significant: t and exact sign-flip]' if ls[0]=='cache' else '4 CACHE SUFFICIENT supported, distribution-sensitive (the exact sign-flip test does not reject)'
    else:
        Ls=ls[1] if ls[0]=='leak' else 0;Lb=min(lt[1],Ls)
        if Lb>0:out['label']=names[Lb]+' [significant: t and exact sign-flip]'
        elif lt[1]>0:out['label']='6 INCONCLUSIVE (t supports '+names[lt[1]][:2]+', the exact sign-flip test does not reject: supported, distribution-sensitive)'
        else:out['label']='6 INCONCLUSIVE'
        if Lb>0 and lt[1]>Lb:sens=f'levels up to {names[lt[1]][:2]} are SUPPORTED, DISTRIBUTION-SENSITIVE (t rejects, the exact sign-flip test does not)'
        if Ls>lt[1]:flag='SIGNFLIP_ONLY (no claim)'
    out['sensitive_levels']=sens;out['signflip_flag']=flag;out['level_exact']=ls;return out
def evaluate(recs,p=P,exact=True):
    for r in recs:validate_input(r)
    A=lambda k:np.array([r[k] for r in recs],float);nc,a,c,b,aP,bP,X=[A(k) for k in ('nc','a','c','b','a_P','b_P','X_MASK')];MN,MM,MR,MP=[A(k) for k in ('M_NATS','M_MASK','M_RECOMP','M_RECOMP_PL')]
    toks=all(len(set(r['n_tokens'].values()))==1 for r in recs);inst=dict(c_nc_ok=bool(all(r['c_minus_nc_abs_max']<=p['c_nc_tol'] for r in recs)),t2_ok=bool(all(r['t2_ok'] for r in recs)),t3_ok=bool(all(r['t3_ok'] for r in recs)),p16_ok=bool(all(r['p16_ok'] for r in recs)),finite=bool(all(r['finite'] for r in recs) and all(np.isfinite(x).all() for x in (nc,a,c,b,X,MN,MM,MR,MP))),determinism_ok=bool(all(r.get('determinism_abs',0.0)<=p['det_tol'] for r in recs)),token_sets_identical=bool(toks),t_ge_8=bool(all(r['measured_chunk_min']>=p['t_min'] for r in recs)),pre_update_timing=bool(all(r['timing_pre_update'] for r in recs)))
    out=evaluate_core(a-nc,b-c,X,MM-MN,MR-MN,MP-MN,A('gain_noprior'),inst_ok=all(inst.values()),p=p,exact=exact);out['instrument_gates']=inst
    hp=int(sum(1 for r in recs if r['half_pl_unmatched']>p['unmatched_max']));out.setdefault('flags',{}).update(HALF_PL_VALID_seeds=int(len(recs)-hp),HALF_PL_excluded_seeds=hp);return out
def final_report(recs,n_planned=N_FIXED,p=P,exact=True):
    """the reported analysis: the evaluation on the available seeds; if the number of seeds is below n_min (n_min = 27, Hans 2026-11-11) the result is labelled EXPLORATORY; if seeds are missing the WORST-CASE missing-seed sensitivity imputes every missing seed as ZERO EFFECT (all contrasts and gate quantities 0) and reports whether the verdict level is unchanged"""
    out=evaluate(recs,p=p,exact=exact);n=len(recs);out['n_available']=n;out['n_planned']=n_planned;nm=p.get('n_min')
    out['below_n_min']=(bool(n<nm) if nm is not None else None);out['n_min_note']='n_min is not set' if nm is None else f'n_min = {nm}'
    if nm is not None and n<nm:out['label_reported']='EXPLORATORY: '+out['label']
    else:out['label_reported']=out['label']
    if n<n_planned and 'tests' in out:
        A=lambda k:np.array([r[k] for r in recs],float);nc,a,c,b,X=[A(k) for k in ('nc','a','c','b','X_MASK')];MN,MM,MR,MP=[A(k) for k in ('M_NATS','M_MASK','M_RECOMP','M_RECOMP_PL')]
        w=evaluate_core(a-nc,b-c,X,MM-MN,MR-MN,MP-MN,A('gain_noprior'),p=p,exact=exact,pad=n_planned-n);out['missing_seed_worst_case']=dict(label=w['label'],imputed=n_planned-n,robust=bool(w['label'][:2]==out['label'][:2]))
    return out
def benchmark_exact(n=30,reps=20,seed=1):
    """CPU benchmark of the exact sign-flip test at n seeds (2^n sign vectors, meet in the middle): wall time per test, per 8-test analysis, and the memory of the two cached sign matrices"""
    import resource;rng=np.random.default_rng(seed);x=[rng.normal(0.2,1,n) for _ in range(reps)];signflip_p_exact(x[0]);t0=time.perf_counter()
    for v in x:signflip_p_exact(v)
    dt=(time.perf_counter()-t0)/reps;m=n//2;mb=((2**m)*m+(2**(n-m))*(n-m))*8/1e6
    return dict(n=n,sign_vectors=2**n,seconds_per_test=dt,seconds_per_analysis_8_tests=8*dt,sign_matrix_MB=mb,peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,cpu=os.cpu_count(),threads_note='single process',fallback='if infeasible: Monte Carlo sign-flip with >= 1e6 sign vectors and a fixed RNG seed (numpy PCG64 seed 20261112), p = (1 + count) / (1 + 1e6)')
# ---------------- seeds and failures ----------------
FAILURE_DEF='a seed whose run diverges or yields a non-finite value'
def fill_seeds(status,planned=None,spares=None,cap=MAX_REPL):
    """status: seed -> 'ok' | 'diverged' | 'non_finite' (anything else raises: no exclusion by outcome); a failed seed is replaced in order by 131..133, at most MAX_REPL replacements; beyond the cap the failed seed is dropped and n is reduced; every failure is reported"""
    planned=list(planned or SEEDS_1280);spares=list(spares or SEEDS_REPL);used=[];fails=[];reps=0
    for sd in planned:
        cand=sd;st=status.get(cand,'ok')
        while st!='ok':
            assert st in('diverged','non_finite'),f'seed {cand}: {st!r} is not a failure by definition ({FAILURE_DEF})';fails.append((cand,st))
            if reps>=cap or not spares:cand=None;break
            cand=spares.pop(0);reps+=1;st=status.get(cand,'ok')
        if cand is not None:used.append(cand)
    return dict(seeds=used,n_final=len(used),failures=fails,replacements=reps,cap_exceeded=bool(len(used)<len(planned)),failure_definition=FAILURE_DEF)
SESOI_NOTE='SESOI_fact = .5 nat/fact is fixed by decision (Hans 2026-11-11), irrespective of the pilot uncertainty; there is no coefficient, no estimator and no replacement rule; SESOI enters no test, it defines the flag G_PRACTICAL only'
# ---------------- design helpers ----------------
def estimate_components(vals):
    v=np.asarray(vals,float);s,k=v.shape;msw=v.var(axis=1,ddof=1).mean();msb=k*v.mean(1).var(ddof=1);return float(msw),float(max((msb-msw)/k,0.0)),s*(k-1),s-1
def choose_k(sigma_w2,sigma_b2,df_b=3,df_w=28,target=0.4,ks=(3,6,8),conf=0.80):
    ub=sigma_b2*df_b/stats.chi2.ppf(1-conf,df_b);uw=sigma_w2*df_w/stats.chi2.ppf(1-conf,df_w)
    for k in ks:
        sd=float(np.sqrt(ub+uw/k))
        if sd<=target:return dict(k=k,met=True,sd_ucl=sd)
    return dict(k=ks[-1],met=False,sd_ucl=float(np.sqrt(ub+uw/ks[-1])),action='k = 8; power recomputed at the achieved SD (target not met, stated)')
def select_branch_chunks(job,seed,salt,eligible,k):
    el=sorted(eligible)
    if not el:return []
    parts=[el[i*len(el)//3:(i+1)*len(el)//3] for i in range(3)];parts=[sorted(q,key=lambda t:hashlib.sha256(f'{job}|{seed}|{salt}|{t}'.encode()).hexdigest()) for q in parts if q];out=[]
    for r in range(max(len(q) for q in parts)):
        for q in parts:
            if r<len(q):out.append(q[r])
    return out[:k]
def assert_isolated(a,b):assert a==b,'branch isolation violated: the natural trajectory checksum changed'
def confound_check(by_arm,tol=None):
    """placebo/oracle confound match per seed: by_arm = {arm: dict(n_masked_tokens, mean_masked_position, attn_mass)}; pairs ORACLE/ORACLE-PL, RECOMP/RECOMP-PL, HALF/HALF-PL; length must be equal, position within tol['position'] tokens, attention mass is a COVARIATE (difference reported, never matched)"""
    tol=tol or dict(position=64);rows=[]
    for a,b in (('ORACLE','ORACLE-PL'),('RECOMP','RECOMP-PL'),('HALF','HALF-PL')):
        if a in by_arm and b in by_arm:
            A,B=by_arm[a],by_arm[b];rows.append(dict(pair=f'{a} vs {b}',d_tokens=int(B['n_masked_tokens']-A['n_masked_tokens']),d_position=float(B['mean_masked_position']-A['mean_masked_position']),d_attn_mass=float(B['attn_mass']-A['attn_mass']),length_matched=bool(A['n_masked_tokens']==B['n_masked_tokens']),position_matched=bool(abs(B['mean_masked_position']-A['mean_masked_position'])<=tol['position'])))
    return dict(rows=rows,all_matched=bool(all(r['length_matched'] and r['position_matched'] for r in rows)),attn_mass_note='covariate: reported, not matched')
# ---------------- timing evidence ----------------
def check_commit_precedes_results(scripts,results_dir,repo='.'):
    """timing evidence = the public commit hash only: every script is committed (clean) and its last commit time precedes the oldest result file in results_dir; returns {script: commit hash}"""
    def git(*a):return subprocess.run(['git',*a],capture_output=True,text=True,cwd=repo)
    files=[os.path.join(r,f) for r,_,fs in os.walk(results_dir) for f in fs];first=min(os.path.getmtime(f) for f in files) if files else float('inf');out={}
    for sc in scripts:
        assert git('ls-files','--error-unmatch',sc).returncode==0,f'{sc} is not committed';assert git('diff','--quiet','HEAD','--',sc).returncode==0,f'{sc} has uncommitted changes'
        t=int(git('log','-1','--format=%ct','--',sc).stdout.strip());assert t<first,f'{sc}: committed after the first result file';out[sc]=git('log','-1','--format=%H','--',sc).stdout.strip()
    return out
def github_release_time(repo,tag):
    """server-side creation time (epoch seconds) of the GitHub release attached to the signed tag in the private registry repo (gh api; the pusher cannot set this time)"""
    import datetime;r=subprocess.run(['gh','api',f'repos/{repo}/releases/tags/{tag}','--jq','.created_at'],capture_output=True,text=True);assert r.returncode==0,'release not found: '+r.stderr[:200]
    return datetime.datetime.fromisoformat(r.stdout.strip().replace('Z','+00:00')).timestamp()
def check_external_record(tag,scripts,results_dir,repo='.',remote_time_fn=None,require_signature=True):
    """external record (Hans 2026-11-11): a SIGNED tag pushed to the private remote; the checker verifies (1) the tag signature (git tag -v), (2) every script is identical in the tag and in the working tree, (3) the REMOTE time of the tag (remote_time_fn(tag) -> epoch seconds, default = the server-side creation time of the release on the private registry repo) precedes the oldest file in results_dir; returns the tag commit"""
    def git(*a):return subprocess.run(['git',*a],capture_output=True,text=True,cwd=repo)
    if require_signature:assert git('tag','-v',tag).returncode==0,'tag signature does not verify'
    commit=git('rev-list','-n','1',tag);assert commit.returncode==0,f'tag {tag} not found';commit=commit.stdout.strip()
    for sc in scripts:assert git('diff','--quiet',tag,'--',sc).returncode==0,f'{sc} differs from the tagged version'
    files=[os.path.join(r,f) for r,_,fs in os.walk(results_dir) for f in fs];first=min(os.path.getmtime(f) for f in files) if files else float('inf');rt=(remote_time_fn or (lambda t:github_release_time(REGISTRY_REPO,t)))(tag)
    assert rt<first,'the remote record is not older than the first result file';return commit
REGISTRY_REPO='PRIVATE-REGISTRY-REPO (to be created by Hans: <owner>/<name>, private)'
# ---------------- power at n = 30 ----------------
def fmt_power(p):return '>=.9999' if p>=0.9999 else f'{p:.3f}'
COMPONENTS=('g2a','g2b','g2c','leak','Q','F','PL')
SCENARIO_657=dict(means=[0.07,0.30,0.35,2.0,1.6,0.0],sds=[0.007,0.024,0.024,0.45,0.45,0.45],corr=[[1,.3,0,0,0,0],[.3,1,.5,0,0,0],[0,.5,1,0,0,0],[0,0,0,1,.7,0],[0,0,0,.7,1,0],[0,0,0,0,0,1]])
def power_n30(means,sds,corr,N=100_000,exact_N=10_000,seed=0,p=P,n=N_FIXED):
    """power at n = 30 (common random numbers): t-only for N datasets (marginal per component, joint 3c, product of marginals, Frechet bounds, joint-minus-product with paired batch SE) and 'significant' = t AND exact sign-flip for the first exact_N datasets; values >= .9999 printed '>=.9999'"""
    rng=np.random.default_rng(seed);mu=np.array(means,float);sd=np.array(sds,float);cov=np.outer(sd,sd)*np.array(corr,float);Lc=np.linalg.cholesky(cov);Z=rng.standard_normal((N,n,6));D=mu+Z@Lc.T;K=len(COMPONENTS);It=np.zeros((N,K),bool);Jt=np.zeros(N,bool);Ie=np.zeros((exact_N,K),bool);Je=np.zeros(exact_N,bool)
    for r in range(N):
        Y=D[r];o=evaluate_core(Y[:,0],Y[:,1],Y[:,2],Y[:,3],Y[:,4],Y[:,5],np.zeros(n),p=p,exact=(r<exact_N));tt=o['tests']
        for k,nm in enumerate(COMPONENTS):It[r,k]=tt[nm]['rej_t']
        Jt[r]=o['label'].startswith('3c')
        if r<exact_N:
            for k,nm in enumerate(COMPONENTS):Ie[r,k]=tt[nm]['rej_t'] and tt[nm]['rej_exact']
            Je[r]=o['label'].startswith('3c') and 'significant' in o['label']
    def summ(I,J):
        pm=I.mean(0);m=len(I);bs=[float(J[b].mean()-np.prod(I[b].mean(0))) for b in np.array_split(np.arange(m),20)];return dict(marginal={nm:fmt_power(float(pm[k])) for k,nm in enumerate(COMPONENTS)},marginal_num={nm:float(pm[k]) for k,nm in enumerate(COMPONENTS)},joint=float(J.mean()),joint_text=fmt_power(float(J.mean())),product=float(np.prod(pm)),joint_minus_product=float(J.mean()-np.prod(pm)),joint_minus_product_se=float(np.std(bs,ddof=1)/np.sqrt(20)),frechet=[float(max(0,pm.sum()-(K-1))),float(pm.min())],N=m)
    return dict(t_only=summ(It,Jt),significant_t_and_exact=summ(Ie,Je),inputs=dict(means=list(map(float,mu)),sds=list(map(float,sd)),corr=np.array(corr,float).tolist(),delta_PL=p['delta_PL_value'],n=n))
POWER_SDS=(0.05,0.225,0.45,0.90)
SD_SOURCE={0.05:'PILOT-BASED: SD of the pilot placebo difference PL - NATS (3 seeds, df = 2; point estimate .050)',0.225:'PILOT-BASED upper limit: .050 * sqrt(2 / chi2_.05(2)) = .22 (95% upper confidence limit of that SD, df = 2)',0.45:'ASSUMPTION (working value of the margin-family SD; origin not re-derived in this thread, not pilot-based)',0.90:'ASSUMPTION (twice the working value; stress case, not pilot-based)'}
def _grid_cell(a):
    msd,rho,N,eN,seed=a;sc=SCENARIO_657;sd=list(sc['sds'][:3])+[msd]*3;corr=[r[:] for r in sc['corr']];corr[3][4]=corr[4][3]=rho;return (msd,rho,power_n30(sc['means'],sd,corr,N=N,exact_N=eN,seed=seed))
def power_grid(N=100_000,exact_N=10_000,seed=2026,procs=9):
    from multiprocessing import Pool
    cells=[(sd,r,N,exact_N,seed) for sd in POWER_SDS for r in (0.0,0.7,0.9)]
    with Pool(procs) as pool:res=pool.map(_grid_cell,cells)
    return dict(grid=[dict(margin_sd=a,sd_source=SD_SOURCE[a],rho_GH=b,**c) for a,b,c in res],n=N_FIXED,note='alt scenario = SCENARIO_657 (means E_cache .07, E_rec .30, E_mask .35, G 2.0, H 1.6, PLd 0), delta_PL .25, token-family SDs fixed; t_only uses N datasets, significant_t_and_exact the first exact_N')
def gpu_hours(n_seeds,per_seed_minutes):return n_seeds*per_seed_minutes/60.0
GPU_ASSUMPTION=dict(basis='measured: NATS 1.9 s/chunk, NATMASK 3.2 s/chunk, one full read about 5 min (291-313 s), 1283 about 10 min per arm; UNMEASURED: the 1280 branch arms (recompute oracle, RECOMP, RECOMP-PL, ORACLE, ORACLE-PL) at k = 8 chunks per seed',per_seed_minutes={'optimistic':45,'central':60,'pessimistic':90})
def gpu_table(n=N_FIXED,old_n=20):return {k:dict(hours_n30=gpu_hours(n,v),hours_n20=gpu_hours(old_n,v)) for k,v in GPU_ASSUMPTION['per_seed_minutes'].items()}
def selftest():
    rng=np.random.default_rng(0);S=30
    x=rng.normal(0.3,1,12);n=len(x);br=sum(1 for m in range(2**n) if sum(x[i]*(1 if (m>>i)&1 else -1) for i in range(n))>=x.sum()-1e-9)/2**n;assert abs(signflip_p_exact(x)-br)<1e-12
    assert abs(signflip_p_exact(np.ones(10))-1/1024)<1e-12 and abs(signflip_p_exact(np.zeros(8))-1.0)<1e-12 and 0<t_p(rng.normal(0,1,30))<1
    pv=np.array([signflip_p_exact(np.random.default_rng(i).normal(0,1,20)) for i in range(2000)]);assert abs((pv<=0.05).mean()-0.05)<0.02
    assert list(np.round(holm([0.01,0.04,0.03,0.2]),3))==[0.04,0.09,0.09,0.2]
    def mk(nc=2.0,a=2.5,c=2.0,b=2.5,aP=2.0,bP=2.0,X=0.4,MN=-3.8,MM=-1.8,MR=-1.8,MP=-3.8,gnp=1.1,sd=0.05,ok=True,um=0,tok=7000,n=S):
        return [dict(nc=nc+rng.normal(0,sd),a=a+rng.normal(0,sd),c=c+rng.normal(0,sd),b=b+rng.normal(0,sd),a_P=aP+rng.normal(0,sd),b_P=bP+rng.normal(0,sd),X_MASK=X+rng.normal(0,sd),M_NATS=MN+rng.normal(0,sd),M_MASK=MM+rng.normal(0,sd),M_RECOMP=MR+rng.normal(0,sd),M_RECOMP_PL=MP+rng.normal(0,sd),gain_noprior=gnp+rng.normal(0,sd),n_tokens=dict(nc=tok,a=tok,c=tok,b=tok,a_P=tok,b_P=tok,X=tok),measured_chunk_min=(8 if ok else 4),timing_pre_update=True,c_minus_nc_abs_max=(5e-5 if ok else 1e-3),t2_ok=ok,t3_ok=ok,p16_ok=ok,finite=True,half_pl_unmatched=um,determinism_abs=1e-6) for _ in range(n)]
    L=lambda **k:evaluate(mk(**k))['label']
    assert L(a=2.5,b=2.5).startswith('4 CACHE SUFFICIENT') and 'significant' in L(a=2.5,b=2.5)
    assert L(a=2.1,b=2.5,X=0.5).startswith('3c LEAK CONFIRMED') and 'significant' in L(a=2.1,b=2.5,X=0.5)
    assert L(a=2.1,b=2.5,X=0.5,MP=-1.8).startswith('3b') and L(a=2.1,b=2.5,X=0.5,MR=-3.5).startswith('3b') and L(a=2.1,b=2.5,X=0.05).startswith('3a')
    assert L(b=2.1).startswith('2a') and L(ok=False).startswith('1 PROTOCOL') and not L(sd=0.9).startswith(('3c','4 '))
    # distribution sensitivity: one extreme seed makes t reject (heavy right tail) while the exact sign-flip test does not
    rs=np.random.default_rng(5);tonly=None
    for _ in range(600):
        x=rs.lognormal(0,1.5,30)-np.exp(1.125)+0.35
        if t_p(x)<=0.05<signflip_p_exact(x):tonly=x;break
    assert tonly is not None and t_p(tonly)<=0.05<signflip_p_exact(tonly)                                                  # skewed contrast: t rejects, the exact test does not
    o=evaluate_core(*[rng.normal(m,s_,30) for m,s_ in zip([0.07,0.30,0.35,2.0,1.6,0.0],[0.007,0.024,0.024,0.45,0.45,0.45])],np.zeros(30));assert set(o['holm'])=={'leak','Q','F','PL'} and 'label' in o
    bad=mk();bad[0]['n_tokens']['a']=1;assert evaluate(bad)['label'].startswith('1 PROTOCOL')
    bad=mk();del bad[0]['nc']
    try:evaluate(bad);raise SystemExit('schema not enforced')
    except AssertionError:pass
    assert evaluate(mk(gnp=1.5))['flags']['H_GEN'] and not evaluate(mk(gnp=0.2))['flags']['H_GEN'] and evaluate(mk(um=4))['flags']['HALF_PL_excluded_seeds']==S and evaluate(mk(a=3.2,b=2.5))['flags']['ANOMALY_E_cache_gt_1_2_E_rec']
    assert evaluate(mk(a=2.1,b=2.5,X=0.5))['flags']['G_PRACTICAL'] is True and evaluate(mk(a=2.1,b=2.5,X=0.5,MM=-3.5))['flags']['G_PRACTICAL'] is False and evaluate(mk(a=2.1,b=2.5,X=0.5),exact=False)['label'].endswith('(t only)')
    assert len(SEEDS_1280)==30 and SEEDS_1280[0]==101 and SEEDS_1280[-1]==130 and SEEDS_REPL==[131,132,133] and not (set(SEEDS_1280)|set(SEEDS_REPL)|set(SEEDS_1284))&(set(SEEDS_INSTR)|set(SEEDS_AUX_SESOI)) and not set(SEEDS_1284)&(set(SEEDS_1280)|set(SEEDS_REPL)|set(SEEDS_INSTR)|set(SEEDS_AUX_SESOI))
    f=fill_seeds({101:'diverged',105:'non_finite'});assert f['n_final']==30 and f['replacements']==2 and 131 in f['seeds'] and 132 in f['seeds'] and f['failures']==[(101,'diverged'),(105,'non_finite')] and not f['cap_exceeded']
    f5=fill_seeds({s_:'diverged' for s_ in (101,102,103,104,105)});assert f5['replacements']==3 and f5['n_final']==28 and f5['cap_exceeded'] and len(f5['failures'])==5
    try:fill_seeds({101:'bad_outcome'});raise SystemExit('outcome exclusion allowed')
    except AssertionError:pass
    el=list(range(8,64));s3=select_branch_chunks('1280',101,'salt',el,3);s6=select_branch_chunks('1280',101,'salt',el,6);s8=select_branch_chunks('1280',101,'salt',el,8);assert s6[:3]==s3 and s8[:6]==s6 and len(set(s8))==8 and sorted((x_-8)*3//56 for x_ in s3)==[0,1,2]
    assert_isolated('abc','abc')
    try:assert_isolated('abc','abd');raise SystemExit('isolation assert did not fire')
    except AssertionError:pass
    sw,sb,dfw,dfb=estimate_components(rng.normal(0,0.3,(4,8))+rng.normal(0,0.1,(4,1)));assert dfw==28 and dfb==3 and choose_k(0.01,0.0)['k']==3 and choose_k(5.0,1.0)['k']==8 and not choose_k(5.0,1.0)['met']
    cc=confound_check({'ORACLE':dict(n_masked_tokens=100,mean_masked_position=500,attn_mass=0.3),'ORACLE-PL':dict(n_masked_tokens=100,mean_masked_position=520,attn_mass=0.1),'RECOMP':dict(n_masked_tokens=90,mean_masked_position=300,attn_mass=0.2),'RECOMP-PL':dict(n_masked_tokens=95,mean_masked_position=300,attn_mass=0.2)});assert cc['rows'][0]['length_matched'] and cc['rows'][0]['position_matched'] and not cc['rows'][1]['length_matched'] and not cc['all_matched'] and abs(cc['rows'][0]['d_attn_mass']+0.2)<1e-9
    import tempfile as tf
    with tf.TemporaryDirectory() as td:
        g=lambda *a:subprocess.run(['git',*a],cwd=td,capture_output=True,text=True,check=True);g('init','-q');g('config','user.email','a@b');g('config','user.name','t');open(os.path.join(td,'s.py'),'w').write('x=1\n');g('add','s.py');g('commit','-qm','s');os.makedirs(os.path.join(td,'res'));time.sleep(1.1);open(os.path.join(td,'res','r.json'),'w').write('{}')
        assert len(check_commit_precedes_results(['s.py'],os.path.join(td,'res'),td)['s.py'])==40
        open(os.path.join(td,'s.py'),'a').write('y=2\n')
        try:check_commit_precedes_results(['s.py'],os.path.join(td,'res'),td);raise SystemExit('uncommitted change accepted')
        except AssertionError:pass
        time.sleep(1.1);g('commit','-qam','late')
        try:check_commit_precedes_results(['s.py'],os.path.join(td,'res'),td);raise SystemExit('commit after results accepted')
        except AssertionError:pass
    with tf.TemporaryDirectory() as td:
        g=lambda *a:subprocess.run(['git',*a],cwd=td,capture_output=True,text=True,check=True);g('init','-q');g('config','user.email','a@b');g('config','user.name','t');open(os.path.join(td,'s.py'),'w').write('x=1\n');g('add','s.py');g('commit','-qm','s');g('tag','-a','reg','-m','reg');os.makedirs(os.path.join(td,'res'));time.sleep(1.1);open(os.path.join(td,'res','r.json'),'w').write('{}')
        old=lambda t:time.time()-100;new=lambda t:time.time()+100
        assert len(check_external_record('reg',['s.py'],os.path.join(td,'res'),td,remote_time_fn=old,require_signature=False))==40
        try:check_external_record('reg',['s.py'],os.path.join(td,'res'),td,remote_time_fn=new,require_signature=False);raise SystemExit('remote time after results accepted')
        except AssertionError:pass
        try:check_external_record('reg',['s.py'],os.path.join(td,'res'),td,remote_time_fn=old,require_signature=True);raise SystemExit('unsigned tag accepted')
        except AssertionError:pass
        open(os.path.join(td,'s.py'),'a').write('y=2\n')
        try:check_external_record('reg',['s.py'],os.path.join(td,'res'),td,remote_time_fn=old,require_signature=False);raise SystemExit('script differs from tag accepted')
        except AssertionError:pass
        try:check_external_record('nope',['s.py'],os.path.join(td,'res'),td,remote_time_fn=old,require_signature=False);raise SystemExit('missing tag accepted')
        except AssertionError:pass
    pr=power_n30(*[SCENARIO_657[k] for k in ('means','sds','corr')],N=300,exact_N=60);a_=pr['t_only'];assert a_['joint']<=a_['frechet'][1]+1e-9 and a_['joint']>=a_['frechet'][0]-1e-9 and pr['inputs']['n']==30 and fmt_power(0.99995)=='>=.9999'
    gt=gpu_table();assert abs(gt['central']['hours_n30']-30.0)<1e-9 and abs(gt['central']['hours_n20']-20.0)<1e-9
    # v11
    try:signflip_p_exact(np.ones(34));raise SystemExit('n above the exact limit accepted')
    except AssertionError:pass
    o=evaluate(mk(a=2.1,b=2.5,X=0.5));assert all(abs(o['holm'][k]['p_max']-max(o['holm'][k]['p_t'],o['holm'][k]['p_exact']))<1e-15 for k in o['holm'])
    assert abs(50**0.5*0+0.050*np.sqrt(2/stats.chi2.ppf(0.05,2))-0.2205)<0.002 and POWER_SDS==(0.05,0.225,0.45,0.90) and all('PILOT-BASED' in SD_SOURCE[k] for k in (0.05,0.225)) and all('ASSUMPTION' in SD_SOURCE[k] for k in (0.45,0.90))
    full=mk(a=2.1,b=2.5,X=0.5);fr=final_report(full);assert fr['below_n_min'] is False and 'n_min = 27' in fr['n_min_note'] and 'missing_seed_worst_case' not in fr
    fr2=final_report(full[:22]);assert fr2['below_n_min'] is True and fr2['label_reported'].startswith('EXPLORATORY') and fr2['missing_seed_worst_case']['imputed']==8 and isinstance(fr2['missing_seed_worst_case']['robust'],bool)
    fr3=final_report(mk(a=2.1,b=2.5,X=0.5,sd=0.02)[:15],p=dict(P,n_min=10));assert fr3['below_n_min'] is False and final_report(full[:28])['below_n_min'] is False and final_report(full[:26])['below_n_min'] is True and fr3['missing_seed_worst_case']['imputed']==15 and fr3['n_available']==15
    h15=full[:15];pd=evaluate_core(np.array([x_['a']-x_['nc'] for x_ in h15]),np.array([x_['b']-x_['c'] for x_ in h15]),np.array([x_['X_MASK'] for x_ in h15]),np.array([x_['M_MASK']-x_['M_NATS'] for x_ in h15]),np.array([x_['M_RECOMP']-x_['M_NATS'] for x_ in h15]),np.array([x_['M_RECOMP_PL']-x_['M_NATS'] for x_ in h15]),np.zeros(15),pad=15);assert pd['n_imputed_zero_effect']==15 and pd['tests']['g2a']['n']==30 and abs(pd['tests']['g2a']['mean']-0.5*np.mean([x_['b']-x_['c'] for x_ in h15]))<1e-9
    bm=benchmark_exact(reps=3);assert bm['sign_vectors']==2**30 and bm['seconds_per_test']>0 and bm['sign_matrix_MB']<20
    assert P['delta_PL_value']==0.25 and P['sesoi_fact']==0.5 and P['n_min']==27 and 'sesoi_coef' not in P and len(IMPLEMENTATION_STATUS)==5
    print('e57_1280_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');ap.add_argument('--power-report');ap.add_argument('--bench',action='store_true');ap.add_argument('--N',type=int,default=100_000);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.bench:print(json.dumps(benchmark_exact(),indent=1))
    elif a.power_report:
        g=power_grid(N=a.N);g['gpu_hours']=gpu_table();g['gpu_assumption']=GPU_ASSUMPTION;json.dump(g,open(a.power_report,'w'),indent=1,default=float);print('power grid written')
    elif a.json:print(json.dumps(evaluate(json.load(open(a.json))),indent=1,default=float))
