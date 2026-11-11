"""E57 direct-access oracle pilot/confirmation analysis (CPU; new file; written and committed BEFORE any result of jobs 1279ff is opened; core-claims thread 11/10 registration draft). Inputs: <dir>/E55_<tag>_<arm>_s<seed>.{saved,read}.json of e57_oracle_run.py (tags orcp4 = cell v3_32_4_a16, orcp16 = cell v3_32_16_a16).
PRIMARY metric (per fact, read at exposure 16, out of window, 16 candidates of equal token length): margin = log-prob(gold) - max log-prob(wrong) in nat (candidate sums; equal candidate length, so sum and length-normalised rankings coincide); secondary: accuracy (argmax = gold), log-softmax margin = log-softmax(gold) - logsumexp(wrong). Unit of inference = SEED (mean over the facts of a seed), paired by seed across arms; t intervals (df = n_seeds - 1). Pilot (3 seeds) = noise / code check, no confirmatory inference.
p=4 contrasts (all margins): ORACLE-NATS (first claim), ORACLE-RAND (second claim, gatekept: tested only if the first holds), ORACLE-PLACEBO, 2x2 factorial effects: same-fact main = 1/2[(ORACLE-NATS)+(MASK-HALF)], general = 1/2[(HALF-NATS)+(MASK-ORACLE)], MASK-HALF (direction corrected: positive = re-opening the same-fact exposures lowers the gain), MASK-NATS at p=4 and at p=16 (Holm over these two). NO p=4 vs p=16 contrast.
Gates (reported before any contrast): G2 exact bit identity: NATBINF == NATS at p=4 (update_norm, ce, gnorm, coef of every chunk, final state hash) and at p=16 seed 62 ORACLE == NATS and HALF == NATMASK; G3 read hashes before == after in every read; G1 manipulation check from token NLL logs: oracle later-exposure fact-token NLL on the update path minus the natural-context NLL, vs the noise tau = 3 x |NATS - NATBINF| fluctuation of the same quantity (if both are bit identical tau = 1e-6); G1 is information only, never an exclusion; RAND hit rate; clip statistics (pre-clip gnorm mean, clip rate) and the fact-gradient share per arm.
  python3 e57_oracle_analyze.py --dir O --seeds 62,63,64 [--json J] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy import stats
P4=['NATS','NATBINF','NATMASK','NATORACLE','NATHALF','NATORACLE_RAND','NATPLACEBO'];P16=['NATS','NATMASK','NATORACLE_RAND','NATPLACEBO']
def lse(x):x=np.asarray(x,float);m=x.max();return m+np.log(np.exp(x-m).sum())
def facts(path,app='16'):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']]);L=np.array(d['read'][app]['out'],float)[fact];g=g[fact]
    gold=L[np.arange(len(g)),g];wrong=L.copy();wrong[np.arange(len(g)),g]=-np.inf
    return dict(margin=gold-wrong.max(1),acc=(L.argmax(1)==g).astype(float),lsm=(gold-lse(L.reshape(-1)) if False else np.array([L[i,g[i]]-lse(np.delete(L[i],g[i])) for i in range(len(g))])))
def seedmeans(D,tag,arm,seeds,key):return np.array([facts(D/f'E55_{tag}_{arm}_s{s}.read.json')[key].mean() for s in seeds])
def tci(d,level=0.95):
    d=np.asarray(d,float);n=len(d);m=d.mean();sd=d.std(ddof=1) if n>1 else float('nan');se=sd/np.sqrt(n);t=stats.t.ppf(0.5+level/2,n-1) if n>1 else float('nan')
    return dict(mean=float(m),ci95=[float(m-t*se),float(m+t*se)],sd=float(sd),p=float(2*stats.t.sf(abs(m/se),n-1)) if se>0 else float('nan'),n=int(n),signflip_p_one_sided=signflip(d)[0],signflip_p_two_sided=signflip(d)[1])
def signflip(d):
    """exact sign-flip permutation test of the mean of the per-seed paired differences d (Council 11/10: exact, n >= 5 one-sided (min p = 1/32), n >= 6 two-sided (min p = 2/64)); returns (p_one_sided for mean > 0, p_two_sided)"""
    d=np.asarray(d,float);n=len(d);obs=d.mean();S=np.array([[1 if (i>>j)&1 else -1 for j in range(n)] for i in range(2**n)]);m=(S*d[None]).mean(1)
    return float((m>=obs-1e-15).mean()),float((np.abs(m)>=abs(obs)-1e-15).mean())
def seeds_needed(sd,sesoi,alpha=0.05,power=0.8,floor=8):
    """number of seeds for a paired design from a fixed SESOI (smallest effect size of interest, same units as sd): n = ((z_{1-alpha} + z_power) sd / sesoi)^2 (one-sided), at least `floor` (Council: 8 to 12)"""
    return int(max(floor,np.ceil(((stats.norm.ppf(1-alpha)+stats.norm.ppf(power))*sd/sesoi)**2)))
def holm(p):
    k=sorted(p,key=lambda z:p[z]);m=len(k);adj={};run=0.0
    for i,z in enumerate(k):run=max(run,min(1.0,(m-i)*p[z]));adj[z]=run
    return adj
def logs(path):
    r=json.load(open(path));return r
def same(a,b,keys=('update_norm','ce','gnorm','coef')):return all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in keys) and len(a['log'])==len(b['log']) and a.get('final_hash')==b.get('final_hash')
ROOT_PARAMS=dict(oracle_k=0.5,half_retain_k=0.5,r2_oracle_k=0.25,equiv_band=0.25,equiv_seed_max=0.5,floor=-0.25,interaction_k=0.5,leak_ratio=0.5,nocopy_se=3.0,nats_invalid_se=3.0,boot_B=2000,boot_seed=20261111,ident_dlogit=1e-4,ident_dmargin_frac=0.01)
def _boot(dfs,rng,B):
    """dfs = list over seeds of per-fact difference arrays; stratified bootstrap (facts resampled within every seed), statistic = mean over seeds of the seed means -> array [B]"""
    return np.array([np.mean([x[rng.integers(0,len(x),len(x))].mean() for x in dfs]) for _ in range(B)])
def _state(dfs,G,kind,k=None,P=ROOT_PARAMS):
    """three-state judgement of a paired difference d (per-fact arrays per seed), G = the registered gain scale:
    greater(d,k): mean >= k G AND d_s > 0 in every seed AND the 95% bootstrap lower bound > 0;  equiv: the 90% interval inside +-0.25 G AND max_s |d_s| <= 0.5 G;  less(d,k): the 95% upper bound < k G.  Anything else = unresolved (goes to R3)."""
    rng=np.random.default_rng(P['boot_seed']);bs=_boot(dfs,rng,P['boot_B']);ds=np.array([x.mean() for x in dfs]);m=float(ds.mean());lo95,hi95=np.quantile(bs,[.025,.975]);lo90,hi90=np.quantile(bs,[.05,.95])
    if kind=='greater':ok=bool(m>=k*G and (ds>0).all() and lo95>0)
    elif kind=='equiv':ok=bool(lo90>=-P['equiv_band']*G and hi90<=P['equiv_band']*G and np.abs(ds).max()<=P['equiv_seed_max']*G)
    elif kind=='less':ok=bool(hi95<k*G)
    return dict(kind=kind,k=k,mean=m,per_seed=[float(x) for x in ds],ci95=[float(lo95),float(hi95)],ci90=[float(lo90),float(hi90)],ok=ok)
def classify_root(F4,F16,X,ident16,seeds,P=ROOT_PARAMS):
    """ROOT CLASSIFICATION v2 (Council 11/11, re-registered before any 1279 result was opened; margin sign: margin = log-prob(gold) - max log-prob(wrong), higher = better; per-seed means are taken inside a seed first; unit of inference = the seed, uncertainty = stratified fact bootstrap B = 2000 within seeds).
 F4[arm][seed] = per-fact margins at p = 4, F16 likewise at p = 16; X[arm][seed] = per-chunk gaps (update-path NLL - natural-context NLL of the later-exposure fact tokens, nat PER TOKEN, chunks >= 8) at p = 4; ident16 = dict arm -> dict(ok=bool) of the p = 16 coherence check.
 ORDER: step 0 -> p=16 coherence -> leak states -> floors -> R1 -> R2a -> R2b -> R3. All 3 seeds must be present, otherwise R3-incomplete.
 STEP 0: G = mean_s(margin NATMASK - NATS) at p = 4; need G_s > 0 in every seed and the 95% lower bound > 0, else 'not applicable'. (No g_min: no justification registered.)
 p=16 COHERENCE (implementation validity, not a root): ORACLE, RAND, PLACEBO must hide 0 keys at p = 16 and equal NATS bit for bit; with a non-deterministic kernel |d logit| <= 1e-4 and |d margin| <= 0.01 G are tolerated. Failure = 'implementation invalid'.
 LEAK (p = 4, per seed s; X in nat per token over chunks >= 8, SE = sd/sqrt(n_chunks), CI95 lower = X - 1.96 SE): invalid run if |X_NATS,s| > 3 SE; 'R3-no-copy' if X_MASK,s < 3 SE; 'R3-leak' if X_ORACLE,s < 0.5 X_MASK,s or the CI95 lower bound of X_ORACLE,s <= 0. Any seed triggering counts; priority invalid > no-copy > leak.
 FLOORS: mean(ORACLE-NATS) >= -0.25 G and mean(HALF-NATS) >= -0.25 G, else R3.
 R1: greater(ORACLE-NATS, 0.5 G) and greater(ORACLE-RAND, 0.5 G) and greater(ORACLE-PLACEBO, 0.5 G) and less(HALF-NATS, 0.5 G) and the interaction greater(G4 - G16, 0.5, G4) (G16 itself is descriptive only).
 R2a: less(ORACLE-NATS, 0.25 G) and equiv(HALF-MASK). R2b: less(ORACLE-NATS, 0.25 G) and equiv(HALF-NATS) (R2a has priority when both hold). Unresolved states -> R3. The multiplicative coefficients (0.5, 0.25, 3, 0.5) are registered and are not changed after registration.
 MATCHING (code facts): ORACLE_RAND hides, per sentence and per source chunk, the same number of keys as the oracle from the SAME chunk (chunk-level distance matched) excluding ALL earlier same-fact keys of that sentence; PLACEBO hides the first L tokens of the other-fact sentence of the same chunk with the closest length and in-chunk position (length and chunk distance matched, same-fact keys stay visible); both are deterministic in (seed, chunk, sentence).
 PILOT STATUS: the 3-seed pilot is for the root classification only, not evidence."""
    out=dict(params=P,seeds=list(seeds))
    if any(s not in F4['NATS'] or s not in F4['NATORACLE'] for s in seeds) or len(seeds)<3:out['label']='R3-incomplete';return out
    L=lambda A,a,b:[A[a][s]-A[b][s] for s in seeds]
    dG=L(F4,'NATMASK','NATS');Gs=np.array([x.mean() for x in dG]);G=float(Gs.mean());g0=_state(dG,G,'greater',0.0,P);out['G']=G;out['step0']=dict(G_per_seed=[float(x) for x in Gs],ci95=g0['ci95'],ok=bool((Gs>0).all() and g0['ci95'][0]>0))
    if not out['step0']['ok']:out['label']='not applicable (step 0)';return out
    out['p16_coherence']=ident16
    if not all(v.get('ok') for v in ident16.values()):out['label']='implementation invalid (p=16 coherence)';return out
    lk={};
    for s in seeds:
        r={}
        for arm in ('NATS','NATMASK','NATORACLE'):
            x=np.asarray(X[arm][s],float);se=x.std(ddof=1)/np.sqrt(len(x));r[arm]=dict(X=float(x.mean()),se=float(se))
        r['invalid']=bool(abs(r['NATS']['X'])>P['nats_invalid_se']*r['NATS']['se']);r['no_copy']=bool(r['NATMASK']['X']<P['nocopy_se']*r['NATMASK']['se'])
        r['leak']=bool(r['NATORACLE']['X']<P['leak_ratio']*r['NATMASK']['X'] or r['NATORACLE']['X']-1.96*r['NATORACLE']['se']<=0);lk[s]=r
    out['leak_per_seed']=lk
    if any(v['invalid'] for v in lk.values()):out['label']='run invalid (|X_NATS| > 3 SE)';return out
    if any(v['no_copy'] for v in lk.values()):out['label']='R3-no-copy';return out
    if any(v['leak'] for v in lk.values()):out['label']='R3-leak';return out
    dO=L(F4,'NATORACLE','NATS');dH=L(F4,'NATHALF','NATS');fl=dict(oracle_minus_nats=float(np.mean([x.mean() for x in dO])),half_minus_nats=float(np.mean([x.mean() for x in dH])));out['floors']=fl
    if fl['oracle_minus_nats']<P['floor']*G or fl['half_minus_nats']<P['floor']*G:out['label']='R3 (floor violated)';return out
    st=dict(oracle_nats=_state(dO,G,'greater',P['oracle_k'],P),oracle_rand=_state(L(F4,'NATORACLE','NATORACLE_RAND'),G,'greater',P['oracle_k'],P),oracle_placebo=_state(L(F4,'NATORACLE','NATPLACEBO'),G,'greater',P['oracle_k'],P),half_retained=_state(dH,G,'less',P['half_retain_k'],P))
    d16=[F16['NATMASK'][s]-F16['NATS'][s] for s in seeds];rng=np.random.default_rng(P['boot_seed']);b4=_boot(dG,rng,P['boot_B']);b16=_boot(d16,rng,P['boot_B']);dd=b4-b16;G16=np.array([x.mean() for x in d16]);ds=Gs-G16
    st['interaction_G4_minus_G16']=dict(mean=float(ds.mean()),per_seed=[float(x) for x in ds],ci95=[float(np.quantile(dd,.025)),float(np.quantile(dd,.975))],ok=bool(ds.mean()>=P['interaction_k']*G and (ds>0).all() and np.quantile(dd,.025)>0),G16_descriptive=float(G16.mean()))
    st['oracle_less_025']=_state(dO,G,'less',P['r2_oracle_k'],P);st['half_equiv_mask']=_state(L(F4,'NATHALF','NATMASK'),G,'equiv',None,P);st['half_equiv_nats']=_state(dH,G,'equiv',None,P);out['states']=st
    if st['oracle_nats']['ok'] and st['oracle_rand']['ok'] and st['oracle_placebo']['ok'] and st['half_retained']['ok'] and st['interaction_G4_minus_G16']['ok']:out['label']='R1'
    elif st['oracle_less_025']['ok'] and st['half_equiv_mask']['ok']:out['label']='R2a'
    elif st['oracle_less_025']['ok'] and st['half_equiv_nats']['ok']:out['label']='R2b'
    else:out['label']='R3'
    return out
def update_signal(D,tag4,seeds):
    """common items (Hans 11/11): descriptive update-signal measures from the logs: (a) mean cosine between the masked and the full-context step (wlog cos_mask_full, mask arms, before / after chunk 8) i.e. the gradient cosine NATS-NATMASK direction agreement, (b) later-exposure fact-token NLL (natural context and update path) per exposure number (learning curve = update utility), (c) the fact-gradient share. Canary did not pass (+0.039 nat, 1276) and DCLM non-inferiority is NOT established: these statements are fixed in the output notes."""
    out={'notes':['canary tail NLL rise NATMASK-NATS did not pass (+0.039 nat in job 1276 > +0.01)','DCLM non-inferiority is not established (job 1274: +0.0091 point estimate, cluster CI upper bound +0.0104)']}
    for arm in ('NATMASK','NATHALF','NATORACLE','NATPLACEBO','NATORACLE_RAND','NATS'):
        cs={'before_chunk8':[],'from_chunk8':[]};cur={};share=[]
        for s in seeds:
            pth=D/f'E55_{tag4}_{arm}_s{s}.saved.json'
            if not pth.exists():continue
            r=logs(pth)
            for e in r.get('wlog',[]):cs['before_chunk8' if e['k']<8 else 'from_chunk8'].append(e['cos_mask_full'])
            for k,x in r.get('diag',{}).items():
                if x.get('nat_nll_fact_later') is not None and x.get('app') is not None:cur.setdefault(x['app'],[]).append((x['nat_nll_fact_later'],x['upd_nll_fact_later']))
                if x.get('fact_grad_share') is not None:share.append(x['fact_grad_share'])
        out[arm]=dict(cos_mask_full={k:(float(np.mean(v)) if v else None) for k,v in cs.items()},fact_grad_share_mean=(float(np.mean(share)) if share else None),later_fact_nll_by_exposure={str(a):dict(natural=float(np.mean([z[0] for z in v])),update_path=float(np.mean([z[1] for z in v]))) for a,v in sorted(cur.items())})
    return out
def analyze(D,seeds,tag4='orcp4',tag16='orcp16',gate_seed=62,run_type='pilot_exploration'):
    D=pathlib.Path(D);out={'run_type':run_type,'seeds':list(seeds),'n_seeds':len(seeds),'gates':{},'p4':{},'p16':{}}
    g2={}
    for s in seeds:g2[f'p4_NATBINF=NATS_s{s}']=bool(same(logs(D/f'E55_{tag4}_NATBINF_s{s}.saved.json'),logs(D/f'E55_{tag4}_NATS_s{s}.saved.json')))
    f=lambda a,b:D/f'E55_{tag16}_{a}_s{gate_seed}.saved.json'
    if f('NATORACLE','').exists() if False else (D/f'E55_{tag16}_NATORACLE_s{gate_seed}.saved.json').exists():
        g2[f'p16_ORACLE=NATS_s{gate_seed}']=bool(same(logs(D/f'E55_{tag16}_NATORACLE_s{gate_seed}.saved.json'),logs(D/f'E55_{tag16}_NATS_s{gate_seed}.saved.json')))
        g2[f'p16_HALF=NATMASK_s{gate_seed}']=bool(same(logs(D/f'E55_{tag16}_NATHALF_s{gate_seed}.saved.json'),logs(D/f'E55_{tag16}_NATMASK_s{gate_seed}.saved.json')))
    k0s={}
    for tag,arms in ((tag4,P4),(tag16,P16)):
        for arm in arms:
            for s in seeds:
                pth=D/f'E55_{tag}_{arm}_s{s}.saved.json'
                if pth.exists():k0s.setdefault((tag,arm),set()).add(logs(pth).get('K0'))
    out['K0_by_arm']={f'{t}:{a_}':sorted(v,key=lambda z:(z is None,z)) for (t,a_),v in k0s.items()};base_k0={k for (t,a_),v in k0s.items() if a_!='NATMASK_K08' for k in v};out['K0']=sorted(base_k0)[0] if len(base_k0)==1 else None;assert len(base_k0)==1,('K0 differs between arms: never pool across K0',base_k0)
    out['gates']['G2_exact_bit_identity']=g2;out['gates']['G2_pass']=bool(all(g2.values()))
    g3={};
    for tag,arms in ((tag4,P4),(tag16,P16)):
        for arm in arms:
            for s in seeds:
                pth=D/f'E55_{tag}_{arm}_s{s}.saved.json'
                if pth.exists():
                    for e in logs(pth).get('read_hash',[]):g3[f'{tag}_{arm}_s{s}_k{e[0]}']=bool(e[1]==e[2])
    out['gates']['G3_read_hash_equal']=bool(all(g3.values())) if g3 else None
    out['gates']['G1_exact_bit_identity']=bool(all(v for k,v in g2.items() if k.startswith('p4_NATBINF=NATS')))      # G1 = NATBINF == NATS on EVERY seed (hard gate, no tolerance tau); a failure = bug, rerun
    def tokstat(tag,arm,s,field):
        d=logs(D/f'E55_{tag}_{arm}_s{s}.saved.json').get('diag',{});v=[x[field] for x in d.values() if x.get(field) is not None];return float(np.mean(v)) if v else None
    g1={}
    for s in seeds:
        ex=lambda arm:(tokstat(tag4,arm,s,'upd_nll_fact_later')-tokstat(tag4,arm,s,'nat_nll_fact_later')) if tokstat(tag4,arm,s,'upd_nll_fact_later') is not None else None
        g1[s]=dict(oracle=ex('NATORACLE'),placebo=ex('NATPLACEBO'),rand=ex('NATORACLE_RAND'),nats=ex('NATS'))
    out['gates']['token_loss_info_only']=dict(per_seed=g1)
    hit=[];cl={}
    for arm in P4:
        pth=[D/f'E55_{tag4}_{arm}_s{s}.saved.json' for s in seeds];R=[logs(x) for x in pth]
        cl[arm]={}
        for nm,sel in (('before_window_fills_chunk_lt8',lambda e:e['k']<8),('after_window_fills_chunk_ge8',lambda e:e['k']>=8)):
            es=[e for r in R for e in r['log'] if sel(e)];cl[arm][nm]=dict(gnorm_pre_clip_mean=float(np.mean([e['gnorm'] for e in es])),clip_rate=float(np.mean([e['coef']<1.0 for e in es])))
        for nm,sel in (('before',lambda k:int(k)<8),('after',lambda k:int(k)>=8)):
            sh=[x['fact_grad_share'] for r in R for k,x in r.get('diag',{}).items() if sel(k) and x.get('fact_grad_share') is not None];cl[arm]['fact_grad_share_'+nm]=float(np.mean(sh)) if sh else None
        if arm=='NATORACLE_RAND':hit=[e['rand_hit_rate'] for r in R for e in r['log'] if 'rand_hit_rate' in e]
    out['p4']['clip_and_grad']=cl;out['p4']['rand_hit_rate_mean']=float(np.mean(hit)) if hit else None
    for key in ('margin','acc','lsm'):
        M={a:seedmeans(D,tag4,a,seeds,key) for a in P4};c={}
        c['ORACLE-NATS']=M['NATORACLE']-M['NATS'];c['ORACLE-RAND']=M['NATORACLE']-M['NATORACLE_RAND'];c['ORACLE-PLACEBO']=M['NATORACLE']-M['NATPLACEBO']
        c['MASK-HALF']=M['NATMASK']-M['NATHALF'];c['same_fact_main']=0.5*((M['NATORACLE']-M['NATS'])+(M['NATMASK']-M['NATHALF']));c['general_main']=0.5*((M['NATHALF']-M['NATS'])+(M['NATMASK']-M['NATORACLE']));c['MASK-NATS_p4']=M['NATMASK']-M['NATS']
        out['p4'][key]={k:tci(v) for k,v in c.items()};out['p4'][key]['arm_means']={a:float(v.mean()) for a,v in M.items()}
        if key=='margin':
            M16={a:seedmeans(D,tag16,a,seeds,key) for a in P16};d16=M16['NATMASK']-M16['NATS'];out['p16'][key]=dict(arm_means={a:float(v.mean()) for a,v in M16.items()},**{'MASK-NATS_p16':tci(d16)})
            ps={'MASK-NATS_p4':out['p4']['margin']['MASK-NATS_p4']['p'],'MASK-NATS_p16':out['p16']['margin']['MASK-NATS_p16']['p']};ad=holm(ps);out['holm_MASK_minus_NATS']=ad
            n_=len(seeds);alpha=0.05;seq=['ORACLE-NATS','ORACLE-RAND','ORACLE-PLACEBO'];res={};ok=True        # fixed-sequence testing: each step is tested at alpha only if the previous step passed
            for c_ in seq:
                if n_<5:res[c_]=dict(tested=False,reason='n < 5: exact one-sided sign-flip has min p = 1/2^n > .05 at n < 5');ok=False;continue
                if not ok:res[c_]=dict(tested=False,reason='previous step in the fixed sequence failed');continue
                pv=out['p4']['margin'][c_]['signflip_p_one_sided'];res[c_]=dict(tested=True,p_one_sided_exact=pv,passed=bool(pv<alpha));ok=bool(pv<alpha)
            def fm(tag,arm,s_):return facts(D/f'E55_{tag}_{arm}_s{s_}.read.json')['margin']
            F4={a:{s_:fm(tag4,a,s_) for s_ in seeds} for a in P4};F16={a:{s_:fm(tag16,a,s_) for s_ in seeds} for a in P16}
            def gaps(arm,s_):
                dd=logs(D/f'E55_{tag4}_{arm}_s{s_}.saved.json').get('diag',{});return [x['upd_nll_fact_later']-x['nat_nll_fact_later'] for k_,x in sorted(dd.items(),key=lambda z:int(z[0])) if int(k_)>=8 and x.get('upd_nll_fact_later') is not None and x.get('nat_nll_fact_later') is not None]
            X={a:{s_:gaps(a,s_) for s_ in seeds} for a in ('NATS','NATMASK','NATORACLE')}
            ident={}
            Gsc=float(np.mean([(F4['NATMASK'][s_]-F4['NATS'][s_]).mean() for s_ in seeds]))
            for arm,sl in (('NATORACLE',[gate_seed]),('NATORACLE_RAND',list(seeds)),('NATPLACEBO',list(seeds))):
                for s_ in sl:
                    pa=D/f'E55_{tag16}_{arm}_s{s_}.saved.json'
                    if not pa.exists():ident[f'{arm}_s{s_}']=dict(ok=False,reason='missing');continue
                    r_a=logs(pa);r_n=logs(D/f'E55_{tag16}_NATS_s{s_}.saved.json');hidden=int(sum(e.get('masked_pairs',0) for e in r_a['log']));bit=bool(same(r_a,r_n))
                    ok=bool(hidden==0 and (bit or float(np.max(np.abs(fm(tag16,arm,s_)-fm(tag16,'NATS',s_))))<=ROOT_PARAMS['ident_dmargin_frac']*Gsc))
                    ident[f'{arm}_s{s_}']=dict(ok=ok,hidden_pairs=hidden,bit_identical=bit)
            out['root_classification']=classify_root(F4,F16,X,ident,list(seeds))
            out['update_signal']=update_signal(D,tag4,seeds)
            out['fixed_sequence']=res;out['interaction_note']='2x2 factorial effects (same_fact_main, general_main, MASK-HALF) are descriptive only; MASK-NATS p4 / p16 are a Holm family; no p4 vs p16 contrast';out['seeds_needed_from_SESOI']='seeds_needed(sd_of_paired_diff, SESOI) with SESOI fixed in the registration (floor 8)'
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    def mk(td,tag,arm,s,gain,same_as=None,hash_ok=True):
        g=[int(x) for x in rng.integers(0,16,36)];kind=['fact']*32+['ctrl']*4;L=rng.normal(size=(36,16));L[np.arange(36),g]+=gain
        json.dump(dict(meta=dict(gold_idx=g,kind=kind),read={'16':{'out':L.tolist()}}),open(td/f'E55_{tag}_{arm}_s{s}.read.json','w'))
        log=[dict(k=i*8,update_norm=1.0,ce=2.0+(0.1 if (arm=='NATMASK' and tag=='orcp4') else 0.0),gnorm=1.5,coef=0.66,rand_hit_rate=0.2) for i in range(3)]
        json.dump(dict(K0=0,log=log,final_hash='h',diag={str(8+j):dict(upd_nll_fact_later=3.0+0.01*j,nat_nll_fact_later=2.0,fact_grad_share=0.3,app=2+j) for j in range(8)},read_hash=[(15,'a','a' if hash_ok else 'b')]),open(td/f'E55_{tag}_{arm}_s{s}.saved.json','w'))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);seeds=[62,63,64,65,66]
        for s in seeds:
            for a in P4:mk(td,'orcp4',a,s,{'NATORACLE':3.0,'NATMASK':3.0,'NATHALF':0.5}.get(a,0.0))
            for a in P16:mk(td,'orcp16',a,s,{'NATMASK':2.0}.get(a,0.0))
        for a in ('NATORACLE','NATHALF'):mk(td,'orcp16',a,62,0.0)
        r=analyze(td,seeds);assert r['gates']['G2_pass'] and r['gates']['G3_read_hash_equal'] and 'same_fact_main' in r['p4']['margin'] and r['p4']['margin']['ORACLE-NATS']['mean']>1 and r['fixed_sequence']['ORACLE-NATS']['tested'] is True
        mk(td,'orcp4','NATBINF',62,0.0,hash_ok=False);r2=analyze(td,seeds);assert r2['gates']['G3_read_hash_equal'] is False
    # ---- classify_root scenarios (pure function; 3 seeds x 32 facts; noise small relative to G = 2.5) ----
    rg=np.random.default_rng(3);seeds3=[62,63,64];nf=32
    def fa(v):return {s_:v+rg.normal(0,0.05,nf) for s_ in seeds3}
    def mkF(o,r,p,h,m,n=0.0):return {'NATS':fa(n),'NATMASK':fa(m),'NATORACLE':fa(o),'NATORACLE_RAND':fa(r),'NATPLACEBO':fa(p),'NATHALF':fa(h)}
    F16={'NATS':fa(0.0),'NATMASK':fa(0.5)}                                           # G16 = 0.5 << G4 = 2.5: interaction holds
    def mkX(o,m,n=0.0,sd=0.02):return {'NATS':{s_:n+rg.normal(0,sd,8) for s_ in seeds3},'NATMASK':{s_:m+rg.normal(0,sd,8) for s_ in seeds3},'NATORACLE':{s_:o+rg.normal(0,sd,8) for s_ in seeds3}}
    idok={'x':dict(ok=True)}
    lab=lambda F,X,F16_=F16,idn=idok,sd_=seeds3:classify_root(F,F16_,X,idn,sd_)['label']
    assert lab(mkF(2.4,0.2,0.3,0.6,2.5),mkX(0.30,0.32))=='R1'
    assert lab(mkF(0.2,0.1,0.1,2.4,2.5),mkX(0.30,0.32))=='R2a'
    assert lab(mkF(0.2,0.1,0.1,0.2,2.5),mkX(0.30,0.32))=='R2b'
    assert lab(mkF(1.0,0.5,0.5,1.5,2.5),mkX(0.30,0.32))=='R3'
    assert lab(mkF(2.4,0.2,0.3,0.6,2.5),mkX(0.02,0.32))=='R3-leak'                 # X_ORACLE < 0.5 X_MASK
    assert lab(mkF(2.4,0.2,0.3,0.6,2.5),mkX(0.30,0.01))=='R3-no-copy'              # X_MASK < 3 SE
    assert lab(mkF(2.4,0.2,0.3,0.6,2.5),mkX(0.30,0.32,n=0.3)).startswith('run invalid')   # |X_NATS| > 3 SE
    assert lab(mkF(-1.0,0.2,0.3,0.6,2.5),mkX(0.30,0.32))=='R3 (floor violated)'
    assert lab(mkF(2.4,0.2,0.3,0.6,-0.5),mkX(0.30,0.32)).startswith('not applicable')
    assert lab(mkF(2.4,0.2,0.3,0.6,2.5),mkX(0.30,0.32),idn={'a':dict(ok=False)}).startswith('implementation invalid')
    assert lab(mkF(2.4,0.2,0.3,0.6,2.5),mkX(0.30,0.32),sd_=[62,63])=='R3-incomplete'
    F16b={'NATS':fa(0.0),'NATMASK':fa(2.4)};assert lab(mkF(2.4,0.2,0.3,0.6,2.5),mkX(0.30,0.32),F16_=F16b)=='R3'      # interaction fails (G16 close to G4)
    assert holm({'a':0.01,'b':0.04})=={'a':0.02,'b':0.04} and abs(signflip([1,1,1,1,1])[0]-1/32)<1e-12 and abs(signflip([1,1,1,1,1,1])[1]-2/64)<1e-12 and seeds_needed(1.0,0.5)>=8
    print('e57_oracle_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--json');ap.add_argument('--run-type',default='pilot_exploration',choices=['confirmation','k0_sensitivity','pilot_exploration','legacy_k08']);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        r=analyze(a.dir,[int(x) for x in a.seeds.split(',')],run_type=a.run_type);print(json.dumps(r,indent=1,default=float))
        if a.json:json.dump(r,open(a.json,'w'),indent=1,default=float)
