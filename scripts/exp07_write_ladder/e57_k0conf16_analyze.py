"""E57 K0=0 CONFIRMATION analysis, synthetic 16-fact worlds, job 1283 (CPU; version 2, Council check 11/10; written and committed BEFORE any result of the new seeds 71..78 is opened; run_type = confirmation, K0 = 0, never pooled with K0 = 8 results).
REGISTRATION (core-claims thread, K0 = 8 data are the design basis only: SD 0.89 nat of the NATMASK-NATS per-seed margin difference, NATS exposure gain .455 nat over 3 doublings):
 SEEDS 71..78 (8), fixed, no second stage. delta (SESOI) = 0.5 nat (= about 3.3 doublings of the exposure count of the original NAT, about 10x exposures; in accuracy terms the uniform margin shift that flips about 10% of the facts is 0.82 nat on the NATS exposure-128 margin distribution, so 0.5 nat is a smaller, conservative effect). MDE (80%, one-sided .05, n = 8): about 0.9 nat if the seed SD is 0.89, about 1.4 nat if it is 1.4 (the SD comes from the K0 = 8 data). A null result is read only as 'inconclusive'.
 PRIMARY: out-of-window read at exposure 128, per-fact margin = log-prob(gold) - max log-prob(wrong) (nat, 16 candidates of equal token length), mean over the 16 facts of a seed, NATMASK - NATS paired by seed (reason: accuracy is the margin cut at a threshold, loses information, is quantised in steps of 6.25 pp and saturates). Sensitivity: the same with the per-seed MEDIAN of the fact margins. Accuracy = pre-specified key secondary (descriptive, 95% t interval of the seed differences).
 DECISION (named 'dual criterion'; it does NOT control the error rate for effects >= delta): confirmed iff the one-sided 95% t lower bound of the mean seed difference is > 0 AND the mean >= 0.5. Stronger step reported alongside: exact shifted sign-flip test of H0: mu <= 0.5 (one-sided). Wording: lower bound <= 0 -> 'inconclusive'; lower bound > 0 but mean < 0.5 -> 'detected but below the SESOI, practical relevance not established'. Disagreement rule: margin significant while the 95% interval of the accuracy difference contains 0 -> 'confidence in the gold answer improved; a retrieval decision improvement is not shown'. SERIAL gate: the secondary family is evaluated only if the primary passes.
 SECONDARY FAMILY (Holm, m = 4, only after the primary passes): (a) NATMASK - NATMASKC (exact two-sided sign-flip), (b) DIRmaskNORMfull - NATS (exact two-sided), (c) DIRfullNORMmask - NATS tested as 'effect < delta' (one-sided t, H0: mu >= 0.5; the power under a true effect 0 is about .42 at SD 0.89 and .23 at SD 1.4, disclosed), (d) the DIRECTION main effect of the 2x2. Descriptive (not in the family): size main effect, direction x size interaction, slope of the margin on log2 exposure (32/64/128, two-sided), accuracy. 2x2 cells: direction {full, mask} x size {full, mask}: NATS = (full, full), DIRfullNORMmask = (full, mask), DIRmaskNORMfull = (mask, full), NATMASK = (mask, mask); 'size' = the post-clip update norm of the step (the swap rescales the clipped step vector, so the norm that is exchanged is the post-clip norm).
 PRESERVATION GATES (separate from the family): in-window read (exposure 128, group 0): harm = margin(NATS) - margin(NATMASK), one-sided 95% t upper bound < 0.25 nat; general text: DCLM gate (job 1283d), one-sided 95% upper bound < +0.01 nat/token (read from its json). Not shown -> 'preservation not established'. Headlines: 'binding improvement' needs only the primary; 'improvement without harm' needs the primary AND both gates.
 CLIP: no conditional rule. NATS_clip10, NATMASK_clip10, NATMASKC_clip10 x 8 seeds are run unconditionally, SENSITIVITY ONLY (cannot change the primary decision; interpretations recorded in advance: robust = NATMASK_clip10 - NATS_clip10 >= 0.5 and NATS_clip10 does not close the gap; clip-driven = NATS_clip10 rises by >= 0.5 relative to NATS; inconclusive otherwise). Clip rates and pre-clip norms are computed from the LOGS ONLY (stage clip), their sha256 must be passed to stage main (--clip-hash) before any margin is read.
 GATES (reported first; a failure = bug, rerun the arms of the seed; a seed that fails again is EXCLUDED (n = 7, min exact p = 1/128); two or more excluded seeds = the confirmatory status is lost): G1 NATBINF == NATS bit for bit (update_norm, ce, gnorm, coef of every chunk, final state hash); ACTIVITY NATMASK != NATS at chunk K0 = 0; read hashes before == after; K0 tag 0 on every mask arm; the regression gate file of job 1279 (e57_orc_regress) and the 256-token unit test (--check-runner); versions.json (library versions and flags) must exist and is reported.
  python3 e57_k0conf16_analyze.py --stage clip --dir O --seeds 71,...,78 --json clip.json   (prints the sha256)
  python3 e57_k0conf16_analyze.py --stage main --dir O --seeds 71,...,78 --clip-hash H [--dcml-json J] [--json J] [--check-runner] | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
from scipy import stats
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ARMS=['NATS','NATBINF','NATMASK','NATMASKC','DIRmaskNORMfull','DIRfullNORMmask'];CLIP_ARMS=['NATS_clip10','NATMASK_clip10','NATMASKC_clip10'];DELTA=0.5;EXPO=['32','64','128']
def signflip(d,shift=0.0):
    d=np.asarray(d,float)-shift;n=len(d);obs=d.mean();S=np.array([[1 if (i>>j)&1 else -1 for j in range(n)] for i in range(2**n)]);m=(S*d[None]).mean(1)
    return float((m>=obs-1e-12).mean()),float((np.abs(m)>=abs(obs)-1e-12).mean())
def holm(p):
    k=sorted(p,key=lambda z:p[z]);m=len(k);adj={};run=0.0
    for i,z in enumerate(k):run=max(run,min(1.0,(m-i)*p[z]));adj[z]=run
    return adj
def t1(d,level=0.95):
    d=np.asarray(d,float);n=len(d);m=d.mean();se=d.std(ddof=1)/np.sqrt(n);t=stats.t.ppf(level,n-1);return dict(mean=float(m),lower_onesided=float(m-t*se),upper_onesided=float(m+t*se),se=float(se),n=int(n),p_less_than_0_5_onesided=float(stats.t.cdf((m-DELTA)/se,n-1)))
def t2(d,level=0.95):
    d=np.asarray(d,float);n=len(d);m=d.mean();se=d.std(ddof=1)/np.sqrt(n);t=stats.t.ppf(0.5+level/2,n-1);return [float(m-t*se),float(m+t*se)]
def fact_margins(path,app,key='out'):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']]);L=np.array(d['read'][app][key],float)
    if key!='out':fact=fact[:len(L)];g=g[:len(L)]
    L=L[fact];gg=g[fact];gold=L[np.arange(len(gg)),gg];w=L.copy();w[np.arange(len(gg)),gg]=-np.inf;return gold-w.max(1),(L.argmax(1)==gg).astype(float)
def logs(p):return json.load(open(p))
def same(a,b):return len(a['log'])==len(b['log']) and all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in ('update_norm','ce','gnorm','coef')) and a.get('final_hash')==b.get('final_hash')
def clip_stats(D,seeds,tag='k0c16'):
    D=pathlib.Path(D);cl={}
    for a in ARMS+CLIP_ARMS:
        cl[a]={}
        for nm,sel in (('before_chunk8',lambda e:e['k']<8),('from_chunk8',lambda e:e['k']>=8)):
            es=[e for s in seeds for e in logs(D/f'E55_{tag}_{a}_s{s}.saved.json')['log'] if sel(e)];cl[a][nm]=dict(gnorm_pre_clip_mean=float(np.mean([e['gnorm'] for e in es])),clip_rate=float(np.mean([e['coef']<1.0 for e in es])),median_update_norm=float(np.median([e['update_norm'] for e in es])))
    return cl
def chash(cl):return hashlib.sha256(json.dumps(cl,sort_keys=True).encode()).hexdigest()
def analyze(D,seeds,tag='k0c16',run_type='confirmation',clip_hash=None,dclm_json=None):
    D=pathlib.Path(D);sv=lambda a,s:logs(D/f'E55_{tag}_{a}_s{s}.saved.json');rd=lambda a,s:D/f'E55_{tag}_{a}_s{s}.read.json'
    cl=clip_stats(D,seeds,tag);assert clip_hash==chash(cl),'clip statistics must be recorded (hash) before any margin is read: run --stage clip first'
    out={'run_type':run_type,'seeds_requested':list(seeds),'clip_hash':clip_hash,'clip':cl}
    k0s={(a,sv(a,s).get('K0')) for a in ARMS+CLIP_ARMS if a!='NATS' and a!='NATS_clip10' for s in seeds};assert {k for _,k in k0s}=={0},('every mask arm must carry K0 = 0 (never pool across K0)',k0s);out['K0']=0
    vf=D/'versions.json';out['versions']=json.load(open(vf)) if vf.exists() else None
    g1={s:bool(same(sv('NATBINF',s),sv('NATS',s))) for s in seeds};act={s:bool(sv('NATMASK',s)['log'][0]['ce']!=sv('NATS',s)['log'][0]['ce'] or sv('NATMASK',s)['log'][0]['update_norm']!=sv('NATS',s)['log'][0]['update_norm']) for s in seeds}
    hs=[e[1]==e[2] for a in ARMS for s in seeds for e in sv(a,s).get('read_hash',[])]
    bad=sorted(s for s in seeds if not (g1[s] and act[s]));use=[s for s in seeds if s not in bad]
    out['gates']=dict(G1_NATBINF_eq_NATS=g1,activity_at_chunk_K0=act,read_hash_equal=(bool(all(hs)) if hs else None),regression_gate_file_found=bool((D.parent/'e57_orc_regress').exists()),versions_json_found=bool(vf.exists()),excluded_seeds=bad,n_effective=len(use),confirmatory_status=bool(len(bad)<=1 and len(use)>=7))
    if len(use)<5:out['abort']='too few seeds';return out
    M={a:np.array([fact_margins(rd(a,s),'128')[0].mean() for s in use]) for a in ARMS+CLIP_ARMS};Md={a:np.array([np.median(fact_margins(rd(a,s),'128')[0]) for s in use]) for a in ARMS}
    A={a:np.array([fact_margins(rd(a,s),'128')[1].mean() for s in use]) for a in ARMS}
    d=M['NATMASK']-M['NATS'];st=t1(d);p0=signflip(d)[0];pd=signflip(d,DELTA)[0];passed=bool(st['lower_onesided']>0 and st['mean']>=DELTA)
    word='confirmed (dual criterion; the error rate for effects >= delta is not controlled)' if passed else ('detected but below the SESOI, practical relevance not established' if st['lower_onesided']>0 else 'inconclusive')
    dacc=A['NATMASK']-A['NATS'];acc_ci=t2(dacc);dis=bool(passed and acc_ci[0]<=0<=acc_ci[1])
    out['primary']=dict(contrast='NATMASK-NATS margin128 (nat)',per_seed=[float(x) for x in d],stats=st,p_signflip_H0_mu_le_0=p0,p_signflip_H0_mu_le_0_5=pd,passed_dual_criterion=passed,wording=word,sensitivity_median_mean_diff=float((Md['NATMASK']-Md['NATS']).mean()),accuracy_diff=dict(mean=float(dacc.mean()),ci95_t=acc_ci),disagreement_rule_fired=dis,disagreement_wording=('confidence in the gold answer improved; a retrieval decision improvement is not shown' if dis else None),min_exact_p=float(1/2**len(use)))
    if passed:
        dirmain=0.5*((M['DIRmaskNORMfull']-M['NATS'])+(M['NATMASK']-M['DIRfullNORMmask']));fam={'a_NATMASK-NATMASKC':signflip(M['NATMASK']-M['NATMASKC'])[1],'b_DIRmaskNORMfull-NATS':signflip(M['DIRmaskNORMfull']-M['NATS'])[1],'c_DIRfullNORMmask-NATS_less_than_delta':t1(M['DIRfullNORMmask']-M['NATS'])['p_less_than_0_5_onesided'],'d_direction_main_effect':signflip(dirmain)[1]}
        ad=holm(fam);out['family_holm_m4']={k:dict(p=fam[k],p_holm=ad[k]) for k in fam}
        out['family_values']={'a':float((M['NATMASK']-M['NATMASKC']).mean()),'b':float((M['DIRmaskNORMfull']-M['NATS']).mean()),'c':t1(M['DIRfullNORMmask']-M['NATS']),'d':float(dirmain.mean())}
    else:out['family_holm_m4']='not evaluated (serial gate: the primary did not pass)'
    sizemain=0.5*((M['DIRfullNORMmask']-M['NATS'])+(M['NATMASK']-M['DIRmaskNORMfull']));inter=(M['NATMASK']-M['DIRmaskNORMfull'])-(M['DIRfullNORMmask']-M['NATS'])
    SL={a:np.array([float(np.polyfit(np.log2([32,64,128]),[fact_margins(rd(a,s),e)[0].mean() for e in EXPO],1)[0]) for s in use]) for a in ('NATS','NATMASK')}
    out['descriptive']=dict(size_main_effect=float(sizemain.mean()),direction_x_size_interaction=float(inter.mean()),slope_diff_NATMASK_minus_NATS=float((SL['NATMASK']-SL['NATS']).mean()),slope_two_sided_signflip_p=signflip(SL['NATMASK']-SL['NATS'])[1],acc128={a:float(A[a].mean()) for a in ARMS},margin128={a:float(M[a].mean()) for a in ARMS+CLIP_ARMS})
    inw={a:np.array([fact_margins(rd(a,s),'128','in_g0')[0].mean() for s in use]) for a in ('NATS','NATMASK')};harm=inw['NATS']-inw['NATMASK'];h=t1(harm)
    out['preservation']=dict(inwindow_harm_NATS_minus_NATMASK=h,inwindow_gate_upper_lt_0_25=bool(h['upper_onesided']<0.25),dclm_gate=None)
    if dclm_json:dj=json.load(open(dclm_json));up=dj.get('gate',{}).get('ci_upper_stream_cluster');out['preservation']['dclm_gate']=dict(upper=up,passed=bool(up is not None and up<0.01))
    gates_ok=out['preservation']['inwindow_gate_upper_lt_0_25'] and bool((out['preservation']['dclm_gate'] or {}).get('passed'))
    out['headlines']=dict(binding_improvement=passed,improvement_without_harm=bool(passed and gates_ok),preservation_note=('preservation not established' if not gates_ok else 'preservation shown'))
    sens=dict(NATMASK_clip10_minus_NATS_clip10=float((M['NATMASK_clip10']-M['NATS_clip10']).mean()),NATMASKC_clip10_minus_NATS_clip10=float((M['NATMASKC_clip10']-M['NATS_clip10']).mean()),NATS_clip10_minus_NATS=float((M['NATS_clip10']-M['NATS']).mean()),NATMASK_clip10_minus_NATMASK=float((M['NATMASK_clip10']-M['NATMASK']).mean()))
    sens['interpretation']='robust' if (sens['NATMASK_clip10_minus_NATS_clip10']>=0.5 and sens['NATS_clip10_minus_NATS']<0.5) else ('clip-driven' if sens['NATS_clip10_minus_NATS']>=0.5 else 'inconclusive');sens['note']='sensitivity only; cannot change the primary decision';out['clip_sensitivity']=sens
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    assert abs(signflip([1.]*8)[0]-1/256)<1e-12 and abs(signflip([1.]*7)[0]-1/128)<1e-12
    def mk(td,tag,arm,s,gain,k0=0):
        g=[int(x) for x in rng.integers(0,16,20)];kind=['fact']*16+['ctrl']*4;R={}
        for e in ('32','64','128'):
            L=rng.normal(size=(20,16));L[np.arange(20),g]+=gain*(1 if e=='128' else 0.5);R[e]={'out':L.tolist()}
        L2=rng.normal(size=(20,16));L2[np.arange(20),g]+=gain;R['128']['in_g0']=L2.tolist()
        json.dump(dict(meta=dict(gold_idx=g,kind=kind),read=R),open(td/f'E55_{tag}_{arm}_s{s}.read.json','w'))
        base=arm.replace('_clip10','');log=[dict(k=i,update_norm=1.0,ce=2.0+(0.1 if base=='NATMASK' else 0),gnorm=1.0+0.1*i,coef=0.5 if i%2 else 1.0) for i in range(16)]
        json.dump(dict(K0=(None if base=='NATS' else 0),log=log,final_hash='h',canary_w0={'_tail_nll':2.0},reads={'127':dict(canary={'_tail_nll':2.01})},read_hash=[(127,'a','a')]),open(td/f'E55_{tag}_{arm}_s{s}.saved.json','w'))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);seeds=list(range(71,79))
        for s in seeds:
            for a in ARMS+CLIP_ARMS:mk(td,'k0c16',a,s,{'NATMASK':3.0,'DIRmaskNORMfull':3.0,'NATMASKC':0.5,'NATMASK_clip10':3.0}.get(a,0.0))
        cl=clip_stats(td,seeds);h=chash(cl)
        try:analyze(td,seeds,clip_hash='wrong');raise SystemExit('clip hash gate not enforced')
        except AssertionError:pass
        r=analyze(td,seeds,clip_hash=h);assert r['gates']['confirmatory_status'] and r['primary']['passed_dual_criterion'] and isinstance(r['family_holm_m4'],dict) and r['clip_sensitivity']['interpretation'] in ('robust','clip-driven','inconclusive') and r['headlines']['binding_improvement']
        for a in ARMS+CLIP_ARMS:mk(td,'k0c16',a,71,0.0)
        r2=analyze(td,seeds,clip_hash=chash(clip_stats(td,seeds)));assert r2['gates']['n_effective']==8
    print('e57_k0conf16_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='main');ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--tag',default='k0c16');ap.add_argument('--json');ap.add_argument('--clip-hash');ap.add_argument('--dclm-json');ap.add_argument('--check-runner',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.stage=='clip':
        cl=clip_stats(a.dir,[int(x) for x in a.seeds.split(',')],a.tag);print(json.dumps(cl,indent=1));print('CLIP_HASH',chash(cl))
        if a.json:json.dump(cl,open(a.json,'w'),indent=1)
    else:
        if a.check_runner:
            import e57_oracle_run as OR;OR.selftest()
        r=analyze(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,clip_hash=a.clip_hash,dclm_json=a.dclm_json);print(json.dumps(r,indent=1,default=float))
        if a.json:json.dump(r,open(a.json,'w'),indent=1,default=float)
