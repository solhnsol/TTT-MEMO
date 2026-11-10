"""E57 K0=0 CONFIRMATION analysis, synthetic 16-fact worlds, job 1283 (CPU; new file; written and committed BEFORE any result of the new seeds 71..78 is opened). Registration draft (core-claims thread, Council check pending; run_type = confirmation, K0 = 0, never pooled with K0 = 8 results):
 PRIMARY: out-of-window read at exposure 128, per-fact margin = log-prob(gold) - max log-prob(wrong) (nat; 16 candidates of equal token length), averaged over the 16 facts of a seed; NATMASK - NATS paired by seed. SUPERIOR if the exact sign-flip test (one-sided, n = 8 seeds, min p = 1/256) gives p < .05 AND the mean difference >= delta = 1.0 nat (SESOI).
 SECONDARY FAMILY (Holm, m = 4, exact two-sided sign-flip p): NATMASK - NATMASKC (margin128), DIRmaskNORMfull - NATS (margin128), DIRfullNORMmask - NATS (margin128), slope of the margin on log2(exposure) over exposures 32/64/128 (NATMASK - NATS). Descriptive only (no test): accuracy at 128, exposure-mean accuracy, the interaction DIRmask-NATS vs DIRfull-NATS.
 GATES (reported before any contrast; a failure = bug, rerun all arms of that seed, no seed is dropped or added): G1 NATBINF == NATS bit for bit on EVERY seed (update_norm, ce, gnorm, coef of every chunk and the final state hash); ACTIVITY: NATMASK differs from NATS at chunk K0 (= 0) in ce or update_norm; read hashes before == after in every read; every mask arm carries the K0 tag (all 0); the regression gate of job 1279 (new code at K0 = 8 reproduces job 1270 bit for bit) must have been run (file e57_orc_regress) and is looked up here; the 256-token unit test is part of the runner selftest (run via --check-runner).
 PRESERVATION (separate from the family): general text = job 1283d (DCLM K0 = 0); in-window read (exposure 128, group 0): margin NATMASK >= margin NATS - 0.5 nat; canary tail NLL rise NATMASK - NATS <= +0.01 nat (information; the DCLM gate decides).
 CLIP (descriptive, no clip arms in this job): per arm pre-clip gnorm mean, clip rate (coef < 1), median update_norm before / after chunk 8; if the clip rate of NATMASK exceeds that of NATS by more than 20 percentage points the conditional clip-sensitivity arms (NATS_clip10, NATMASK_clip10) are to be run on the same seeds (flag clip_sensitivity_triggered).
  python3 e57_k0conf16_analyze.py --dir O --seeds 71,72,73,74,75,76,77,78 [--tag k0c16] [--json J] [--check-runner] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ARMS=['NATS','NATBINF','NATMASK','NATMASKC','DIRmaskNORMfull','DIRfullNORMmask'];DELTA=1.0;EXPO=['32','64','128']
def signflip(d):
    d=np.asarray(d,float);n=len(d);obs=d.mean();S=np.array([[1 if (i>>j)&1 else -1 for j in range(n)] for i in range(2**n)]);m=(S*d[None]).mean(1)
    return float((m>=obs-1e-12).mean()),float((np.abs(m)>=abs(obs)-1e-12).mean())
def holm(p):
    k=sorted(p,key=lambda z:p[z]);m=len(k);adj={};run=0.0
    for i,z in enumerate(k):run=max(run,min(1.0,(m-i)*p[z]));adj[z]=run
    return adj
def read_stats(path,app,key='out'):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']]);grp=np.array(d['meta'].get('group',[0]*len(g)))
    L=np.array(d['read'][app][key],float)
    if key!='out':ix=np.arange(len(g))[:len(L)];fact=fact[ix];g=g[ix]
    L=L[fact];gg=g[fact];gold=L[np.arange(len(gg)),gg];w=L.copy();w[np.arange(len(gg)),gg]=-np.inf
    return dict(margin=float((gold-w.max(1)).mean()),acc=float((L.argmax(1)==gg).mean()))
def slope(m):return float(np.polyfit(np.log2([32,64,128]),np.array(m),1)[0])
def logs(p):return json.load(open(p))
def same(a,b):return len(a['log'])==len(b['log']) and all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in ('update_norm','ce','gnorm','coef')) and a.get('final_hash')==b.get('final_hash')
def analyze(D,seeds,tag='k0c16',run_type='confirmation'):
    D=pathlib.Path(D);sv=lambda a,s:logs(D/f'E55_{tag}_{a}_s{s}.saved.json');rd=lambda a,s:D/f'E55_{tag}_{a}_s{s}.read.json'
    out={'run_type':run_type,'seeds':list(seeds),'n_seeds':len(seeds)}
    k0s={(a,sv(a,s).get('K0')) for a in ARMS if a!='NATS' for s in seeds};ks={k for _,k in k0s};assert ks=={0},('every mask arm must carry K0 = 0 (never pool across K0)',k0s);out['K0']=0
    g={};g['G1_NATBINF_eq_NATS']={s:bool(same(sv('NATBINF',s),sv('NATS',s))) for s in seeds}
    g['activity_NATMASK_differs_from_NATS_at_chunk_K0']={s:bool(sv('NATMASK',s)['log'][0]['ce']!=sv('NATS',s)['log'][0]['ce'] or sv('NATMASK',s)['log'][0]['update_norm']!=sv('NATS',s)['log'][0]['update_norm']) for s in seeds}
    hs=[e[1]==e[2] for a in ARMS for s in seeds for e in sv(a,s).get('read_hash',[])];g['read_hash_equal']=(bool(all(hs)) if hs else None)
    g['regression_gate_file_found']=bool((D.parent/'e57_orc_regress').exists());g['all_hard_gates_pass']=bool(all(g['G1_NATBINF_eq_NATS'].values()) and all(g['activity_NATMASK_differs_from_NATS_at_chunk_K0'].values()) and g['read_hash_equal']!=False)
    out['gates']=g
    M={a:np.array([read_stats(rd(a,s),'128')['margin'] for s in seeds]) for a in ARMS};A={a:np.array([read_stats(rd(a,s),'128')['acc'] for s in seeds]) for a in ARMS}
    SL={a:np.array([slope([read_stats(rd(a,s),e)['margin'] for e in EXPO]) for s in seeds]) for a in ARMS}
    d=M['NATMASK']-M['NATS'];p1,_=signflip(d);out['primary']=dict(contrast='NATMASK-NATS margin128 (nat)',mean=float(d.mean()),per_seed=[float(x) for x in d],p_one_sided_exact=p1,delta=DELTA,superior=bool(p1<0.05 and d.mean()>=DELTA))
    fam={'NATMASK-NATMASKC':M['NATMASK']-M['NATMASKC'],'DIRmaskNORMfull-NATS':M['DIRmaskNORMfull']-M['NATS'],'DIRfullNORMmask-NATS':M['DIRfullNORMmask']-M['NATS'],'slope(NATMASK)-slope(NATS)':SL['NATMASK']-SL['NATS']}
    ps={k:signflip(v)[1] for k,v in fam.items()};ad=holm(ps);out['family']={k:dict(mean=float(v.mean()),per_seed=[float(x) for x in v],p_two_sided_exact=ps[k],p_holm=ad[k]) for k,v in fam.items()}
    out['descriptive']={'acc128':{a:float(A[a].mean()) for a in ARMS},'margin128':{a:float(M[a].mean()) for a in ARMS},'interaction_DIRmask_minus_DIRfull_vs_NATS':float((M['DIRmaskNORMfull']-M['NATS']).mean()-(M['DIRfullNORMmask']-M['NATS']).mean()),'acc128_NATMASK_minus_NATS':float((A['NATMASK']-A['NATS']).mean())}
    inw={a:np.array([read_stats(rd(a,s),'128','in_g0')['margin'] for s in seeds]) for a in ('NATS','NATMASK')};di=inw['NATMASK']-inw['NATS']
    can={a:np.array([ (lambda r:r['reads'][str(max(int(k) for k in r['reads']))]['canary']['_tail_nll']-r['canary_w0']['_tail_nll'])(sv(a,s)) for s in seeds]) for a in ('NATS','NATMASK')}
    out['preservation']={'inwindow_margin_NATMASK_minus_NATS':float(di.mean()),'inwindow_pass_ge_minus_0.5nat':bool(di.mean()>=-0.5),'canary_tail_rise_NATMASK_minus_NATS':float((can['NATMASK']-can['NATS']).mean()),'note':'general-text gate = job 1283d (DCLM K0 = 0)'}
    cl={}
    for a in ARMS:
        cl[a]={}
        for nm,sel in (('before_chunk8',lambda e:e['k']<8),('from_chunk8',lambda e:e['k']>=8)):
            es=[e for s in seeds for e in sv(a,s)['log'] if sel(e)];cl[a][nm]=dict(gnorm_pre_clip_mean=float(np.mean([e['gnorm'] for e in es])),clip_rate=float(np.mean([e['coef']<1.0 for e in es])),median_update_norm=float(np.median([e['update_norm'] for e in es])))
    out['clip']=cl;out['clip_sensitivity_triggered']=bool(cl['NATMASK']['from_chunk8']['clip_rate']-cl['NATS']['from_chunk8']['clip_rate']>0.20)
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    def mk(td,tag,arm,s,gain):
        g=[int(x) for x in rng.integers(0,16,20)];kind=['fact']*16+['ctrl']*4;R={}
        for e in ('32','64','128'):
            L=rng.normal(size=(20,16));L[np.arange(20),g]+=gain*(1 if e=='128' else 0.5);R[e]={'out':L.tolist()}
        L2=rng.normal(size=(20,16));L2[np.arange(20),g]+=gain;R['128']['in_g0']=L2.tolist()
        json.dump(dict(meta=dict(gold_idx=g,kind=kind),read=R),open(td/f'E55_{tag}_{arm}_s{s}.read.json','w'))
        log=[dict(k=i,update_norm=1.0,ce=2.0+(0.1 if arm=='NATMASK' else 0),gnorm=1.0+0.1*i,coef=0.5 if i%2 else 1.0) for i in range(16)]
        json.dump(dict(K0=(None if arm=='NATS' else 0),log=log,final_hash='h',canary_w0={'_tail_nll':2.0},reads={'127':dict(canary={'_tail_nll':2.01})},read_hash=[(127,'a','a')]),open(td/f'E55_{tag}_{arm}_s{s}.saved.json','w'))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);seeds=list(range(71,79))
        for s in seeds:
            for a in ARMS:mk(td,'k0c16',a,s,{'NATMASK':3.0,'DIRmaskNORMfull':3.0,'NATMASKC':0.5}.get(a,0.0))
        r=analyze(td,seeds);assert r['gates']['all_hard_gates_pass'] and r['primary']['superior'] and abs(r['primary']['p_one_sided_exact']-1/256)<1e-12 and 'NATMASK-NATMASKC' in r['family'] and r['K0']==0
    assert abs(signflip([1.]*8)[0]-1/256)<1e-12
    print('e57_k0conf16_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--tag',default='k0c16');ap.add_argument('--json');ap.add_argument('--check-runner',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        if a.check_runner:
            import e57_oracle_run as OR;OR.selftest()
        r=analyze(a.dir,[int(x) for x in a.seeds.split(',')],a.tag);print(json.dumps(r,indent=1,default=float))
        if a.json:json.dump(r,open(a.json,'w'),indent=1,default=float)
