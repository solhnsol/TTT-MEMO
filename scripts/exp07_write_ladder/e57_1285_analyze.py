"""E57 job 1285b analysis (CPU; written before any K0 = 8 output was opened; pre-specification reports/E57-1285-prespec-v6.md). Inputs: K0 = 0 records of job 1283 (E55_k0c16_<arm>_s<seed>.read.json, arms NATS and NATMASK) and K0 = 8 records of job 1285b (E55_k0s16_NATMASK_K08_s<seed>.read.json), seeds 71..78. Margin of a fact = log-prob(gold) - max log-prob(wrong) (fact_margins of e57_k0conf16_analyze.py, the convention of the 1283 analysis; candidate score = summed span log-probabilities). G_K0 = mean over the 16 fact items of the out-of-window read at exposure 128 of margin(NATMASK at K0) - margin(NATS); H_K0 = mean over the 16 fact items of the in-window read in_g0 at exposure 128 of margin(NATS) - margin(NATMASK at K0) (positive = harm); NATS = the NATS of job 1283. D_G = G_0 - G_8, D_H = H_0 - H_8, N_K0 = n_out G_K0 - n_in H_K0 (n_out = n_in = 16), D_N = N_0 - N_8, all paired by seed. Rules (prespec sections 4.1 to 4.5): equivalence margin +-1.5 nat/fact by the 90% two-sided paired t interval (inclusive); states (a)..(d) and 'not evaluable' when the half-width exceeds 1.5; harm label 'cold start' / 'mask's own cost' / undetermined; exact sign-flip next to the t interval (one-sided minimum .0039, two-sided .0078).
  python3 e57_1285_analyze.py --k0-dir O1283 --k8-dir O1285 --seeds 71,...,78 --json out.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy import stats
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
MARGIN=1.5;H_THRESH=0.25
def fm(path,app,key):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']]);L=np.array(d['read'][app][key],float)
    n=len(L);fact=fact[:n];g=g[:n];L=L[fact];gg=g[fact];gold=L[np.arange(len(gg)),gg];w=L.copy();w[np.arange(len(gg)),gg]=-np.inf;return gold-w.max(1)
def tci(d,level):
    d=np.asarray(d,float);n=len(d);m=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(n));t=stats.t.ppf(0.5+level/2,n-1);return [m-t*se,m+t*se],m,se,n
def one_sided(d,level=0.95):
    d=np.asarray(d,float);n=len(d);m=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(n));t=stats.t.ppf(level,n-1);return dict(mean=m,lower=m-t*se,upper=m+t*se,se=se,n=n)
def signflip(d):
    d=np.asarray(d,float);n=len(d);obs=d.mean();idx=np.arange(2**n,dtype=np.uint32);S=((idx[:,None]>>np.arange(n,dtype=np.uint32))&1).astype(float)*2-1;m=(S*d[None]).mean(1);return float((m>=obs-1e-12).mean()),float((np.abs(m)>=abs(obs)-1e-12).mean())
def equivalence(d):
    """4.1 / 4.2: states for the paired differences d (D_G)"""
    ci90,m,se,n=tci(d,0.90);ci95,_,_,_=tci(d,0.95);half=(ci90[1]-ci90[0])/2;detected=bool(ci95[0]>0 or ci95[1]<0)
    if half>MARGIN:state='equivalence not evaluable (half-width of the 90% interval above 1.5); detection reported separately';equiv=None
    else:
        equiv=bool(ci90[0]>=-MARGIN and ci90[1]<=MARGIN)
        state=('(a) equivalent and not detected' if equiv and not detected else '(b) equivalent and detected (small shift)' if equiv and detected else '(c) detected and not equivalent' if detected else '(d) neither equivalent nor detected: undetermined')
    t95=stats.t.ppf(0.95,n-1);return dict(mean=m,ci90=ci90,ci95=ci95,half_width_90=half,equivalent=equiv,detected=detected,state=state,feasibility_sd_threshold=MARGIN*np.sqrt(n)/t95,sd=float(np.std(d,ddof=1)))
def harm_label(H8,DH):
    u=one_sided(H8);lo=one_sided(H8);dl=one_sided(DH)
    if u['upper']<=H_THRESH and dl['lower']>0:lab='cold start'
    elif lo['lower']>H_THRESH:lab="mask's own cost"
    else:lab='undetermined'
    return dict(label=lab,H8_upper_95_one_sided=u['upper'],H8_lower_95_one_sided=lo['lower'],DH_lower_95_one_sided=dl['lower'])
def analyze(d0,d8,seeds,fmargin=fm):
    rd0=lambda a,s:f'{d0}/E55_k0c16_{a}_s{s}.read.json';rd8=lambda s:f'{d8}/E55_k0s16_NATMASK_K08_s{s}.read.json'
    G0=[];G8=[];H0=[];H8=[]
    for s in seeds:
        nats_out=fmargin(rd0('NATS',s),'128','out');nats_in=fmargin(rd0('NATS',s),'128','in_g0');m0_out=fmargin(rd0('NATMASK',s),'128','out');m0_in=fmargin(rd0('NATMASK',s),'128','in_g0');m8_out=fmargin(rd8(s),'128','out');m8_in=fmargin(rd8(s),'128','in_g0')
        G0.append((m0_out-nats_out).mean());G8.append((m8_out-nats_out).mean());H0.append((nats_in-m0_in).mean());H8.append((nats_in-m8_in).mean());nout=len(nats_out);nin=len(nats_in)
    G0,G8,H0,H8=map(np.array,(G0,G8,H0,H8));DG=G0-G8;DH=H0-H8;N0=nout*G0-nin*H0;N8=nout*G8-nin*H8;DN=N0-N8
    out=dict(seeds=list(seeds),n=len(seeds),n_out=int(nout),n_in=int(nin),G_0=one_sided(G0),G_8=one_sided(G8),H_0=one_sided(H0),H_8=one_sided(H8),per_seed=dict(G_0=G0.tolist(),G_8=G8.tolist(),H_0=H0.tolist(),H_8=H8.tolist(),D_G=DG.tolist(),D_H=DH.tolist(),D_N=DN.tolist()))
    out['D_G']=equivalence(DG);out['D_G']['signflip_p_one_sided_two_sided']=signflip(DG)
    ci95,m,se,n=tci(DH,0.95);out['D_H']=dict(mean=m,ci95=ci95,signflip_p_one_sided_two_sided=signflip(DH));ci95,m,se,n=tci(DN,0.95);out['D_N']=dict(mean=m,ci95=ci95,note='derived linear combination, no test of its own')
    out['harm_label']=harm_label(H8,DH);out['N_0_mean']=float(N0.mean());out['N_8_mean']=float(N8.mean());out['status']='exploratory secondary evidence; the primary regime is K0 = 0 of job 1283';return out
def selftest():
    import tempfile,os
    rng=np.random.default_rng(0);seeds=list(range(71,79))
    def mk(td,name,seed,gold_gap_out,gold_gap_in,nfact=16):
        kind=['fact']*nfact+['ctrl']*4;g=[0]*20;out=rng.normal(0,.3,(20,16));inn=rng.normal(0,.3,(20,16));out[:,0]+=gold_gap_out;inn[:,0]+=gold_gap_in
        json.dump(dict(meta=dict(kind=kind,gold_idx=g),read={'128':{'out':out.tolist(),'in_g0':inn.tolist()}}),open(os.path.join(td,name),'w'))
    with tempfile.TemporaryDirectory() as td:
        for s in seeds:
            mk(td,f'E55_k0c16_NATS_s{s}.read.json',s,0.0,1.0);mk(td,f'E55_k0c16_NATMASK_s{s}.read.json',s,3.0,0.0);mk(td,f'E55_k0s16_NATMASK_K08_s{s}.read.json',s,3.1,0.9)
        o=analyze(td,td,seeds);assert o['n']==8 and o['n_out']==16 and o['n_in']==16
        assert abs(o['G_0']['mean']-3.0)<0.4 and abs(o['D_G']['mean'])<0.5 and o['D_G']['equivalent'] is True and o['D_G']['state'].startswith('(a)') or o['D_G']['state'].startswith('(b)') or o['D_G']['state'].startswith('(d)')
        assert o['harm_label']['label'] in ('cold start',"mask's own cost",'undetermined') and abs(o['D_N']['mean']-(o['N_0_mean']-o['N_8_mean']))<1e-9
    e=equivalence(np.array([0.1,-0.1,0.05,0.0,0.02,-0.05,0.0,0.03]));assert e['equivalent'] is True and e['state'].startswith('(a)') and abs(e['feasibility_sd_threshold']-2.24)<0.01
    e2=equivalence(np.array([3.0,-3.0,2.5,-2.5,3.2,-3.1,2.9,-3.3]));assert e2['equivalent'] is None and 'not evaluable' in e2['state']
    e3=equivalence(np.array([1.0,1.1,0.9,1.05,0.95,1.0,1.02,0.98]));assert e3['state'].startswith('(b)')
    e4=equivalence(np.array([2.0,2.1,1.9,2.05,1.95,2.0,2.02,1.98]));assert e4['state'].startswith('(c)')
    h=harm_label(np.array([0.05,0.0,0.1,0.02,-0.03,0.04,0.06,0.0]),np.array([1.0,1.1,1.2,0.9,1.0,1.1,1.0,1.05]));assert h['label']=='cold start'
    h2=harm_label(np.array([1.0,1.1,0.9,1.0,1.05,0.95,1.0,1.0]),np.array([0.1,-0.1,0.0,0.05,-0.05,0.0,0.02,0.0]));assert h2['label']=="mask's own cost"
    assert abs(signflip(np.ones(8))[0]-1/256)<1e-12 and abs(signflip(np.ones(8))[1]-1/128)<1e-12
    print('e57_1285_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--k0-dir');ap.add_argument('--k8-dir');ap.add_argument('--seeds');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        o=analyze(a.k0_dir,a.k8_dir,[int(x) for x in a.seeds.split(',')]);s=json.dumps(o,indent=1,default=float);print(s)
        if a.json:open(a.json,'w').write(s)
