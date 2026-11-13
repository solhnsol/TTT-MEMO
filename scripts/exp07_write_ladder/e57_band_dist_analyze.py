"""E57 band-distance experiment (K0 = 0) analysis (CPU; written before any new-width output exists; pre-specification reports/E57-distB-prespec.md). Inputs: job 1283 records (tag k0c16: NATS and NATMASK = band 256, K0 = 0) and the new widths (dirs <D>/w<W>, tag bd<W>, arm NATMASK, band W), seeds 71..78. Primary margin = log-prob(gold) - max log-prob(wrong) (the 1283 convention, per fact, out-of-window read at exposure 128 and in-window read in_g0 at exposure 128); secondary = logsumexp margin (e57_1280_margin), descriptive; 1280 values are not comparable with these. G_W = mean over the 16 facts of margin(NATMASK_W) - margin(NATS) (out-of-window); H_W = margin(NATS) - margin(NATMASK_W) (in-window, positive = harm); N_W = 16 (G_W - H_W) nat per sequence; R_W = mean(G_W) / mean(H_W) with a seed bootstrap 95% interval (undefined and not reported if the bootstrap interval of mean(H_W) includes 0); W* = the width with the largest mean N_W with paired differences to the neighbouring widths (95% t interval, distinguishable only if it excludes 0); monotonicity = mean over seeds of the Spearman correlation of G_W and of H_W with log2 W (width order) with a 95% t interval. Exploratory; no multiplicity correction; K0 = 0 only.
  python3 e57_band_dist_analyze.py --k0-dir O1283 --band-dir O1290 --widths 64,128,256,512 --seeds 71,...,78 --json out.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy import stats
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_1280_margin as MG
def scores(path,key):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']]);L=np.array(d['read']['128'][key],float);n=len(L);return L[fact[:n]],g[:n][fact[:n]]
def margins(path,key,conv):
    L,g=scores(path,key);return MG.margin_secondary(L,g) if conv=='max' else MG.margin_primary(L,g)      # 'max' = log p(gold) - max log p(wrong) (1283 convention); 'lse' = logsumexp
def tci(d,lvl=0.95):
    d=np.asarray(d,float);n=len(d);m=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(n));t=stats.t.ppf(.5+lvl/2,n-1);return dict(mean=m,sd=float(d.std(ddof=1)),ci95=[m-t*se,m+t*se],n=n)
def boot_ratio(G,H,B=5000,seed=20261113):
    rng=np.random.default_rng(seed);n=len(G);r=[];h=[]
    for _ in range(B):
        i=rng.integers(0,n,n);hm=H[i].mean();h.append(hm);r.append(G[i].mean()/hm if hm!=0 else np.nan)
    hlo,hhi=np.quantile(h,[.025,.975]);ok=bool(hlo>0 or hhi<0);rr=np.array(r);return dict(point=float(G.mean()/H.mean()),ci95=[float(np.nanquantile(rr,.025)),float(np.nanquantile(rr,.975))] if ok else None,defined=ok)
def analyze(d0,db,widths,seeds,conv='max'):
    path=lambda w,a,s:(f'{d0}/E55_k0c16_{a}_s{s}.read.json' if (a=='NATS' or w==256) else f'{db}/w{w}/E55_bd{w}_NATMASK_s{s}.read.json')
    G={w:[] for w in widths};H={w:[] for w in widths}
    for s in seeds:
        nO=margins(path(0,'NATS',s),'out',conv);nI=margins(path(0,'NATS',s),'in_g0',conv)
        for w in widths:
            p=path(w,'NATMASK',s);G[w].append((margins(p,'out',conv)-nO).mean());H[w].append((nI-margins(p,'in_g0',conv)).mean())
    G={w:np.array(v) for w,v in G.items()};H={w:np.array(v) for w,v in H.items()};N={w:16*(G[w]-H[w]) for w in widths}
    out=dict(convention=conv,seeds=list(seeds),widths=list(widths),G={w:tci(G[w]) for w in widths},H={w:tci(H[w]) for w in widths},N={w:tci(N[w]) for w in widths},R={w:boot_ratio(G[w],H[w]) for w in widths})
    wstar=max(widths,key=lambda w:N[w].mean());out['W_star']=wstar;out['neighbour_diffs']={}
    ws=sorted(widths)
    for a,b in zip(ws[:-1],ws[1:]):d=N[b]-N[a];ci=tci(d);out['neighbour_diffs'][f'{a}->{b}']=dict(**ci,distinguishable=bool(ci['ci95'][0]>0 or ci['ci95'][1]<0))
    lw=np.log2(ws)
    for nm,X in (('G',G),('H',H)):
        rho=np.array([stats.spearmanr(lw,[X[w][i] for w in ws])[0] for i in range(len(seeds))]);rho=np.nan_to_num(rho);out[f'monotonicity_{nm}']=dict(per_seed=rho.tolist(),**tci(rho))
    out['per_seed']=dict(G={w:G[w].tolist() for w in widths},H={w:H[w].tolist() for w in widths});out['status']='exploratory secondary evidence; K0 = 0 only';return out
def selftest():
    import tempfile,os
    rng=np.random.default_rng(0);seeds=list(range(71,79));widths=[64,128,256,512]
    with tempfile.TemporaryDirectory() as td:
        def mk(path,gout,gin):
            o=rng.normal(0,.3,(20,16));i=rng.normal(0,.3,(20,16));o[:,0]+=gout;i[:,0]+=gin
            json.dump(dict(meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=[0]*20),read={'128':{'out':o.tolist(),'in_g0':i.tolist()}}),open(path,'w'))
        eff={64:(1.0,0.2),128:(2.0,0.6),256:(3.0,1.2),512:(3.1,2.0)}
        for s in seeds:
            mk(f'{td}/E55_k0c16_NATS_s{s}.read.json',0,1.0);mk(f'{td}/E55_k0c16_NATMASK_s{s}.read.json',3.0,1.0-1.2)
            for w in (64,128,512):
                os.makedirs(f'{td}/w{w}',exist_ok=True);g,h=eff[w];mk(f'{td}/w{w}/E55_bd{w}_NATMASK_s{s}.read.json',g,1.0-h)
        o=analyze(td,td,widths,seeds);assert o['W_star'] in (256,512) and abs(o['G'][256]['mean']-3.0)<0.4 and abs(o['H'][256]['mean']-1.2)<0.4 and o['monotonicity_G']['mean']>0.8 and o['monotonicity_H']['mean']>0.8
        assert o['R'][256]['defined'] is True and o['R'][64]['point']>0 and all(k in o['neighbour_diffs'] for k in ('64->128','128->256','256->512'))
        o2=analyze(td,td,widths,seeds,'lse');assert o2['convention']=='lse'
    r=boot_ratio(np.array([1.0,1.0,1.0,1.0]),np.array([0.01,-0.01,0.02,-0.02]));assert r['defined'] is False and r['ci95'] is None
    print('e57_band_dist_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--k0-dir');ap.add_argument('--band-dir');ap.add_argument('--widths');ap.add_argument('--seeds');ap.add_argument('--conv',default='max');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        o=analyze(a.k0_dir,a.band_dir,[int(x) for x in a.widths.split(',')],[int(x) for x in a.seeds.split(',')],a.conv);s=json.dumps(o,indent=1,default=float);print(s)
        if a.json:open(a.json,'w').write(s)
