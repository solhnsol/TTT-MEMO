"""E57 block-stream dilution experiment analysis (new file, CPU; pre-specified before any output exists). Inputs: read records of e57_blockstream_run.py (tag T, arms NATS and NATMASK, seeds), NB blocks of F facts, read points a_j = LC (j+1) (after block j). Per seed, read point j, block i, margin per fact = log p(gold) - max log p(wrong) among the 16 same-block candidates (1283 primary; --conv lse = logsumexp margin, secondary). Lag = j - i (blocks since the end of block i's exposures); lag 0 is reported twice: 'out' (neutral tail, weights only) and 'in' (in-window read, key in_g0, last 8 chunks); lag >= 1: 'out'. lag -1 = block not yet exposed (untrained control, same read point). Per lag: mean over the blocks with this lag and the facts, then per seed; summary = mean over seeds with a 95% t interval, for NATS, NATMASK and the paired difference NATMASK - NATS (positive = masking helps), plus the dilution slope of NATS (and of the difference) over lag 0..NB-1 (mean over seeds of the per-seed least-squares slope, 95% t interval). Exploratory; K0 = 0 only; no pooling with other cells.
  python3 e57_blockstream_analyze.py --dir D --tag bs --seeds 81,...,88 [--nb 4 --facts 16] [--conv max|lse] [--json out] | --selftest"""
import sys,json,argparse,pathlib,tempfile,numpy as np
from scipy import stats
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_1280_margin as MG
def tci(v,lvl=0.95):
    v=np.asarray(v,float);n=len(v);m=float(v.mean());
    if n<2:return dict(mean=m,ci95=[None,None],n=n)
    h=float(stats.t.ppf(0.5+lvl/2,n-1)*v.std(ddof=1)/np.sqrt(n));return dict(mean=m,ci95=[m-h,m+h],n=n)
def per_lag(path,nb,F,conv):
    d=json.load(open(path));meta=d['meta'];g=np.array(meta['gold_idx']);fact=np.array([k=='fact' for k in meta['kind']]);fid=np.array(meta['fact']);mg=MG.margin_secondary if conv=='max' else MG.margin_primary;apps=sorted(int(a) for a in d['read']);assert len(apps)==nb,(apps,nb);out={}
    for j,a in enumerate(apps):
        r=d['read'][str(a)]
        for key,idx in (('out',None),('in_g0',d['meta']['item_idx'][str(a)]['in_g0'] if 'in_g0' in r else None)):
            if key not in r:continue
            L=np.array(r[key],float);sel=np.arange(len(L)) if idx is None else np.array(idx)
            for i in range(nb):
                rows=[q for q,it in enumerate(sel) if fact[it] and fid[it]//F==i];m=mg(L[rows],g[sel[rows]])
                out.setdefault(('out' if key=='out' else 'in',j-i),[]).append(float(m.mean()))
    return {k:float(np.mean(v)) for k,v in out.items()}
def analyze(D,tag,seeds,nb=4,F=16,conv='max'):
    R={arm:[per_lag(f'{D}/E55_{tag}_{arm}_s{s}.read.json',nb,F,conv) for s in seeds] for arm in ('NATS','NATMASK')};keys=sorted(R['NATS'][0]);res={}
    for k in keys:
        a=np.array([r[k] for r in R['NATS']]);b=np.array([r[k] for r in R['NATMASK']]);res[f'{k[0]}_lag{k[1]}']=dict(NATS=tci(a),NATMASK=tci(b),diff=tci(b-a))
    lags=[l for l in range(nb) if ('out',l) in keys]
    if len(lags)>=2:
        sl=lambda arm:np.array([np.polyfit(lags,[r[('out',l)] for l in lags],1)[0] for r in R[arm]]);sa,sb=sl('NATS'),sl('NATMASK');res['slope_out_lag']=dict(NATS=tci(sa),NATMASK=tci(sb),diff=tci(sb-sa))
    return dict(convention=conv,seeds=list(seeds),nb=nb,F=F,result=res,status='exploratory; K0 = 0 only')
def selftest():
    rng=np.random.default_rng(0);nb,F=4,16;seeds=[1,2,3,4]
    with tempfile.TemporaryDirectory() as td:
        for arm,bonus in (('NATS',0.0),('NATMASK',1.0)):
            for s in seeds:
                nit=64+4;kind=['fact']*64+['ctrl']*4;fact=list(range(64))+[-1]*4;gold=[int(x) for x in rng.integers(0,16,nit)];read={};ii=list(range(nit))
                for j in range(nb):
                    a=16*(j+1);L=rng.normal(0,1,(nit,16))
                    for f in range(64):
                        blk=f//16
                        if blk<=j:L[f,gold[f]]+=4-1.0*(j-blk)+bonus*(j-blk)*0.5
                    read[str(a)]={'out':L.tolist(),'in_g0':L.tolist()}
                json.dump(dict(meta=dict(kind=kind,fact=fact,gold_idx=gold,item_idx={str(16*(j+1)):{'in_g0':ii} for j in range(nb)}),read=read),open(f'{td}/E55_t_{arm}_s{s}.read.json','w'))
        o=analyze(td,'t',seeds)['result'];assert o['out_lag0']['NATS']['mean']>2.0 and o['out_lag-3']['NATS']['mean']<0.0 and o['slope_out_lag']['NATS']['mean']<-0.3 and o['slope_out_lag']['diff']['mean']>0.1 and o['out_lag3']['diff']['mean']>0.8 and 'in_lag0' in o
    print('e57_blockstream_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--tag',default='bs');ap.add_argument('--seeds');ap.add_argument('--nb',type=int,default=4);ap.add_argument('--facts',type=int,default=16);ap.add_argument('--conv',default='max');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        o=analyze(a.dir,a.tag,[int(x) for x in a.seeds.split(',')],a.nb,a.facts,a.conv);s=json.dumps(o,indent=1,default=float);print(s)
        if a.json:open(a.json,'w').write(s)
