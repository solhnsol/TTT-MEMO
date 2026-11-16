"""E55 ISO 8-seed extension (job 1303; EXPLORATORY; new file, CPU): identity check and analysis. Seeds 61..68 of cell v3_16_1: ISO-STD / ISO-MOM / ISO-ADAM (e55_run.py q1a, tag q1a8, cumulative update norm matched to the NAT run of the same seed) against the NAT runs (tag v3a, seeds 61..65 stored, 66..68 run by the job). Margin = summed span log-prob of the gold candidate minus the best wrong candidate (16 candidates of equal length), mean over the 16 fact items = seed value; out-of-window neutral-tail read at exposures 16 / 32 / 64 / 128 (the definition of the existing E55 rows, job 185 / 192). Summary = mean over seeds with a two-sided 95% t interval; ISO - NAT paired by seed.
IDENTITY CHECK (--check): the reruns of seeds 61, 62, 63 (tag q1a8) must be bit identical to the stored runs of job 192 (tag q1a): every read value (read, W0) and every logged step value (grad_norm, clip_coef, ce, raw_update_norm, update_norm, cum_update_norm) EXACTLY equal; any difference exits with 13 (the extension is then not tied to the earlier rows and is not interpreted).
  python3 e55_iso8_check.py --check --dir D --seeds 61 [--arms ISO-STD,ISO-MOM,ISO-ADAM] | --analyze --dir D --seeds 61,...,68 --json J | --selftest"""
import sys,json,argparse,pathlib,math
import numpy as np
TQ={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228}
ARMS=('ISO-STD','ISO-MOM','ISO-ADAM');EXPO=('16','32','64','128');LOGK=('grad_norm','clip_coef','ce','raw_update_norm','update_norm','cum_update_norm')
def margin(sc,meta):
    g=np.array(meta['gold_idx']);fact=np.array([k=='fact' for k in meta['kind']]);L=np.array(sc,float)[fact];gg=g[fact];gold=L[np.arange(len(gg)),gg];w=L.copy();w[np.arange(len(gg)),gg]=-np.inf;return float((gold-w.max(1)).mean())
def tci(v):
    v=np.asarray(v,float);n=len(v);m=float(v.mean());h=TQ[n-1]*float(v.std(ddof=1))/math.sqrt(n) if n>1 else float('nan');return dict(mean=m,ci95=[m-h,m+h],n=n,per_seed=[float(x) for x in v])
def load(D,tag,s,kind):return json.load(open(pathlib.Path(D)/f'E55_{tag}_s{s}.{kind}.json'))
def check(D,seeds,arms):
    bad=[]
    for s in seeds:
        for ar in arms:
            a=load(D,f'q1a8_{ar}',s,'read');b=load(D,f'q1a_{ar}',s,'read')
            if a['read']!=b['read']:bad.append((s,ar,'read values'))
            if a['W0']!=b['W0']:bad.append((s,ar,'W0 values'))
            la=load(D,f'q1a8_{ar}',s,'saved')['log'];lb=load(D,f'q1a_{ar}',s,'saved')['log']
            if len(la)!=len(lb) or any(x[k]!=y[k] for x,y in zip(la,lb) for k in LOGK):bad.append((s,ar,'step log'))
    return dict(ok=not bad,checked=[(s,a) for s in seeds for a in arms],mismatches=bad)
def analyze(D,seeds):
    out={'seeds':list(seeds),'definition':__doc__.split('IDENTITY')[0].strip(),'rows':{}}
    def vals(tag,key,e):return np.array([margin(load(D,tag,s,'read')['read'][e]['out'],load(D,tag,s,'read')['meta']) for s in seeds])
    w0=[]
    for s in seeds:
        d=load(D,'v3a_NAT',s,'read');w0.append(margin(d['W0']['out'],d['meta']))
    out['W0']=tci(w0);nat={e:vals('v3a_NAT','out',e) for e in EXPO};out['rows']['NAT']={e:tci(v) for e,v in nat.items()}
    for ar in ARMS:
        iso={e:vals(f'q1a8_{ar}','out',e) for e in EXPO};out['rows'][ar]={e:tci(v) for e,v in iso.items()};out['rows'][ar+' - NAT (paired)']={e:tci(iso[e]-nat[e]) for e in EXPO}
    return out
def selftest():
    import tempfile
    rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        seeds=[61,62,63,64];meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=rng.integers(0,16,20).tolist())
        def rec(eff):
            rd={}
            for e in EXPO:
                L=rng.normal(0,1,(20,16));
                for i in range(16):L[i,meta['gold_idx'][i]]+=eff*int(e)/128
                rd[e]={'out':L.tolist()}
            return dict(meta=meta,read=rd,W0={'out':rng.normal(0,1,(20,16)).tolist()})
        for s in seeds:
            nat=rec(0.0);json.dump(nat,open(f'{td}/E55_v3a_NAT_s{s}.read.json','w'))
            for ar in ARMS:
                r=rec(4.0);log=[dict(grad_norm=1.0+i,clip_coef=1.0,ce=2.0,raw_update_norm=.5,update_norm=.5,cum_update_norm=.5*(i+1)) for i in range(5)]
                for tag in (f'q1a_{ar}',f'q1a8_{ar}'):
                    json.dump(r,open(f'{td}/E55_{tag}_s{s}.read.json','w'));json.dump(dict(log=log),open(f'{td}/E55_{tag}_s{s}.saved.json','w'))
        c=check(td,[61,62,63],ARMS);assert c['ok'],c
        o=analyze(td,seeds);assert o['rows']['ISO-STD']['128']['mean']>o['rows']['NAT']['128']['mean']+2 and len(o['rows']['ISO-ADAM - NAT (paired)']['128']['per_seed'])==4
        j=json.load(open(f'{td}/E55_q1a8_ISO-MOM_s62.saved.json'));j['log'][2]['ce']=2.0000001;json.dump(j,open(f'{td}/E55_q1a8_ISO-MOM_s62.saved.json','w'));assert not check(td,[61,62,63],ARMS)['ok']
    print('e55_iso8_check selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--analyze',action='store_true');ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--arms',default=','.join(ARMS));ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    seeds=[int(x) for x in a.seeds.split(',')]
    if a.check:
        c=check(a.dir,seeds,a.arms.split(','));print(json.dumps(c,indent=1))
        if not c['ok']:print('IDENTITY CHECK FAILED: the rerun is not bit identical to the stored run');sys.exit(13)
    if a.analyze:
        o=analyze(a.dir,seeds);s=json.dumps(o,indent=1);print(s[:1500])
        if a.json:pathlib.Path(a.json).write_text(s)
if __name__=='__main__':main()
