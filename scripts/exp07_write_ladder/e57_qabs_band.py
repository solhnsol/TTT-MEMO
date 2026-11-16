"""E57 QA block-stream, band extension (job 1300; EXPLORATORY; K0 = 0; new file; e57_qabs_run.py (sha adefe1cf...) is imported and NOT edited). Same plan / stream ids / 128-paragraph assignment as job 1299 (e57_qabs/qabs_s<seed>_plan.json, qabs_s<seed>_ids.npy, seeds 101..104, 6 bundles), the arms NATMASK ALLQ256 and NATMASK ALLQ2048 are added; NATS and NATMASK ALLQ1024 are reused from job 1299 (reads in qabs_s<seed>_reads.json). The write and read stages are the functions of e57_qabs_run.py with exactly three textual substitutions made at import (arm list, output file names writes_band / reads_band; each substitution is counted and asserted), so the code path is that of job 1299. Analysis: the same six measures as e57_qabs_margin.py (first-token dA / dAB, margin S / N dA / dAB), contrast band - NATS per lag -1..5 and in0, t 95% interval over seeds, for bands 256 / 1024 / 2048 side by side.
  python3 e57_qabs_band.py --stage write|read|analyze --out O --seeds 101,102,103,104 --nb 6 [--json J] | --selftest"""
import sys,re,json,inspect,textwrap,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_qabs_run as Q
BAND_ARMS=(('NATMASK256','ALLQ256'),('NATMASK2048','ALLQ2048'));NAMES=tuple(a for a,_ in BAND_ARMS)
def patched(fn_name):
    src=textwrap.dedent(inspect.getsource(getattr(Q,fn_name)));n=0
    def sub(old,new,cnt):
        nonlocal src;assert src.count(old)==cnt,(fn_name,old,src.count(old),cnt);src=src.replace(old,new)
    if fn_name=='write_stage':
        sub("(('NATS','NATQ'),('NATMASK','ALLQ1024'))","BAND_ARMS",1)
        sub("masks={'band_ALLQ1024':np.asarray(1024),'all_ALLQ1024':np.asarray(1)}","masks={f'band_{m}':np.asarray(int(m[4:])) for _,m in BAND_ARMS}|{f'all_{m}':np.asarray(1) for _,m in BAND_ARMS}",1)
        sub("qabs_s{s}_writes.json","qabs_s{s}_writes_band.json",1)
    else:
        sub("('NATS','NATMASK')","NAMES",2)
        sub("qabs_s{s}_writes.json","qabs_s{s}_writes_band.json",1)
        sub("qabs_s{s}_reads.json","qabs_s{s}_reads_band.json",1)
    ns=dict(Q.__dict__);ns.update(BAND_ARMS=BAND_ARMS,NAMES=NAMES);exec(compile(src,f'<{fn_name} band patch>','exec'),ns);return ns[fn_name]
def analyze(a):
    import e57_pqa2_stream as S,e57_qabs_margin as M
    from scipy import stats
    out=pathlib.Path(a['out']);tk=S.tokenizer();dec=lambda t:tk.decode(t);NB=a['nb'];seeds=a['seeds'];NV=Q.NV;D={}
    SRC={'NATS':('reads','NATS'),'NATMASK1024':('reads','NATMASK'),'NATMASK256':('band','NATMASK256'),'NATMASK2048':('band','NATMASK2048')}
    for s in seeds:
        R={'reads':json.load(open(out/f'qabs_s{s}_reads.json')),'band':json.load(open(out/f'qabs_s{s}_reads_band.json'))};meta=R['reads']['meta']
        assert R['band']['meta']==meta and R['band']['W0']==R['reads']['W0'],'W0 / meta of the band reads differ from job 1299'
        w0={}
        for bi in range(NB):
            for k,v in M.item_scores(R['reads']['W0'][str(bi)],meta,bi,dec).items():w0[(bi,k)]=v
        for lab,(f,key) in SRC.items():
            r=R[f]
            for j in range(NB):
                st={}
                for bi in range(NB):
                    for k,v in M.item_scores(r[key][str(NV*j+NV-1)][str(bi)],meta,bi,dec).items():st[(bi,k)]=v
                for kk,d in M.deltas(st,w0).items():
                    lag=j-kk[0];lag=lag if lag>=0 else -1;D.setdefault((lab,lag,s),[]).append(d)
                w0j={(j,k):v for k,v in M.item_scores(r['IN'][str(j)]['W0'],meta,j,dec).items()};stj={(j,k):v for k,v in M.item_scores(r['IN'][str(j)][key],meta,j,dec).items()}
                for kk,d in M.deltas(stj,w0j).items():D.setdefault((lab,'in0',s),[]).append(d)
    res={'definition':M.__doc__.split('Statistic')[0].strip(),'bands':{}}
    lags=sorted({k[1] for k in D},key=lambda x:(isinstance(x,str),str(x)))
    for lab in ('NATMASK256','NATMASK1024','NATMASK2048'):
        res['bands'][lab]={}
        for lag in lags:
            row={}
            for mi,m in enumerate(M.MEAS):
                v=np.asarray([np.mean([x[mi] for x in D[(lab,lag,s)]]) for s in seeds]);u=np.asarray([np.mean([x[mi] for x in D[('NATS',lag,s)]]) for s in seeds]);d=v-u;n=len(d);se=float(d.std(ddof=1)/np.sqrt(n));tq=float(stats.t.ppf(0.975,n-1))
                row[m]=dict(NATS=float(u.mean()),arm=float(v.mean()),diff=float(d.mean()),ci95=[float(d.mean()-tq*se),float(d.mean()+tq*se)],per_seed=[float(x) for x in d],n_seeds=n)
            res['bands'][lab][str(lag)]=row
    pathlib.Path(a['json']).write_text(json.dumps(res,indent=1))
    for lab,t in res['bands'].items():
        print(lab+' (band minus NATS): lag | '+' | '.join(M.MEAS))
        for lag,row in t.items():print(' ',lag,'| '+' | '.join(f"{row[m]['diff']:+.2f} [{row[m]['ci95'][0]:+.2f},{row[m]['ci95'][1]:+.2f}]" for m in M.MEAS))
def selftest():
    for f in ('write_stage','read_stage'):
        fn=patched(f);assert callable(fn)
    import e57_qabs_margin as M;M.selftest();print('e57_qabs_band selftest ok (substitutions applied once / as counted)')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage');ap.add_argument('--out');ap.add_argument('--seeds',default='101,102,103,104');ap.add_argument('--nb',type=int,default=6);ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');x=ap.parse_args()
    if x.selftest:selftest();return
    a=dict(out=x.out,seeds=[int(s) for s in x.seeds.split(',')],nb=x.nb,nbundles=x.nb,json=x.json)
    if x.stage=='analyze':analyze(a)
    else:patched('write_stage' if x.stage=='write' else 'read_stage')(a)
if __name__=='__main__':main()
