"""E57 ALIGNMENT SET, seeds 71..78, cell v3_16_1, K0 = 0 (CPU; new file; exploratory, descriptive: NO pass / fail rule, no threshold). One table with the arms of the exposure curve on the SAME seeds and the SAME margin: NATS (job 1283, E55_k0c16_NATS), NATMASK256 (1283), NATMASK1024 (job 1293, e57_band_dist2/w1024), NATORACLE and NATPLACEBO (job 1302o, tag orc16: cache-path oracle / cache placebo = the update-path masks OracleUV / PlaceboOracleUV over the cached window K,V, the masks of the NLL conditions A / A_P of job 1280c), ISO-STD (job 1302i, tag q1a71, cumulative update norm matched to the NAT-norm source of the seed). Margin = summed span log-prob of the gold candidate minus the best wrong candidate (16 candidates of equal length), mean over the 16 fact items = seed value (NOT the logsumexp margin of 1280c / 1302). Reads: out-of-window (neutral tail) at exposures 16 / 32 / 64 / 128; in-window (in_g0) at exposure 128 where the file has it. Summary per arm and exposure: mean over seeds, two-sided 95% t interval (df = n - 1); paired by seed against NATS: NATORACLE - NATS, NATPLACEBO - NATS, NATORACLE - NATPLACEBO, NATMASK1024 - NATS, NATORACLE - NATMASK1024, ISO-STD - NATS (positive = better storage). Arms whose files are missing are listed under 'missing' and skipped.
  python3 e57_align71_analyze.py --dir O --seeds 71,...,78 --json J | --selftest"""
import sys,json,argparse,math,pathlib
import numpy as np
TQ={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228}
EXPO=('16','32','64','128')
SRC={'NATS':'e57_k0conf16/E55_k0c16_NATS_s{s}.read.json','NATMASK256':'e57_k0conf16/E55_k0c16_NATMASK_s{s}.read.json','NATMASK1024':'e57_band_dist2/w1024/E55_bd1024_NATMASK_s{s}.read.json','NATORACLE':'e57_orc16/E55_orc16_NATORACLE_s{s}.read.json','NATPLACEBO':'e57_orc16/E55_orc16_NATPLACEBO_s{s}.read.json','ISO-STD':'E55_q1a71_ISO-STD_s{s}.read.json'}
PAIRS=[('NATORACLE','NATS'),('NATPLACEBO','NATS'),('NATORACLE','NATPLACEBO'),('NATMASK1024','NATS'),('NATORACLE','NATMASK1024'),('ISO-STD','NATS')]
def margin(sc,meta):
    L=np.array(sc,float);g=np.array(meta['gold_idx'])[:len(L)];fact=np.array([k=='fact' for k in meta['kind']])[:len(L)];L=L[fact];gg=g[fact];gold=L[np.arange(len(gg)),gg];w=L.copy();w[np.arange(len(gg)),gg]=-np.inf;return float((gold-w.max(1)).mean())
def tci(v):
    v=np.asarray(v,float);n=len(v);m=float(v.mean());h=TQ[n-1]*float(v.std(ddof=1))/math.sqrt(n) if n>1 else float('nan');return dict(mean=m,ci95=[m-h,m+h],n=n,per_seed=[float(x) for x in v])
def collect(D,seeds):
    D=pathlib.Path(D);val={};miss={}
    for arm,pat in SRC.items():
        files=[D/pat.format(s=s) for s in seeds]
        if not all(f.exists() for f in files):miss[arm]=[str(f) for f in files if not f.exists()];continue
        recs=[json.load(open(f)) for f in files];val[arm]={'out':{e:np.array([margin(r['read'][e]['out'],r['meta']) for r in recs]) for e in EXPO},'in':{e:np.array([margin(r['read'][e]['in_g0'],r['meta']) for r in recs]) for e in EXPO if all('in_g0' in r['read'][e] for r in recs)},'W0':np.array([margin(r['W0']['out'],r['meta']) for r in recs])}
    return val,miss
def analyze(D,seeds):
    val,miss=collect(D,seeds);out={'seeds':list(seeds),'definition':__doc__.split('Summary per arm')[0].strip(),'missing':miss,'arms':{},'paired':{}}
    for arm,v in val.items():
        out['arms'][arm]={'out':{e:tci(x) for e,x in v['out'].items()},'in':{e:tci(x) for e,x in v['in'].items() if len(x)},'W0':tci(v['W0'])}
    for a,b in PAIRS:
        if a in val and b in val:out['paired'][f'{a} - {b}']={'out':{e:tci(val[a]['out'][e]-val[b]['out'][e]) for e in EXPO},'in':{e:tci(val[a]['in'][e]-val[b]['in'][e]) for e in val[a]['in'] if e in val[b]['in']}}
    return out
def selftest():
    import tempfile
    rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        seeds=list(range(71,79));meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=rng.integers(0,16,20).tolist())
        for arm,pat in SRC.items():
            for s in seeds:
                eff={'NATS':0,'NATMASK256':3,'NATMASK1024':2,'NATORACLE':1,'NATPLACEBO':0.05,'ISO-STD':4}[arm];rd={}
                for e in EXPO:
                    L=rng.normal(0,1,(20,16))
                    for i in range(16):L[i,meta['gold_idx'][i]]+=eff*int(e)/128
                    rd[e]={'out':L.tolist()}
                rd['128']['in_g0']=rng.normal(0,1,(20,16)).tolist()
                f=pathlib.Path(td)/pat.format(s=s);f.parent.mkdir(parents=True,exist_ok=True);json.dump(dict(meta=meta,read=rd,W0={'out':rng.normal(0,1,(20,16)).tolist()}),open(f,'w'))
        o=analyze(td,seeds);assert not o['missing'] and o['arms']['NATORACLE']['out']['128']['n']==8 and o['paired']['NATORACLE - NATS']['out']['128']['mean']>0.5 and 'in' in o['paired']['ISO-STD - NATS']
        (pathlib.Path(td)/SRC['ISO-STD'].format(s=71)).unlink();o2=analyze(td,seeds);assert 'ISO-STD' in o2['missing'] and 'ISO-STD' not in o2['arms']
    print('e57_align71_analyze selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    o=analyze(a.dir,[int(x) for x in a.seeds.split(',')]);s=json.dumps(o,indent=1);print(json.dumps({'missing':o['missing'],'arms':{k:{e:round(v['mean'],2) for e,v in x['out'].items()} for k,x in o['arms'].items()}}))
    if a.json:pathlib.Path(a.json).write_text(s)
if __name__=='__main__':main()
