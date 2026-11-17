"""E57 NATALLPREV analysis (CPU; new file; EXPLORATORY, descriptive, NO pass / fail rule; job 1307 proposal). Cell v3_16_1, seeds 71..78, K0 = 0. Reads the stored read.json of NATS (1283), NATMASK256 (1283), NATMASK1024 (1293), NATORACLE / NATPLACEBO (1302o), ISO-STD (1302i) and NATALLPREV (job 1307, tag allprev). MARGIN = summed span log-prob of the gold candidate minus the best wrong candidate (16 candidates of equal length), mean over the 16 fact items = seed value (NOT the logsumexp margin of 1280c / 1302). TOP1 = share of the 16 fact items whose gold candidate has the highest summed score (chance 1/16). Reads: out-of-window (neutral tail) at exposures 16 / 32 / 64 / 128, in-window (in_g0) at exposure 128. Per arm and exposure: mean over seeds with the two-sided 95% t interval (df = n - 1); paired by seed against NATS and against NATORACLE; GAP RECOVERY at exposure 128 out = (arm - NATS) / (ISO-STD - NATS) per seed (mean and t interval over seeds, plus the ratio of the means).
  python3 e57_allprev_analyze.py --dir experiments/exp07_write_ladder --json J | --selftest"""
import sys,json,math,pathlib,argparse
import numpy as np
TQ={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131}
SRC={'NATS':'e57_k0conf16/E55_k0c16_NATS_s{s}.read.json','NATMASK256':'e57_k0conf16/E55_k0c16_NATMASK_s{s}.read.json','NATMASK1024':'e57_band_dist2/w1024/E55_bd1024_NATMASK_s{s}.read.json','NATORACLE':'e57_orc16/E55_orc16_NATORACLE_s{s}.read.json','NATPLACEBO':'e57_orc16/E55_orc16_NATPLACEBO_s{s}.read.json','ISO-STD':'E55_q1a71_ISO-STD_s{s}.read.json','NATALLPREV':'e57_allprev/E55_allprev_NATALLPREV_s{s}.read.json'}
EXPO=('16','32','64','128')
def scores(r,e,k):
    m=r['meta'];x=np.asarray(r['read'][e][k] if e!='W0' else r['W0'][k],float);g=np.asarray(m['gold_idx'])[:len(x)];f=np.asarray([t=='fact' for t in m['kind']])[:len(x)];return x[f],g[f]
def margin(r,e,k):
    x,g=scores(r,e,k);gold=x[np.arange(len(g)),g];y=x.copy();y[np.arange(len(g)),g]=-np.inf;return float((gold-y.max(1)).mean())
def top1(r,e,k):
    x,g=scores(r,e,k);return float((x.argmax(1)==g).mean())
def tci(v):
    v=np.asarray(v,float);n=len(v);m=float(v.mean());h=TQ[n-1]*float(v.std(ddof=1))/math.sqrt(n) if n>1 else float('nan');return dict(mean=m,ci95=[m-h,m+h],n=n,per_seed=[float(x) for x in v])
def analyze(D,seeds):
    D=pathlib.Path(D);R={};miss={}
    for a,p in SRC.items():
        fs=[D/p.format(s=s) for s in seeds]
        if not all(f.exists() for f in fs):miss[a]=[str(f) for f in fs if not f.exists()];continue
        R[a]=[json.load(open(f)) for f in fs]
    out=dict(seeds=list(seeds),missing=miss,margin={},top1={},paired={},gap_recovery={})
    for fn,key in ((margin,'margin'),(top1,'top1')):
        for a,rs in R.items():
            o={'out':{e:tci([fn(r,e,'out') for r in rs]) for e in EXPO}}
            if all('in_g0' in r['read']['128'] for r in rs):o['in']={'128':tci([fn(r,'128','in_g0') for r in rs])}
            o['W0']={'out':tci([fn(r,'W0','out') for r in rs])['mean'],'in128':tci([fn(r,'W0','in_g0@128') for r in rs])['mean']} if all('in_g0@128' in r['W0'] for r in rs) else None
            out[key][a]=o
    def pair(a,b):
        res={'out':{e:tci([margin(x,e,'out')-margin(y,e,'out') for x,y in zip(R[a],R[b])]) for e in EXPO}}
        if all('in_g0' in r['read']['128'] for r in R[a]+R[b]):res['in']={'128':tci([margin(x,'128','in_g0')-margin(y,'128','in_g0') for x,y in zip(R[a],R[b])])}
        t1={'out':tci([top1(x,'128','out')-top1(y,'128','out') for x,y in zip(R[a],R[b])])}
        if 'in' in res:t1['in']=tci([top1(x,'128','in_g0')-top1(y,'128','in_g0') for x,y in zip(R[a],R[b])])
        res['top1_128']=t1;return res
    for a,b in (('NATALLPREV','NATS'),('NATALLPREV','NATORACLE'),('NATALLPREV','NATPLACEBO'),('NATORACLE','NATS')):
        if a in R and b in R:out['paired'][f'{a} - {b}']=pair(a,b)
    if 'ISO-STD' in R and 'NATS' in R:
        nats=np.array([margin(r,'128','out') for r in R['NATS']]);iso=np.array([margin(r,'128','out') for r in R['ISO-STD']]);den=iso-nats
        for a in R:
            if a in ('NATS','ISO-STD'):continue
            m=np.array([margin(r,'128','out') for r in R[a]]);rec=(m-nats)/den;t=tci(rec);t['ratio_of_means']=float((m-nats).mean()/den.mean());out['gap_recovery'][a]=t
        out['gap_denominator_ISO_minus_NATS_per_seed']=[float(x) for x in den]
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0);seeds=list(range(71,79))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td)
        for a,p in SRC.items():
            eff={'NATS':0,'NATMASK256':3,'NATMASK1024':2,'NATORACLE':1,'NATPLACEBO':0.05,'ISO-STD':4,'NATALLPREV':1.5}[a]
            for s in seeds:
                g=rng.integers(0,16,20);rd={}
                for e in EXPO:
                    L=rng.normal(0,1,(20,16));o=L.copy()
                    for i in range(16):o[i,g[i]]+=eff
                    rd[e]={'out':o.tolist()}
                rd['128']['in_g0']=rd['128']['out']
                f=td/p.format(s=s);f.parent.mkdir(parents=True,exist_ok=True);json.dump(dict(meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g.tolist()),read=rd,W0={'out':L.tolist(),'in_g0@128':L.tolist()}),open(f,'w'))
        o=analyze(td,seeds);assert not o['missing'] and o['paired']['NATALLPREV - NATS']['out']['128']['mean']>0.8 and 0<o['gap_recovery']['NATALLPREV']['mean']<0.7 and 'in' in o['paired']['NATALLPREV - NATS'],o['gap_recovery']
        (td/SRC['NATALLPREV'].format(s=71)).unlink();o2=analyze(td,seeds);assert 'NATALLPREV' in o2['missing'] and 'NATALLPREV' not in o2['margin']
    print('e57_allprev_analyze selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='71,72,73,74,75,76,77,78');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    o=analyze(a.dir,[int(x) for x in a.seeds.split(',')]);pathlib.Path(a.json).write_text(json.dumps(o,indent=1))
    for k,v in o['paired'].items():print(k,{e:(round(x['mean'],2),[round(c,2) for c in x['ci95']]) for e,x in v['out'].items()})
if __name__=='__main__':main()
