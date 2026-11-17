"""E57 band-1024 + loss-weight arms on the clipped-SGD engine, analysis (CPU; new file; EXPLORATORY, descriptive, NO pass / fail rule; job 1308 proposal v2). Cell v3_16_1, seeds 71..78, K0 = 0, SAME engine / reads / mask as the band-1024 arm NATMASK1024 of jobs 1283 / 1293 (reused from e57_band_dist2/w1024 when the bit-identity gate of job 1308 passed, else the band-1024 arm rerun by the job in e57_b1024s/, tag b1024s). Arms: NATB1024G15W8 / NATB1024G30W8 (loss weight 8 on the top-15% / top-30% rows by g1024), NATB1024R15W8 / NATB1024R30W8 (weight 8 on the same number of random rows), and the AUXILIARY norm-matched NATB1024G30W8NM / NATB1024R30W8NM (update rescaled to the update norm of the band-1024 run of the seed in every chunk). Definitions as e57_allprev_analyze.py (margin = summed span log-prob gold minus best wrong, 16 candidates, mean over the 16 facts; top-1 share; out-of-window at 16 / 32 / 64 / 128, in-window at 128; t 95% intervals over seeds). Paired by seed: G - M, R - M, G - R (M = NATMASK1024) and the same with the NM arms, with the top-1 share at 128. Update statistics per arm (mean over seeds of summary.mean_update_norm, cum_update_norm, clip_frac_coef_lt1 from the saved json): the weights enlarge the gradient, the clip is active more often.
  python3 e57_b1024s_analyze.py --dir experiments/exp07_write_ladder --json J [--band-src gate|stored] | --selftest"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_allprev_analyze as A
def sources(band_stored=True):
    s={'NATS':A.SRC['NATS'],'NATMASK1024':(A.SRC['NATMASK1024'] if band_stored else 'e57_b1024s/E55_b1024s_NATMASK_s{s}.read.json'),'NATB1024G15W8':'e57_b1024s/E55_b1024s_NATB1024G15W8_s{s}.read.json','NATB1024R15W8':'e57_b1024s/E55_b1024s_NATB1024R15W8_s{s}.read.json','NATB1024G30W8':'e57_b1024s/E55_b1024s_NATB1024G30W8_s{s}.read.json','NATB1024R30W8':'e57_b1024s/E55_b1024s_NATB1024R30W8_s{s}.read.json','NATB1024G30W8NM':'e57_b1024s/E55_b1024s_NATB1024G30W8NM_s{s}.read.json','NATB1024R30W8NM':'e57_b1024s/E55_b1024s_NATB1024R30W8NM_s{s}.read.json','ISO-STD':A.SRC['ISO-STD']}
    return s
PAIRS=(('NATB1024G15W8','NATMASK1024'),('NATB1024R15W8','NATMASK1024'),('NATB1024G15W8','NATB1024R15W8'),('NATB1024G30W8','NATMASK1024'),('NATB1024R30W8','NATMASK1024'),('NATB1024G30W8','NATB1024R30W8'),('NATB1024G30W8NM','NATMASK1024'),('NATB1024R30W8NM','NATMASK1024'),('NATB1024G30W8NM','NATB1024R30W8NM'))
def saved_path(read_path):return read_path.replace('.read.json','.saved.json')
def analyze(D,seeds,band_stored=True):
    SRC=sources(band_stored);A.SRC=SRC;out=A.analyze(D,seeds);D=pathlib.Path(D)
    R={a:[json.load(open(D/p.format(s=s))) for s in seeds] for a,p in SRC.items() if all((D/p.format(s=s)).exists() for s in seeds)};out['paired']={}
    for a,b in PAIRS:
        if a in R and b in R:
            res={'out':{e:A.tci([A.margin(x,e,'out')-A.margin(y,e,'out') for x,y in zip(R[a],R[b])]) for e in A.EXPO}}
            if all('in_g0' in r['read']['128'] for r in R[a]+R[b]):res['in']={'128':A.tci([A.margin(x,'128','in_g0')-A.margin(y,'128','in_g0') for x,y in zip(R[a],R[b])])}
            t1={'out':A.tci([A.top1(x,'128','out')-A.top1(y,'128','out') for x,y in zip(R[a],R[b])])}
            if 'in' in res:t1['in']=A.tci([A.top1(x,'128','in_g0')-A.top1(y,'128','in_g0') for x,y in zip(R[a],R[b])])
            res['top1_128']=t1;out['paired'][f'{a} - {b}']=res
    out['update_stats']={}
    for a,p in SRC.items():
        fs=[D/saved_path(p.format(s=s)) for s in seeds]
        if all(f.exists() for f in fs):
            sm=[json.load(open(f))['summary'] for f in fs];out['update_stats'][a]={k:float(np.mean([x[k] for x in sm])) for k in ('mean_update_norm','cum_update_norm','clip_frac_coef_lt1','mean_gnorm','mean_ce')}
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0);seeds=list(range(71,79))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);SRC=sources(True)
        for a,p in SRC.items():
            eff={'NATS':0,'NATMASK1024':2,'NATB1024G15W8':3.1,'NATB1024R15W8':2.05,'NATB1024G30W8':3,'NATB1024R30W8':2.1,'NATB1024G30W8NM':2.8,'NATB1024R30W8NM':2.0,'ISO-STD':4}[a]
            for s in seeds:
                g=rng.integers(0,16,20);rd={}
                for e in A.EXPO:
                    L=rng.normal(0,1,(20,16));o=L.copy()
                    for i in range(16):o[i,g[i]]+=eff
                    rd[e]={'out':o.tolist()}
                rd['128']['in_g0']=rd['128']['out'];f=td/p.format(s=s);f.parent.mkdir(parents=True,exist_ok=True);json.dump(dict(meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g.tolist()),read=rd,W0={'out':L.tolist(),'in_g0@128':L.tolist()}),open(f,'w'))
                json.dump({'summary':dict(mean_update_norm=0.7+eff/10,cum_update_norm=90.0,clip_frac_coef_lt1=0.03,mean_gnorm=0.7,mean_ce=2.2)},open(td/saved_path(p.format(s=s)),'w'))
        o=analyze(td,seeds);assert not o['missing'] and set(o['paired'])=={f'{a} - {b}' for a,b in PAIRS} and o['paired']['NATB1024G30W8 - NATMASK1024']['out']['128']['mean']>0.5 and 'NATB1024G30W8' in o['gap_recovery'] and set(o['update_stats'])==set(SRC),o['paired']
        o2=analyze(td,seeds,band_stored=False);assert 'NATMASK1024' in o2['missing']                     # the rerun band arm is read from e57_b1024s/ (absent here)
    print('e57_b1024s_analyze selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='71,72,73,74,75,76,77,78');ap.add_argument('--json');ap.add_argument('--band-src',default='stored',choices=['stored','gate']);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    o=analyze(a.dir,[int(x) for x in a.seeds.split(',')],a.band_src=='stored');pathlib.Path(a.json).write_text(json.dumps(o,indent=1))
    for k,v in o['paired'].items():print(k,{e:(round(x['mean'],2),[round(c,2) for c in x['ci95']]) for e,x in v['out'].items()})
if __name__=='__main__':main()
