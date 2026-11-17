"""E57 NATOTHERPREV / NATRANDPREV analysis (CPU; new file; EXPLORATORY, descriptive, NO pass / fail rule; job 1308z proposal). Cell v3_16_1, seeds 71..78, K0 = 0, exposure 128 primary. Same definitions as e57_allprev_analyze.py (margin = summed span log-prob gold minus best wrong, 16 candidates, mean over the 16 facts; top-1 share, chance 1/16; out-of-window at 16 / 32 / 64 / 128, in-window at 128; t 95% intervals over seeds; gap recovery (arm - NATS) / (ISO-STD - NATS) at exposure 128 out). Arms: NATS (1283), NATORACLE / NATPLACEBO (1302o), NATMASK256 / NATMASK1024, ISO-STD, NATALLPREV (1307) and the two controls of job 1308z (NATOTHERPREV: only other-fact earlier sentences hidden; NATRANDPREV: the ALLPREV number of random filler keys hidden, chunk by chunk). Paired by seed: OTHERPREV - NATS, RANDPREV - NATS, ALLPREV - NATS, OTHERPREV - ALLPREV, RANDPREV - ALLPREV, OTHERPREV - ORACLE, OTHERPREV - RANDPREV (margin out at all exposures, in-window at 128, top-1 at 128). Update statistics per arm (summary.mean_update_norm, cum_update_norm, clip_frac_coef_lt1, mean_masked_row_frac, mean_hidden_keys_row from the saved json).
  python3 e57_prevctrl_analyze.py --dir experiments/exp07_write_ladder --json J | --selftest"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_allprev_analyze as A
SRC=dict(A.SRC);SRC['NATOTHERPREV']='e57_prevctrl/E55_prevctrl_NATOTHERPREV_s{s}.read.json';SRC['NATRANDPREV']='e57_prevctrl/E55_prevctrl_NATRANDPREV_s{s}.read.json'
PAIRS=(('NATALLPREV','NATS'),('NATOTHERPREV','NATS'),('NATRANDPREV','NATS'),('NATOTHERPREV','NATALLPREV'),('NATRANDPREV','NATALLPREV'),('NATOTHERPREV','NATORACLE'),('NATOTHERPREV','NATRANDPREV'))
def analyze(D,seeds):
    A.SRC=SRC;out=A.analyze(D,seeds);D=pathlib.Path(D)
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
        fs=[D/p.format(s=s).replace('.read.json','.saved.json') for s in seeds]
        if all(f.exists() for f in fs):
            sm=[json.load(open(f))['summary'] for f in fs];out['update_stats'][a]={k:float(np.mean([x.get(k,0.0) for x in sm])) for k in ('mean_update_norm','cum_update_norm','clip_frac_coef_lt1','mean_masked_row_frac','mean_hidden_keys_row','mean_ce')}
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0);seeds=list(range(71,79))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td)
        for a,p in SRC.items():
            eff={'NATS':0,'NATMASK256':3,'NATMASK1024':2,'NATORACLE':1,'NATPLACEBO':0.05,'ISO-STD':4,'NATALLPREV':1.5,'NATOTHERPREV':1.4,'NATRANDPREV':0.1}[a]
            for s in seeds:
                g=rng.integers(0,16,20);rd={}
                for e in A.EXPO:
                    L=rng.normal(0,1,(20,16));o=L.copy()
                    for i in range(16):o[i,g[i]]+=eff
                    rd[e]={'out':o.tolist()}
                rd['128']['in_g0']=rd['128']['out'];f=td/p.format(s=s);f.parent.mkdir(parents=True,exist_ok=True);json.dump(dict(meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g.tolist()),read=rd,W0={'out':L.tolist(),'in_g0@128':L.tolist()}),open(f,'w'))
                json.dump({'summary':dict(mean_update_norm=0.7,cum_update_norm=90.0,clip_frac_coef_lt1=0.03,mean_masked_row_frac=0.36,mean_hidden_keys_row=2848.7,mean_ce=2.2)},open(str(f).replace('.read.json','.saved.json'),'w'))
        o=analyze(td,seeds);assert not o['missing'] and set(o['paired'])=={f'{a} - {b}' for a,b in PAIRS} and o['paired']['NATOTHERPREV - NATS']['out']['128']['mean']>0.8 and abs(o['paired']['NATRANDPREV - NATS']['out']['128']['mean'])<0.6 and 'NATOTHERPREV' in o['gap_recovery'] and set(o['update_stats'])==set(SRC),o['paired']
        (td/SRC['NATRANDPREV'].format(s=71)).unlink();o2=analyze(td,seeds);assert 'NATRANDPREV' in o2['missing'] and 'NATRANDPREV' not in o2['margin']
    print('e57_prevctrl_analyze selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='71,72,73,74,75,76,77,78');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    o=analyze(a.dir,[int(x) for x in a.seeds.split(',')]);pathlib.Path(a.json).write_text(json.dumps(o,indent=1))
    for k,v in o['paired'].items():print(k,{e:(round(x['mean'],2),[round(c,2) for c in x['ci95']]) for e,x in v['out'].items()})
if __name__=='__main__':main()
