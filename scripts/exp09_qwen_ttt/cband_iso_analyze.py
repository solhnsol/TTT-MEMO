"""Qwen ISO analysis (CPU; new file; EXPLORATORY, descriptive, NO pass / fail rule; job 1309 proposal). Does Qwen3-4B-Base (LoRA r16 last 9 layers, SGD lr 3e-2, clip 1) learn the binding when it sees only the fact sentences without context (ISO-STD of E55 ported, cell_iso_ladder.py; update norm per segment matched to the NATS run of the same seed)? Reads the stored files of job 1309 (cband_iso/cband_ISO_s<seed>.json, cband_W0_s<seed>.json) and of job 1306 (cband_main/cband_NATS_s<seed>.json, cband_NATMASK1024_s<seed>.json, cband_NATMASK256_s<seed>.json), seeds 161..168, same world / contexts / candidates (summed span log-prob, 16 candidates of equal length, 16 fact items per seed). Per arm (W0, NATS, NATMASK1024, NATMASK256, ISO), per exposure 16 / 32 / 64 / 128, out-of-window and in-window reads: margin (gold minus best wrong), top-1 share (chance 1/16 = 0.0625), mean gold rank, MRR, gold span log-prob; mean over seeds with the two-sided 95% t interval (df = n - 1) and, paired by seed, the difference to NATS and to W0; ISO top-1 against chance (t interval of top-1 minus 0.0625). Update statistics (summed update norm per segment, clip activity) of ISO against NATS from the logs.
  python3 cband_iso_analyze.py --iso DIR --main DIR --json J [--seeds 161,...,168] | --selftest"""
import sys,json,math,pathlib,argparse
import numpy as np
TQ={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131}
EXPO=('16','32','64','128');ARMS=('NATS','NATMASK1024','NATMASK256','ISO');CHANCE=1/16
def stats(x,g):
    gold=x[np.arange(len(g)),g];y=x.copy();y[np.arange(len(g)),g]=-np.inf;rank=1+(x>gold[:,None]).sum(1);return dict(margin=gold-y.max(1),top1=(rank==1).astype(float),rank=rank.astype(float),mrr=1.0/rank,gold=gold)
def get(r,e,k):
    m=r['meta'];x=np.asarray(r['reads'][e][k],float);g=np.asarray(m['gold_idx'])[:len(x)];f=np.asarray([t=='fact' for t in m['kind']])[:len(x)];return stats(x[f],g[f])
def tci(v):
    v=np.asarray(v,float);n=len(v);m=float(v.mean());h=TQ[n-1]*float(v.std(ddof=1))/math.sqrt(n) if n>1 else float('nan');return dict(mean=m,ci95=[m-h,m+h],n=n,per_seed=[float(x) for x in v])
def analyze(iso,main,seeds):
    iso=pathlib.Path(iso);main=pathlib.Path(main);R={};miss=[]
    for a,d in (('NATS',main),('NATMASK1024',main),('NATMASK256',main),('ISO',iso)):
        fs=[d/f'cband_{a}_s{s}.json' for s in seeds]
        if all(f.exists() for f in fs):R[a]=[json.load(open(f)) for f in fs]
        else:miss.append(a)
    w0f=[main/f'cband_W0_s{s}.json' for s in seeds]
    if all(f.exists() for f in w0f):R['W0']=[json.load(open(f)) for f in w0f]
    else:miss.append('W0')
    out=dict(seeds=list(seeds),missing=miss,chance_top1=CHANCE,arms={},paired={},iso_top1_vs_chance={},update_stats={})
    for a,rs in R.items():
        out['arms'][a]={}
        for e in EXPO:
            row={}
            for k in ('out','in'):
                st=[get(r,e,k) for r in rs];row[k]={met:tci([s[met].mean() for s in st]) for met in ('margin','top1','rank','mrr','gold')}
            out['arms'][a][e]=row
    for a in ('NATMASK1024','NATMASK256','ISO'):
        for b in ('NATS','W0'):
            if a in R and b in R:
                out['paired'][f'{a} - {b}']={e:{k:{met:tci([get(x,e,k)[met].mean()-get(y,e,k)[met].mean() for x,y in zip(R[a],R[b])]) for met in ('margin','top1','rank','mrr','gold')} for k in ('out','in')} for e in EXPO}
    if 'ISO' in R:
        out['iso_top1_vs_chance']={e:{k:tci([get(r,e,k)['top1'].mean()-CHANCE for r in R['ISO']]) for k in ('out','in')} for e in EXPO}
        us={}
        for a,rs in R.items():
            if a in ('W0',):continue
            us[a]=dict(cum_update_norm=float(np.mean([sum(e['update_norm'] for e in r['log']) for r in rs])),clip_frac=float(np.mean([np.mean([e['clip_coef']<1 for e in r['log']]) for r in rs])),n_steps=float(np.mean([len(r['log']) for r in rs])))
        out['update_stats']=us
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0);seeds=list(range(161,169))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);(td/'m').mkdir();(td/'i').mkdir()
        for s in seeds:
            g=rng.integers(0,16,20).tolist();meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g)
            for a,d,eff in (('NATS',td/'m',0.5),('NATMASK1024',td/'m',0.7),('NATMASK256',td/'m',0.9),('ISO',td/'i',2.0),('W0',td/'m',0.0)):
                rd={}
                for e in EXPO:
                    o=rng.normal(0,1,(20,16));i=rng.normal(0,1,(20,16))
                    for q in range(16):o[q,g[q]]+=eff;i[q,g[q]]+=3
                    rd[e]={'out':o.tolist(),'in':i.tolist()}
                name=f'cband_W0_s{s}.json' if a=='W0' else f'cband_{a}_s{s}.json'
                json.dump(dict(meta=meta,reads=rd,log=[dict(update_norm=0.5,clip_coef=1.0)]*3),open(d/name,'w'))
        o=analyze(td/'i',td/'m',seeds);assert not o['missing'] and o['paired']['ISO - NATS']['128']['out']['margin']['mean']>0.8 and o['iso_top1_vs_chance']['128']['out']['mean']>0.1 and set(o['update_stats'])=={'NATS','NATMASK1024','NATMASK256','ISO'}
        (td/'i'/'cband_ISO_s161.json').unlink();o2=analyze(td/'i',td/'m',seeds);assert 'ISO' in o2['missing'] and 'ISO' not in o2['arms']
    print('cband_iso_analyze selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--iso');ap.add_argument('--main');ap.add_argument('--seeds',default='161,162,163,164,165,166,167,168');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    o=analyze(a.iso,a.main,[int(x) for x in a.seeds.split(',')]);pathlib.Path(a.json).write_text(json.dumps(o,indent=1))
    for a_,t in o['arms'].items():print(a_,'out margin / top1 @128:',round(t['128']['out']['margin']['mean'],2),round(t['128']['out']['top1']['mean'],3),'| in margin / top1:',round(t['128']['in']['margin']['mean'],2),round(t['128']['in']['top1']['mean'],3))
if __name__=='__main__':main()
