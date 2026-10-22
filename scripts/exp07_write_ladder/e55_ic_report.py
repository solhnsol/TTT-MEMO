"""E55 design 19 report (exploratory, ONE seed): 2 frame conditions (unique = every fact its own 64 frames, shared64 = the same 64 frames for every fact) x 3 arms (NAT real window, KMASK = a query inside the sentence of fact f does not see the earlier sentences of f in the window, SENT = no query sees any earlier fact sentence in the window). Per cell and read point: mean gain u = (W0 rank - rank) / 15 over the 16 facts (W0 subtracted, fact-level bootstrap CI), paired differences KMASK-NAT, SENT-NAT, SENT-KMASK and shared64-unique (same facts, same names / places / background, only the frame assignment differs); masked-token counts and update sizes from the saved logs. No preregistered verdict: one seed, so nothing here is claimed as seed-level evidence.
  python3 e55_ic_report.py --dir OUT --seed 61 --out report.md        (files E55_icu_<ARM>_s<seed>.read.json / E55_ics_...; tags icu = unique, ics = shared64)"""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_metrics as MT
U=1/15;NB=10000
ARMS=('NAT','KMASK','SENT');TAGS={'unique':'icu','shared64':'ics'}
LAB={'unique':'low frame sharing (0.45x)','shared64':'shared64 frames (1.95x)'}       # labels in the report: 'unique' = every fact its own 64 frames, still 0.45 same-frame sentences of other facts per window
def fact_g(rec,a,ctx):
    it=MT.rec_items(rec,a,ctx);m=MT._sel(rec,a,ctx);return np.asarray(it['G_rank'][m],float)*U
def boot(x,seed=5):
    x=np.asarray(x,float);r=np.random.default_rng(seed);b=x[r.integers(0,len(x),(NB,len(x)))].mean(1);return float(x.mean()),*np.percentile(b,[2.5,97.5]).tolist()
def fmt(t):return f'{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]'
def report(d,seed,apps=(16,32,64),ctxs=('out','in'),out=None,load=None):
    load=load or (lambda f:json.load(open(f)));G={};S={}
    for fr,tg in TAGS.items():
        for ar in ARMS:
            f=Path(d)/f'E55_{tg}_{ar}_s{seed}.read.json'
            if f.exists():G[fr,ar]=load(f)
            f=Path(d)/f'E55_{tg}_{ar}_s{seed}.saved.json'
            if f.exists():S[fr,ar]=json.load(open(f))
    L=[f'# E55 design 19 report: frames x mask arms, seed {seed} (exploratory, one seed)','',f'u = (W0 rank - rank) / 15, mean over facts, 95% fact-bootstrap CI; chance 0; the preregistered 0.15 u threshold is shown only as a reference line, no verdict is drawn from one seed.','']
    for ctx in ctxs:
        L+=[f'## {"out-of-window" if ctx=="out" else "in-window"} read','','| frames | arm | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
        for fr in TAGS:
            for ar in ARMS:
                if (fr,ar) not in G:continue
                row=[]
                for a in apps:
                    try:row.append(fmt(boot(fact_g(G[fr,ar],a,ctx))))
                    except Exception as e:row.append('n/a')
                L.append(f'| {LAB[fr]} | {ar} | '+' | '.join(row)+' |')
        L+=['','paired differences (same facts):','','| contrast | '+' | '.join(f'a={a}' for a in apps)+' |','|---|'+'---|'*len(apps)]
        cons=[(f'{LAB[fr]}: {b}-{a_}',(fr,b),(fr,a_)) for fr in TAGS for a_,b in (('NAT','KMASK'),('NAT','SENT'),('KMASK','SENT'))]+[(f'{ar}: shared64 - low frame sharing',('shared64',ar),('unique',ar)) for ar in ARMS]
        for nm,x,y in cons:
            if x not in G or y not in G:continue
            row=[]
            for a in apps:
                try:row.append(fmt(boot(fact_g(G[x],a,ctx)-fact_g(G[y],a,ctx))))
                except Exception:row.append('n/a')
            L.append(f'| {nm} | '+' | '.join(row)+' |')
        L.append('')
    L+=['## write side (from the saved logs)','','| frames | arm | mean update norm | mean CE | clip fraction | mean hidden distinct window keys / chunk (share of the 8192 window) |','|---|---|---|---|---|---|']
    for (fr,ar),r in S.items():
        lg=r['log'];nm=np.mean([e.get('n_masked_keys',0) for e in lg[8:]]) if len(lg)>8 else 0.0
        L.append(f"| {LAB[fr]} | {ar} | {np.mean([e['update_norm'] for e in lg]):.4g} | {np.mean([e['ce'] for e in lg]):.3f} | {r.get('clip_bind_frac',float('nan')):.2f} | {nm:.0f} ({100*nm/8192:.1f}%) |")
    L+=['','KMASK hides the same distinct keys as SENT but only from the 1/16 of the query tokens that sit inside the sentence of the same fact (see e55_swap.token_masks); the mean number of hidden keys per fact-sentence query token is about 1/16 of the SENT count x 1 sentence.']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def selftest():
    rng=np.random.default_rng(0);fake={}
    class R(dict):pass
    orig=(MT.rec_items,MT._sel)
    def ri(rec,a,ctx):return {'G_rank':rec['g']+a*0}
    MT.rec_items=ri;MT._sel=lambda rec,a,ctx:np.ones(16,bool)
    try:
        import tempfile;d=Path(tempfile.mkdtemp());base=rng.normal(size=16)
        data={(tg,ar):{'g':base+(0.5*i if ar!='NAT' else 0)} for tg in TAGS.values() for i,ar in enumerate(ARMS)}
        ld=lambda f:data[(Path(f).name.split('_')[1],Path(f).name.split('_')[2])]
        for (tg,ar) in data:(d/f'E55_{tg}_{ar}_s61.read.json').write_text('{}')
        t=report(d,61,load=ld);assert 'KMASK-NAT' in t and 'shared64 - low frame sharing' in t and 'low frame sharing (0.45x)' in t and 'NAT' in t
        x=fact_g({'g':np.array([15.0]*16)},16,'out') if False else None
    finally:MT.rec_items,MT._sel=orig
    MT.rec_items=lambda rec,a,ctx:{'G_rank':rec['g']};MT._sel=lambda rec,a,ctx:np.ones(16,bool)
    try:
        d2=Path(tempfile.mkdtemp());dt={(sd,ar):{'g':base+(1.0 if ar=='SENT' else 0)} for sd in (61,62,63) for ar in ('NAT','SENT')}
        for (sd,ar) in dt:(d2/f'E55_icr_{ar}_s{sd}.read.json').write_text('{}')
        t=rep_report(d2,[61,62,63],load=lambda f:dt[(int(Path(f).name.split('_s')[1].split('.')[0]),Path(f).name.split('_')[2])]);assert 'pooled' in t and 'SENT - NAT' in t and 'n/a' not in t,t
    finally:MT.rec_items,MT._sel=orig
    print('e55_ic_report SELFTEST OK')
def rep_report(d,seeds,tag='icr',apps=(16,32,64),out=None,load=None):
    """E55 design 21.2 (shared16rep: 16 shared frames, 4 laps, repeats of (fact, frame) allowed at gap 16): arms NAT and SENT, seeds 61-63; per seed and pooled over seeds (facts of all seeds, fact bootstrap): u of NAT, SENT and the paired difference SENT - NAT, out-of-window and in-window. Exploratory; per-seed signs are listed because three seeds cannot carry seed-level significance."""
    load=load or (lambda f:json.load(open(f)));G={}
    for sd in seeds:
        for ar in ('NAT','SENT'):
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
    L=[f'# E55 design 21.2 report: 16 shared frames x 4 laps (repeats allowed), arms NAT / SENT, seeds {list(seeds)} (exploratory)','','u = (W0 rank - rank)/15 over facts, 95% fact-bootstrap CI; the 0.15 u line is a reference only; three seeds cannot carry a seed-level significance claim.','']
    for ctx in ('out','in'):
        L+=[f'## {"out-of-window" if ctx=="out" else "in-window"} read','','| seed | quantity | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
        pooled={}
        for sd in list(seeds)+['pooled']:
            for nm,fn in (('NAT',lambda a,sd:fact_g(G[sd,'NAT'],a,ctx)),('SENT',lambda a,sd:fact_g(G[sd,'SENT'],a,ctx)),('SENT - NAT',lambda a,sd:fact_g(G[sd,'SENT'],a,ctx)-fact_g(G[sd,'NAT'],a,ctx))):
                row=[]
                for a in apps:
                    try:
                        x=np.concatenate([fn(a,s_) for s_ in seeds if (s_,'NAT') in G and (s_,'SENT') in G]) if sd=='pooled' else fn(a,sd);row.append(fmt(boot(x)))
                    except Exception:row.append('n/a')
                L.append(f'| {sd} | {nm} | '+' | '.join(row)+' |')
        L.append('')
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
if __name__=='__main__':
    import sys as _s
    if '--rep' in _s.argv:
        _s.argv.remove('--rep');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--tag',default='icr');a=ap.parse_args();print(rep_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,out=a.out));_s.exit(0)
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dir');ap.add_argument('--seed',type=int,default=61);ap.add_argument('--out');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    print(report(a.dir,a.seed,out=a.out))

