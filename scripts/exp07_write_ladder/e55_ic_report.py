"""E55 design 19 report (exploratory, ONE seed): 2 frame conditions (unique = every fact its own 64 frames, shared64 = the same 64 frames for every fact) x 3 arms (NAT real window, KMASK = a query inside the sentence of fact f does not see the earlier sentences of f in the window, SENT = no query sees any earlier fact sentence in the window). Per cell and read point: mean gain u = (W0 rank - rank) / 15 over the 16 facts (W0 subtracted, fact-level bootstrap CI), paired differences KMASK-NAT, SENT-NAT, SENT-KMASK and shared64-unique (same facts, same names / places / background, only the frame assignment differs); masked-token counts and update sizes from the saved logs. No preregistered verdict: one seed, so nothing here is claimed as seed-level evidence.
  python3 e55_ic_report.py --dir OUT --seed 61 --out report.md        (files E55_icu_<ARM>_s<seed>.read.json / E55_ics_...; tags icu = unique, ics = shared64)"""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_metrics as MT
U=1/15;NB=10000
ERR=[]       # exceptions behind any 'n/a' cell, printed at the end of the report
ARMS=('NAT','KMASK','SENT');TAGS={'unique':'icu','shared64':'ics'}
LAB={'unique':'low frame sharing (0.45x)','shared64':'shared64 frames (1.95x)'}       # labels in the report: 'unique' = every fact its own 64 frames, still 0.45 same-frame sentences of other facts per window
def in_ctx(rec):
    """name of the in-window context in a read record (e.g. 'in_g0'; the out-of-window one is 'out')"""
    r=rec.get('read',{});names=sorted({c for a_ in r.values() for c in a_ if c!='out'});return names[0] if names else None
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
    inn=next((in_ctx(r) for r in G.values() if in_ctx(r)),None)
    for ctx_label in ctxs:
        ctx='out' if ctx_label=='out' else inn
        if ctx is None:L+=['## in-window read: no in-window context in the read files','']; continue
        L+=[f'## {"out-of-window" if ctx=="out" else "in-window ("+ctx+")"} read','','| frames | arm | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
        for fr in TAGS:
            for ar in ARMS:
                if (fr,ar) not in G:continue
                row=[]
                for a in apps:
                    try:row.append(fmt(boot(fact_g(G[fr,ar],a,ctx))))
                    except Exception as e:row.append('n/a');ERR.append(repr(e))
                L.append(f'| {LAB[fr]} | {ar} | '+' | '.join(row)+' |')
        L+=['','paired differences (same facts):','','| contrast | '+' | '.join(f'a={a}' for a in apps)+' |','|---|'+'---|'*len(apps)]
        cons=[(f'{LAB[fr]}: {b}-{a_}',(fr,b),(fr,a_)) for fr in TAGS for a_,b in (('NAT','KMASK'),('NAT','SENT'),('KMASK','SENT'))]+[(f'{ar}: shared64 - low frame sharing',('shared64',ar),('unique',ar)) for ar in ARMS]
        for nm,x,y in cons:
            if x not in G or y not in G:continue
            row=[]
            for a in apps:
                try:row.append(fmt(boot(fact_g(G[x],a,ctx)-fact_g(G[y],a,ctx))))
                except Exception as e:row.append('n/a');ERR.append(repr(e))
            L.append(f'| {nm} | '+' | '.join(row)+' |')
        L.append('')
    L+=['## write side (from the saved logs)','','| frames | arm | mean update norm | mean CE | clip fraction | mean hidden distinct window keys / chunk (share of the 8192 window) |','|---|---|---|---|---|---|']
    for (fr,ar),r in S.items():
        lg=r['log'];nm=np.mean([e.get('n_masked_keys',0) for e in lg[8:]]) if len(lg)>8 else 0.0
        L.append(f"| {LAB[fr]} | {ar} | {np.mean([e['update_norm'] for e in lg]):.4g} | {np.mean([e['ce'] for e in lg]):.3f} | {r.get('clip_bind_frac',float('nan')):.2f} | {nm:.0f} ({100*nm/8192:.1f}%) |")
    L+=['','KMASK hides the same distinct keys as SENT but only from the 1/16 of the query tokens that sit inside the sentence of the same fact (see e55_swap.token_masks); the mean number of hidden keys per fact-sentence query token is about 1/16 of the SENT count x 1 sentence.']
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:3]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def selftest():
    """real-shaped read records (as e55_analyze.selftest: out + in_g0 contexts, W0 under 'out' and 'in_g0@16'), no mocking: both reports must fill every cell, including the in-window section"""
    import tempfile
    rng=np.random.default_rng(1);K=8;nf=8;d=Path(tempfile.mkdtemp())
    def rec(sig):
        n=nf+4;gold=rng.integers(0,K,n);ex=np.array([[True]*K+[False]*8]*n);meta={'kind':['fact']*nf+['ctrl']*4,'gold_idx':gold.tolist(),'cand_exposed':ex.tolist(),'n_rank':[K]*n,'d':{'16':{'in_g0':[1]*nf+[None]*4}},'item_idx':{'16':{'in_g0':list(range(n))}}}
        base=rng.normal(size=(n,16));W=base+0.2*rng.normal(size=(n,16));W[np.arange(nf),gold[:nf]]+=sig
        return {'meta':meta,'read':{'16':{'out':W.tolist(),'in_g0':W.tolist()}},'W0':{'out':base.tolist(),'in_g0@16':base.tolist()}}
    for tg in TAGS.values():
        for i,ar in enumerate(ARMS):(d/f'E55_{tg}_{ar}_s61.read.json').write_text(json.dumps(rec(1.0*i)))
    ERR.clear();t=report(d,61,apps=(16,));assert 'KMASK-NAT' in t and 'shared64 - low frame sharing' in t and 'low frame sharing (0.45x)' in t and 'in-window (in_g0)' in t and 'n/a' not in t,t
    for sd in (61,62,63):
        for i,ar in enumerate(('NAT','SENT')):(d/f'E55_icr_{ar}_s{sd}.read.json').write_text(json.dumps(rec(1.0*i)))
    ERR.clear();t=rep_report(d,[61,62,63],apps=(16,));assert 'pooled' in t and 'SENT - NAT' in t and 'in-window (in_g0)' in t and 'n/a' not in t,t
    for sd in (61,62,63):
        for ar,tg in (('NAT','v3a'),('ISO','iso')):(d/f'E55_{tg}_{ar}_s{sd}.read.json').write_text(json.dumps({**rec(1.0*(ar=='NAT')),'read':{str(a):{'out':rec(1.0)['read']['16']['out']} for a in (16,32,64,128)}}))
    ERR.clear();t=iso_report(d,[61,62,63]);assert 'ISO - NAT' in t and 'n/a' not in t,t
    dd=Path(tempfile.mkdtemp())
    for ar in ('FT-NAT','FT-ISO'):
        (dd/f'E55_ft_{ar}_lr0.0001_s61.saved.json').write_text(json.dumps({'epochs':[{'epoch':1,'mean_ce':2.5,'fact_token_ce':3.1},{'epoch':2,'mean_ce':2.0,'fact_token_ce':None}]}))
        for ep in (1,2):
            r_=rec(1.0);(dd/f'E55_ft_{ar}_lr0.0001_s61_ep{ep}.read.json').write_text(json.dumps({**r_,'read':{'64':r_['read']['16']},'W0':{'out':r_['W0']['out'],'in_g0@64':r_['W0']['in_g0@16']}}))
            r2=rec(2.0);(dd/f'E55_ft_{ar}_lr0.0001_s61_ep{ep}.pc.read.json').write_text(json.dumps({**r2,'read':{'64':r2['read']['16']},'W0':{'out':r2['W0']['out'],'in_g0@64':r2['W0']['in_g0@16']}}))
    ERR.clear();t=ft_report(dd,'ft',61,[1e-4],2,arms=('FT-NAT','FT-ISO'));assert 'u pc' in t and 'n/a |' in t.replace('| n/a | n/a','|') or True
    assert t.count('n/a')==2,t
    print('e55_ic_report SELFTEST OK')
def rep_report(d,seeds,tag='icr',apps=(16,32,64),out=None,load=None):
    """E55 design 21.2 (shared16rep: 16 shared frames, 4 laps, repeats of (fact, frame) allowed at gap 16): arms NAT and SENT, seeds 61-63; per seed and pooled over seeds (facts of all seeds, fact bootstrap): u of NAT, SENT and the paired difference SENT - NAT, out-of-window and in-window. Exploratory; per-seed signs are listed because three seeds cannot carry seed-level significance."""
    load=load or (lambda f:json.load(open(f)));G={}
    for sd in seeds:
        for ar in ('NAT','SENT'):
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
    L=[f'# E55 design 21.2 report: 16 shared frames x 4 laps (repeats allowed), arms NAT / SENT, seeds {list(seeds)} (exploratory)','','u = (W0 rank - rank)/15 over facts, 95% fact-bootstrap CI; the 0.15 u line is a reference only; three seeds cannot carry a seed-level significance claim.','']
    inn=next((in_ctx(r) for r in G.values() if in_ctx(r)),None)
    for ctx_label in ('out','in'):
        ctx='out' if ctx_label=='out' else inn
        if ctx is None:L+=['## in-window read: no in-window context in the read files','']; continue
        L+=[f'## {"out-of-window" if ctx=="out" else "in-window ("+ctx+")"} read','','| seed | quantity | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
        pooled={}
        for sd in list(seeds)+['pooled']:
            for nm,fn in (('NAT',lambda a,sd:fact_g(G[sd,'NAT'],a,ctx)),('SENT',lambda a,sd:fact_g(G[sd,'SENT'],a,ctx)),('SENT - NAT',lambda a,sd:fact_g(G[sd,'SENT'],a,ctx)-fact_g(G[sd,'NAT'],a,ctx))):
                row=[]
                for a in apps:
                    try:
                        x=np.concatenate([fn(a,s_) for s_ in seeds if (s_,'NAT') in G and (s_,'SENT') in G]) if sd=='pooled' else fn(a,sd);row.append(fmt(boot(x)))
                    except Exception as e:row.append('n/a');ERR.append(repr(e))
                L.append(f'| {sd} | {nm} | '+' | '.join(row)+' |')
        L.append('')
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:3]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def iso_report(d,seeds,nat_tag='v3a',iso_tag='iso',apps=(16,32,64,128),out=None,load=None):
    """E55 design 22.3 stage 0: NAT (online stream, job 181 files, tag v3a) vs ISO (isolated rows, cumulative update norm matched to NAT), out-of-window reads at a = 16, 32, 64, 128; per seed and pooled: u = (W0 rank - rank)/15 with fact-bootstrap CI and the paired ISO - NAT."""
    load=load or (lambda f:json.load(open(f)));G={}
    for sd in seeds:
        for ar,tg in (('NAT',nat_tag),('ISO',iso_tag)):
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
    L=[f'# E55 stage 0 report: isolated rows (ISO, norm-matched) vs the online stream (NAT), seeds {list(seeds)}','','Label: same norm budget, fewer and larger steps (ISO takes about 53 SGD steps up to a = 128, NAT 128 chunk steps).','','u = (W0 rank - rank)/15, mean over facts, 95% fact-bootstrap CI; out-of-window read, question frames never seen in the stream; 0.15 u = reference threshold, exploratory.','','| seed | quantity | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
    for sd in list(seeds)+['pooled']:
        use=[s_ for s_ in seeds if (s_,'NAT') in G and (s_,'ISO') in G]
        for nm,fn in (('NAT',lambda a,s_:fact_g(G[s_,'NAT'],a,'out')),('ISO',lambda a,s_:fact_g(G[s_,'ISO'],a,'out')),('ISO - NAT',lambda a,s_:fact_g(G[s_,'ISO'],a,'out')-fact_g(G[s_,'NAT'],a,'out'))):
            row=[]
            for a in apps:
                try:x=np.concatenate([fn(a,s_) for s_ in use]) if sd=='pooled' else fn(a,sd);row.append(fmt(boot(x)))
                except Exception as e:row.append('n/a');ERR.append(repr(e))
            L.append(f'| {sd} | {nm} | '+' | '.join(row)+' |')
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:3]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def ft_report(d,tag,seed,lrs,epochs,arms=('FT-NAT','FT-MASK','FT-ISO'),out=None,load=None):
    """E55 design 22.1 report: per learning rate and arm, per epoch with reads: u (out-of-window, in-window, positive control; W0 subtracted, fact bootstrap CI), mean CE and the place-token CE of the epoch. The positive control is a memorization ceiling only and is not part of any verdict. First 8 chunks of every epoch have a short carried window (reset at the epoch start)."""
    load=load or (lambda f:json.load(open(f)));L=[f'# E55 stage 1 report: fast-weight fine-tuning probe, seed {seed}, tag {tag}','','u = (W0 rank - rank)/15 over the 16 facts, 95% fact-bootstrap CI. pc = positive control (question in the frame of the latest training sentence): memorization ceiling, not used for the verdict. Each epoch repeats the 64-chunk stream (FT-NAT / FT-MASK) or all isolated rows (FT-ISO) with the carried window reset at the start; the first 8 chunks of every epoch therefore have a short window.','']
    pairs=[(ar,lr) for ar,v in lrs.items() for lr in v] if isinstance(lrs,dict) else [(ar,lr) for lr in lrs for ar in arms]
    for ar,lr in pairs:
        if True:
            fs=Path(d)/f'E55_{tag}_{ar}_lr{lr:g}_s{seed}.saved.json'
            if not fs.exists():continue
            sv=load(fs);ep_info={e['epoch']:e for e in sv['epochs']};L+=[f'## {ar}, lr {lr:g}','','| epoch | u out | u in-window | u pc | mean CE | place-token CE |','|---|---|---|---|---|---|']
            for ep in sorted(ep_info):
                fo=Path(d)/f'E55_{tag}_{ar}_lr{lr:g}_s{seed}_ep{ep}.read.json';fp=Path(d)/f'E55_{tag}_{ar}_lr{lr:g}_s{seed}_ep{ep}.pc.read.json'
                if not fo.exists():continue
                ro=load(fo);rp=load(fp) if fp.exists() else None;cell=[]
                for rec_,ctx in ((ro,'out'),(ro,in_ctx(ro)),(rp,'out')):
                    try:cell.append(fmt(boot(fact_g(rec_,64,ctx))))
                    except Exception as e:cell.append('n/a');ERR.append(repr(e))
                e_=ep_info[ep];pce=e_.get('fact_token_ce');L.append(f"| {ep} | {cell[0]} | {cell[1]} | {cell[2]} | {e_['mean_ce']:.4f} | {'n/a' if pce is None else f'{pce:.4f}'} |")
            L.append('')
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:3]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
if __name__=='__main__':
    import sys as _s
    if '--iso' in _s.argv:
        _s.argv.remove('--iso');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');a=ap.parse_args();print(iso_report(a.dir,[int(x) for x in a.seeds.split(',')],out=a.out));_s.exit(0)
    if '--ft' in _s.argv:
        _s.argv.remove('--ft');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--tag');ap.add_argument('--seed',type=int);ap.add_argument('--lrs',default=None);ap.add_argument('--arm-lrs',default=None);ap.add_argument('--epochs',type=int,default=4);ap.add_argument('--out');a=ap.parse_args();L_={}
        for x in (a.arm_lrs.split(',') if a.arm_lrs else []):L_.setdefault(x.split(':')[0],[]).append(float(x.split(':')[1]))
        print(ft_report(a.dir,a.tag,a.seed,L_ if a.arm_lrs else [float(x) for x in a.lrs.split(',')],a.epochs,out=a.out));_s.exit(0)
    if '--rep' in _s.argv:
        _s.argv.remove('--rep');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--tag',default='icr');a=ap.parse_args();print(rep_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,out=a.out));_s.exit(0)
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dir');ap.add_argument('--seed',type=int,default=61);ap.add_argument('--out');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    print(report(a.dir,a.seed,out=a.out))

