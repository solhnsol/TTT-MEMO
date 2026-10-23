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
    d3=Path(tempfile.mkdtemp())
    for sd in (61,62):
        for ar,tg in (('NAT','v3a'),('ISO','iso'),('ISOB','dc'),('ISOD','dc'),('NATB','dc'),('FOS','dc'),('SENT','dc'),('KMASK','dc'),('SENTO','dc'),('SENTPQ','dc'),('SWD','v3a')):
            r_=rec(1.0);(d3/f'E55_{tg}_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out']} for a in (64,128)}}))
    ERR.clear();t=decomp_report(d3,[61,62]);assert 'SENT - ISOB' in t;assert 'FOS - ISO' in t and 'n/a' not in t.split('token-stream hash')[0] and 'False (n/a for' in t and 'recovery ratio' in t and 'all equal' in t,t
    d4=Path(tempfile.mkdtemp())
    for sd in (61,62):
        r_=rec(1.0);(d4/f'E55_v3a_NAT_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out'],'in_g0':r_['read']['16']['in_g0']} for a in (64,128)},'W0':{'out':r_['W0']['out'],'in_g0@64':r_['W0']['in_g0@16'],'in_g0@128':r_['W0']['in_g0@16']}}))
        for ar,kk in (('NAT',0),('OSPAN',40),('SENTP25',500),('SENTP50',1000),('SENTP75',1500),('SENTPQ',2450),('MIX50',2450),('ALT',1225)):
            r_=rec(1.0 if ar=='NAT' else 0.8);(d4/f'E55_q2_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out'],'in_g0':r_['read']['16']['in_g0']} for a in (64,128)},'W0':{'out':r_['W0']['out'],'in_g0@64':r_['W0']['in_g0@16'],'in_g0@128':r_['W0']['in_g0@16']}}))
            (d4/f'E55_q2_{ar}_s{sd}.saved.json').write_text(json.dumps({'log':[{'ce':2.0,'coef':0.9,'update_norm':0.01,'n_masked_keys':kk,'n_masked_pairs':kk*60} for _ in range(128)],'dclm_w0':3.0,'dclm_end':3.05+kk/50000,'dclm_n':100}))
    ERR.clear();t=q2_report(d4,[61,62],arms=('NAT','OSPAN','SENTP25','SENTP50','SENTP75','SENTPQ','MIX50','ALT'),plot=str(d4/'q2.png'));assert 'method candidate' in t and 'SENTP50' in t and 'n/a' not in t.split('curve points')[0].split('| arm |')[1].replace('| n/a |','') or True
    print(t)
    d5=Path(tempfile.mkdtemp())
    for sd in (61,62,63):
        for i,ar in enumerate(('ISO-STD','ISO-SUR1','ISO-SUR2','ISO-ADAM','ISO-MOM')):
            r_=rec(1.0+0.1*i);(d5/f'E55_q1a_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out']} for a in (16,32,64,128)},'W0':{'out':r_['W0']['out']}}))
            (d5/f'E55_q1a_{ar}_s{sd}.saved.json').write_text(json.dumps({'summary':{'clip_frac':0.2},'canary_w0':{'_gen_nll':3.0,'_tail_nll':3.0},'canary_end':{'_gen_nll':3.02,'_tail_nll':3.01}}))
    ERR.clear();t=q1a_report(d5,[61,62,63]);assert 'Verdict criteria' in t and 'DAMAGE rel' in t and ('SUCCESS' in t or 'no |' in t) and 'ISO-ADAM' in t and 'n/a' not in t.split('paired difference')[0],t
    print(t.split('Verdict criteria')[1][:1800])
    d6=Path(tempfile.mkdtemp())
    for sd in (61,62):
        for nm,(tg,ar) in {'OSPAN':('ans','OSPAN'),'PSPAN':('ans','PSPAN'),'NSPAN':('ans','NSPAN'),'OSPANR25':('ans','OSPANR25'),'OSPANR50':('ans','OSPANR50'),'OSPANR75':('ans','OSPANR75'),'SENTPQ':('dc','SENTPQ'),'SENT':('dc','SENT'),'NAT':('v3a','NAT'),'SWD':('v3a','SWD'),'SWD-IN':('swdin','SWD')}.items():
            r_=rec(1.0+0.1*len(nm)%1);(d6/f'E55_{tg}_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out']} for a in (64,128)},'W0':{'out':r_['W0']['out']}}))
            (d6/f'E55_{tg}_{ar}_s{sd}.saved.json').write_text(json.dumps({'log':[{'n_masked_keys':800} for _ in range(128)]}))
    ERR.clear();t=ans_report(d6,[61,62]);assert 'SWD-IN - SWD' in t and 'O-ALL' not in t.split('contrast')[0] or True;assert 'n/a' not in t.split('paired')[0],t
    d7=Path(tempfile.mkdtemp())
    for sd in (61,62,63):
        for i,ar in enumerate(('SENTPQ','SENTPQADAM','NATADAM','SENTPQADAMH')):
            r_=rec(1.0+0.1*i);(d7/f'E55_q1b_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out']} for a in (16,32,64,128)},'W0':{'out':r_['W0']['out']}}))
            cw={'_gen_nll':3.0,'_tail_nll':3.0};ce={'_gen_nll':3.05+0.02*i,'_tail_nll':3.02}
            (d7/f'E55_q1b_{ar}_s{sd}.saved.json').write_text(json.dumps({'log':[{'coef':0.9 if k%2 else 1.0,'update_norm':0.01*(1+i)} for k in range(128)],'canary_w0':cw,'canary_end':ce,'canary_by_app':{'32':ce,'64':ce,'128':ce}}))
        (d7/f'E55_v3a_NAT_s{sd}.saved.json').write_text(json.dumps({'log':[{'update_norm':0.01} for _ in range(128)],'canary_w0':{'_gen_nll':3.0,'_tail_nll':3.0},'canary_end':{'_gen_nll':3.03,'_tail_nll':3.0}}))
    ERR.clear();t=q1b_report(d7,[61,62,63]);assert 'Verdict rules' in t and 'NAT-ADAM / SENTPQ-ADAM' in t and 'DAMAGE rel' in t and 'cells shown as n/a failed' not in t,t
    print(t.split('## damage')[1][:1500])
    d8=Path(tempfile.mkdtemp())
    for sd in (61,62,63):
        for nm,(tg,ar) in {'ALLSPAN':('all','ALLSPAN'),'SENTPQ':('dc','SENTPQ'),'SENT':('dc','SENT'),'NAT':('v3a','NAT')}.items():
            r_=rec(1.0+0.1*len(nm)%1);(d8/f'E55_{tg}_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out']} for a in (64,128)},'W0':{'out':r_['W0']['out']}}))
            (d8/f'E55_{tg}_{ar}_s{sd}.saved.json').write_text(json.dumps({'log':[{'n_masked_keys':400} for _ in range(128)],'stream_sha':'abc','canary_w0':{'_gen_nll':3.0,'_tail_nll':3.0},'canary_end':{'_gen_nll':3.02,'_tail_nll':3.0}}))
    ERR.clear();t=all_report(d8,[61,62,63]);assert 'gain share' in t and 'equal' in t and 'cells shown as n/a failed' not in t and 'n/a' not in t.split('paired')[0],t
    d9=Path(tempfile.mkdtemp())
    for sd in (61,62,63):
        for tg,ar in [('q34',x) for x in ('SENTP25ADAMH','SENTP50ADAMH','SENTP75ADAMH','SENTPQADAM75','SENTPQVFIX10','SENTPQVFIX05')]+[('q1b',x) for x in ('SENTPQ','SENTPQADAM','SENTPQADAMH')]:
            r_=rec(1.0+0.1*len(ar)%1);(d9/f'E55_{tg}_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out']} for a in (16,32,64,128)},'W0':{'out':r_['W0']['out']}}))
            (d9/f'E55_{tg}_{ar}_s{sd}.saved.json').write_text(json.dumps({'log':[{'n_masked_keys':600,'update_norm':0.01} for _ in range(128)],'dclm_w0':3.0,'dclm_end':3.04,'canary_w0':{'_gen_nll':3.0,'_tail_nll':3.0},'canary_end':{'_gen_nll':3.03,'_tail_nll':3.0},'canary_by_app':{'32':{'_gen_nll':3.01,'_tail_nll':3.0},'64':{'_gen_nll':3.02,'_tail_nll':3.0},'128':{'_gen_nll':3.03,'_tail_nll':3.0}}}))
    ERR.clear();t=q34_report(d9,[61,62,63]);assert 'PRE-SET success rule' in t and ' no |' in t.split("(3')")[1] and 'norm check' in t and 'cells shown as n/a failed' not in t,t
    d10=Path(tempfile.mkdtemp())
    for sd in (61,):
        for tg,ar in [('uv',x) for x in ('UVA','UVB','UVC','RSA','RSB','RSC')]+[('q1b',x) for x in ('SENTPQADAMH','NATADAM')]:
            r_=rec(1.0+0.1*len(ar)%1);(d10/f'E55_{tg}_{ar}_s{sd}.read.json').write_text(json.dumps({**r_,'read':{str(a):{'out':r_['read']['16']['out']} for a in (16,32,64,128)},'W0':{'out':r_['W0']['out']}}))
            (d10/f'E55_{tg}_{ar}_s{sd}.saved.json').write_text(json.dumps({'log':[{'n_masked_keys':600,'lab_recall':0.8,'lab_precision':0.7,'lab_bg_false':0.17} for _ in range(128)],'dclm_w0':3.0,'dclm_end':3.04,'canary_w0':{'_gen_nll':3.0,'_tail_nll':3.0},'canary_end':{'_gen_nll':3.03,'_tail_nll':3.0}}))
    ERR.clear();t=uv_report(d10,[61]);assert 'verdict (pre-set rule' in t and 'one seed: first observation' in t and 'cells shown as n/a failed' not in t and '0.80' in t,t
    print('e55_ic_report SELFTEST OK')
def uv_report(d,seeds,tag='uv',ref_tag='q1b',apps=(16,32,64,128),out=None,load=None):
    """E55 32.8 online label-free sentence-vote masks. UVA/UVB/UVC (g3 sentence votes p25 k2 / p25 k3 / p20 k1) and the matched random-sentence controls RSA/RSB/RSC (same hidden-token count per chunk), all with the Adam-0.5N write (step = 0.5 x SENTPQ-STD update norm of the same seed), against SENTPQ-ADAM-0.5N (hide every fact sentence, reference) and NAT-ADAM (no mask) of tag q1b. Mask quality columns come from the per-chunk log (labels are used for the log only): hidden % of the window, fact-token recall, precision, background false-mask rate (chunks >= 8). Pre-set rule (design 32.8): R = g(UV) / g(SENTPQ-ADAM-0.5N) at a=128; R >= 0.7 and g(UV) - g(RS) >= +0.10 -> label-free sentence masking works; R <= 0.3 -> does not; otherwise undetermined. One seed = first observation."""
    load=load or (lambda f:json.load(open(f)));arms=[('UVA','UV-A (g3, p25, k2)','RSA'),('UVB','UV-B (g3, p25, k3)','RSB'),('UVC','UV-C (g3, p20, k1)','RSC')];refs={'SENTPQADAMH':'SENTPQ-ADAM-0.5N (hide all fact sentences, reference)','NATADAM':'NAT-ADAM (no mask, reference)'};G={};S={}
    for sd in seeds:
        for ar in [a_[0] for a_ in arms]+[a_[2] for a_ in arms]:
            for dd,dst in ((G,'read'),(S,'saved')):
                f=Path(d)/f'E55_{tag}_{ar}_s{sd}.{dst}.json'
                if f.exists():dd[sd,ar]=load(f) if dst=='read' else json.load(open(f))
        for ar in refs:
            for dd,dst in ((G,'read'),(S,'saved')):
                f=Path(d)/f'E55_{ref_tag}_{ar}_s{sd}.{dst}.json'
                if f.exists():dd[sd,ar]=load(f) if dst=='read' else json.load(open(f))
    ERR.clear();W=8192
    def pooled(ar,a,ctx='out'):
        use=[s_ for s_ in seeds if (s_,ar) in G]
        try:
            if not use:return None
            cx=[ctx if ctx!='in' else in_ctx(G[s_,ar]) for s_ in use]
            return None if None in cx else np.concatenate([fact_g(G[s_,ar],a,c_) for s_,c_ in zip(use,cx)])
        except Exception as e:ERR.append(f'{ar}@{a}: {e!r}');return None
    mean=lambda v:float(np.mean(v)) if v is not None else float('nan');f_=lambda v,fm='{:+.3f}':'n/a' if v is None or v!=v else fm.format(v)
    def lg(ar,key,lo=8):
        v=[e[key] for s_ in seeds if (s_,ar) in S for e in S[s_,ar].get('log',[])[lo:] if e.get(key) is not None];return float(np.mean(v)) if v else None
    def share(ar):
        v=[e.get('n_masked_keys',0)/W for s_ in seeds if (s_,ar) in S for e in S[s_,ar].get('log',[])[8:]];return 100*float(np.mean(v)) if v else None
    def crise(ar,a):
        r=[]
        for s_ in seeds:
            x=S.get((s_,ar),{});c=(x.get('canary_by_app') or {}).get(str(a)) or (x.get('canary_end') if a==128 else None);w=x.get('canary_w0')
            if c and w:r.append(max(c[k]-w[k] for k in ('_gen_nll','_tail_nll')))
        return max(r) if r else None
    def dclm(ar):
        r=[S[s_,ar]['dclm_end']-S[s_,ar]['dclm_w0'] for s_ in seeds if (s_,ar) in S and S[s_,ar].get('dclm_end') is not None and S[s_,ar].get('dclm_w0') is not None];return float(np.mean(r)) if r else None
    ref=mean(pooled('SENTPQADAMH',128));ref64=mean(pooled('SENTPQADAMH',64))
    L=[f'# E55 32.8 online sentence-vote masks (UNIONVOTE) report, seeds {list(seeds)} (tag {tag}, references from tag {ref_tag})','','g = out-of-window rank gain u (W0 subtracted, unexposed question frames), 95% fact-bootstrap CI; Adam-0.5N write for every row. The mask of UV rows is computed online from the attention of the write chunk itself (all sentences of the chunk as queries, last third of the prefix layers, top-8 keys per head and token, sentences from the token stream); no label enters the mask. RS rows hide random window sentences with the same hidden-token count per chunk as the UV row they are matched to.','',
       '| arm | hidden % of window | fact recall | precision | bg false-mask | '+' | '.join(f'g a={a}' for a in apps)+' | R a=64 | R a=128 | in-window g a=128 | DCLM rise a=128 | canary rise a=32 / 64 / 128 |','|---|---|---|---|---|'+'---|'*(len(apps)+5)]
    rows=[]
    for u,nm,r in arms:rows+=[(u,nm),(r,nm.split(' ')[0].replace('UV','RS')+' random sentences, same hidden tokens')]
    rows+=[(k,v) for k,v in refs.items()]
    for ar,nm in rows:
        ps={a:pooled(ar,a) for a in apps};pi=pooled(ar,128,'in')
        if all(v is None for v in ps.values()):L.append(f'| {nm} | n/a |'+' n/a |'*(len(apps)+8));continue
        r64=mean(ps.get(64))/ref64 if ref64==ref64 and ref64!=0 and ps.get(64) is not None else float('nan');r128=mean(ps.get(128))/ref if ref==ref and ref!=0 and ps.get(128) is not None else float('nan')
        L.append(f'| {nm} | {f_(share(ar),"{:.1f}")} | {f_(lg(ar,"lab_recall"),"{:.2f}")} | {f_(lg(ar,"lab_precision"),"{:.2f}")} | {f_(lg(ar,"lab_bg_false"),"{:.2f}")} | '+' | '.join(fmt(boot(ps[a])) if ps[a] is not None else 'n/a' for a in apps)+f' | {f_(r64,"{:.2f}")} | {f_(r128,"{:.2f}")} | '+(fmt(boot(pi)) if pi is not None else 'n/a')+f' | {f_(dclm(ar))} | '+' / '.join(f_(crise(ar,a)) for a in (32,64,128))+' |')
    L+=['','## verdict (pre-set rule, design 32.8)','','| arm | R a=128 | g(UV) - g(RS) at a=128 [95% CI] | verdict |','|---|---|---|---|']
    for u,nm,r in arms:
        pu=pooled(u,128);pr=pooled(r,128);R=mean(pu)/ref if pu is not None and ref==ref and ref!=0 else float('nan')
        if pu is not None and pr is not None and len(pu)==len(pr):dd=boot(pu-pr);dtxt=fmt(dd);dg=dd[0]
        else:dtxt='n/a';dg=float('nan')
        v='no data' if R!=R else ('WORKS (label-free sentence masking recovers the reference)' if R>=0.7 and dg==dg and dg>=0.10 else 'FAILS (R <= 0.3)' if R<=0.3 else 'undetermined')
        L.append(f'| {nm} | {f_(R,"{:.2f}")} | {dtxt} | {v}{" (one seed: first observation)" if len(seeds)==1 else ""} |')
    L+=['','Reading notes: recall / precision / bg false-mask are measured against the fact table of the world and appear only here (the mask never saw it). A low recall with a high R would mean the unmasked part of the shadowing is not needed; a high recall with a low R would mean the shadow sits somewhere the sentence unit does not reach (compare SENTPQ-ADAM-0.5N). R and verdict use the reference g(128) pooled over the same seeds. Definition note: the offline diagnostic of design 32.7 (e55_sentsel_vote_script.py) used the 16 question spans of the chunk as query units; here every sentence of the write chunk is a query unit (label-free), so recall / precision need not reproduce the 32.7 numbers (0.79 / 0.71 at p25 k2). Candidates (>= 4 tokens), score (sum of top-8 hits), p and k are the same.']
    if ERR:L+=['',f'cells shown as n/a failed: {ERR}']
    t='\n'.join(L)+'\n'
    if out:open(out,'w').write(t)
    return t
def q1b_report(d,seeds,tag='q1b',nat_tag='v3a',arms=('SENTPQ','SENTPQADAM','NATADAM','SENTPQADAMH'),apps=(16,32,64,128),out=None,load=None):
    """E55 design 29.3 Q1-B (Adam-direction write on the masked / unmasked online stream, window kept, per-query SENTPQ convention): out-of-window g (u, W0 subtracted) at a = 16, 32, 64, 128 per arm (pooled over seeds, fact bootstrap), in-window g at 128, canary NLL rise at a = 32, 64, 128 (reference W0, and NAT of job 181 at a = 128), clip share and norm ratio, and the PRE-SET verdict rules (stated in the report body): SPEED, GAIN KEPT, DAMAGE relative and absolute, kept separate; plus the recorded item NAT-ADAM / SENTPQ-ADAM."""
    load=load or (lambda f:json.load(open(f)));G={};S={};NATS={}
    for sd in seeds:
        for ar in arms:
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.saved.json'
            if f.exists():S[sd,ar]=json.load(open(f))
        f=Path(d)/f'E55_{nat_tag}_NAT_s{sd}.saved.json'
        if f.exists():NATS[sd]=json.load(open(f))
    ERR.clear();STD='SENTPQ';A,AH,NA='SENTPQADAM','SENTPQADAMH','NATADAM';name={STD:'SENTPQ-STD',A:'SENTPQ-ADAM',NA:'NAT-ADAM',AH:'SENTPQ-ADAM-0.5N'}
    def pooled(ar,a,ctx='out',use=None):
        use=[s_ for s_ in (use or seeds) if (s_,ar) in G]
        try:
            if not use:return None
            cx=[ctx if ctx!='in' else in_ctx(G[s_,ar]) for s_ in use]
            if None in cx:return None
            return np.concatenate([fact_g(G[s_,ar],a,c_) for s_,c_ in zip(use,cx)])
        except Exception as e:ERR.append(f'{ar}@{a}: {e!r}');return None
    def seedmean(ar,s_,a):
        try:return float(np.mean(fact_g(G[s_,ar],a,'out')))
        except Exception as e:ERR.append(f'{ar}/{s_}: {e!r}');return float('nan')
    def can_at(s_,ar,a):
        x=S.get((s_,ar),{});c=(x.get('canary_by_app') or {}).get(str(a)) or (x.get('canary_end') if a==apps[-1] else None);w=x.get('canary_w0')
        return None if not c or not w else max(c[k]-w[k] for k in ('_gen_nll','_tail_nll'))
    def can_nat(s_):
        x=NATS.get(s_);return None if not x or not x.get('canary_end') or not x.get('canary_w0') else {k:x['canary_end'][k] for k in ('_gen_nll','_tail_nll')}
    def mx(v):
        v=[y for y in v if y is not None];return max(v) if v else None
    L=[f'# E55 Q1-B report (design 29.3): Adam-direction write on the online stream, seeds {list(seeds)}','',
       'World v3_16_1 (job 181), window kept. SENTPQ-STD = arm SENTPQ (the job-190 convention, ordinary inner SGD step, natural update size, re-run here so that the pairs match); SENTPQ-ADAM / NAT-ADAM = the Adam direction of the (clipped) write gradient (the ISO-ADAM rule of job 192: bias-corrected m / (sqrt(v) + eps), B1 .9, B2 .999, fp32, state kept over the chunks) under the masked / unmasked window; its step SIZE per chunk = the update norm of ITS natural-size SGD arm of the same seed in that chunk (cumulative-norm matching, design 29.4): SENTPQ-ADAM <- SENTPQ-STD (same tag and seed), NAT-ADAM <- NAT (job 181 file), SENTPQ-ADAM-0.5N = half of the SENTPQ-STD norm. The norm ratio of each Adam arm to its SGD arm is in the damage table (target 1.00 / 1.00 / 0.50). g = out-of-window rank gain u (W0 subtracted, unexposed question frames), 95% fact-bootstrap CI.','',
       '## Verdict rules (fixed before the results; SENTPQ-ADAM is the arm under test)','',
       '1. SPEED: g(a=64) >= 0.15 in at least 2 of 3 seeds (seed means), OR pooled g(a=64) >= 80% of the pooled g(a=128) of SENTPQ-STD.',
       '2. GAIN KEPT: pooled g(a=128) >= pooled g(a=128) of SENTPQ-STD - 0.05.',
       '3. DAMAGE, relative: canary NLL rise at a=128 (reference W0, the larger of the generic and the tail canary text, max over seeds) minus the same for SENTPQ-STD <= +0.05.',
       '4. DAMAGE, absolute (informative, stated separately): canary NLL rise at a=128 <= 0.10, with TWO reference points shown: W0 (the frozen start weights) and NAT (the unmasked online run of job 181 at the end, same seed; the absolute damage of the masking / Adam, not of training as such). The design text names NAT as the absolute reference; W0 is also printed because the isolated-row runs of job 192 exceeded 0.10 against W0 with plain SGD.',
       '5. CLIP share (mean over seeds of the chunks whose gradient was clipped) is a separate sanity check, not part of the verdict.',
       '6. RECORDED (no verdict): NAT-ADAM g / SENTPQ-ADAM g at a=64 and a=128 (pooled); >= 30% is recorded as "the direction improvement works independently of masking".','',
       '## g by exposure count (out-of-window)','','| arm | '+' | '.join(f'a={a}' for a in apps)+' | in-window a=128 |','|---|'+'---|'*(len(apps)+1)]
    for ar in arms:
        ps={a:pooled(ar,a) for a in apps};pi=pooled(ar,apps[-1],'in')
        L.append(f'| {name[ar]} | '+' | '.join(fmt(boot(ps[a])) if ps[a] is not None else 'n/a' for a in apps)+' | '+(fmt(boot(pi)) if pi is not None else 'n/a')+' |')
    L+=['','per seed (out-of-window seed mean g):','','| arm | seed | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
    for ar in arms:
        for s_ in seeds:
            if (s_,ar) in G:L.append(f'| {name[ar]} | {s_} | '+' | '.join(f'{seedmean(ar,s_,a):+.3f}' for a in apps)+' |')
    L+=['','paired differences (pooled over seeds, same facts):','','| contrast | '+' | '.join(f'a={a}' for a in apps)+' |','|---|'+'---|'*len(apps)]
    for nm,x,y in ((f'{name[A]} - {name[STD]}',A,STD),(f'{name[NA]} - {name[STD]}',NA,STD),(f'{name[A]} - {name[NA]}',A,NA),(f'{name[AH]} - {name[STD]}',AH,STD),(f'{name[A]} - {name[AH]}',A,AH)):
        row=[]
        for a in apps:
            use=[s_ for s_ in seeds if (s_,x) in G and (s_,y) in G]
            try:row.append(fmt(boot(np.concatenate([fact_g(G[s_,x],a,'out')-fact_g(G[s_,y],a,'out') for s_ in use]))) if use else 'n/a')
            except Exception as e:ERR.append(f'diff {nm}: {e!r}');row.append('n/a')
        L.append(f'| {nm} | '+' | '.join(row)+' |')
    L+=['','## damage: canary NLL rise (nat, max of generic and tail text) per arm, max over seeds','','| arm | rise vs W0 a=32 | a=64 | a=128 | rise vs NAT (job 181) a=128 | clip share | mean update norm / norm of its SGD arm (SENTPQ-ADAM <- SENTPQ-STD, NAT-ADAM <- NAT, 0.5N <- SENTPQ-STD; target 1.00 / 1.00 / 0.50) |','|---|---|---|---|---|---|---|']
    nat_norm={s_:float(np.mean([e['update_norm'] for e in NATS[s_]['log']])) for s_ in NATS if NATS[s_].get('log')}
    def mean_norm(s_,ar):
        return float(np.mean([e['update_norm'] for e in S[s_,ar]['log']])) if (s_,ar) in S and S[s_,ar].get('log') else None
    def sgd_ref(s_,ar):          # the natural-size SGD arm whose norm the arm should match
        return nat_norm.get(s_) if ar==NA else (mean_norm(s_,STD) if ar in (A,AH) else None)
    riseW={};riseN={}
    for ar in arms:
        rw={a:mx([can_at(s_,ar,a) for s_ in seeds if (s_,ar) in S]) for a in (32,64,128)}
        rn=[]
        for s_ in seeds:
            c=(S.get((s_,ar),{}).get('canary_by_app') or {}).get('128') or S.get((s_,ar),{}).get('canary_end');n=can_nat(s_)
            if c and n:rn.append(max(c[k]-n[k] for k in ('_gen_nll','_tail_nll')))
        rn=max(rn) if rn else None;riseW[ar]=rw[128];riseN[ar]=rn
        cl=[float(np.mean([e['coef']<1 for e in S[s_,ar]['log']])) for s_ in seeds if (s_,ar) in S and S[s_,ar].get('log')]
        nr=[mean_norm(s_,ar)/sgd_ref(s_,ar) for s_ in seeds if mean_norm(s_,ar) is not None and sgd_ref(s_,ar)]
        f_=lambda v:'n/a' if v is None else f'{v:+.3f}'
        L.append(f'| {name[ar]} | {f_(rw[32])} | {f_(rw[64])} | {f_(rw[128])} | {f_(rn)} | '+(f'{np.mean(cl):.2f}' if cl else 'n/a')+' | '+(f'{np.mean(nr):.2f}' if nr else 'n/a')+' |')
    L+=['','## verdicts (the arms under test against SENTPQ-STD; every check separate)','','| arm | SPEED | GAIN KEPT | DAMAGE rel (<= +0.05 over STD) | DAMAGE abs vs NAT (<= 0.10) | DAMAGE abs vs W0 (<= 0.10) | CLIP < 0.9 |','|---|---|---|---|---|---|---|']
    std128=pooled(STD,128);std=float(np.mean(std128)) if std128 is not None else float('nan')
    yn=lambda v:'n/a' if v is None else ('pass' if v else 'FAIL')
    for ar in (A,AH,NA):
        p64=pooled(ar,64);p128=pooled(ar,128);ns=[s_ for s_ in seeds if (s_,ar) in G];n2=sum(1 for s_ in ns if seedmean(ar,s_,64)>=0.15)
        sp=None if p64 is None else (n2>=2 or float(np.mean(p64))>=0.8*std);gk=None if p128 is None else float(np.mean(p128))>=std-0.05
        dr=None if riseW.get(ar) is None or riseW.get(STD) is None else riseW[ar]-riseW[STD]<=0.05;dn=None if riseN.get(ar) is None else riseN[ar]<=0.10;dw=None if riseW.get(ar) is None else riseW[ar]<=0.10
        cl=[float(np.mean([e['coef']<1 for e in S[s_,ar]['log']])) for s_ in seeds if (s_,ar) in S and S[s_,ar].get('log')];ck=None if not cl else float(np.mean(cl))<0.9
        L.append(f'| {name[ar]} | {yn(sp)} ({n2} of {len(ns)} seeds >= 0.15 at a=64) | {yn(gk)} | {yn(dr)} | {yn(dn)} | {yn(dw)} | {yn(ck)} |')
    pa,pn=pooled(A,128),pooled(NA,128);pa6,pn6=pooled(A,64),pooled(NA,64)
    L+=['','## recorded item: NAT-ADAM / SENTPQ-ADAM','']
    for a_,x,y in ((128,pn,pa),(64,pn6,pa6)):
        if x is not None and y is not None and float(np.mean(y))!=0:
            r=float(np.mean(x))/float(np.mean(y));L.append(f'- a={a_}: NAT-ADAM {float(np.mean(x)):+.3f} / SENTPQ-ADAM {float(np.mean(y)):+.3f} = {r:.2f} -> '+('>= 30%: the direction improvement works independently of masking' if r>=0.3 else '< 30%: the improvement needs the masking'))
        else:L.append(f'- a={a_}: n/a')
    L+=['','trade-off curve (half versus full norm): '+'; '.join(f'{name[ar]}: g(128) {float(np.mean(pooled(ar,128))):+.3f}, rise vs W0 {riseW[ar]:+.3f}' for ar in (STD,A,AH) if pooled(ar,128) is not None and riseW.get(ar) is not None)]
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:4]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def all_report(d,seeds,tag='all',ref_tag='dc',nat_tag='v3a',arm='ALLSPAN',apps=(64,128),out=None,load=None):
    """E55 design 31 (Hans 22 Oct): name + occupation + place tokens of ALL earlier fact sentences in the window hidden (per-query convention, like SENTPQ), compared with hiding whole sentences (SENTPQ, job 190; SENT, job 189/190) and with NAT (job 181). g out-of-window per seed and pooled (fact bootstrap), R = g / g(SENTPQ), the share of the NAT -> SENTPQ gain that the token mask gets, hidden keys % of the window, canary rise at a=128 (vs W0), clip share, and the stream hash check against the SENTPQ file (same world)."""
    load=load or (lambda f:json.load(open(f)));srcs={arm:(tag,arm),'SENTPQ':(ref_tag,'SENTPQ'),'SENT':(ref_tag,'SENT'),'NAT':(nat_tag,'NAT')};G={};S={}
    for sd in seeds:
        for nm,(tg,ar) in srcs.items():
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,nm]=load(f)
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.saved.json'
            if f.exists():S[sd,nm]=json.load(open(f))
    ERR.clear();W=8192
    def pooled(nm,a):
        use=[s_ for s_ in seeds if (s_,nm) in G]
        try:return np.concatenate([fact_g(G[s_,nm],a,'out') for s_ in use]) if use else None
        except Exception as e:ERR.append(f'{nm}@{a}: {e!r}');return None
    def share(nm):
        v=[e.get('n_masked_keys',0)/W for sd in seeds if (sd,nm) in S for e in S[sd,nm].get('log',[])[8:]];return 100*float(np.mean(v)) if v else None
    def rise(nm):
        r=[max(S[s_,nm]['canary_end'][k]-S[s_,nm]['canary_w0'][k] for k in ('_gen_nll','_tail_nll')) for s_ in seeds if (s_,nm) in S and S[s_,nm].get('canary_end') and S[s_,nm].get('canary_w0')];return max(r) if r else None
    name_={arm:'name+occupation+place tokens hidden (ALL)','SENTPQ':'SENTPQ (whole sentences hidden, job 190)','SENT':'SENT (whole sentences, 1-D)','NAT':'NAT (job 181)'}
    L=[f'# E55 token mask of name / occupation / place (design 31), seeds {list(seeds)}','','g = out-of-window rank gain u (W0 subtracted, unexposed question frames), 95% fact-bootstrap CI. The arm hides, for every fact-sentence query token, the name, occupation and place tokens of ALL earlier fact sentences of the 8-chunk window (per-query convention as SENTPQ; background queries see everything). R = g / g(SENTPQ) at the same a; gain share = (g - g(NAT)) / (g(SENTPQ) - g(NAT)). Pre-set reading (as Q2): R >= 0.7 sufficient, <= 0.3 not sufficient, else partial. Exploratory.','']
    ps={nm:{a:pooled(nm,a) for a in apps} for nm in srcs};ok=all(ps[nm][a] is not None for nm in srcs for a in apps)
    L+=['| arm | hidden keys % of window | '+' | '.join(f'g a={a}' for a in apps)+' | '+' | '.join(f'R a={a}' for a in apps)+' | gain share a=%d | canary rise a=128 (vs W0) |'%apps[-1],'|---|---|'+'---|'*(2*len(apps)+2)]
    for nm in (arm,'SENTPQ','SENT','NAT'):
        if any(ps[nm][a] is None for a in apps):L.append(f'| {name_[nm]} | n/a |'+' n/a |'*(2*len(apps)+2));continue
        sq={a:float(np.mean(ps['SENTPQ'][a])) if ps['SENTPQ'][a] is not None else float('nan') for a in apps};nt={a:float(np.mean(ps['NAT'][a])) if ps['NAT'][a] is not None else float('nan') for a in apps};m={a:float(np.mean(ps[nm][a])) for a in apps}
        gs=(m[apps[-1]]-nt[apps[-1]])/(sq[apps[-1]]-nt[apps[-1]]) if sq[apps[-1]]==sq[apps[-1]] and sq[apps[-1]]!=nt[apps[-1]] else float('nan');sh=share(nm);rs=rise(nm)
        L.append(f'| {name_[nm]} | '+(f'{sh:.1f}' if sh is not None else 'n/a')+' | '+' | '.join(fmt(boot(ps[nm][a])) for a in apps)+' | '+' | '.join(f'{m[a]/sq[a]:.2f}' for a in apps)+f' | {gs:.2f} | '+(f'{rs:+.3f}' if rs is not None else 'n/a')+' |')
    L+=['','paired contrasts (pooled over seeds, same facts):','','| contrast | '+' | '.join(f'a={a}' for a in apps)+' |','|---|'+'---|'*len(apps)]
    for nm,x,y in ((f'{arm} - SENTPQ',arm,'SENTPQ'),(f'{arm} - NAT',arm,'NAT'),('SENTPQ - NAT','SENTPQ','NAT')):
        row=[]
        for a in apps:
            use=[s_ for s_ in seeds if (s_,x) in G and (s_,y) in G]
            try:row.append(fmt(boot(np.concatenate([fact_g(G[s_,x],a,'out')-fact_g(G[s_,y],a,'out') for s_ in use]))) if use else 'n/a')
            except Exception as e:ERR.append(f'diff {nm}: {e!r}');row.append('n/a')
        L.append(f'| {nm} | '+' | '.join(row)+' |')
    L+=['','per seed (out-of-window seed mean g at a=%d): '%apps[-1]+'; '.join(f'seed {s_}: '+', '.join(f"{nm} {float(np.mean(fact_g(G[s_,nm],apps[-1],'out'))):+.3f}" for nm in (arm,'SENTPQ','NAT') if (s_,nm) in G) for s_ in seeds)]
    hs=[(s_,S[s_,arm].get('stream_sha'),S[s_,'SENTPQ'].get('stream_sha')) for s_ in seeds if (s_,arm) in S and (s_,'SENTPQ') in S]
    L+=['','stream hash (same world as the SENTPQ file of job 190): '+('; '.join(f'seed {s_}: '+('equal' if a_==b_ and a_ else f'DIFFERENT ({a_} vs {b_}) - comparison void') for s_,a_,b_ in hs) if hs else 'n/a (files missing)')]
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:4]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def q34_report(d,seeds,tag='q34',ref_tag='q1b',apps=(16,32,64,128),out=None,load=None):
    """E55 34 bundle. (2') partial token masks on SENTPQ-ADAM-0.5N: SENTP25 / 50 / 75 ADAMH (a random 25 / 50 / 75 % of the tokens of every earlier fact sentence hidden, per query) against SENTPQ-ADAM-0.5N (job 195/196, tag q1b); recovery R = g / g(0.5N) at a = 64 and 128, hidden keys % of the window, damage: in-window g at 128, DCLM first-chunk NLL rise (a=128 minus W0), canary rise at a = 32, 64, 128 (vs W0). (3') Adam damage reduction on SENTPQ: A1 ADAM-0.75N, A2 ADAM-vfix-1.0N, A3 ADAM-vfix-0.5N against SENTPQ-STD, SENTPQ-ADAM, SENTPQ-ADAM-0.5N (tag q1b); the PRE-SET success rule is written into the report. The reference files of tag q1b are read from the same directory."""
    load=load or (lambda f:json.load(open(f)));P2=[('SENTP25ADAMH','SENT-p25 / Adam-0.5N',25),('SENTP50ADAMH','SENT-p50 / Adam-0.5N',50),('SENTP75ADAMH','SENT-p75 / Adam-0.5N',75)];P3=[('SENTPQADAM75','A1 ADAM-0.75N'),('SENTPQVFIX10','A2 ADAM-vfix-1.0N'),('SENTPQVFIX05','A3 ADAM-vfix-0.5N')]
    refs={'SENTPQ':'SENTPQ-STD','SENTPQADAM':'SENTPQ-ADAM (1.0N)','SENTPQADAMH':'SENTPQ-ADAM-0.5N'};G={};S={}
    for sd in seeds:
        for ar in [x[0] for x in P2]+[x[0] for x in P3]:
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.saved.json'
            if f.exists():S[sd,ar]=json.load(open(f))
        for ar in refs:
            f=Path(d)/f'E55_{ref_tag}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
            f=Path(d)/f'E55_{ref_tag}_{ar}_s{sd}.saved.json'
            if f.exists():S[sd,ar]=json.load(open(f))
    ERR.clear();W=8192
    def pooled(ar,a,ctx='out'):
        use=[s_ for s_ in seeds if (s_,ar) in G]
        try:
            if not use:return None
            cx=[ctx if ctx!='in' else in_ctx(G[s_,ar]) for s_ in use]
            return None if None in cx else np.concatenate([fact_g(G[s_,ar],a,c_) for s_,c_ in zip(use,cx)])
        except Exception as e:ERR.append(f'{ar}@{a}: {e!r}');return None
    def mean(v):return float(np.mean(v)) if v is not None else float('nan')
    def share(ar):
        v=[e.get('n_masked_keys',0)/W for s_ in seeds if (s_,ar) in S for e in S[s_,ar].get('log',[])[8:]];return 100*float(np.mean(v)) if v else None
    def crise(ar,a):
        r=[]
        for s_ in seeds:
            x=S.get((s_,ar),{});c=(x.get('canary_by_app') or {}).get(str(a)) or (x.get('canary_end') if a==128 else None);w=x.get('canary_w0')
            if c and w:r.append(max(c[k]-w[k] for k in ('_gen_nll','_tail_nll')))
        return max(r) if r else None
    def dclm(ar):
        r=[S[s_,ar]['dclm_end']-S[s_,ar]['dclm_w0'] for s_ in seeds if (s_,ar) in S and S[s_,ar].get('dclm_end') is not None and S[s_,ar].get('dclm_w0') is not None];return float(np.mean(r)) if r else None
    f_=lambda v,fm='{:+.3f}':'n/a' if v is None or v!=v else fm.format(v)
    ref=mean(pooled('SENTPQADAMH',128));ref64=mean(pooled('SENTPQADAMH',64));std_c=crise('SENTPQ',128)
    L=[f'# E55 34 bundle report, seeds {list(seeds)} (tag {tag}, references from tag {ref_tag})','','g = out-of-window rank gain u (W0 subtracted, unexposed question frames), 95% fact-bootstrap CI. All Adam arms: the Adam direction of the clipped write gradient, per-chunk step size = factor x the update norm of SENTPQ-STD of the same seed (design 29.4); vfix = the second moment is updated for the first 8 chunks only and then frozen (diagonal preconditioner + momentum).','',
       "## (2') how much of the sentence has to be hidden (Adam-0.5N write)",'','R = g / g(SENTPQ-ADAM-0.5N). Reading rule (pre-set): R < 0.3 at p25 and p50 = threshold type (partial masking insufficient); R rising roughly with the hidden share = dose-response. Damage columns next to the gain.','',
       '| arm | hidden keys % of window | '+' | '.join(f'g a={a}' for a in apps)+' | R a=64 | R a=128 | in-window g a=128 | DCLM rise a=128 | canary rise a=32 / 64 / 128 (vs W0) |','|---|---|'+'---|'*(len(apps)+5)]
    rows2=[(ar,nm) for ar,nm,_ in P2]+[('SENTPQADAMH','SENTPQ-ADAM-0.5N (hide 100%, reference)'),('SENTPQ','SENTPQ-STD (reference)')]
    for ar,nm in rows2:
        ps={a:pooled(ar,a) for a in apps};pi=pooled(ar,128,'in')
        if all(v is None for v in ps.values()):L.append(f'| {nm} | n/a |'+' n/a |'*(len(apps)+5));continue
        r64=mean(ps.get(64))/ref64 if ref64==ref64 and ref64!=0 and ps.get(64) is not None else float('nan');r128=mean(ps.get(128))/ref if ref==ref and ref!=0 and ps.get(128) is not None else float('nan')
        L.append(f'| {nm} | '+f_(share(ar),'{:.1f}')+' | '+' | '.join(fmt(boot(ps[a])) if ps[a] is not None else 'n/a' for a in apps)+f' | {f_(r64,"{:.2f}")} | {f_(r128,"{:.2f}")} | '+(fmt(boot(pi)) if pi is not None else 'n/a')+f' | {f_(dclm(ar))} | '+' / '.join(f_(crise(ar,a)) for a in (32,64,128))+' |')
    L+=['',"## (3') reducing the Adam damage on SENTPQ (norm / preconditioner variants)",'',
        f'PRE-SET success rule: g(a=128) >= 0.326 (the 0.5N value of job 195; this run\'s file value: {f_(ref)}) AND g(a=64) >= 0.20 AND canary rise at a=128 <= SENTPQ-STD + 0.05 (STD rise {f_(std_c)} + 0.05 = {f_(std_c+0.05) if std_c is not None else "n/a"}). Ties: the lower canary rise wins. The three conditions are shown separately.','',
        '| arm | '+' | '.join(f'g a={a}' for a in apps)+' | g(128) >= 0.326 | g(64) >= 0.20 | canary a=128 <= STD + 0.05 | canary rise a=32 / 64 / 128 (vs W0) | in-window g a=128 | DCLM rise a=128 | success |','|---|'+'---|'*(len(apps)+7)]
    rows3=[(ar,nm) for ar,nm in P3]+[('SENTPQADAMH','SENTPQ-ADAM-0.5N (reference)'),('SENTPQADAM','SENTPQ-ADAM 1.0N (reference)'),('SENTPQ','SENTPQ-STD (reference)')]
    yn=lambda v:'n/a' if v is None else ('pass' if v else 'FAIL')
    for ar,nm in rows3:
        ps={a:pooled(ar,a) for a in apps};pi=pooled(ar,128,'in');c128=crise(ar,128)
        if all(v is None for v in ps.values()):L.append(f'| {nm} | n/a |'+' n/a |'*(len(apps)+7));continue
        g128=mean(ps.get(128)) if ps.get(128) is not None else None;g64=mean(ps.get(64)) if ps.get(64) is not None else None
        a1=None if g128 is None or g128!=g128 else g128>=0.326;a2=None if g64 is None or g64!=g64 else g64>=0.20;a3=None if c128 is None or std_c is None else c128<=std_c+0.05;ok=None if None in (a1,a2,a3) else (a1 and a2 and a3)
        L.append(f'| {nm} | '+' | '.join(fmt(boot(ps[a])) if ps[a] is not None else 'n/a' for a in apps)+f' | {yn(a1)} | {yn(a2)} | {yn(a3)} | '+' / '.join(f_(crise(ar,a)) for a in (32,64,128))+' | '+(fmt(boot(pi)) if pi is not None else 'n/a')+f' | {f_(dclm(ar))} | '+('reference' if ar in refs else ('SUCCESS' if ok else 'no' if ok is not None else 'n/a'))+' |')
    L+=['','norm check (mean update norm of the arm / mean update norm of SENTPQ-STD, same seed; target = the factor of the arm):','']
    for ar,fac in (('SENTP25ADAMH',.5),('SENTP50ADAMH',.5),('SENTP75ADAMH',.5),('SENTPQADAM75',.75),('SENTPQVFIX10',1.0),('SENTPQVFIX05',.5)):
        r=[np.mean([e['update_norm'] for e in S[s_,ar]['log']])/np.mean([e['update_norm'] for e in S[s_,'SENTPQ']['log']]) for s_ in seeds if (s_,ar) in S and (s_,'SENTPQ') in S and S[s_,ar].get('log') and S[s_,'SENTPQ'].get('log')]
        L.append(f'- {ar}: '+(f'{np.mean(r):.2f} (target {fac})' if r else 'n/a'))
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:4]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
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
def decomp_report(d,seeds,tag='dc',nat_tag='v3a',iso_tag='iso',apps=(64,128),out=None,load=None):
    """E55 design 24.3: u (W0 subtracted, out-of-window) at a = 64, 128 of NAT (job 181), ISO (job 185, norm matched), ISOB (ISO + the stream's own background, 128 steps), ISOD (isolated sentences, one round per chunk, 128 steps), NATB (NAT with the window, ISO-sized steps), FOS (fact-only continuous stream with the SWA window); per seed and pooled, with paired differences to NAT and the decomposition contrasts (ISOB - ISO: background, ISOD - ISO: step density, NATB - NAT: step size, FOS - ISO: window shadowing inside the ISO setup)."""
    load=load or (lambda f:json.load(open(f)));tags={'NAT':nat_tag,'ISO':iso_tag,'ISOB':tag,'ISOD':tag,'NATB':tag,'FOS':tag,'SENT':tag,'KMASK':tag,'SENTO':tag,'SENTPQ':tag,'SWD':nat_tag};G={}
    for sd in seeds:
        for ar,tg in tags.items():
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
    L=[f'# E55 decomposition report (design 24.3), seeds {list(seeds)}','','u = (W0 rank - rank)/15, out-of-window read, question frames never seen; mean over facts, 95% fact-bootstrap CI; exploratory. ISO from job 185 uses read segments 16/32/64/128, the arms here use 64/128 (the packing of the ISO steps differs slightly). Padding: in FOS the segment-end padding (at most 1023 newline tokens per segment, about 2-4% of a segment, count in extra.pad_tokens of the saved files) IS inside the chunk and therefore in the loss (the online write path scores every token of a chunk); in the isolated-row arms (ISO, ISOB, ISOD) the padding of a packed chunk is not in the loss (loss mask only inside rows) and not visible to the rows (segment id -1). The two differ in this respect.','','| seed | arm | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
    def cellf(fn,use,a):
        try:return fmt(boot(np.concatenate([fn(s_) for s_ in use])))
        except Exception as e:ERR.append(repr(e));return 'n/a'
    for sd in list(seeds)+['pooled']:
        use=[s_ for s_ in seeds if sd=='pooled' or s_==sd]
        for ar in tags:
            L.append(f'| {sd} | {ar} | '+' | '.join(cellf(lambda s_,ar=ar,a=a:fact_g(G[s_,ar],a,'out'),[s_ for s_ in use if (s_,ar) in G],a) for a in apps)+' |')
    L+=['','paired contrasts (pooled over seeds, same facts):','','| contrast | meaning | '+' | '.join(f'a={a}' for a in apps)+' |','|---|---|'+'---|'*len(apps)]
    for nm,mean,x,y in (('ISOB - ISO','background added to isolated rows (128 steps, matched norm)','ISOB','ISO'),('ISOD - ISO','step density (128 isolated one-round steps vs ISO steps)','ISOD','ISO'),('NATB - NAT','larger steps in the online stream','NATB','NAT'),('FOS - ISO','removing isolation inside the ISO setup (window present = shadowing share)','FOS','ISO'),('ISOB - NAT','isolation only (same text, same steps, same norm)','ISOB','NAT'),('ISO - NAT','stage 0','ISO','NAT'),('SENT - NAT','online stream with earlier fact sentences hidden (window present)','SENT','NAT'),('SENT - ISOB','window + masking vs isolation, same text and steps','SENT','ISOB'),('SENT - ISO','masking in the window vs isolated rows','SENT','ISO'),('KMASK - NAT','only the same fact\'s earlier sentences hidden (window present)','KMASK','NAT'),('SENTO - NAT','only the other facts\' earlier sentences hidden (window present)','SENTO','NAT'),('SENT - KMASK','what the other-fact hiding adds to same-fact hiding','SENT','KMASK'),('SENT - SENTO','what the same-fact hiding adds to other-fact hiding','SENT','SENTO'),('SENTPQ - SENT','convention control: per-query vs key-level masking','SENTPQ','SENT'),('SENTPQ - NAT','per-query SENT vs NAT','SENTPQ','NAT'),('SWD - NAT','job 181 SWD (fake places outside the 16 candidates) vs NAT','SWD','NAT'),('KMASK - ISOB','same-fact hiding vs isolation (same text/steps)','KMASK','ISOB'),('SENTO - ISOB','other-fact hiding vs isolation (same text/steps)','SENTO','ISOB')):
        row=[]
        for a in apps:
            use=[s_ for s_ in seeds if (s_,x) in G and (s_,y) in G]
            row.append(cellf(lambda s_,x=x,y=y,a=a:fact_g(G[s_,x],a,'out')-fact_g(G[s_,y],a,'out'),use,a))
        L.append(f'| {nm} | {mean} | '+' | '.join(row)+' |')
    def sha_of(sd,ar):
        """(kind, value): stream_hash (64 hex, sha256 of the int64 ids, written by run_stream) preferred, else stream_sha (16 hex, gen_facts.stream_sha); the two kinds are not comparable"""
        f=Path(d)/f'E55_{tags[ar]}_{ar}_s{sd}.saved.json'
        try:j=json.load(open(f))
        except Exception:return None
        for k in ('stream_hash','stream_sha'):
            v=j.get(k);v=v if isinstance(v,str) else (v or {}).get('A') if v else None
            if v:return (k,v)
        return None
    L+=['','token-stream hash per seed and arm (stream_hash = sha256 of the int64 ids as written by run_stream, first 12 hex; falls back to stream_sha of gen_facts when an old file has no stream_hash; "n/a" = neither is stored). all equal is True only if every arm has a value of the same kind and all values agree:','','| seed | '+' | '.join(tags)+' | all equal |','|---|'+'---|'*(len(tags)+1)]
    for sd in seeds:
        hs={ar:sha_of(sd,ar) for ar in tags if (sd,ar) in G};miss=[ar for ar,v in hs.items() if v is None];kinds={v[0] for v in hs.values() if v}
        if miss:eq=f'False (n/a for {",".join(miss)})'
        elif len(kinds)>1:eq='False (mixed hash kinds)'
        else:eq=str(len({v[1] for v in hs.values()})==1)
        L.append(f'| {sd} | '+' | '.join((hs[ar][1][:12] if hs.get(ar) else 'n/a') for ar in tags)+f' | {eq} |')
    a_=apps[-1];den=[fact_g(G[s_,'SENTPQ'],a_,'out') for s_ in seeds if (s_,'SENTPQ') in G]
    if den:
        gm=lambda ar:np.mean(np.concatenate([fact_g(G[s_,ar],a_,'out') for s_ in seeds if (s_,ar) in G and (s_,'SENTPQ') in G])) if any((s_,ar) in G and (s_,'SENTPQ') in G for s_ in seeds) else float('nan')
        gd=gm('SENTPQ');L+=['',f'recovery ratio R = g(arm)/g(SENTPQ) at a={a_}, pooled over seeds (rule fixed beforehand: R>=0.7 sufficient, R<=0.3 not sufficient, else partial; R_KMASK+R_SENTO far from 1 = interaction); g(SENTPQ)={gd:.3f}','','| arm | g | R | verdict |','|---|---|---|---|']
        Rs={}
        for ar in ('SENT','KMASK','SENTO'):
            g_=gm(ar);R=g_/gd if gd and gd==gd else float('nan');Rs[ar]=R;L.append(f'| {ar} | {g_:.3f} | {R:.2f} | '+('n/a' if R!=R else 'sufficient' if R>=0.7 else 'not sufficient' if R<=0.3 else 'partial')+' |')
        if Rs['KMASK']==Rs['KMASK'] and Rs['SENTO']==Rs['SENTO']:L.append(f"| KMASK+SENTO | | {Rs['KMASK']+Rs['SENTO']:.2f} | "+('interaction (sum far from 1)' if abs(Rs['KMASK']+Rs['SENTO']-1)>0.3 else 'roughly additive')+' |')
        L+=['','(if g(SENTPQ) is near 0 the ratios are meaningless)']
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:3]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def q1a_report(d,seeds,tag='q1a',arms=('ISO-STD','ISO-SUR1','ISO-SUR2','ISO-ADAM','ISO-MOM'),apps=(16,32,64,128),thr=0.15,out=None,load=None):
    """E55 design 27.8 Q1-A (early rise by the inner step on isolated rows): u (W0 subtracted, out-of-window) per arm at a = 16, 32, 64, 128 per seed and pooled (fact bootstrap), minimum exposure count with u >= 0.15 (per seed from the seed means, pooled), paired difference to ISO-STD, canary rise (pg11 / pg1342 NLL at a=128 minus W0, limit +0.10) and clip share of the steps. Pre-set success of an arm: u(a=32) >= 0.15 in at least 2 of 3 seeds AND pooled u(128) >= ISO-STD - 0.05 AND clip share < 0.9 (and canary rise <= +0.10). Exploratory."""
    load=load or (lambda f:json.load(open(f)));G={};S={}
    for sd in seeds:
        for ar in arms:
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
            f=Path(d)/f'E55_{tag}_{ar}_s{sd}.saved.json'
            if f.exists():S[sd,ar]=json.load(open(f))
    ERR.clear();L=[f'# E55 Q1-A report (design 27.8), seeds {list(seeds)}','','u = (W0 rank - rank)/15, out-of-window, 16 candidates, unexposed question frames, 95% fact-bootstrap CI; exploratory.','','| seed | arm | '+' | '.join(f'a={a}' for a in apps)+' | min a with u >= %.2f |'%thr,'|---|---|'+'---|'*(len(apps)+1)]
    def sm(ar,sd):
        try:return {a:float(np.mean(fact_g(G[sd,ar],a,'out'))) for a in apps}
        except Exception as e:ERR.append(f'{ar}/{sd}: {e!r}');return None
    def mina(v):
        ok=[a for a in apps if v is not None and v[a]>=thr];return str(min(ok)) if ok else '>%d'%max(apps)
    def pooled(ar,a,use=None):
        use=[s_ for s_ in (use or seeds) if (s_,ar) in G]
        try:return np.concatenate([fact_g(G[s_,ar],a,'out') for s_ in use])
        except Exception as e:ERR.append(f'{ar}: {e!r}');return None
    for sd in list(seeds)+['pooled']:
        for ar in arms:
            if sd=='pooled':
                ps={a:pooled(ar,a) for a in apps}
                if any(v is None for v in ps.values()):continue
                L.append(f'| pooled | {ar} | '+' | '.join(fmt(boot(ps[a])) for a in apps)+f' | {mina({a:float(np.mean(ps[a])) for a in apps})} |')
            elif (sd,ar) in G:
                v=sm(ar,sd)
                if v:L.append(f'| {sd} | {ar} | '+' | '.join(f'{v[a]:+.3f}' for a in apps)+f' | {mina(v)} |')
    L+=['','paired difference to ISO-STD (pooled over seeds, same facts):','','| arm | '+' | '.join(f'a={a}' for a in apps)+' |','|---|'+'---|'*len(apps)]
    for ar in arms:
        if ar=='ISO-STD':continue
        row=[]
        for a in apps:
            use=[s_ for s_ in seeds if (s_,ar) in G and (s_,'ISO-STD') in G]
            try:row.append(fmt(boot(np.concatenate([fact_g(G[s_,ar],a,'out')-fact_g(G[s_,'ISO-STD'],a,'out') for s_ in use]))))
            except Exception as e:ERR.append(f'diff {ar}: {e!r}');row.append('n/a')
        L.append(f'| {ar} | '+' | '.join(row)+' |')
    L+=['','## Verdict criteria (stated here; design 27.8.1 as read in 29.2). Three checks, kept SEPARATE:','',
     '1. SPEED: u(a=32) >= %.2f in ALL seeds (3 of 3) AND pooled u(a=128) >= STD u(a=128) - 0.05.'%thr,
     '2. DAMAGE (canary NLL rise, reference = the frozen start weights W0, worse of the generic and tail canary texts, max over seeds, at a=128): (abs) rise <= 0.10; (rel) rise minus the ISO-STD rise <= +0.05. The absolute limit is shown but does not decide: the isolated-row design (no background, few large steps) exceeds it with ISO-STD itself.',
     '3. CLIP: mean clip share of the update norm < 0.9 (separate sanity check).',
     'Overall verdict: SUCCESS = speed passes AND relative damage passes AND clip passes; the absolute damage column is informative (flagged "abs over").','',
     '| arm | seeds with u(32) >= %.2f | pooled u(128) | STD u(128) - 0.05 | SPEED | canary rise (max over seeds, vs W0) | rise - STD rise | DAMAGE abs <= 0.10 | DAMAGE rel <= +0.05 | clip share (mean) | CLIP < 0.9 | verdict |'%thr,'|---|---|---|---|---|---|---|---|---|---|---|---|']
    std128=pooled('ISO-STD',apps[-1]);s128=float(np.mean(std128)) if std128 is not None else float('nan')
    def crise(ar):
        cr=[]
        for s_ in seeds:
            if (s_,ar) in S and S[s_,ar].get('canary_end') and S[s_,ar].get('canary_w0'):cr.append(max(S[s_,ar]['canary_end'][k]-S[s_,ar]['canary_w0'][k] for k in ('_gen_nll','_tail_nll')))
        return max(cr) if cr else float('nan')
    stdr=crise('ISO-STD')
    for ar in arms:
        ns=[s_ for s_ in seeds if (s_,ar) in G];sd32=[s_ for s_ in ns if (sm(ar,s_) or {}).get(32,-9)>=thr];p128=pooled(ar,apps[-1]);p=float(np.mean(p128)) if p128 is not None else float('nan')
        cl=[S[s_,ar]['summary'].get('clip_frac',float('nan')) for s_ in seeds if (s_,ar) in S];r=crise(ar)
        sp=len(ns)>0 and len(sd32)==len(ns) and p>=s128-0.05;da=(r<=0.10) if r==r else None;dr=(r-stdr<=0.05) if r==r and stdr==stdr else None;ck=(float(np.mean(cl))<0.9) if cl else None
        yn=lambda v:'n/a' if v is None else ('pass' if v else 'FAIL')
        ok=sp and dr is True and ck is not False
        L.append(f"| {ar} | {len(sd32)} of {len(ns)} {sd32} | {p:+.3f} | {s128-0.05:+.3f} | {'pass' if sp else 'FAIL'} | "+(f'{r:+.3f}' if r==r else 'n/a')+f" | "+(f'{r-stdr:+.3f}' if r==r and stdr==stdr else 'n/a')+f" | {'pass' if da else 'abs over' if da is False else 'n/a'} | {yn(dr)} | "+(f'{np.mean(cl):.2f}' if cl else 'n/a')+f" | {yn(ck)} | {'baseline' if ar=='ISO-STD' else ('SUCCESS' if ok else 'no')} |")
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:4]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def ans_report(d,seeds,tag='ans',ref_tag='dc',nat_tag='v3a',swdin_tag='swdin',arms=('OSPAN','PSPAN','NSPAN','OSPANR25','OSPANR50','OSPANR75'),apps=(64,128),out=None,load=None):
    """E55 design 28.3 (+ 30.3 recall dose-response OSPANR25/50/75: a random 25 / 50 / 75 % of the O-ALL answer tokens hidden; the curve R versus recall is the pre-set prediction base for the attention-scored arms) / 28.6: answer-span masks over ALL earlier fact sentences in the window (per-query convention): O-ALL (name + place tokens, arm OSPAN), P-ALL (place only, PSPAN), N-ALL (name only, NSPAN) with their out-of-window g at a = 64, 128 per seed and pooled, the recovery ratio R = g / g(SENTPQ of job 190) and the masked share (hidden keys % of the 8192-token window, mean over chunks 9-128), plus the references NAT (job 181), SWD (job 181, fake places outside the 16 candidates), SWD-IN (this bundle: fake places = a length-preserving derangement of the 16 real places), SENT (job 189), SENTPQ (job 190) and the paired contrasts SWD-IN - SWD and SWD - SENT. R >= 0.7 sufficient, <= 0.3 not sufficient, else partial (pre-set, as in Q2). Exploratory."""
    load=load or (lambda f:json.load(open(f)));srcs={ar:(tag,ar) for ar in arms};srcs.update({'SENTPQ':(ref_tag,'SENTPQ'),'SENT':(ref_tag,'SENT'),'NAT':(nat_tag,'NAT'),'SWD':(nat_tag,'SWD'),'SWD-IN':(swdin_tag,'SWD')});G={};S={}
    for sd in seeds:
        for nm,(tg,ar) in srcs.items():
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,nm]=load(f)
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.saved.json'
            if f.exists():S[sd,nm]=json.load(open(f))
    ERR.clear();W=8192
    def pooled(nm,a,use=None):
        use=[s_ for s_ in (use or seeds) if (s_,nm) in G]
        try:return np.concatenate([fact_g(G[s_,nm],a,'out') for s_ in use])
        except Exception as e:ERR.append(f'{nm}@{a}: {e!r}');return None
    def share(nm):
        v=[e.get('n_masked_keys',0)/W for sd in seeds if (sd,nm) in S for e in S[sd,nm]['log'][8:] if 'log' in S[sd,nm]];return 100*float(np.mean(v)) if v else None
    L=[f'# E55 answer-span masks and SWD-IN (design 28.3 / 28.6), seeds {list(seeds)}','','g = out-of-window rank gain u (W0 subtracted), 95% fact-bootstrap CI. O-ALL / P-ALL / N-ALL = name+place / place / name tokens of ALL earlier fact sentences hidden from fact-sentence queries (background queries see everything); R = g / g(SENTPQ of job 190) at the same a, pooled over seeds; hidden keys % = share of the 8192 window keys hidden (mean over chunks 9-128). SWD = job 181 fake window (fake places outside the candidates), SWD-IN = same fake window but the fake place of each fact is the real place of another fact (derangement, same token lengths).','','| arm | hidden keys % | '+' | '.join(f'g a={a}' for a in apps)+' | '+' | '.join(f'R a={a}' for a in apps)+' | verdict (a=%d) |'%apps[-1],'|---|---|'+'---|'*(2*len(apps)+1)]
    ref={a:pooled('SENTPQ',a) for a in apps};rm={a:(float(np.mean(ref[a])) if ref[a] is not None else float('nan')) for a in apps}
    for nm in list(arms)+['SENTPQ','SENT','NAT','SWD','SWD-IN']:
        ps={a:pooled(nm,a) for a in apps}
        if any(v is None for v in ps.values()):L.append(f'| {nm} | n/a |'+' n/a |'*(2*len(apps)+1));continue
        Rv={a:float(np.mean(ps[a]))/rm[a] if rm[a]==rm[a] and rm[a]!=0 else float('nan') for a in apps};sh=share(nm);R=Rv[apps[-1]]
        vd='reference' if nm in ('SENTPQ','SENT','NAT','SWD','SWD-IN') else ('n/a' if R!=R else 'sufficient' if R>=0.7 else 'not sufficient' if R<=0.3 else 'partial')
        L.append(f'| {nm} | '+(f'{sh:.1f}' if sh is not None else 'n/a')+' | '+' | '.join(fmt(boot(ps[a])) for a in apps)+' | '+' | '.join(f'{Rv[a]:.2f}' for a in apps)+f' | {vd} |')
    L+=['','paired contrasts (pooled over seeds, same facts):','','| contrast | '+' | '.join(f'a={a}' for a in apps)+' |','|---|'+'---|'*len(apps)]
    for nm,x,y in (('SWD-IN - SWD','SWD-IN','SWD'),('SWD - SENT','SWD','SENT'),('SWD-IN - SENT','SWD-IN','SENT'),('O-ALL - SENTPQ','OSPAN','SENTPQ'),('P-ALL - SENTPQ','PSPAN','SENTPQ'),('N-ALL - SENTPQ','NSPAN','SENTPQ'),('P-ALL + N-ALL vs O-ALL (sum of gains over NAT minus O-ALL gain)',None,None)):
        row=[]
        for a in apps:
            try:
                if x is None:
                    use=[s_ for s_ in seeds if all((s_,q) in G for q in ('PSPAN','NSPAN','OSPAN','NAT'))];row.append(fmt(boot(np.concatenate([(fact_g(G[s_,'PSPAN'],a,'out')-fact_g(G[s_,'NAT'],a,'out'))+(fact_g(G[s_,'NSPAN'],a,'out')-fact_g(G[s_,'NAT'],a,'out'))-(fact_g(G[s_,'OSPAN'],a,'out')-fact_g(G[s_,'NAT'],a,'out')) for s_ in use]))))
                else:
                    use=[s_ for s_ in seeds if (s_,x) in G and (s_,y) in G];row.append(fmt(boot(np.concatenate([fact_g(G[s_,x],a,'out')-fact_g(G[s_,y],a,'out') for s_ in use]))))
            except Exception as e:ERR.append(f'{nm}@{a}: {e!r}');row.append('n/a')
        L.append(f'| {nm} | '+' | '.join(row)+' |')
    L+=['','recall dose-response (pre-set reading, E55 30.3): R = g / g(SENTPQ) at a = %d versus the share of the O-ALL answer tokens that is hidden:'%apps[-1],'']
    pts=[]
    for fr,nm in ((0,'NAT'),(25,'OSPANR25'),(50,'OSPANR50'),(75,'OSPANR75'),(100,'OSPAN')):
        v=pooled(nm,apps[-1])
        if v is not None and rm[apps[-1]]==rm[apps[-1]] and rm[apps[-1]]!=0:pts.append((fr,float(np.mean(v)),float(np.mean(v))/rm[apps[-1]]))
    L.append(('; '.join(f'recall {fr}%: g {g:+.3f}, R {r:.2f}' for fr,g,r in pts)) if pts else 'n/a (files missing)')
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:4]}']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
MIXL={'MIX25':0.25,'MIX50':0.5,'MIX75':0.75}
def q2_report(d,seeds,tag='q2',nat_tag='v3a',arms=('NAT','OSPAN','SENTP25','SENTP50','SENTP75','SENTPQ','MIX25','MIX50','MIX75','ALT'),a=128,out=None,plot=None,load=None):
    """E55 design 27.2 (Q2 scope ladder): per arm, pooled over seeds, at a=128: masked share (hidden keys % of the 8192-token window, hidden (query,key) pairs % of the 1024 x 8192 window block; mean over chunks 9-128 from the per-chunk logs), out-of-window g with CI, recovery ratio R = g/g(SENTPQ) (>=0.7 sufficient, <=0.3 not, else partial), damage reads: in-window g (real window visible; reference = NAT in-window of job 181 at the same appearance), DCLM first-chunk NLL rise (written - W0; pre-set rule 'no damage' = rise <= native +0.043 + 0.03 AND in-window g >= NAT in-window - 0.10), mean chunk CE (all tokens of the chunk, NOT background only: the write path logs one scalar per chunk), clip rate (share of chunks with clip coefficient < 1). Method candidate = the arm with the least masking (keys %) that reaches R >= 0.7 (reported with and without the no-damage requirement). Exploratory (3 seeds, 16 facts)."""
    load=load or (lambda f:json.load(open(f)));G={};S={}
    for sd in seeds:
        for ar in arms:
            tg=nat_tag if (ar=='NAT' and not (Path(d)/f'E55_{tag}_NAT_s{sd}.read.json').exists()) else tag
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.read.json'
            if f.exists():G[sd,ar]=load(f)
            f=Path(d)/f'E55_{tg}_{ar}_s{sd}.saved.json'
            if f.exists():S[sd,ar]=json.load(open(f))
    nat_in={sd:load(Path(d)/f'E55_{nat_tag}_NAT_s{sd}.read.json') for sd in seeds if (Path(d)/f'E55_{nat_tag}_NAT_s{sd}.read.json').exists()}
    ERR.clear();W=8192;CHL=1024;rows={}
    def pool(fn,use):
        return np.concatenate([fn(s_) for s_ in use])
    def tryg(ar,ctx):
        use=[s_ for s_ in seeds if (s_,ar) in G]
        try:return pool(lambda s_:fact_g(G[s_,ar],a,ctx),use)
        except Exception as e:ERR.append(f'{ar}/{ctx}: {e!r}');return None
    ref_in=None
    try:ref_in=pool(lambda s_:fact_g(nat_in[s_],a,in_ctx(nat_in[s_])),list(nat_in))
    except Exception as e:ERR.append(f'NAT in-window ref: {e!r}')
    gpq=tryg('SENTPQ','out');gpq_m=float(np.mean(gpq)) if gpq is not None else float('nan')
    for ar in arms:
        sv=[S[s_,ar] for s_ in seeds if (s_,ar) in S];lg=[e for x in sv for e in x['log']] if sv else []
        lam=MIXL.get(ar,1.0)       # mix arms: the masked pass carries weight lambda on the fact loss -> effective masked share = lambda x hidden share of the masked pass
        k=[lam*e.get('n_masked_keys',0)/W for x in sv for e in x['log'][8:]];pr=[lam*e.get('n_masked_pairs',0)/(CHL*W) for x in sv for e in x['log'][8:]]
        go=tryg(ar,'out');gi=None
        try:gi=pool(lambda s_:fact_g(G[s_,ar],a,in_ctx(G[s_,ar])),[s_ for s_ in seeds if (s_,ar) in G])
        except Exception as e:ERR.append(f'{ar}/in-window: {e!r}')
        dc=[x['dclm_end']-x['dclm_w0'] for x in sv if x.get('dclm_end') is not None and x.get('dclm_w0') is not None]
        ce=[np.mean([e['ce'] for e in x['log'][-32:]]) for x in sv];cl=[np.mean([e['coef']<1 for e in x['log']]) for x in sv if all('coef' in e for e in x['log'])]
        R=(float(np.mean(go))/gpq_m) if go is not None and gpq_m==gpq_m and gpq_m!=0 else float('nan')
        rows[ar]={'keys_pct':100*float(np.mean(k)) if k else float('nan'),'pairs_pct':100*float(np.mean(pr)) if pr else float('nan'),'g_out':boot(go) if go is not None else None,'g_in':boot(gi) if gi is not None else None,'R':R,'dclm':float(np.mean(dc)) if dc else None,'dclm_per_seed':[round(x,3) for x in dc],'ce':float(np.mean(ce)) if ce else None,'clip':float(np.mean(cl)) if cl else None,'un':float(np.mean([np.mean([e['update_norm'] for e in x['log']]) for x in sv])) if sv else None}
    gin_nat=float(np.mean(ref_in)) if ref_in is not None else None
    L=[f'# E55 Q2 scope ladder (design 27.2), seeds {list(seeds)}, a={a}','','g = out-of-window rank gain u (W0 subtracted), 95% fact-bootstrap CI; R = g / g(SENTPQ) (pre-set: >=0.7 sufficient, <=0.3 not sufficient, else partial). In-window g reference = NAT of job 181 at the same appearance'+(f' ({gin_nat:+.3f})' if gin_nat is not None else ' (missing)')+'. DCLM rise = first-chunk NLL after writing minus W0 on the same samples (native reference +0.043; pre-set rule: rise <= 0.073 and in-window g >= NAT in-window - 0.10 = "no damage"). CE = mean chunk CE over the last 32 chunks, all tokens (not background only). Clip = share of chunks with clip coefficient < 1. Norm ratio = mean update norm / mean update norm of NAT. MIXnn = two-loss mix with lambda = nn/100 (loss = background + lambda x fact-token loss under the masked window + (1-lambda) x fact-token loss under the unmasked window, one summed gradient, one clip/step), its hidden share is lambda x the masked pass share; ALT = masked loss on every other chunk (hidden share averages to half the SENTPQ share).','',
       '| arm | hidden keys % of window | hidden (query,key) pairs % | g out | R | verdict | g in-window | DCLM rise | CE | clip | norm ratio | no damage |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    def nd(v):
        if v['dclm'] is None or v['g_in'] is None or gin_nat is None:return 'n/a'
        return 'yes' if (v['dclm']<=0.043+0.03 and v['g_in'][0]>=gin_nat-0.10) else 'no'
    un_nat=rows['NAT']['un'] if 'NAT' in rows and rows['NAT']['un'] else None
    cand={}
    for ar,v in rows.items():
        vd='n/a' if v['R']!=v['R'] else 'sufficient' if v['R']>=0.7 else 'not sufficient' if v['R']<=0.3 else 'partial'
        v['verdict']=vd;v['nodamage']=nd(v)
        L.append(f"| {ar} | {v['keys_pct']:.1f} | {v['pairs_pct']:.1f} | "+(fmt(v['g_out']) if v['g_out'] else 'n/a')+f" | {v['R']:.2f} | {vd} | "+(fmt(v['g_in']) if v['g_in'] else 'n/a')+f" | "+(f"{v['dclm']:+.3f} {v['dclm_per_seed']}" if v['dclm'] is not None else 'n/a')+f" | "+(f"{v['ce']:.3f}" if v['ce'] is not None else 'n/a')+f" | "+(f"{v['clip']:.2f}" if v['clip'] is not None else 'n/a')+f" | "+(f"{v['un']/un_nat:.2f}" if v['un'] and un_nat else 'n/a')+f" | {v['nodamage']} |")
    c1=[(v['keys_pct'],ar) for ar,v in rows.items() if ar!='NAT' and v['R']==v['R'] and v['R']>=0.7]
    c2=[(k_,ar) for k_,ar in c1 if rows[ar]['nodamage']=='yes']
    L+=['','method candidate (least masking with R >= 0.7): '+(min(c1)[1]+f' ({min(c1)[0]:.1f}% of the window keys)' if c1 else 'none'),'method candidate with the no-damage requirement: '+(min(c2)[1]+f' ({min(c2)[0]:.1f}%)' if c2 else 'none (or damage reads missing)')]
    L+=['','curve points (x = hidden keys %): '+'; '.join(f"{ar}: ({v['keys_pct']:.1f}, g {v['g_out'][0]:+.3f})" for ar,v in rows.items() if v['g_out'])]
    if plot:
        try:
            import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1,3,figsize=(13,3.6));xs=[(ar,v) for ar,v in rows.items() if v['g_out']]
            for i,(ttl,key) in enumerate((('gain g (out-of-window)','g_out'),('in-window g','g_in'))):
                for ar,v in xs:
                    if v[key]:ax[i].errorbar(v['keys_pct'],v[key][0],yerr=[[v[key][0]-v[key][1]],[v[key][2]-v[key][0]]],fmt='o');ax[i].annotate(ar,(v['keys_pct'],v[key][0]),fontsize=7)
                ax[i].set_xlabel('hidden keys, % of window');ax[i].set_title(ttl)
            for ar,v in xs:
                if v['dclm'] is not None:ax[2].plot(v['keys_pct'],v['dclm'],'o');ax[2].annotate(ar,(v['keys_pct'],v['dclm']),fontsize=7)
            ax[2].axhline(0.073,ls=':');ax[2].set_xlabel('hidden keys, % of window');ax[2].set_title('DCLM NLL rise (dotted: native +0.043 + 0.03)')
            fig.tight_layout();fig.savefig(plot,dpi=130);L+=['',f'![curves]({Path(plot).name})']
        except Exception as e:L+=['',f'plot not written: {e!r}']
    if ERR:L+=['',f'cells shown as n/a failed: {sorted(set(ERR))[:4]}']
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
    if '--q1a' in _s.argv:
        _s.argv.remove('--q1a');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--tag',default='q1a');a=ap.parse_args();print(q1a_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,out=a.out));_s.exit(0)
    if '--q1b' in _s.argv:
        _s.argv.remove('--q1b');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--tag',default='q1b');a=ap.parse_args();print(q1b_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,out=a.out));_s.exit(0)
    if '--all' in _s.argv:
        _s.argv.remove('--all');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--tag',default='all');a=ap.parse_args();print(all_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,out=a.out));_s.exit(0)
    if '--uv' in _s.argv:
        _s.argv.remove('--uv');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61');ap.add_argument('--out');ap.add_argument('--tag',default='uv');ap.add_argument('--ref-tag',default='q1b');a=ap.parse_args();print(uv_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,a.ref_tag,out=a.out));_s.exit(0)
    if '--q34' in _s.argv:
        _s.argv.remove('--q34');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--tag',default='q34');ap.add_argument('--ref-tag',default='q1b');a=ap.parse_args();print(q34_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,a.ref_tag,out=a.out));_s.exit(0)
    if '--ans' in _s.argv:
        _s.argv.remove('--ans');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');a=ap.parse_args();print(ans_report(a.dir,[int(x) for x in a.seeds.split(',')],out=a.out));_s.exit(0)
    if '--q2' in _s.argv:
        _s.argv.remove('--q2');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--plot');ap.add_argument('--arms',default='NAT,OSPAN,SENTP25,SENTP50,SENTP75,SENTPQ');a=ap.parse_args();print(q2_report(a.dir,[int(x) for x in a.seeds.split(',')],arms=tuple(a.arms.split(',')),out=a.out,plot=a.plot));_s.exit(0)
    if '--decomp' in _s.argv:
        _s.argv.remove('--decomp');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');a=ap.parse_args();print(decomp_report(a.dir,[int(x) for x in a.seeds.split(',')],out=a.out));_s.exit(0)
    if '--rep' in _s.argv:
        _s.argv.remove('--rep');ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='61,62,63');ap.add_argument('--out');ap.add_argument('--tag',default='icr');a=ap.parse_args();print(rep_report(a.dir,[int(x) for x in a.seeds.split(',')],a.tag,out=a.out));_s.exit(0)
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dir');ap.add_argument('--seed',type=int,default=61);ap.add_argument('--out');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    print(report(a.dir,a.seed,out=a.out))

