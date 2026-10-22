"""E55 revised metric system (design 3.1b / 4c, registered 2026-10-22 14:30 KST before any result), numpy only, computed from the raw 16-candidate score matrices of the e55_gpu records ({'meta','read','W0'}; see e55_gpu.py).
Per item and world w (A: bound target = gi, partner = fi; B: bound target = fi, partner = gi), candidates = 16 exposed places of the world, rank = 1 + #(candidates with a strictly higher score):
  PM1 mean rank of the bound target (chance 8.5; reported as the gain G_rank = rank(W0) - rank(W), positive = better), PM2 top-1 gain = top1(W) - top1(W0) (chance 1/16), PM3 = [score(target) - mean score of the other 15] (W) - same (W0).
  Partner stratification (W and W0 and change): mean rank of the partner place vs mean rank of the other 14 non-target candidates; share of wrong top-1 answers that is the partner (uniform: 1/15); share of items with the target ranked above the partner (uniform: 0.5).
  Secondary (always alongside): D_alt = score(target) - score(partner) averaged over A and B (nats), 2-candidate accuracy (target > partner).
Two aggregations (design 3.1b proposal): (i) 'w0sub': per world change vs the SAME world's W0, averaged over A and B; (ii) 'AB': the counterfactual A-B contrast of the same candidate strings, [(f_A(gi) - f_B(gi)) - (f_A(fi) - f_B(fi))] / 2 (sign so that positive = better, i.e. in rank units [(r_B(gi) - r_A(gi)) + (r_A(fi) - r_B(fi))] / 2); it needs no W0 for 'out' (the W0 reads of A and B are the same there) and cancels any binding-independent change. The two are reported side by side.
Unit = pair (its two names, worlds A and B averaged), several seeds of a cell are concatenated; CI = 95% pair bootstrap (4000, seed 55); contrasts between cells of the SAME world (NAT vs SWD, window vs out) are paired by pair, contrasts between cells of different worlds (different p) are unpaired.
Thresholds of 4c (rank units): binding / shadowing G_rank >= 1.5 with CI excluding 0, PM2 >= +0.05, PM3 same sign; p = 9 negative control |diff| <= 1 with CI containing 0; prediction A (window - out) >= +3 ranks with CI excluding 0 and PM2 >= +0.10 for p in {1,2,4}, 'similar' = CI of the p=1 vs p=4 difference inside +-1.5, p=9 control |diff| <= 1; prediction B 'flat' = 90% CI of NAT(p) - NAT(9) inside +-1.5 for p in {1,2,4} AND NAT above chance by >= 1.5 ranks, 'rise under masking' = SWD(1) - SWD(9) >= +1.5 with CI excluding 0, point order SWD(1) >= SWD(2) >= SWD(4) >= SWD(9), PM2 same sign. A verdict that PM2 or PM3 disagrees in sign with PM1 is 'metric mismatch' and is NOT accepted. Single-seed resolution (3.1b): only large effects (about 2 ranks at N = 16) can be judged; equivalence ('flat') cannot be shown with one seed and is reported as 'cannot be judged'.
Window readout (rule 9 in rank units): the window effect of a context is read on the items whose pair has information distance d in 1..8 in that context (meta['d']); Delta_exit = rank(W, out) - rank(W, in) on the same items (raw, as rule 9 (ii) 'metric(W, in window) - metric(W, out)'), 'write share' = G_rank(W, in) - G_rank(W0-subtracted, same context) i.e. rank(W0, in) - rank(W, in) beyond W0 is reported separately; the W0-relative window effect G_in - G_out is reported alongside."""
import json,sys,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e49_stats as ES
NB=4000;NC=16
def ci(v,nb=NB,level=95):
    v=np.asarray(v,float);rng=np.random.default_rng(55);b=v[rng.integers(0,len(v),(nb,len(v)))].mean(1);lo=(100-level)/2;return float(v.mean()),*np.percentile(b,[lo,100-lo]).tolist()
def ci_unpaired(x,y,nb=NB,level=95):
    x=np.asarray(x,float);y=np.asarray(y,float);rng=np.random.default_rng(56);b=x[rng.integers(0,len(x),(nb,len(x)))].mean(1)-y[rng.integers(0,len(y),(nb,len(y)))].mean(1);lo=(100-level)/2;return float(x.mean()-y.mean()),*np.percentile(b,[lo,100-lo]).tolist()
def ratio_ci(num,den,nb=NB):
    num=np.asarray(num,float);den=np.asarray(den,float);rng=np.random.default_rng(57);idx=rng.integers(0,len(num),(nb,len(num)));d=den[idx].sum(1);b=np.where(d>0,num[idx].sum(1)/np.maximum(d,1e-12),np.nan)
    pt=num.sum()/den.sum() if den.sum()>0 else float('nan');return float(pt),*np.nanpercentile(b,[2.5,97.5]).tolist()
def excl0(c):return c[1]>0 or c[2]<0
def ranks(S):
    S=np.asarray(S,float);return 1+(S[:,None,:]>S[:,:,None]).sum(2)     # (n,16) rank of every candidate
def item_stats(S,gi,fi,world):
    """S (n,16) -> per-item arrays for the bound target (A: gi, B: fi) and its partner"""
    S=np.asarray(S,float);n=len(S);r=np.arange(n);R=ranks(S)
    if fi is None:   # no pairs (single-fact items): only the target statistics
        t=gi;rt=R[r,t];nan=np.full(n,np.nan);return {'rank':rt.astype(float),'top1':(rt==1).astype(float),'pm3':S[r,t]-(S.sum(1)-S[r,t])/(NC-1),'rank_partner':nan,'rank_other14':nan,'wrong':(rt!=1).astype(float),'wrong_partner':nan,'above_partner':nan,'margin':nan}
    t,o=(gi,fi) if world=='A' else (fi,gi);rt,ro=R[r,t],R[r,o];top=S.argmax(1)
    oth=np.ones_like(S,bool);oth[r,t]=False;oth[r,o]=False
    wrong=(rt!=1)
    return {'rank':rt.astype(float),'top1':(rt==1).astype(float),'pm3':S[r,t]-(S.sum(1)-S[r,t])/(NC-1),'rank_partner':ro.astype(float),'rank_other14':(R*oth).sum(1)/oth.sum(1),
            'wrong':wrong.astype(float),'wrong_partner':(wrong&(top==o)).astype(float),'above_partner':(S[r,t]>S[r,o]).astype(float),'margin':S[r,t]-S[r,o]}
def place_stats(S,idx):
    """f(S, place) for a given candidate index per item: (rank, top1, pm3)"""
    S=np.asarray(S,float);r=np.arange(len(S));R=ranks(S);return R[r,idx].astype(float),(R[r,idx]==1).astype(float),S[r,idx]-(S.sum(1)-S[r,idx])/(NC-1)
def _w0(rec,ctx,a):
    W=rec['W0'];S=W[f'{ctx}@{a}'] if f'{ctx}@{a}' in W else W[ctx];return S if isinstance(S,dict) else {'A':S,'B':S}
def _sel(rec,a=None,ctx=None,dmin=None,dmax=None):
    """exposed items (optionally with information distance in [dmin, dmax] in context ctx)"""
    m=(np.asarray(rec['meta']['kind'])=='exposed') if 'kind' in rec['meta'] else np.ones(len(rec['meta']['gi']),bool)
    if dmin is not None:
        d=rec['meta']['d'][str(a)][ctx];m=m&np.array([x is not None and dmin<=x<=dmax for x in d])
    return m
def _pairs(rec,m):
    pr=rec['meta'].get('pair') or [f'p{i}' for i in range(len(m))]   # no pair structure: the unit is the item
    return [p for p,s in zip(pr,m) if s]
def rec_items(rec,a,ctx,w0ctx=None,mask=None):
    """-> dict name -> per-item arrays (all items, masked later): W and W0 stats per world and the (i)/(ii) aggregations. w0ctx: context whose W0 is used (default = ctx)"""
    meta=rec['meta'];gi=np.asarray(meta['gi']);fi=np.asarray(meta['fi']) if meta.get('fi') is not None else None;S=rec['read'][str(a)][ctx];W0=_w0(rec,w0ctx or ctx,a);o={};ws=[w for w in 'AB' if w in S and S[w] is not None];nw=len(ws)
    for w in ws:
        s=item_stats(S[w],gi,fi,w);s0=item_stats(W0[w],gi,fi,w)
        for n_,v in s.items():o[(n_,'W',w)]=v
        for n_,v in s0.items():o[(n_,'W0',w)]=v
    out={}
    for n_ in ('rank','top1','pm3','rank_partner','rank_other14','above_partner','margin'):
        out[n_+'_W']=sum(o[(n_,'W',w)] for w in ws)/nw;out[n_+'_W0']=sum(o[(n_,'W0',w)] for w in ws)/nw
    for n_ in ('wrong','wrong_partner'):
        out[n_+'_W']=sum(o[(n_,'W',w)] for w in ws);out[n_+'_W0']=sum(o[(n_,'W0',w)] for w in ws)   # counts over the worlds (ratio of sums later)
    out['G_rank']=out['rank_W0']-out['rank_W'];out['PM2']=out['top1_W']-out['top1_W0'];out['PM3']=out['pm3_W']-out['pm3_W0']
    out['G_rank_partner']=out['rank_partner_W0']-out['rank_partner_W'];out['G_rank_other14']=out['rank_other14_W0']-out['rank_other14_W']
    # (ii) A-B counterfactual contrast on the same candidate strings (optional: needs both worlds and pairs)
    if nw<2 or fi is None:
        for n_ in ('AB_G_rank','AB_PM2','AB_PM3'):out[n_]=np.full(len(gi),np.nan)
        out['Dalt']=(sum(o[('margin','W',w)] for w in ws)-sum(o[('margin','W0',w)] for w in ws))/nw;out['acc2']=out['above_partner_W']-out['above_partner_W0'];return out
    rA,tA,pA=place_stats(S['A'],gi);rB,tB,pB=place_stats(S['B'],gi);rA2,tA2,pA2=place_stats(S['A'],fi);rB2,tB2,pB2=place_stats(S['B'],fi)
    out['AB_G_rank']=((rB-rA)+(rA2-rB2))/2;out['AB_PM2']=((tA-tB)+(tB2-tA2))/2;out['AB_PM3']=((pA-pB)+(pB2-pA2))/2
    out['Dalt']=(o[('margin','W','A')]+o[('margin','W','B')])/2-(o[('margin','W0','A')]+o[('margin','W0','B')])/2;out['acc2']=out['above_partner_W']-out['above_partner_W0']
    return out
def _pv(rec,a,ctx,name,mask,w0ctx=None):
    it=rec_items(rec,a,ctx,w0ctx);return ES.pair_means(it[name][mask],_pairs(rec,mask))
def vec(recs,a,ctx,name,w0ctx=None,**sel):return np.concatenate([_pv(r,a,ctx,name,_sel(r,a,ctx,**sel) if sel else _sel(r),w0ctx) for r in recs])
def cell_table(recs,a=16,ctx='out',**sel):
    """headline numbers of one cell (change vs W0): mean [95% CI]"""
    g=lambda n:ci(vec(recs,a,ctx,n,**sel));t={n:g(n) for n in ('G_rank','PM2','PM3','AB_G_rank','AB_PM2','AB_PM3','Dalt','acc2','G_rank_partner','G_rank_other14')}
    t['rank_W']=g('rank_W');t['rank_W0']=g('rank_W0')
    for k in ('W','W0'):
        num=np.concatenate([_pv_sum(r,a,ctx,'wrong_partner_'+k,**sel) for r in recs]);den=np.concatenate([_pv_sum(r,a,ctx,'wrong_'+k,**sel) for r in recs])
        t['wrong_top1_is_partner_'+k]=ratio_ci(num,den) if den.sum()>0 else (float('nan'),)*3
    t['above_partner_W']=g('above_partner_W');t['above_partner_W0']=g('above_partner_W0');return t
def _pv_sum(rec,a,ctx,name,**sel):
    m=_sel(rec,a,ctx,**sel) if sel else _sel(rec);it=rec_items(rec,a,ctx);d={}
    for x,p in zip(it[name][m],_pairs(rec,m)):d[p]=d.get(p,0)+x
    return np.asarray([d[k] for k in sorted(d,key=lambda s:int(str(s).lstrip('px')))])
def contrast(recs_x,recs_y,a=16,ctx='out',ctx_y=None,a_y=None,paired=True,w0ctx_x=None,w0ctx_y=None,**sel):
    """X - Y of the change-vs-W0 metrics: paired by pair when both are the same world (same seeds, same order), else unpaired"""
    o={}
    for n in ('G_rank','PM2','PM3','AB_G_rank'):
        x=vec(recs_x,a,ctx,n,w0ctx_x,**sel);y=vec(recs_y,a_y or a,ctx_y or ctx,n,w0ctx_y,**sel)
        o[n]=ci(x-y) if paired else ci_unpaired(x,y)
    return o
def signs_agree(c):
    """PM2 and PM3 must have the sign of PM1 (G_rank); else 'metric mismatch'"""
    s=np.sign(c['G_rank'][0]);return bool(np.sign(c['PM2'][0])==s and np.sign(c['PM3'][0])==s)
# ------------------------------------------------------------------------------------------------ rules (4c)
def binding(recs,a=16,ctx='out'):
    c=cell_table(recs,a,ctx);g=c['G_rank'];ok=bool(excl0(g) and g[0]>=1.5 and c['PM2'][0]>=0.05 and c['PM3'][0]>0)
    return {'binding':ok,'metric_mismatch':(not signs_agree(c)) and excl0(g),'table':c}
def shadowing(nat,swd,a=16,ctx='out'):
    c=contrast(swd,nat,a,ctx);g=c['G_rank'];ok=bool(excl0(g) and g[0]>=1.5 and c['PM2'][0]>=0.05 and np.sign(c['PM3'][0])==np.sign(g[0]) and c['PM3'][0]>0)
    return {'shadowing':ok,'negctrl_pass':bool(abs(g[0])<=1.0 and not excl0(g)),'metric_mismatch':(not signs_agree(c)) and excl0(g),'S':c}
def window_effect(recs,a=16,ctx='in_e0',dmin=1,dmax=8):
    """items with information in the window (d in [dmin, dmax]) -> raw exit drop rank(W,out) - rank(W,in) [positive = in-window better], its PM2 version, the W0-relative G_in - G_out, and the write share (W in - W0 in in rank units = G_rank in)"""
    o={}
    for n_,fn in (('exit_rank',lambda r:_pv(r,a,'out','rank_W',_sel(r,a,ctx,dmin=dmin,dmax=dmax))-_pv(r,a,ctx,'rank_W',_sel(r,a,ctx,dmin=dmin,dmax=dmax))),
                  ('exit_top1',lambda r:_pv(r,a,ctx,'top1_W',_sel(r,a,ctx,dmin=dmin,dmax=dmax))-_pv(r,a,'out','top1_W',_sel(r,a,ctx,dmin=dmin,dmax=dmax))),
                  ('G_in_minus_G_out',lambda r:_pv(r,a,ctx,'G_rank',_sel(r,a,ctx,dmin=dmin,dmax=dmax))-_pv(r,a,'out','G_rank',_sel(r,a,ctx,dmin=dmin,dmax=dmax))),
                  ('write_share_G_in',lambda r:_pv(r,a,ctx,'G_rank',_sel(r,a,ctx,dmin=dmin,dmax=dmax))),
                  ('W0_in_rank',lambda r:_pv(r,a,ctx,'rank_W0',_sel(r,a,ctx,dmin=dmin,dmax=dmax))),
                  ('W0_in_top1',lambda r:_pv(r,a,ctx,'top1_W0',_sel(r,a,ctx,dmin=dmin,dmax=dmax)))):
        o[n_]=ci(np.concatenate([fn(r) for r in recs]))
    # decision column (4c, 14:58 KST): the raw rank difference exit_rank = rank(W, out) - rank(W, in) (positive = in-window better); G_in_minus_G_out is secondary; opposite directions -> no call
    o['metric_mismatch']=bool(np.sign(o['exit_rank'][0])!=np.sign(o['G_in_minus_G_out'][0]) or np.sign(o['exit_top1'][0])!=np.sign(o['exit_rank'][0]))
    return o
def prediction_A(win_by_p):
    """win_by_p: {p: window_effect output (NAT cells)}. p=9 only as the construction check"""
    res={'per_p':{}};ok=True
    for p in (1,2,4):
        w=win_by_p.get(p)
        if w is None:res['per_p'][p]='missing';ok=None;continue
        g=w['exit_rank'];t=w['exit_top1'];good=bool(excl0(g) and g[0]>=3 and t[0]>=0.10 and not w['metric_mismatch']);res['per_p'][p]={'exit_rank':g,'exit_top1':t,'metric_mismatch':w['metric_mismatch'],'pass':good};ok=ok and good if ok is not None else None
    if all(isinstance(res['per_p'].get(p),dict) for p in (1,4)):
        d=np.asarray(win_by_p[1]['exit_rank'][0])-np.asarray(win_by_p[4]['exit_rank'][0]);res['p1_minus_p4_rank_point']=float(d);res['falsified_p4_lower_by_3']=bool(win_by_p[4]['exit_rank'][0]<win_by_p[1]['exit_rank'][0]-3)
        res['similar_point_within_1.5']=bool(abs(d)<=1.5)   # the registered test needs the CI of the p1-p4 difference (records of different worlds): use similar_ci below
    if any(isinstance(v,dict) and v['metric_mismatch'] for v in res['per_p'].values()):ok=None
    res['verdict']='supported' if ok else ('not supported/falsified' if ok is False else 'cannot judge (cells missing or 지표 불일치)')
    if win_by_p.get(9):res['negctrl_p9_pass']=bool(abs(win_by_p[9]['exit_rank'][0])<=1.0 and not excl0(win_by_p[9]['exit_rank']))
    return res
def prediction_B(nat_by_p,swd_by_p,a=16,ctx='out',n_seeds=1):
    """nat_by_p / swd_by_p: {p: [records]} -> flatness of NAT in p (90% equivalence CI +-1.5 against p=9; not judgeable with a single seed) and the rise of SWD toward frequent periods"""
    res={}
    nat_g={p:vec(r,a,ctx,'G_rank') for p,r in nat_by_p.items()};swd_g={p:vec(r,a,ctx,'G_rank') for p,r in swd_by_p.items()}
    if 9 in nat_g:
        eq={};above=True
        for p in (1,2,4):
            if p in nat_g:c=ci_unpaired(nat_g[p],nat_g[9],level=90);eq[p]=c
        res['NAT_minus_NAT9_90ci']=eq;res['NAT_above_chance_1.5']={p:bool(nat_g[p].mean()>=1.5) for p in nat_g}
        res['flat']='cannot be judged with one seed' if n_seeds<3 else bool(eq and all(-1.5<=c[1] and c[2]<=1.5 for c in eq.values()) and all(res['NAT_above_chance_1.5'].values()))
    if 1 in swd_g and 9 in swd_g:
        c=ci_unpaired(swd_g[1],swd_g[9]);pts=[swd_g[p].mean() for p in (1,2,4,9) if p in swd_g]
        res['SWD1_minus_SWD9']=c;res['SWD_order_ok']=bool(all(pts[i]>=pts[i+1] for i in range(len(pts)-1)));res['rise_under_masking']=bool(excl0(c) and c[0]>=1.5 and res['SWD_order_ok'])
    return res
# ------------------------------------------------------------------------------------------------ report
def fmt(c):return '%+.2f [%+.2f, %+.2f]'%tuple(c)
def table(cells,a=16,ctx='out'):
    L=[f'| cell ({ctx}) | G_rank (PM1) | PM2 | PM3 (nats) | AB contrast G_rank | D_alt | acc2 | partner: rank gain | other-14 rank gain | wrong top-1 = partner (W) | target above partner (W / W0) |','|'+'---|'*11]
    for n,recs in cells.items():
        t=cell_table(recs,a,ctx);L.append(f'| {n} | {fmt(t["G_rank"])} | {fmt(t["PM2"])} | {fmt(t["PM3"])} | {fmt(t["AB_G_rank"])} | {fmt(t["Dalt"])} | {fmt(t["acc2"])} | {fmt(t["G_rank_partner"])} | {fmt(t["G_rank_other14"])} | {fmt(t["wrong_top1_is_partner_W"])} | {t["above_partner_W"][0]:.2f} / {t["above_partner_W0"][0]:.2f} |')
    return '\n'.join(L)
def stage0(nat_files,swd_files=None,a=16,out=None):
    load=lambda fs:[json.load(open(f)) for f in fs.split(',')];nat=load(nat_files);swd=load(swd_files) if swd_files else None
    L=['# E55 revised metrics (design 3.1b / 4c), '+('NAT vs SWD' if swd else 'NAT only'),'',table({'NAT':nat,**({'SWD':swd} if swd else {})},a),'']
    for n,r in (('NAT',nat),)+((('SWD',swd),) if swd else ()):
        b=binding(r,a);L.append(f'binding {n}: {b["binding"]}'+(' (METRIC MISMATCH: PM2/PM3 disagree in sign with PM1)' if b['metric_mismatch'] else ''))
    if swd:
        s=shadowing(nat,swd,a);L.append(f'shadowing S=SWD-NAT: G_rank {fmt(s["S"]["G_rank"])}, PM2 {fmt(s["S"]["PM2"])}, PM3 {fmt(s["S"]["PM3"])} -> {s["shadowing"]}'+(' (METRIC MISMATCH)' if s['metric_mismatch'] else '')+f'; negative-control pass (|S|<=1 rank, CI has 0; meaningful at p=9 only) = {s["negctrl_pass"]}')
    if all('in_e0' in r['read'].get(str(a),{}) for r in nat):
        L.append('');L.append('window readout (NAT; decision column = exit drop rank(out)-rank(in), G_in-G_out secondary; opposite directions = 지표 불일치), items with information distance 1..8:')
        for ctx in ('in_e0','in_e1','in_e3','in_e7'):
            if ctx in nat[0]['read'][str(a)]:
                w=window_effect(nat,a,ctx);L.append(f'  {ctx}: exit drop rank(out)-rank(in) {fmt(w["exit_rank"])}, top1 in-out {fmt(w["exit_top1"])}, G_in-G_out {fmt(w["G_in_minus_G_out"])}, write share (G_rank in) {fmt(w["write_share_G_in"])}{" [지표 불일치: no call]" if w["metric_mismatch"] else ""}, W0 in-window rank {fmt(w["W0_in_rank"])}, W0 in-window top-1 {fmt(w["W0_in_top1"])}')
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
# ------------------------------------------------------------------------------------------------ self test
def _rec(N,rng,signal=0.0,partner_boost=0.0,inwin=0.0,w0_signal=0.0,with_d=False,pair_boost=0.0):
    """synthetic record: n = 2N exposed + 8 control items, 16 candidates; the bound target of an item gets +signal in its world (A: gi, B: fi), the partner of the bound target gets +partner_boost (so it is a different candidate in A and B), both places of the pair get +pair_boost in both worlds (binding-independent chunk-topic boost)"""
    n=2*N+8;gi=np.zeros(n,int);fi=np.ones(n,int)
    meta={'kind':['exposed']*(2*N)+['ctrl']*8,'pair':[f'p{i//2}' for i in range(2*N)]+[f'c{i//2}' for i in range(8)],'gi':gi.tolist(),'fi':fi.tolist(),'qframe':[i%3 for i in range(n)],'name_id':list(range(n))}
    base=rng.normal(size=(n,16))*1.5     # fixed per-item preferences (the same in W0 and W)
    def S(sig,mask=None):
        out={}
        for w,(t,o) in (('A',(0,1)),('B',(1,0))):
            x=base+0.3*rng.normal(size=(n,16));m=np.ones(n) if mask is None else mask
            x[:2*N,t]+=sig*m[:2*N];x[:2*N,o]+=partner_boost;x[:2*N,0]+=pair_boost;x[:2*N,1]+=pair_boost;out[w]=x
        return out
    rec={'meta':meta,'W0':{'out':base+0.3*rng.normal(size=(n,16))},'read':{'16':{'out':S(signal)}}}
    if with_d:
        d=[1+(i//2)%8 for i in range(2*N)]+[None]*8;meta['d']={'16':{'in_e0':d}};m=np.array([1.0 if (x is not None and x<=4) else 0.0 for x in d])
        rec['W0']['in_e0@16']={'A':base+0.3*rng.normal(size=(n,16)),'B':base+0.3*rng.normal(size=(n,16))}
        for w,(t,o) in (('A',(0,1)),('B',(1,0))):rec['W0']['in_e0@16'][w][:2*N,t]+=w0_signal*m[:2*N]
        rec['read']['16']['in_e0']=S(signal+inwin,m)
    return rec
def selftest():
    rng=np.random.default_rng(3)
    # 1 rank computation and item stats on a hand example: target (idx 0) has the 3rd highest score, partner (idx 1) the highest
    S=np.zeros((1,16));S[0,0]=2;S[0,1]=5;S[0,2]=3;st=item_stats(S,np.array([0]),np.array([1]),'A');assert st['rank'][0]==3 and st['rank_partner'][0]==1 and st['top1'][0]==0 and st['wrong_partner'][0]==1 and st['above_partner'][0]==0
    stB=item_stats(S,np.array([0]),np.array([1]),'B');assert stB['rank'][0]==1 and stB['top1'][0]==1 and stB['wrong'][0]==0
    # 2 chance: no signal -> G_rank, PM2, PM3 near 0; real signal: binding, shadowing, p9 negative control
    nat=[_rec(16,rng,0.0) for _ in range(3)];swd=[_rec(16,rng,2.5) for _ in range(3)]
    b0=binding(nat);assert not b0['binding'];b1=binding(swd);assert b1['binding'] and not b1['metric_mismatch'] and b1['table']['G_rank'][0]>1.5
    sh=shadowing(nat,swd);assert sh['shadowing'] and not sh['negctrl_pass']
    n9=[_rec(72,rng,0.0) for _ in range(2)];s9=[_rec(72,rng,0.0) for _ in range(2)];assert shadowing(n9,s9)['negctrl_pass'] and not shadowing(n9,s9)['shadowing']
    # 3 the A-B contrast agrees with the W0-subtracted gain when the signal is binding-specific (+-1 rank), and cancels a binding-independent boost
    t=b1['table'];assert abs(t['AB_G_rank'][0]-t['G_rank'][0])<1.0,(t['AB_G_rank'],t['G_rank'])
    # 4 partner stratification: a planted partner boost (both worlds) raises the partner's rank gain over the other 14 and the share of wrong top-1 answers that are the partner
    pb=[_rec(16,rng,0.0,partner_boost=3.0) for _ in range(3)];tp=cell_table(pb);assert tp['G_rank_partner'][0]>tp['G_rank_other14'][0]+3 and tp['wrong_top1_is_partner_W'][0]>0.5 
    # binding-independent topic boost of both places of the pair: the W0-subtracted gain sees it, the A-B contrast cancels it
    tb=[_rec(16,rng,0.0,pair_boost=3.0) for _ in range(3)];tt=cell_table(tb);assert tt['G_rank'][0]>2 and abs(tt['AB_G_rank'][0])<1.0,(tt['G_rank'],tt['AB_G_rank'])
    # 5 metric mismatch flag: PM1 up but PM3 down cannot be planted with random data; check the sign test directly
    assert signs_agree({'G_rank':(1,0,2),'PM2':(0.1,0,0.2),'PM3':(0.2,0,0.3)}) and not signs_agree({'G_rank':(1,0,2),'PM2':(0.1,0,0.2),'PM3':(-0.2,-0.3,0)})
    # 6 window readout and prediction A
    w=[_rec(16,rng,0.5,inwin=6.0,with_d=True) for _ in range(3)];we=window_effect(w,16,'in_e0',1,4);assert excl0(we['exit_rank']) and we['exit_rank'][0]>3 and we['exit_top1'][0]>0.1
    win={1:we,2:we,4:window_effect([_rec(16,rng,0.5,inwin=6.0,with_d=True) for _ in range(3)],16,'in_e0',1,4)};pa=prediction_A(win);assert pa['verdict']=='supported'
    weak=window_effect([_rec(16,rng,0.5,inwin=0.0,with_d=True) for _ in range(3)],16,'in_e0',1,4);assert prediction_A({1:we,2:weak,4:we})['verdict']!='supported'
    # 7 prediction B: NAT flat in p (all same weak signal), SWD rising toward frequent periods
    mk=lambda sig,N:[_rec(N,rng,sig) for _ in range(3)]
    nb=prediction_B({1:mk(2.0,8),2:mk(2.0,16),4:mk(2.0,32),9:mk(2.0,72)},{1:mk(4.0,8),2:mk(3.0,16),4:mk(2.0,32),9:mk(0.0,72)},n_seeds=3);assert nb['rise_under_masking'] and nb['flat'] is True,nb
    assert prediction_B({1:mk(0.0,8),9:mk(0.0,72)},{},n_seeds=1)['flat']=='cannot be judged with one seed'
    # 8 single-world (A only, no pair structure, no partner): same headline numbers, A-B contrast and partner stats are NaN
    def a_only(r):
        r2={'meta':{k:v for k,v in r['meta'].items() if k not in ('pair','fi','kind')},'W0':{k:(v['A'] if isinstance(v,dict) else v) for k,v in r['W0'].items()},'read':{a:{c:{'A':d['A']} for c,d in cs.items()} for a,cs in r['read'].items()}};r2['meta']['fi']=None;r2['meta']['gi']=r['meta']['gi'][:2*16];return r2
    def trim(r):
        n=32;r2=a_only(r);r2['meta']['gi']=r['meta']['gi'][:n];r2['W0']={'out':np.asarray(r['W0']['out'])[:n]};r2['read']={'16':{'out':{'A':np.asarray(r['read']['16']['out']['A'])[:n]}}};return r2
    sa=[trim(r) for r in swd];bs=binding(sa);assert bs['binding'] and np.isnan(bs['table']['AB_G_rank'][0]) and np.isnan(bs['table']['G_rank_partner'][0]),bs['table']
    # 9 window mismatch flag: raw rank drop and G_in-G_out of opposite sign -> no call
    wm=dict(we);wm['metric_mismatch']=True;assert prediction_A({1:wm,2:we,4:we})['verdict'].startswith('cannot judge')
    print(table({'NAT':nat,'SWD':swd,'partner-boost only':pb}))
    print('e55_metrics SELFTEST OK (synthetic scores)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--nat',default=None,help='comma list of NAT records (e55_gpu make_records files, W0 merged under rec["W0"])');ap.add_argument('--swd',default=None);ap.add_argument('--a',type=int,default=16);ap.add_argument('--out',default=None);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.nat:print(stage0(a.nat,a.swd,a.a,a.out))
    else:print(__doc__)
