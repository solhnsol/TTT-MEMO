"""E55 metric system v2 (design 3.1b + section 10.2/10.3/4c, pairless: one world, unit = fact), numpy only, from the raw 16-candidate score matrices of the e55_gpu records.
Record: {'meta': {kind ('fact'|'ctrl'), fact, group, qframe, name_id, place_id, gold_idx, cand_fact, cand_group, cand_same, k, n, p, bank, seed, d: {a: {ctx: [d per item or None]}}, arm}, 'read': {a: {ctx: S (n_items,16)}}, 'W0': {ctx or ctx@a: S}}.
Group contexts: an item is read only in the in-window context of its own group; S may have all n_items rows (other rows ignored) or only the rows listed in meta['item_idx'][a][ctx]; an in-window metric is taken over the items with a distance d in [1, 8] in that context (meta['d']); a LIST of contexts (all groups) is combined item by item.
Per item: rank = 1 + #(candidates with a strictly higher score) of the gold place (chance 8.5). PM1 = G_rank = rank(W0) - rank(W) (positive = better), PM2 = top1(W) - top1(W0) (chance 1/16), PM3 = [score(gold) - mean of the other 15](W) - same(W0). All are changes against the same items read at W0 (the W0 read of the same context).
Stratification (p >= 2): same-group candidates (cand_same) vs other-group candidates, excluding the gold: mean rank (W, W0, gain), share of wrong top-1 answers that is a same-group candidate (uniform = 7/15), share of (gold, same-group candidate) comparisons with the gold above (uniform 0.5). Control items (kind != 'fact': never-shown name, hypothetical gold = a random shown place, expected mean rank 8.5) are reported separately and never enter the fact statistics: a deviation of their change from 0 is a name-independent place-preference shift.
CI = 95% bootstrap over facts (several seeds concatenated). Contrasts between cells of the same world (NAT vs SWD, window vs out) are paired by fact (same seeds, same order); cells of different worlds (different p, k) are unpaired.
Rules (4c, rank units; thresholds raised to the minimum detectable effect, see mde()/thresholds()): binding G_rank >= thr (1.5) with CI excluding 0, PM2 >= +0.05, PM3 > 0; shadowing S = SWD - NAT the same; p = 9 control |S| <= 1 rank with CI containing 0; prediction A (decision column = raw rank(W,out) - rank(W,in) on items with d in 1..8; G_in - G_out secondary; opposite directions = metric mismatch, no call) >= +3 ranks with CI excluding 0 and top-1 difference >= +0.10 for p in {1,2,4}; prediction B 'flat' = 90% CI of NAT(p) - NAT(9) inside +-thr for p in {1,2,4} and NAT above chance by thr (needs >= 3 seeds, otherwise 'cannot be judged'), 'rise under masking' = SWD(1) - SWD(9) >= thr with CI excluding 0 and the point order SWD(1) >= SWD(2) >= SWD(4) >= SWD(9). PM2/PM3 disagreeing in sign with PM1 = metric mismatch (no call)."""
import json,sys,argparse
from pathlib import Path
import numpy as np
NB=4000;NC=16;THR_RANK=1.5;THR_A=3.0
def ci(v,nb=NB,level=95):
    v=np.asarray(v,float);rng=np.random.default_rng(55);b=v[rng.integers(0,len(v),(nb,len(v)))].mean(1);lo=(100-level)/2;return float(v.mean()),*np.percentile(b,[lo,100-lo]).tolist()
def ci_unpaired(x,y,nb=NB,level=95):
    x=np.asarray(x,float);y=np.asarray(y,float);rng=np.random.default_rng(56);b=x[rng.integers(0,len(x),(nb,len(x)))].mean(1)-y[rng.integers(0,len(y),(nb,len(y)))].mean(1);lo=(100-level)/2;return float(x.mean()-y.mean()),*np.percentile(b,[lo,100-lo]).tolist()
def ratio_ci(num,den,nb=NB):
    num=np.asarray(num,float);den=np.asarray(den,float);rng=np.random.default_rng(57);idx=rng.integers(0,len(num),(nb,len(num)));d=den[idx].sum(1);b=np.where(d>0,num[idx].sum(1)/np.maximum(d,1e-12),np.nan)
    return (float(num.sum()/den.sum()) if den.sum()>0 else float('nan')),*np.nanpercentile(b,[2.5,97.5]).tolist()
def excl0(c):return c[1]>0 or c[2]<0
def ranks(S):
    S=np.asarray(S,float);return 1+(S[:,None,:]>S[:,:,None]).sum(2)+0.5*((S[:,None,:]==S[:,:,None]).sum(2)-1)       # ties = average rank (analysis plan 11h)
def _gold(meta):return np.asarray(meta['gold_idx'] if 'gold_idx' in meta else meta['gi'])
def item_stats(S,gold,same=None):
    """S (n,16), gold (n,), same (n,16) bool or None -> per-item arrays"""
    S=np.asarray(S,float);n=len(S);r=np.arange(n);R=ranks(S);rt=R[r,gold];top=S.argmax(1);wrong=rt!=1
    o={'rank':rt.astype(float),'top1':(rt==1).astype(float),'pm3':S[r,gold]-(S.sum(1)-S[r,gold])/(NC-1),'wrong':wrong.astype(float)}
    if same is None:nan=np.full(n,np.nan);o.update(rank_same=nan,rank_other=nan,wrong_same=nan,gold_above_same=nan,n_same=nan);return o
    same=np.asarray(same,bool).copy();same[r,gold]=False;oth=~same;oth[r,gold]=False;ns=same.sum(1)
    o['rank_same']=np.where(ns>0,(R*same).sum(1)/np.maximum(ns,1),np.nan);o['rank_other']=(R*oth).sum(1)/np.maximum(oth.sum(1),1)
    o['wrong_same']=np.where(ns>0,(wrong&same[r,top]).astype(float),np.nan);o['gold_above_same']=np.where(ns>0,((S[r,gold][:,None]>S)&same).sum(1)/np.maximum(ns,1),np.nan);o['n_same']=ns.astype(float);return o
def _full(rec,S,a,ctx):
    """S as an (n_items,16) matrix with NaN rows for items that were not read in this context"""
    S=np.asarray(S,float);n=len(rec['meta']['kind'])
    if len(S)==n:return S
    idx=rec['meta']['item_idx'][str(a)][ctx];F=np.full((n,NC),np.nan);F[idx]=S;return F
def _read(rec,a,ctx):return _full(rec,rec['read'][str(a)][ctx],a,ctx)
def _w0(rec,ctx,a):
    W=rec['W0'];S=W[f'{ctx}@{a}'] if f'{ctx}@{a}' in W else W[ctx];return _full(rec,S,a,ctx)
def _valid(rec,a,ctx,dmin=None,dmax=None):
    """items usable in this context: has a read; for in-window contexts d in [dmin, dmax]"""
    v=~np.isnan(_read(rec,a,ctx)).any(1)
    if dmin is not None:
        d=rec['meta']['d'][str(a)][ctx];v=v&np.array([x is not None and dmin<=x<=dmax for x in d])
    return v
def rec_items(rec,a,ctx):
    """-> dict of per-item arrays (all items, NaN where not read): W/W0 stats and the three changes. ctx may be a list of contexts (item-wise combination: the first context in which the item is valid)"""
    if isinstance(ctx,(list,tuple)):
        parts=[rec_items(rec,a,c) for c in ctx];vs=[_valid(rec,a,c) for c in ctx];out={}
        for k in parts[0]:
            x=np.full(len(parts[0][k]),np.nan)
            for p_,v in reversed(list(zip(parts,vs))):x=np.where(v,p_[k],x)
            out[k]=x
        return out
    meta=rec['meta'];g=_gold(meta);same=np.asarray(meta['cand_same'],bool) if meta.get('cand_same') is not None and meta.get('p',2)!=1 else None
    s=item_stats(_read(rec,a,ctx),g,same);s0=item_stats(_w0(rec,ctx,a),g,same);o={}
    for k,v in s.items():o[k+'_W']=v
    for k,v in s0.items():o[k+'_W0']=v
    o['G_rank']=s0['rank']-s['rank'];o['PM2']=s['top1']-s0['top1'];o['PM3']=s['pm3']-s0['pm3']
    o['G_rank_same']=s0['rank_same']-s['rank_same'];o['G_rank_other']=s0['rank_other']-s['rank_other'];return o
def _kind(rec,fact=True):
    k=np.asarray(rec['meta']['kind']);return (k=='fact') if fact else (k!='fact')
def _sel(rec,a,ctx,dmin=None,dmax=None,fact=True):
    m=_kind(rec,fact)
    for c in (ctx if isinstance(ctx,(list,tuple)) else [ctx]):
        pass
    if isinstance(ctx,(list,tuple)):v=np.zeros(len(m),bool);[v.__ior__(_valid(rec,a,c,dmin,dmax)) for c in ctx]
    else:v=_valid(rec,a,ctx,dmin,dmax)
    return m&v
def vec(recs,a,ctx,name,**sel):return np.concatenate([rec_items(r,a,ctx)[name][_sel(r,a,ctx,**sel)] for r in recs])
def cell_table(recs,a=16,ctx='out',**sel):
    g=lambda n:ci(vec(recs,a,ctx,n,**sel));t={n:g(n) for n in ('G_rank','PM2','PM3','G_rank_same','G_rank_other')};t['rank_W']=g('rank_W');t['rank_W0']=g('rank_W0')
    ok=lambda n:np.concatenate([rec_items(r,a,ctx)[n][_sel(r,a,ctx,**sel)] for r in recs])
    for k in ('W','W0'):
        ws=ok('wrong_same_'+k);wr=ok('wrong_'+k);m=~np.isnan(ws);t['wrong_top1_is_same_group_'+k]=ratio_ci(ws[m],wr[m]) if wr[m].sum()>0 else (float('nan'),)*3
        ga=ok('gold_above_same_'+k);t['gold_above_same_group_'+k]=ci(ga[~np.isnan(ga)]) if (~np.isnan(ga)).any() else (float('nan'),)*3
        t['rank_same_'+k]=ci(ok('rank_same_'+k)[~np.isnan(ok('rank_same_'+k))]) if (~np.isnan(ok('rank_same_'+k))).any() else (float('nan'),)*3
        t['rank_other_'+k]=g('rank_other_'+k)
    t['n_facts']=int(sum(_sel(r,a,ctx,**sel).sum() for r in recs));return t
def ctrl_table(recs,a=16,ctx='out'):
    """control items (never-shown names; expected mean rank 8.5): rank at W and W0 and the change; a change away from 0 = name-independent place-preference shift"""
    x=lambda n:np.concatenate([rec_items(r,a,ctx)[n][_sel(r,a,ctx,fact=False)] for r in recs])
    if len(x('rank_W'))==0:return None
    return {'n':int(len(x('rank_W'))),'rank_W':ci(x('rank_W')),'rank_W0':ci(x('rank_W0')),'G_rank':ci(x('G_rank')),'PM2':ci(x('PM2')),'PM3':ci(x('PM3'))}
def contrast(recs_x,recs_y,a=16,ctx='out',ctx_y=None,a_y=None,paired=True,**sel):
    o={}
    for n in ('G_rank','PM2','PM3'):
        x=vec(recs_x,a,ctx,n,**sel);y=vec(recs_y,a_y or a,ctx_y or ctx,n,**sel);o[n]=ci(x-y) if paired else ci_unpaired(x,y)
    return o
def signs_agree(c):
    s=np.sign(c['G_rank'][0]);return bool(np.sign(c['PM2'][0])==s and np.sign(c['PM3'][0])==s)
# ------------------------------------------------------------------------------------------------ minimum detectable effect
def mde(sd_diff,n_items,n_seeds=1,z=2.8):
    """MDE (80% power, two-sided 5%) of a paired contrast in rank units: 2.8 * sd(per-fact paired difference) / sqrt(items * seeds)"""
    return z*sd_diff/np.sqrt(n_items*n_seeds)
def sd_from_records(recs_x,recs_y,a=16,ctx='out',name='G_rank'):
    """sd of the per-fact paired difference X - Y of a change metric (CPU: from stored margins/records of two cells of the same world), pooled over the records"""
    d=np.concatenate([rec_items(rx,a,ctx)[name][_sel(rx,a,ctx)]-rec_items(ry,a,ctx)[name][_sel(ry,a,ctx)] for rx,ry in zip(recs_x,recs_y)]);return float(d.std(ddof=1))
def sd_from_ranks(rank_sd=4.6,rho=0.0):
    """analytic fallback: rank sd 4.6 at chance, paired difference sd = sqrt(2 (1-rho)) * 4.6 (rho = correlation of the per-fact change between the two cells)"""
    return rank_sd*np.sqrt(2*(1-rho))
def thresholds(sd_diff,n_items,n_seeds=1,base_rank=THR_RANK,base_A=THR_A):
    """4c: a threshold below the MDE is raised to the MDE. -> {'mde','thr_rank','thr_A','raised'}"""
    m=mde(sd_diff,n_items,n_seeds);return {'mde':float(m),'thr_rank':float(max(base_rank,m)),'thr_A':float(max(base_A,m)),'raised':bool(m>base_rank)}
# ------------------------------------------------------------------------------------------------ rules (4c)
def binding(recs,a=16,ctx='out',thr=THR_RANK):
    c=cell_table(recs,a,ctx);g=c['G_rank'];ok=bool(excl0(g) and g[0]>=thr and c['PM2'][0]>=0.05 and c['PM3'][0]>0)
    return {'binding':ok,'metric_mismatch':(not signs_agree(c)) and excl0(g),'table':c,'ctrl':ctrl_table(recs,a,ctx)}
def shadowing(nat,swd,a=16,ctx='out',thr=THR_RANK):
    c=contrast(swd,nat,a,ctx);g=c['G_rank'];ok=bool(excl0(g) and g[0]>=thr and c['PM2'][0]>=0.05 and c['PM3'][0]>0)
    return {'shadowing':ok,'negctrl_pass':bool(abs(g[0])<=1.0 and not excl0(g)),'metric_mismatch':(not signs_agree(c)) and excl0(g),'S':c}
def window_effect(recs,a=16,ctx=('in',),dmin=1,dmax=8):
    """ctx: in-window context name(s) (list = all group contexts, combined item by item). Items with d in [dmin, dmax]. exit_rank = rank(W,out) - rank(W,in) is the DECISION column; G_in_minus_G_out is secondary"""
    ctxs=list(ctx) if isinstance(ctx,(list,tuple)) else [ctx];o={}
    def per(fn):return np.concatenate([fn(r,rec_items(r,a,ctxs),rec_items(r,a,'out'),_sel(r,a,ctxs,dmin,dmax)&_sel(r,a,'out')) for r in recs])
    o['exit_rank']=ci(per(lambda r,i,u,m:u['rank_W'][m]-i['rank_W'][m]));o['exit_top1']=ci(per(lambda r,i,u,m:i['top1_W'][m]-u['top1_W'][m]))
    o['G_in_minus_G_out']=ci(per(lambda r,i,u,m:i['G_rank'][m]-u['G_rank'][m]));o['write_share_G_in']=ci(per(lambda r,i,u,m:i['G_rank'][m]))
    o['W0_in_rank']=ci(per(lambda r,i,u,m:i['rank_W0'][m]));o['W0_in_top1']=ci(per(lambda r,i,u,m:i['top1_W0'][m]));o['n_facts']=int(sum((_sel(r,a,ctxs,dmin,dmax)&_sel(r,a,'out')).sum() for r in recs))
    o['metric_mismatch']=bool(np.sign(o['exit_rank'][0])!=np.sign(o['G_in_minus_G_out'][0]) or np.sign(o['exit_top1'][0])!=np.sign(o['exit_rank'][0]));return o
def prediction_A(win_by_p,thr=THR_A):
    res={'per_p':{}};ok=True
    for p in (1,2,4):
        w=win_by_p.get(p)
        if w is None:res['per_p'][p]='missing';ok=None;continue
        g=w['exit_rank'];t=w['exit_top1'];good=bool(excl0(g) and g[0]>=thr and t[0]>=0.10 and not w['metric_mismatch']);res['per_p'][p]={'exit_rank':g,'exit_top1':t,'metric_mismatch':w['metric_mismatch'],'pass':good}
        if ok is not None:ok=ok and good
    if isinstance(res['per_p'].get(1),dict) and isinstance(res['per_p'].get(4),dict):
        res['p1_minus_p4_rank_point']=float(win_by_p[1]['exit_rank'][0]-win_by_p[4]['exit_rank'][0]);res['falsified_p4_lower_by_thr']=bool(win_by_p[4]['exit_rank'][0]<win_by_p[1]['exit_rank'][0]-thr)
    if any(isinstance(v,dict) and v['metric_mismatch'] for v in res['per_p'].values()):ok=None
    res['verdict']='supported' if ok else ('not supported/falsified' if ok is False else 'cannot judge (cells missing or metric mismatch)')
    if win_by_p.get(9):res['negctrl_p9_pass']=bool(abs(win_by_p[9]['exit_rank'][0])<=1.0 and not excl0(win_by_p[9]['exit_rank']))
    return res
def prediction_B(nat_by_p,swd_by_p,a=16,ctx='out',n_seeds=1,thr=THR_RANK):
    res={};ng={p:vec(r,a,ctx,'G_rank') for p,r in nat_by_p.items()};sg={p:vec(r,a,ctx,'G_rank') for p,r in swd_by_p.items()}
    if 9 in ng:
        eq={p:ci_unpaired(ng[p],ng[9],level=90) for p in (1,2,4) if p in ng};res['NAT_minus_NAT9_90ci']=eq;res['NAT_above_chance']={p:bool(ng[p].mean()>=thr) for p in ng}
        res['flat']='cannot be judged with fewer than 3 seeds' if n_seeds<3 else bool(eq and all(-thr<=c[1] and c[2]<=thr for c in eq.values()) and all(res['NAT_above_chance'].values()))
    if 1 in sg and 9 in sg:
        c=ci_unpaired(sg[1],sg[9]);pts=[sg[p].mean() for p in (1,2,4,9) if p in sg];res['SWD1_minus_SWD9']=c;res['SWD_order_ok']=bool(all(pts[i]>=pts[i+1] for i in range(len(pts)-1)));res['rise_under_masking']=bool(excl0(c) and c[0]>=thr and res['SWD_order_ok'])
    return res
# ------------------------------------------------------------------------------------------------ report
def fmt(c):return '%+.2f [%+.2f, %+.2f]'%tuple(c)
def table(cells,a=16,ctx='out'):
    L=[f'| cell ({ctx}) | facts | G_rank (PM1) | PM2 | PM3 (nats) | rank gain same-group | rank gain other-group | wrong top-1 = same group (W) | gold above same-group (W / W0) |','|'+'---|'*9]
    for n,recs in cells.items():
        t=cell_table(recs,a,ctx);L.append(f'| {n} | {t["n_facts"]} | {fmt(t["G_rank"])} | {fmt(t["PM2"])} | {fmt(t["PM3"])} | {fmt(t["G_rank_same"])} | {fmt(t["G_rank_other"])} | {fmt(t["wrong_top1_is_same_group_W"])} | {t["gold_above_same_group_W"][0]:.2f} / {t["gold_above_same_group_W0"][0]:.2f} |')
    return '\n'.join(L)
def stage0(nat_files,swd_files=None,a=16,out=None,thr=THR_RANK):
    load=lambda fs:[json.load(open(f)) for f in fs.split(',')];nat=load(nat_files);swd=load(swd_files) if swd_files else None
    L=['# E55 metrics v2 (design 10.3 / 4c), '+('NAT vs SWD' if swd else 'NAT only'),'',table({'NAT':nat,**({'SWD':swd} if swd else {})},a),'']
    for n,r in (('NAT',nat),)+((('SWD',swd),) if swd else ()):
        b=binding(r,a,thr=thr);L.append(f'binding {n}: {b["binding"]}'+(' (METRIC MISMATCH)' if b['metric_mismatch'] else ''))
        if b['ctrl']:c=b['ctrl'];L.append(f'  control items {n} (n={c["n"]}, expected rank 8.5): rank W {fmt(c["rank_W"])}, change G_rank {fmt(c["G_rank"])}')
    if swd:
        s=shadowing(nat,swd,a,thr=thr);L.append(f'shadowing S=SWD-NAT: G_rank {fmt(s["S"]["G_rank"])}, PM2 {fmt(s["S"]["PM2"])}, PM3 {fmt(s["S"]["PM3"])} -> {s["shadowing"]}'+(' (METRIC MISMATCH)' if s['metric_mismatch'] else '')+f'; p=9 negative control pass = {s["negctrl_pass"]}')
    ins=sorted(c for c in nat[0]['read'].get(str(a),{}) if c!='out')
    if ins:
        w=window_effect(nat,a,ins);L+=['',f'window readout (NAT; contexts {ins}; decision column exit drop rank(out)-rank(in), G_in-G_out secondary), {w["n_facts"]} facts with d in 1..8:',f'  exit drop {fmt(w["exit_rank"])}, top1 in-out {fmt(w["exit_top1"])}, G_in-G_out {fmt(w["G_in_minus_G_out"])}, write share (G_rank in) {fmt(w["write_share_G_in"])}, W0 in-window rank {fmt(w["W0_in_rank"])}, W0 in-window top-1 {fmt(w["W0_in_top1"])}'+(' [metric mismatch: no call]' if w['metric_mismatch'] else '')]
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
# ------------------------------------------------------------------------------------------------ self test
def _rec(k,rng,signal=0.0,same_boost=0.0,p=2,inwin=0.0,ctrl_shift=0.0,with_d=False):
    """synthetic v2 record: k facts + 4 ctrl items, 16 candidates (gold at index 0; half of the others same-group when p>=2), fixed per-item preferences shared by W0 and W; gold gets +signal, same-group candidates +same_boost"""
    n=k+4;kind=['fact']*k+['ctrl']*4;group=(np.arange(n)%p).tolist();cs=np.zeros((n,16),bool)
    if p>=2:cs[:,1:8]=True
    meta={'kind':kind,'fact':list(range(n)),'group':group,'qframe':[i%3 for i in range(n)],'name_id':list(range(n)),'place_id':list(range(n)),'gold_idx':[0]*n,'cand_fact':[[0]*16]*n,'cand_group':[[0]*16]*n,'cand_same':cs.tolist(),'k':k,'n':1,'p':p,'bank':'50','seed':0,'arm':'x'}
    base=rng.normal(size=(n,16))*1.5;noise=lambda:0.3*rng.normal(size=(n,16))
    def S(sig,mask=None):
        x=base+noise();m=np.ones(n) if mask is None else mask;x[:k,0]+=sig*m[:k];x[:k]+=same_boost*cs[:k];x[k:,0]+=ctrl_shift;return x
    rec={'meta':meta,'W0':{'out':base+noise()},'read':{'16':{'out':S(signal)}}}
    if with_d:
        d=[1+(i%4) for i in range(k)]+[None]*4;meta['d']={'16':{'in':d}};m=np.array([1.0 if x is not None else 0.0 for x in d]);rec['W0']['in@16']=base+noise();rec['read']['16']['in']=S(signal+inwin,m)
    return rec
def selftest():
    rng=np.random.default_rng(3)
    S=np.zeros((1,16));S[0,0]=2;S[0,1]=5;S[0,2]=3;same=np.zeros((1,16),bool);same[0,1]=True;st=item_stats(S,np.array([0]),same);assert st['rank'][0]==3 and st['wrong'][0]==1 and st['wrong_same'][0]==1 and st['gold_above_same'][0]==0 and st['rank_same'][0]==1
    nat=[_rec(32,rng,0.0) for _ in range(3)];swd=[_rec(32,rng,2.5) for _ in range(3)]
    assert not binding(nat)['binding'];b=binding(swd);assert b['binding'] and not b['metric_mismatch'] and b['table']['G_rank'][0]>1.5 and b['ctrl']['n']==12
    sh=shadowing(nat,swd);assert sh['shadowing'] and not sh['negctrl_pass']
    n9=[_rec(144,rng,0.0,p=9) for _ in range(2)];s9=[_rec(144,rng,0.0,p=9) for _ in range(2)];assert shadowing(n9,s9)['negctrl_pass'] and not shadowing(n9,s9)['shadowing']
    # same-group boost: same-group candidates gain, wrong top-1s are mostly same-group
    sg=cell_table([_rec(32,rng,0.0,same_boost=3.0) for _ in range(3)]);assert sg['G_rank_same'][0]>sg['G_rank_other'][0]+2 and sg['wrong_top1_is_same_group_W'][0]>7/15+0.2
    # control items: a name-independent place shift moves the control rank (expected 8.5 rank at W0)
    ct=binding([_rec(32,rng,0.0,ctrl_shift=4.0) for _ in range(3)])['ctrl'];assert ct['G_rank'][0]>2 and abs(ct['rank_W0'][0]-8.5)<2.5
    # p = 1: no same-group stratum -> NaN, headline numbers still work
    one=cell_table([_rec(16,rng,2.5,p=1) for _ in range(3)]);assert one['G_rank'][0]>1.5 and np.isnan(one['G_rank_same'][0])
    assert signs_agree({'G_rank':(1,0,2),'PM2':(0.1,0,0.2),'PM3':(0.2,0,0.3)}) and not signs_agree({'G_rank':(1,0,2),'PM2':(0.1,0,0.2),'PM3':(-0.2,-0.3,0)})
    # window readout and prediction A
    w=[_rec(32,rng,0.5,inwin=6.0,with_d=True) for _ in range(3)];we=window_effect(w,16,['in'],1,4);assert excl0(we['exit_rank']) and we['exit_rank'][0]>3 and we['exit_top1'][0]>0.1 and not we['metric_mismatch']
    win={1:we,2:we,4:window_effect([_rec(32,rng,0.5,inwin=6.0,with_d=True) for _ in range(3)],16,['in'],1,4)};assert prediction_A(win)['verdict']=='supported'
    weak=window_effect([_rec(32,rng,0.5,inwin=0.0,with_d=True) for _ in range(3)],16,['in'],1,4);assert prediction_A({1:we,2:weak,4:we})['verdict']!='supported'
    wm=dict(we);wm['metric_mismatch']=True;assert prediction_A({1:wm,2:we,4:we})['verdict'].startswith('cannot judge')
    # rows given only for the items read in a context (item_idx)
    r=_rec(32,rng,0.5,inwin=6.0,with_d=True);idx=list(range(0,36,2));r['read']['16']['in']=np.asarray(r['read']['16']['in'])[idx];r['W0']['in@16']=np.asarray(r['W0']['in@16'])[idx];r['meta']['item_idx']={'16':{'in':idx}};r['meta']['d']['16']['in']=[x if i%2==0 else None for i,x in enumerate(r['meta']['d']['16']['in'])];window_effect([r],16,['in'],1,4)
    # prediction B
    mk=lambda sig,k:[_rec(k,rng,sig) for _ in range(3)]
    nb=prediction_B({1:mk(2.0,16),2:mk(2.0,32),4:mk(2.0,64),9:mk(2.0,144)},{1:mk(4.0,16),2:mk(3.0,32),4:mk(2.0,64),9:mk(0.0,144)},n_seeds=3);assert nb['rise_under_masking'] and nb['flat'] is True,nb
    assert prediction_B({1:mk(0.0,16),9:mk(0.0,144)},{},n_seeds=1)['flat'].startswith('cannot')
    # MDE: design text (rank sd 4.6, paired difference ~1.1 per sqrt(32)...): 2.8 * sd / sqrt(items * seeds)
    sd=sd_from_ranks(4.6,0.0);assert abs(mde(sd,32,1)-mde(sd,32,1))<1e-12 and mde(sd,32,3)<mde(sd,32,1) and thresholds(sd,32,1)['raised'] and thresholds(sd,32,3)['thr_rank']>=1.5
    sdr=sd_from_records(nat,swd);assert sdr>0
    print(table({'NAT':nat,'SWD':swd,'same-group boost':[_rec(32,rng,0.0,same_boost=3.0) for _ in range(3)]}))
    print('MDE (rank sd 4.6, independent cells): 1 seed %.2f, 2 seeds %.2f, 3 seeds %.2f at 32 facts'%tuple(mde(sd,32,s) for s in (1,2,3)))
    print('e55_metrics SELFTEST OK (synthetic scores)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--nat',default=None,help='comma list of NAT records');ap.add_argument('--swd',default=None);ap.add_argument('--a',type=int,default=16);ap.add_argument('--out',default=None);ap.add_argument('--thr',type=float,default=THR_RANK,help='rank threshold (raise to the MDE: see thresholds())');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.nat:print(stage0(a.nat,a.swd,a.a,a.out,a.thr))
    else:print(__doc__)
