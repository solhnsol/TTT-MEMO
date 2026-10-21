"""E49 full-candidate readout, CPU part (no jax): candidate bookkeeping and statistics.
Input per cell and world w in {A,B}: S[w][readout] = array (n_items, n_cand) of span log-probs of every candidate place after each L3 prompt
(readout 'R1' = neutral pg1342 tail, 'alt' = SQuAD tail, the headline readout of D_alt).
An L3 item has gold = the place of the named person in world A and foil = the swap-pair partner's place; in world B the binding is flipped, so the BOUND target of an item is
the gold place in world A and the foil place in world B (the same convention as E47 'bound lp' in e47_terms.py).
Per item and world: abs_lp = log-prob of the bound target, lognorm = log softmax over all candidates at the target, p_norm, rank (1 = best, 1 + #candidates with a higher score), top1, top3,
mass_partner (normalized prob of the OTHER place of the swap pair), acc2 ((gold>foil) in A, (foil>gold) in B), abs_lp_unbound (the other pair place).
Item values are the mean over the two worlds ('AB'); pair value = mean over the pair's items; CI = bootstrap over pairs (15). Changes vs the reference cell are paired by pair.
Pre-registered rule (E49): a change counts only if the CI excludes 0 AND |delta| >= 0.2 nats (abs_lp, lognorm) / 0.10 (accuracy-like: acc2, top1, top3, p_norm) ; rank change is descriptive."""
import json,numpy as np
METRICS=['abs_lp','abs_lp_unbound','lognorm','p_norm','rank','top1','top3','mass_partner','acc2']
THR={'abs_lp':0.2,'abs_lp_unbound':0.2,'lognorm':0.2,'p_norm':0.10,'top1':0.10,'top3':0.10,'acc2':0.10,'mass_partner':0.10,'rank':None}
def place_index(places,texts):
    """places: list of candidate strings (e.g. ' '+P), texts: item gold/foil strings -> indices; every text must be exactly one candidate and candidates must be distinct."""
    assert len(set(places))==len(places),'candidate places are not distinct'
    ix={p:i for i,p in enumerate(places)};out=[]
    for t in texts:
        if t not in ix:raise ValueError(f'item text {t!r} is not a candidate')
        out.append(ix[t])
    return np.asarray(out)
def item_metrics(S,gi,fi,world):
    """S (n,C); gi, fi = gold/foil candidate index per item; world 'A' (bound target = gold) or 'B' (bound target = foil) -> dict metric -> (n,) array."""
    S=np.asarray(S,float);n=len(S);r=np.arange(n);tg,ot=(gi,fi) if world=='A' else (fi,gi)
    assert S.shape[0]==n and (gi!=fi).all()
    m=S-S.max(1,keepdims=True);ls=m-np.log(np.exp(m).sum(1,keepdims=True));tv=S[r,tg]
    rank=1+(S>tv[:,None]).sum(1)
    return {'abs_lp':tv,'abs_lp_unbound':S[r,ot],'lognorm':ls[r,tg],'p_norm':np.exp(ls[r,tg]),'rank':rank.astype(float),'top1':(rank==1).astype(float),'top3':(rank<=3).astype(float),
            'mass_partner':np.exp(ls[r,ot]),'acc2':(S[r,tg]>S[r,ot]).astype(float)}
def cell_items(SA,SB,gi,fi):
    """-> dict metric -> {'A':..,'B':..,'AB':..} per item."""
    a=item_metrics(SA,gi,fi,'A');b=item_metrics(SB,gi,fi,'B');return {k:{'A':a[k],'B':b[k],'AB':(a[k]+b[k])/2} for k in METRICS}
def pair_means(v,pairs):
    d={}
    for x,p in zip(v,pairs):d.setdefault(p,[]).append(x)
    ks=sorted(d,key=lambda s:int(str(s).lstrip('px')));return np.asarray([np.mean(d[k]) for k in ks])
def ci(v,nboot=4000,seed=49):
    v=np.asarray(v,float);rng=np.random.default_rng(seed);idx=rng.integers(0,len(v),(nboot,len(v)));b=v[idx].mean(1);return float(v.mean()),*np.percentile(b,[2.5,97.5]).tolist()
def counts(delta_ci,metric):
    m,lo,hi=delta_ci;t=THR[metric]
    if t is None:return None
    return bool((lo>0 or hi<0) and abs(m)>=t)
def summarize(items_by_cell,pairs,ref=None,nboot=4000):
    """items_by_cell: cell -> output of cell_items (one readout). -> {'cell':{metric:{'AB':ci,'A':ci,'B':ci}}, 'change':{cell:{metric:(ci,counts)}} vs ref}"""
    out={'cell':{},'change':{}}
    for c,im in items_by_cell.items():out['cell'][c]={k:{w:ci(pair_means(im[k][w],pairs),nboot) for w in ('AB','A','B')} for k in METRICS}
    if ref is not None:
        for c,im in items_by_cell.items():
            if c==ref:continue
            ch={}
            for k in METRICS:
                d=pair_means(im[k]['AB'],pairs)-pair_means(items_by_cell[ref][k]['AB'],pairs);r=ci(d,nboot);ch[k]=(r,counts(r,k))
            out['change'][c]=ch
    return out
def fmt(o,readout,ref):
    L=[f'### E49 full-candidate readout ({readout}); item = mean of world A (target = gold place) and world B (target = flipped place); mean [95% pair-bootstrap CI]; chance: rank 15.5, top1 0.033, top3 0.10, p_norm 0.033']
    ks=['abs_lp','lognorm','p_norm','rank','top1','top3','mass_partner','acc2']
    L.append('| cell | '+' | '.join(ks)+' |');L.append('|'+'---|'*(len(ks)+1))
    for c,m in o['cell'].items():L.append(f'| {c} | '+' | '.join('%.3f [%.3f, %.3f]'%tuple(m[k]['AB']) for k in ks)+' |')
    if o['change']:
        L.append(f'\n change vs {ref}: delta [CI] and * where the pre-registered rule counts it (CI excludes 0 and |delta| >= 0.2 nats / 0.10)');L.append('| cell | '+' | '.join(ks)+' |');L.append('|'+'---|'*(len(ks)+1))
        for c,m in o['change'].items():L.append(f'| {c} | '+' | '.join('%+.3f [%+.3f, %+.3f]%s'%(*m[k][0],'*' if m[k][1] else '') for k in ks)+' |')
    return '\n'.join(L)

def candidate_items(l3_items,triples):
    """The 30-place candidate set of the L3 items: candidates = ' '+P of every triple in id order (ids 0..n-1, places distinct) -> (places, gi, fi, rows, pairs); rows = [(prefix, candidate text)] item-major (row i*C+c = item i, candidate c).
    Checks: gold / foil of every item are candidates, are the places of two different triples of the item's own swap pair (gold = the named person's place, foil = the partner's)."""
    tr=sorted(triples,key=lambda t:t['id']);assert [t['id'] for t in tr]==list(range(len(tr))),'triple ids are not 0..n-1'
    places=[' '+t['P'] for t in tr];gi=place_index(places,[it['gold'] for it in l3_items]);fi=place_index(places,[it['foil'] for it in l3_items])
    for it,g,f in zip(l3_items,gi,fi):
        assert tr[g]['pair']==it['pair']==tr[f]['pair'] and g!=f and tr[g]['id']==it['tid'],('item gold/foil are not the pair places',it)
    rows=[(it['prefix'],c) for it in l3_items for c in places];return places,gi,fi,rows,[it['pair'] for it in l3_items]
def cell_record(cell,places,gi,fi,pairs,S,meta=None):
    """S: {world: {readout: (n,C) array}} -> json-able record (raw scores kept, so every statistic can be recomputed on the CPU)."""
    return {'cell':cell,'places':places,'gi':gi.tolist(),'fi':fi.tolist(),'pairs':list(pairs),'S':{w:{r:np.asarray(v).tolist() for r,v in d.items()} for w,d in S.items()},'meta':meta or {}}
def analyze(records,ref=None,nboot=4000):
    """records: list of cell_record dicts (same items) -> {readout: summarize(...)} and a printable report; the first record is the reference unless ref is given."""
    ref=ref or records[0]['cell'];R0=records[0]
    for r in records:assert r['pairs']==R0['pairs'] and r['gi']==R0['gi'] and r['fi']==R0['fi'],'cells were read on different items'
    out={};text=[]
    for ro in sorted(R0['S']['A']):
        ib={r['cell']:cell_items(r['S']['A'][ro],r['S']['B'][ro],np.asarray(r['gi']),np.asarray(r['fi'])) for r in records}
        out[ro]=summarize(ib,R0['pairs'],ref,nboot);text.append(fmt(out[ro],ro,ref))
    return out,'\n\n'.join(text)
def selftest():
    # 2 pairs, 4 places; items: person a of pair 0 (gold place 0, foil 1), b (gold 1, foil 0), pair 1: places 2,3
    places=[' P0',' P1',' P2',' P3'];gi=place_index(places,[' P0',' P1',' P2',' P3']);fi=place_index(places,[' P1',' P0',' P3',' P2']);pairs=['p0','p0','p1','p1']
    SA=np.array([[-1,-2,-5,-6],[-3,-1,-5,-6],[-5,-6,-2,-1],[-5,-6,-3,-1.]])   # world A: items 0,1 pick the gold; item 2 picks the foil (3), item 3 gold
    SB=np.array([[-2,-1,-5,-6],[-1,-3,-5,-6],[-5,-6,-2,-1],[-5,-6,-1,-2.]])    # world B: bound target = the foil: item0 -> P1, item1 -> P0, item2 -> P3, item3 -> P2 (all best)
    a=item_metrics(SA,gi,fi,'A');b=item_metrics(SB,gi,fi,'B')
    assert list(a['rank'])==[1,1,2,1],a['rank'];assert list(a['top1'])==[1,1,0,1];assert list(a['acc2'])==[1,1,0,1]
    assert list(b['rank'])==[1,1,1,1],b['rank'];assert list(b['acc2'])==[1,1,1,1]
    assert abs(a['abs_lp'][0]+1)<1e-12 and abs(a['abs_lp_unbound'][0]+2)<1e-12 and abs(b['abs_lp'][2]+1)<1e-12   # item 2 in B: bound target = foil P3 with lp -1
    # hand-computed normalized prob for item 0 in A: exp(-1)/(exp(-1)+exp(-2)+exp(-5)+exp(-6))
    z=np.exp(-1)+np.exp(-2)+np.exp(-5)+np.exp(-6);assert abs(a['p_norm'][0]-np.exp(-1)/z)<1e-12 and abs(a['mass_partner'][0]-np.exp(-2)/z)<1e-12 and abs(a['lognorm'][0]-(-1-np.log(z)))<1e-12
    ci_=cell_items(SA,SB,gi,fi);assert list(ci_['top1']['AB'])==[1,1,.5,1]
    # ties do not help the target: equal scores -> rank 1 + #strictly higher
    t=item_metrics(np.array([[-1,-1,-1,-1.]]),np.array([0]),np.array([1]),'A');assert t['rank'][0]==1 and t['top1'][0]==1
    # summary / change with a reference of identical items: delta 0, never counts
    s=summarize({'ref':ci_,'x':ci_},pairs,'ref',200);assert s['change']['x']['top1'][0][0]==0 and not s['change']['x']['top1'][1]
    pm=pair_means(np.array([1.,3,5,7]),pairs);assert list(pm)==[2,6]
    try:place_index(places,[' PX']);raise SystemExit('missing candidate not rejected')
    except ValueError:pass
    print(fmt(s,'selftest','ref'));print('e49_stats selftest ok')
if __name__=='__main__':selftest()
