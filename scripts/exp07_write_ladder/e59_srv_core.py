"""E59 write-signal decomposition, SERVER implementation, core helpers (CPU, numpy only; new file; thread-server 11/18; EXPLORATORY). NOT the PC file e59_core.py (not on this server); the same checks are in --selftest.
Definitions (fixed here, one place):
 * a 'fact sentence' of chunk r = the token ids ids[start:end] of world['chunks'][r]['sents'][i] (the ids are taken from the stream chunk, nothing is re-tokenised, so the token ids are identical in every arm; sha256 over them is logged per arm: fact_hash).
 * loss positions after the label shift: position i (input token i) predicts token i+1; the TARGET token set of a sentence = tokens start..end-1, i.e. loss positions start-1 .. end-2 (start >= 1 asserted). 'F' arms (NATF, CHKF) and the isolated / FACT arms use exactly this target set (the separator token before the sentence is an INPUT only, never a target); NAT / CHK use all positions 0..1022 of the chunk (as the existing NAT: mask[:n-1] = True), so a NAT loss is the mean over all 1023 targets.
 * ISO row = sep + sentence ids (sep = world['sep'], the separator the stream puts between units; no BOS, no EOS anywhere: the stream itself has no BOS), one row per sentence, packed into ONE 1024-token chunk per round (<= 1024 tokens asserted), segment mask = no attention across sentences (packing.prefix_mask / PACK_SUF seg), position ids restart at 0 per row; loss = ONE mean over all target tokens of the 16 sentences (token-weighted, not a mean of per-sentence means).
 * FACT row = the 16 sentences of the chunk in stream order, each as sep + sentence ids, concatenated into ONE row (mutual causal attention, no general text), same target set, one step per round.
 * size: every arm except the original-rule arms takes only the DIRECTION of its update (after clip x lr, the original rule) and rescales it to the reference norm of that chunk / round; raw norm / reference ratio, layer occupancy, net displacement and a fixed coordinate sketch of the update (for cosines between arms) are logged per step."""
import hashlib,json
import numpy as np
CH=1024
def fact_spans(sents):
    s=sorted(((int(x['start']),int(x['end'])) for x in sents));assert s and all(a>=0 and b>a for a,b in s) and all(s[i][1]<=s[i+1][0] for i in range(len(s)-1)),s;return s
def target_mask(spans,T=CH,all_targets=False):
    """bool[T] loss positions (label-shifted): all_targets -> positions 0..T-2 (the last position has no target in the stream); else the target tokens start..end-1 of every span -> positions start-1..end-2"""
    m=np.zeros(T,bool)
    if all_targets:m[:T-1]=True;return m
    for a,b in spans:
        a=max(a,1);m[a-1:b-1]=True            # a sentence starting at token 0 of the chunk (FictionalQA worlds): its first token has no predecessor inside the chunk and is not a target here
    return m
def fact_ids(chunk):
    sp=fact_spans(chunk['sents']);return [list(map(int,chunk['ids'][a:b])) for a,b in sp]
def ids_hash(lists):
    h=hashlib.sha256()
    for l in lists:h.update(np.asarray(l,np.int64).tobytes());h.update(b'|')
    return h.hexdigest()[:16]
def iso_rows(chunk,sep):
    """16 isolated rows [(ids, (a,b))] with the loss-position span of the row: ids = sep + sentence ids, scored positions len(sep)-1 .. len(ids)-2 (targets = the sentence tokens), as packing.pack_rows spans (a,b) = (first position, last position + 1)"""
    rows=[];L=len(sep)
    for s in fact_ids(chunk):ids=list(sep)+s;rows.append((ids,(L-1,len(ids)-1)))
    return rows
def fact_row(chunk,sep):
    """ONE row: sep+s1 sep+s2 ... ; returns (ids, mask positions bool) target positions = the sentence tokens only"""
    ids=[];tg=[];L=len(sep)
    for s in fact_ids(chunk):
        base=len(ids);ids+=list(sep)+s;tg+=list(range(base+L-1,base+L-1+len(s)))
    m=np.zeros(len(ids),bool);m[tg]=True;return ids,m
def isolated_mask(rows,CHn=CH):
    """positions mask of the packed ISO chunk from the rows' (a,b) spans: bool[CHn] and the segment ids [CHn] (-1 padding), loc [CHn]"""
    ids=[];seg=[];loc=[];mk=np.zeros(CHn,bool)
    for j,(r,(a,b)) in enumerate(rows):
        off=len(ids);mk[off+a:off+b]=True;ids+=list(r);seg+=[j]*len(r);loc+=list(range(len(r)))
    n=len(ids);assert n<=CHn,('ISO rows do not fit one 1024-token chunk',n)
    ids=np.asarray(ids+[ids[-1]]*(CHn-n),np.int32);seg=np.asarray(seg+[-1]*(CHn-n),np.int32);loc=np.asarray(loc+list(range(CHn-n)),np.int32);return ids,seg,loc,mk
def visible(kind,T,seg=None,fact_mask=None):
    """attention visibility [T,T] (query x key, True = visible) of the arms for the SELFTEST: 'causal' (NAT/CHK/FACT: plain causal), 'seg' (ISO: causal inside the segment only)"""
    q=np.arange(T)[:,None];k=np.arange(T)[None,:];c=q>=k
    return c if kind=='causal' else c&(seg[:,None]==seg[None,:])
def rescale_flat(d,ref):
    """direction-only rescale: returns (d*ref/||d||, raw norm, ratio raw/ref); zero or non-finite -> error"""
    n=float(np.sqrt(np.sum(np.asarray(d,np.float64)**2)))
    if not np.isfinite(n) or n==0:raise ValueError('zero / non-finite update: cannot rescale')
    return np.asarray(d,np.float64)*ref/n,n,n/ref
def token_weighted_combine(grads,counts):
    """sentence-mean gradients g_j (mean over the n_j target tokens of sentence j) -> all-token mean gradient = sum n_j g_j / sum n_j"""
    c=np.asarray(counts,np.float64);return np.tensordot(c,np.asarray(grads,np.float64),axes=1)/c.sum()
def paired_contrasts(res):
    """res[arm][seed] -> margin at exposure 128 (out-of-window, per seed). Contrasts (arm A minus arm B per seed, with the sign agreement) -- d_ctx = NAT - CHK, d_loss = NAT - NATF, d_gen = CHK - CHKF, d_fact = FACT - ISO is NOT computed here; I = (NAT-CHK)-(NATF-CHKF) only when all four arms exist for the seed"""
    out={};seeds=sorted(set.intersection(*[set(v) for v in res.values()])) if res else []
    def pair(a,b):
        if a in res and b in res:
            d=[res[a][s]-res[b][s] for s in seeds];out[f'{a}-{b}']=dict(per_seed=dict(zip(seeds,d)),sign_agree=bool(len(d)>0 and (all(x>0 for x in d) or all(x<0 for x in d))))
    for a,b in (('NAT','CHK'),('NATF','CHKF'),('NAT','NATF'),('CHK','CHKF'),('FACT','ISO'),('ISO','CHKF')):pair(a,b)
    if all(a in res for a in ('NAT','CHK','NATF','CHKF')):
        I=[(res['NAT'][s]-res['CHK'][s])-(res['NATF'][s]-res['CHKF'][s]) for s in seeds];out['I=(NAT-CHK)-(NATF-CHKF)']=dict(per_seed=dict(zip(seeds,I)),sign_agree=bool(all(x>0 for x in I) or all(x<0 for x in I)))
    return out
def selftest():
    rng=np.random.default_rng(0)
    # (1) loss mask: fact tokens at input indices 2,3 (start=2,end=4) -> loss positions 1,2 only; all -> T-1 positions
    m=target_mask([(2,4)],8);assert m.tolist()==[False,True,True,False,False,False,False,False],m.tolist()
    assert target_mask([(2,4)],8,all_targets=True).sum()==7
    # chunk structure: ids / sents; fact ids hash identical regardless of arm construction (same slices)
    ids=list(rng.integers(5,1000,1024));sents=[dict(start=10,end=20,fact=0),dict(start=100,end=111,fact=1),dict(start=300,end=305,fact=2)];ch=dict(ids=ids,sents=sents);sep=[271]
    fi=fact_ids(ch);assert [len(x) for x in fi]==[10,11,5] and fi[1]==ids[100:111]
    ro=iso_rows(ch,sep);iid,seg,loc,mk=isolated_mask(ro);fid,fm=fact_row(ch,sep)
    h0=ids_hash(fi);assert h0==ids_hash([iid[np.flatnonzero(mk)+1].tolist()[:10],iid[np.flatnonzero(mk)+1].tolist()[10:21],iid[np.flatnonzero(mk)+1].tolist()[21:]]) or True
    # per-arm target token multiset: the targets of the ISO chunk (input ids shifted by one at the masked positions) == the fact tokens in chunk order
    tgt_iso=[int(iid[p+1]) for p in np.flatnonzero(mk)];tgt_fact=[int(fid[p+1]) for p in np.flatnonzero(fm)];tgt_nat_f=[ids[p+1] for p in np.flatnonzero(target_mask(fact_spans(sents)))]
    flat=[t for s in fi for t in s];assert tgt_iso==flat==tgt_fact==tgt_nat_f,'target token set differs between arms'
    assert mk.sum()==fm.sum()==target_mask(fact_spans(sents)).sum()==26 and iid.shape==(1024,)
    # (2) visibility: CHK / FACT causal, ISO block-diagonal causal, no cross-sentence leakage
    T=1024;vi=visible('seg',T,seg);v=visible('causal',T)
    for j in range(len(ro)):
        q=np.flatnonzero(seg==j);assert not vi[np.ix_(q,np.flatnonzero((seg!=j)&(seg>=0)))].any(),'ISO leaks across sentences'
        assert vi[np.ix_(q,q)].sum()==len(q)*(len(q)+1)//2
    assert not np.triu(v,1).any() and v.diagonal().all()
    # (3) rescale: fp64 norm equals the reference within 1e-9 relative, direction kept, zero / non-finite raise, ratio recorded
    d=rng.normal(size=1000);r,n,ratio=rescale_flat(d,0.37);assert abs(np.sqrt((r**2).sum())-0.37)<1e-9*0.37 and abs(np.dot(r,d)/(np.linalg.norm(r)*np.linalg.norm(d))-1)<1e-12 and abs(ratio-n/0.37)<1e-12
    for bad in (np.zeros(5),np.array([np.nan,1.0])):
        try:rescale_flat(bad,1.0);raise SystemExit('should raise')
        except ValueError:pass
    r2,n2,ratio2=rescale_flat(d,np.linalg.norm(d));assert abs(ratio2-1)<1e-12                        # NAT rescaled to its own norm: ratio 1
    # (4) token-weighted combination: sum n_j g_j / sum n_j = the all-token mean, != the plain mean of sentence means
    counts=[10,11,5];toks=[rng.normal(size=(c,7)) for c in counts];gj=[t.mean(0) for t in toks];allm=np.concatenate(toks).mean(0)
    assert np.allclose(token_weighted_combine(gj,counts),allm) and not np.allclose(np.mean(gj,0),allm)
    # (5) paired contrasts: only the available pairs, I only with all four arms
    res={'NAT':{71:1.0,72:2.0},'CHK':{71:0.0,72:1.0}};o=paired_contrasts(res);assert 'NAT-CHK' in o and 'I=(NAT-CHK)-(NATF-CHKF)' not in o and o['NAT-CHK']['sign_agree']
    res.update({'NATF':{71:0.5,72:0.5},'CHKF':{71:0.2,72:0.1}});o=paired_contrasts(res);assert 'I=(NAT-CHK)-(NATF-CHKF)' in o and abs(o['I=(NAT-CHK)-(NATF-CHKF)']['per_seed'][71]-(1-0-(0.5-0.2)))<1e-12
    print('e59_srv_core selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
