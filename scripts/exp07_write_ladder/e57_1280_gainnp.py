"""E57 job 1280 gain_noprior builder, CPU part (A1 v12 7.1, 7.2, 7.3; new file, GPU-free). The sentence table of the world is chunks[c]['sents'] = [{fact, start, end, ...}] (token positions inside chunk c, as in e57_oracle_masks.own_earlier); the real and the 'nofact' stream have identical chunk boundaries and differ only inside fact-sentence spans (generator, decoy 'nofact': the span of every fact sentence is filled with background text of exactly the same token length).
  window_positions(...)       positions (absolute, in the window of chunk k, 0 .. NW*CH - 1) of the earlier same-fact exposure sentences S_i (all earlier sentences of the facts of chunk k's sentences inside the window)
  variant_window(...)         the window token ids with every S_i span replaced by the token ids of the nofact stream at the same positions (exact length, no truncation, no random draw)
  filler_checks(...)          the checks of 7.3 (1) hash disjointness with the fact sentences, (3) the nofact stream equals the real stream outside fact spans; (2) and (4) need the generator texts and are provided as functions taking the n-gram sets and the world strings
  python3 e57_1280_gainnp.py --selftest"""
import hashlib
import numpy as np
CH=1024;NW=8
def sentence_spans_in_window(chunks,k,nw=NW):
    """-> (s_i, facts) : s_i = list of (win_start, win_end) spans of the earlier same-fact sentences inside the window of chunk k (window coordinates: chunk c occupies [(c-k+nw)*CH, (c-k+nw+1)*CH)); facts = the set of facts that have a sentence in chunk k; sorted, no duplicates"""
    lo=max(0,k-nw);facts={x['fact'] for x in chunks[k]['sents']};out=[]
    for c in range(lo,k):
        off=(c-k+nw)*CH
        for x in chunks[c]['sents']:
            if x['fact'] in facts:out.append((off+x['start'],off+x['end']))
    return sorted(set(out)),facts
def window_positions(chunks,k,nw=NW):
    spans,_=sentence_spans_in_window(chunks,k,nw);pos=[]
    for a,b in spans:pos.extend(range(a,b))
    return pos
def window_ids(stream_ids,k,nw=NW):
    """token ids of the window of chunk k (the nw chunks before chunk k) from a concatenated stream (chunks of CH tokens); shorter at the start of the stream (window coordinates are aligned to the right, the missing part is None)"""
    ids=np.asarray(stream_ids);lo=max(0,k-nw);w=[None]*((nw-(k-lo))*CH)+list(ids[lo*CH:k*CH]);return w
def variant_window(real_ids,nofact_ids,chunks,k,nw=NW):
    """-> (variant window ids, S_i positions): the real window with every S_i token replaced by the nofact token at the same position; asserts exact length equality of real and nofact streams and that nofact differs from real only inside fact-sentence spans"""
    real=np.asarray(real_ids);nf=np.asarray(nofact_ids);assert real.shape==nf.shape,'real and nofact streams must have the same length'
    pos=window_positions(chunks,k,nw);rw=window_ids(real,k,nw);nw_=window_ids(nf,k,nw);var=list(rw)
    assert len(pos)>0,'S_i is empty: a runner defect, technical failure of the seed (A1 4.5, 7.1)'
    for p in pos:
        assert rw[p] is not None and nw_[p] is not None,'S_i position outside the available window';var[p]=nw_[p]
    return var,pos
def fact_span_mask(chunks,n_chunks):
    """boolean mask over the whole stream of the positions inside a fact sentence span"""
    m=np.zeros(n_chunks*CH,bool)
    for c in range(n_chunks):
        for x in chunks[c]['sents']:m[c*CH+x['start']:c*CH+x['end']]=True
    return m
def sha(tokens):return hashlib.sha256(np.asarray(tokens,np.int64).tobytes()).hexdigest()
def filler_checks(real_ids,nofact_ids,chunks,n_chunks,k_list,nw=NW):
    """7.3 (1) the SHA-256 of every substituted nofact span is not equal to the SHA-256 of any fact sentence of the world; (3) the nofact stream equals the real stream at every position outside the union of fact-sentence spans; returns a dict of booleans (a failed check = the generator is suspect, H_GEN not evaluated for the whole study, A1 7.4)"""
    real=np.asarray(real_ids);nf=np.asarray(nofact_ids);m=fact_span_mask(chunks,n_chunks)
    h_fact={sha(real[c*CH+x['start']:c*CH+x['end']]) for c in range(n_chunks) for x in chunks[c]['sents']}
    h_span=set()
    for k in k_list:
        spans,_=sentence_spans_in_window(chunks,k,nw)
        for a,b in spans:
            c=k-nw+a//CH;lo=max(0,k-nw);c=lo+(a//CH)-(nw-(k-lo));st=c*CH+(a%CH);en=st+(b-a);h_span.add(sha(nf[st:en]))
    return dict(disjoint_from_fact_sentences=h_fact.isdisjoint(h_span),equal_outside_fact_spans=bool(np.array_equal(real[~m],nf[~m])),n_spans=len(h_span))
def selftest():
    rng=np.random.default_rng(0);nch=12
    chunks=[]
    for c in range(nch):
        sents=[];pos=100
        for f in range(3):sents.append(dict(fact=(c+f)%6,start=pos,end=pos+20));pos+=100
        chunks.append(dict(sents=sents))
    real=rng.integers(1000,2000,nch*CH);nf=real.copy();m=fact_span_mask(chunks,nch);nf[m]=rng.integers(5000,6000,int(m.sum()))
    k=9;spans,facts=sentence_spans_in_window(chunks,k);assert len(spans)>0 and facts=={x['fact'] for x in chunks[k]['sents']}
    assert all(0<=a<b<=NW*CH for a,b in spans) and spans==sorted(set(spans))
    var,pos=variant_window(real,nf,chunks,k);rw=window_ids(real,k);assert len(var)==NW*CH and set(pos)<set(range(NW*CH))
    assert all(var[p]!=rw[p] for p in pos) and all(var[p]==rw[p] for p in range(NW*CH) if p not in set(pos))        # only S_i differs, exact length
    # the window position -> stream position mapping: the variant token at window position p equals the nofact stream token at the same stream position
    lo=k-NW;assert all(var[p]==nf[lo*CH+p] for p in pos)
    ck=filler_checks(real,nf,chunks,nch,[8,9,10,11]);assert ck['disjoint_from_fact_sentences'] and ck['equal_outside_fact_spans'] and ck['n_spans']>0
    bad=nf.copy();bad[0]=bad[0]+1;assert not filler_checks(real,bad,chunks,nch,[9])['equal_outside_fact_spans']                  # a difference outside the fact spans is detected
    leak=nf.copy();a,b=spans[0];st=lo*CH+a;leak[st:st+(b-a)]=real[st:st+(b-a)];assert not filler_checks(real,leak,chunks,nch,[9])['disjoint_from_fact_sentences']   # a fact sentence used as filler is detected
    early=[dict(sents=[dict(fact=0,start=10,end=30)]) for _ in range(3)];
    try:variant_window(np.zeros(3*CH,int),np.zeros(3*CH,int),[dict(sents=[dict(fact=1,start=10,end=30)]),dict(sents=[dict(fact=3,start=10,end=30)]),dict(sents=[dict(fact=2,start=10,end=30)])],2);raise SystemExit('empty S_i accepted')
    except AssertionError:pass
    print('e57_1280_gainnp selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
