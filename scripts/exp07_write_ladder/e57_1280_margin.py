"""E57 job 1280 margin code (A1 v12 2.6; new file, GPU-free). Input: candidate score arrays from the e55 reader (summed span log-probabilities, shape (n_items, 16), one row per item, the gold candidate at gold_idx; the e55 reader files compute the candidate scores only). Output per fact item:
  primary margin   = log p(gold) - logsumexp(log p(wrong))   (A1 2.6; used by every validity check, gate and record: M_NATS, M_MASK, M_RECOMP, M_RECOMP_PL, G_s, gain_noprior)
  secondary margin = log p(gold) - max log p(wrong)           (the convention of the 1283 analysis; descriptive only, written to e57_1280_secondary_margin.json after the validity confirmation, never used by a validity check, gate or option choice)
Aggregation of the seed values (A1 7.1): mean over the fact items within a chunk, then mean over the analysis-set chunks, then the seed value; `aggregate` implements this for lists of per-chunk margin arrays.
  python3 e57_1280_margin.py --selftest"""
import numpy as np
def _split(scores,gold_idx):
    L=np.asarray(scores,float);g=np.asarray(gold_idx,int);assert L.ndim==2 and len(g)==len(L) and L.shape[1]>=2 and np.isfinite(L).all(),'finite score matrix (items x candidates) required'
    gold=L[np.arange(len(g)),g];w=L.copy();w[np.arange(len(g)),g]=-np.inf;return gold,w
def margin_primary(scores,gold_idx):
    gold,w=_split(scores,gold_idx);m=w.max(1);lse=m+np.log(np.exp(w-m[:,None]).sum(1));return gold-lse
def margin_secondary(scores,gold_idx):
    gold,w=_split(scores,gold_idx);return gold-w.max(1)
def aggregate(per_chunk_margins):
    """per_chunk_margins: list over analysis-set chunks of 1-D arrays (the margins of the fact items of the chunk); mean over facts within a chunk, then mean over chunks"""
    assert len(per_chunk_margins)>0 and all(len(np.asarray(x))>0 for x in per_chunk_margins),'an empty chunk or an empty analysis set is a technical failure (A1 4.5, 7.1)'
    return float(np.mean([np.mean(np.asarray(x,float)) for x in per_chunk_margins]))
def selftest():
    S=np.array([[2.0,0.0,-1.0,-3.0],[0.5,1.0,0.0,0.0]]);g=[0,1]
    p=margin_primary(S,g);s=margin_secondary(S,g)
    assert abs(s[0]-2.0)<1e-12 and abs(s[1]-(1.0-0.5))<1e-12                                   # largest wrong: 0.0 and 0.5
    assert abs(p[0]-(2.0-np.log(np.exp(0.0)+np.exp(-1.0)+np.exp(-3.0))))<1e-12                 # logsumexp of the three wrong scores
    assert abs(p[1]-(1.0-np.log(np.exp(0.5)+np.exp(0.0)+np.exp(0.0))))<1e-12
    assert (p<=s+1e-12).all()                                                                  # logsumexp(wrong) >= max(wrong), so primary <= secondary
    big=np.array([[0.0,-1000.0,-1000.5]]);assert np.isfinite(margin_primary(big,[0])).all()    # stable for very negative scores
    L=np.array([[-10000.0,-10001.0,-10002.0]]);assert np.isfinite(margin_primary(L,[0])).all() and abs(margin_primary(L,[0])[0]-(-10000.0-(-10001.0+np.log(1+np.exp(-1.0)))))<1e-9
    assert abs(aggregate([np.array([1.0,3.0]),np.array([5.0])])-3.5)<1e-12                    # chunk means 2 and 5, then 3.5 (not the pooled mean 3.0)
    try:aggregate([]);raise SystemExit('empty analysis set accepted')
    except AssertionError:pass
    try:margin_primary(np.array([[1.0,np.nan]]),[0]);raise SystemExit('non-finite accepted')
    except AssertionError:pass
    print('e57_1280_margin selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
