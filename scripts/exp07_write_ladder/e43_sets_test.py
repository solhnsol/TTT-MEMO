"""E43 unit test (CPU, no model): Sets.sentencecover (K3), K3b, K3f on a synthetic sentence table with token ranges, and bit-identity of the existing cells.
Checks: (1) O is inside K3 on every O chunk, chunks without O mask nothing; (2) every K3 token lies in [window start, the fact sentence's place token) and in O or in an EARLIER round sentence of the SAME triple as one of the chunk's fact sentences (no other sentence is touched);
(3) K3b contains K3 and its extra tokens belong only to sentences of the same pair and another triple; (4) K3f contains K3, has the fill_total size where it is larger than K3, fills outside K3, and is deterministic; (5) the existing cells (oracle, othent, randdist, randfill, oraclefill, k6, k6p, recency, prefixall, oraclesub, othentsub) return exactly the stored golden hashes
(computed with the s3_sets.py before the E43 change). Prints the per-chunk n_masked of K3 / K3b on the synthetic table. With the real table the same numbers come from k3_counts.py (server)."""
import sys,hashlib,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W
GOLDEN='6f27039023c45328919c8763db77dbb3eae7e254ae69743fe3894496d8df0724'
def table():
    t=[]
    for tr in range(30):
        for r in range(1,5):
            pos=(r-1)*6144+tr*150+15;toks=list(range(pos-5,pos+3))
            t.append(dict(idx=len(t),t=f't{tr}',pair=f'p{tr//2}',round=r,place=[pos],names=[pos-3,pos-2],spans=[[pos],[pos-3,pos-2]],text='',toks=toks))
    return sorted(t,key=lambda s:s['place'][0])
def old_hash(sets):
    h=hashlib.sha256()
    for cell in ('oracle','othent','randdist','randfill','oraclefill','k6','k6p','recency','prefixall','oraclesub','othentsub','native3'):
        for k in range(63):
            S,fr=sets.chunk_set(cell,k);h.update(json.dumps([cell,k,[int(x) for x in S],float(fr)]).encode())
    return h.hexdigest()
def main():
    tb=table();sets=SETS.Sets(tb,101,2,0.35)
    assert old_hash(sets)==GOLDEN,'an existing cell changed'
    owner={};byidx={u['idx']:u for u in tb}
    for u in tb:
        for p in u['toks']:owner[p]=u['idx']
    rows=[];nO=0
    for k in range(63):
        O=sets.oracle(k);K3=sets.sentencecover(k);K3b=sets.sentencecover(k,True)
        if not O:assert K3==[] and K3b==[];continue
        nO+=1;lo=max(0,k*CH-W);assert set(O)<=set(K3) and set(K3)<=set(K3b) and K3==sorted(set(K3))
        cur=sets.cur_sentences(k);allowed={}
        for s in cur:
            for u in tb:
                if u['t']==s['t'] and u['round']<s['round']:allowed[u['idx']]=max(allowed.get(u['idx'],0),s['place'][0])
        for p in K3:
            assert p>=lo and p<k*CH+CH
            if p in set(O):continue
            assert p in owner and owner[p] in allowed and p<allowed[owner[p]],(k,p)   # the sentence is an earlier same-triple sentence of a fact sentence of the chunk; the token is before that fact sentence's place token
        pairs={s['pair'] for s in cur};ts={s['t'] for s in cur}
        for p in set(K3b)-set(K3):
            u=byidx[owner[p]];assert any(u['pair']==s['pair'] and u['t']!=s['t'] for s in cur),(k,p)
        tot=len(K3)+300;sets.fill_total={k:tot};sets.fill_attn_cands=True;F=sets.sentencecover_fill(k);F2=sets.sentencecover_fill(k)
        assert F==F2 and set(K3)<=set(F) and len(F)==tot and all(p>=4 and p<k*CH-128 for p in set(F)-set(K3)),(k,len(F),tot)
        sets.fill_total={k:len(K3)-1};assert sets.sentencecover_fill(k)==K3   # target below K3: nothing added
        sets.fill_total=None;sets.fill_attn_cands=False
        sets.sentencecover(k);rows.append((k,len(O),len(K3),len(K3b),sets.last_nsent))
    assert nO>0
    sel=SETS.select_by_ctx([-0.9,-0.9,-0.2,-0.65,-0.66,0.1],-0.65);assert sel==[False,True,False,False,True,False],sel   # chunk 0 never, strict inequality at tau
    d_=[0.0]+[-0.1*(k%7)-0.001*k for k in range(1,63)];q=SETS.select_by_ctx_quantile(d_,0.30);assert sum(q)==19 and not q[0] and all(d_[k]<=max(d_[j] for j in range(63) if q[j]) for k in range(63) if q[k]) and min(d_[k] for k in range(1,63) if not q[k])>=max(d_[k] for k in range(1,63) if q[k]),q   # 19 = round(0.30 x 62), the lowest ones
    print('k6ctx test OK: select_by_ctx tau boundary strict, chunk 0 never; k6ctxq selects',sum(q),'lowest-delta chunks of 62 (frac 0.30), chunk 0 never')
    k_=next(k for k in range(8,63) if sets.oracle(k));assert sets.prefixall(k_)==sets.chunk_set('k6p',k_)[0]   # a selected k6ctx chunk is masked exactly like a K6' chunk
    print('k6ctx / k6ctxq masks identical to K6\' prefixall on selected chunk',k_,' (sets.prefixall == chunk_set k6p); the s3.py wiring (S = prefixall if selected else empty) is GPU-side and not exercised here')
    print('E43 sets test OK; synthetic table, chunks with O',nO,'(chunk, |O|, |K3|, |K3b|, covered sentences):',rows[:6],'...')
    print('old-cell hash',old_hash(sets))
if __name__=='__main__':main()
