"""E57 EXPQ row score (Council 11/09; pure numpy, the attention maps come from a separate GPU forward and are passed in). A query row i is 'repeat-exposed' when its attention goes to keys that follow an earlier occurrence of the current bigram, more than 256 tokens back:
  match[i,j] = ids[j-2]==ids[i-1] and ids[j-1]==ids[i], j < i, i-j > 256        (key j's two previous tokens equal the current bigram)
  s_i = mean over the selected induction heads of sum_j attn[h,i,j] x match[i,j];  targets: rows with global position i > 256 only.
Heads: from a random-token sequence of length T=512 repeated twice (256 + 256), induction score of a head = mean attention from the second-half position t to the key one after the first occurrence (t-255); heads with score > 0.3, else the top 8; fixed, uniform average.
EXPQ = the top 50% of the target rows of each chunk by s (ties broken by a seeded random key); RANDQ = a random 50% of the SAME target rows. Logs: share of selected rows with s>0, overlap EXPQ/RANDQ, share of target rows with s>0.
  python3 e57_expq_scores.py --selftest"""
import numpy as np
CHUNK=1024;MINDIST=256;THR=0.3;TOPK=8
def induction_heads(attn_rep,T=512):
    """attn_rep [H,T,T] (H = layers x heads flattened) on the repeated random sequence (period T/2) -> sorted indices of the selected heads, per-head scores"""
    H=attn_rep.shape[0];P=T//2;t=np.arange(P,T);sc=attn_rep[:,t,t-P+1].mean(1)
    sel=np.nonzero(sc>THR)[0]
    if len(sel)==0:sel=np.argsort(-sc,kind='stable')[:TOPK]
    return np.sort(sel),sc
def match_matrix(ids,mindist=MINDIST):
    """bool [L,L]: match[i,j] as above (row i, key j)"""
    ids=np.asarray(ids);L=len(ids);m=np.zeros((L,L),bool)
    if L<3:return m
    # key j (2..L-1): previous two tokens (j-2,j-1); query i (1..L-1): bigram (i-1,i)
    kj=np.stack([ids[:-2],ids[1:-1]],1)                  # keys j=2..L-1 -> index j-2
    qi=np.stack([ids[:-1],ids[1:]],1)                    # queries i=1..L-1 -> index i-1
    eq=(qi[:,None,0]==kj[None,:,0])&(qi[:,None,1]==kj[None,:,1])   # [L-1, L-2]
    m[1:,2:]=eq
    d=np.arange(L)[:,None]-np.arange(L)[None,:]
    return m&(d>mindist)
def row_scores(attn,ids,heads):
    """attn [H,L,L] (query, key), ids [L] -> s [L] (mean over heads of the attention mass on matching keys), target mask [L] (i > MINDIST)"""
    M=match_matrix(ids);a=attn[heads];s=(a*M[None]).sum(2).mean(0);tgt=np.arange(len(ids))>MINDIST;s=np.where(tgt,s,0.0);return s,tgt
def select_rows(s,tgt,seed,frac=0.5):
    """per chunk: EXPQ = top frac of the target rows by s, RANDQ = a random frac of the same target rows -> (expq bool [L], randq bool [L], log)"""
    L=len(s);rng=np.random.default_rng(seed);ex=np.zeros(L,bool);rd=np.zeros(L,bool);ov=[];sp=[];tp=[]
    for c0 in range(0,L,CHUNK):
        ix=np.nonzero(tgt[c0:c0+CHUNK])[0]+c0
        if len(ix)==0:continue
        n=int(np.ceil(frac*len(ix)));key=rng.random(len(ix))
        o=np.lexsort((key,-s[ix]));e=ix[o[:n]];r=rng.choice(ix,n,replace=False);ex[e]=True;rd[r]=True
        sp.append(float((s[e]>0).mean()));tp.append(float((s[ix]>0).mean()));ov.append(len(set(e)&set(r))/n)
    return ex,rd,{'expq_s_pos_frac':float(np.mean(sp)) if sp else float('nan'),'target_s_pos_frac':float(np.mean(tp)) if tp else float('nan'),'expq_randq_overlap':float(np.mean(ov)) if ov else float('nan')}
def auroc(pos,neg):
    pos=np.asarray(pos,float);neg=np.asarray(neg,float);a=np.concatenate([pos,neg]);r=np.empty(len(a));o=np.argsort(a,kind='stable');r[o]=np.arange(1,len(a)+1)
    for v in np.unique(a):
        k=a==v
        if k.sum()>1:r[k]=r[k].mean()
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def validate(s,tgt,rep_rows,far_rows,ex):
    """Council check on a synthetic sequence: rep_rows = rows inside a 128-token span repeated at distance 300-1000, far_rows = rows with no repeat. Returns AUROC (rep vs no-repeat targets), share of repeated rows selected, 99th percentile of s on no-repeat target rows; (rows with a repeat at distance < 256 are expected to have s ~ 0 by construction, since the match requires distance > 256)."""
    rep=np.asarray(rep_rows)&tgt;non=np.asarray(far_rows)&tgt
    return {'auroc':auroc(s[rep],s[non]),'rep_selected':float(ex[rep].mean()),'norep_p99':float(np.percentile(s[non],99))}
def selftest():
    rng=np.random.default_rng(0)
    # match_matrix on a hand case: ids = A B C ... with a repeat of (X,Y) then Z later than 256
    L=700;ids=rng.integers(100,10000,L);ids[10:13]=[7,8,9];ids[600:603]=[7,8,9]   # key j=600+... : bigram (7,8) at 10,11 -> key 12 holds 9; query i=11 has bigram (7,8)? use query i with (ids[i-1],ids[i])=(7,8) at i=601 -> keys j with (ids[j-2],ids[j-1])=(7,8): j=12
    M=match_matrix(ids);assert M[601,12] and not M[601,13] and not M[100,12]
    ids2=ids.copy();ids2[300:303]=[7,8,9];M2=match_matrix(ids2);assert not M2[301,12] or (301-12)>MINDIST   # distance 289 > 256 matches
    assert M2[301,12]==((301-12)>MINDIST)
    ids3=ids.copy();ids3[200:203]=[7,8,9];assert not match_matrix(ids3)[201,12]    # distance 189 < 256: no match
    # row scores with a fake attention that puts mass 0.6 on the matching key
    H=3;A=np.zeros((H,L,L));A[:,np.arange(L),np.maximum(np.arange(L)-1,0)]=0.5;A[:,601,12]=0.6
    s,tgt=row_scores(A,ids,np.array([0,1,2]));assert abs(s[601]-0.6)<1e-9 and s[100]==0 and not tgt[100] and tgt[601]
    # head selection: heads 1 and 4 are induction-like
    T=512;AR=rng.random((6,T,T))*1e-3
    for h in (1,4):AR[h,np.arange(256,512),np.arange(1,257)]=0.8
    sel,sc=induction_heads(AR,T);assert list(sel)==[1,4]
    AR2=rng.random((12,T,T))*1e-3;sel2,_=induction_heads(AR2,T);assert len(sel2)==TOPK
    # selection per chunk, RANDQ on the same target set
    Ls=2048;s_=rng.random(Ls)*(rng.random(Ls)<0.2);tg=np.arange(Ls)>MINDIST;ex,rd,lg=select_rows(s_,tg,1)
    assert not ex[:MINDIST+1].any() and not rd[:MINDIST+1].any() and ex.sum()==rd.sum() and (ex&~tg).sum()==0
    assert lg['expq_s_pos_frac']>=lg['target_s_pos_frac']
    # validation metric on a toy: repeated rows get s>0.3, others 0
    sv=np.zeros(Ls);rep=np.zeros(Ls,bool);rep[1000:1128]=True;sv[rep]=0.4;ex2,_,_=select_rows(sv,tg,2);v=validate(sv,tg,rep,~rep,ex2)
    assert v['auroc']==1.0 and v['rep_selected']==1.0 and v['norep_p99']==0.0
    print('e57_expq_scores selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
