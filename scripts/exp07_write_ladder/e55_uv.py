"""E55 32.8 online label-free sentence-vote masks (UNIONVOTE) and their random-sentence controls.
Per write chunk k (before its write, current weights, NAT window): (1) forward-only 'att' pass of the prefix layers with ALL tokens of the chunk as queries (nothing from the world's labels); every (layer, head, query token) gives its top-m WINDOW keys; (2) sentence boundaries are found from the token stream only (end punctuation followed by a space/newline token, newline tokens, and chunk starts) - the fact table is NOT used for them;
(3) per query sentence s and window sentence j: score V[s,j] = number of top-m hits of the tokens of s on the tokens of j, summed over the heads of the layer group (default g3 = last third of the prefix layers); 'sum' (default, as the lab's sentsel vote variant: count of top-8 hits, zero scores excluded) or 'density' = V / len(j); window sentences shorter than MIN_TOK=4 tokens are never candidates; (4) each query sentence selects the top p % of the candidate window sentences with a positive score; (5) a window sentence is hidden when it is selected by >= k different query sentences; hidden keys are hidden from EVERY query of the chunk (1-D keep, like SENT). RS arms: random window sentences (seeded) until the hidden-token count of the matched UV arm at the same chunk is reached.
The fact table (chunks_meta) is used ONLY for the per-chunk log (recall / precision / background false-mask rate); it never enters the mask. CPU tested with the tiny model (e55_units.py U40); the 3B GPU path is NOT run."""
import numpy as np
RS_SEED=55061;MIN_TOK=4
# Difference to the lab's offline diagnostic (e55_sentsel_vote_script.py): there the query units were the 16 question spans of the chunk (qfact rows, label-dependent); here the query units are ALL sentences of the write chunk (label-free). Candidates, score (sum of top-8 hits), p and k are the same.
def sentence_ends(enc,ids):
    """bool [N]: token i ends a sentence. Rule (label-free): the decoded token, minus trailing quote/bracket characters, ends with . ? ! and the NEXT token starts with a space or a newline (so 3.5 and e.g.x do not split); a token containing a newline ends a sentence; the last token of the stream ends one."""
    ids=np.asarray(ids);toks=[enc.id_to_token(int(i)) or '' for i in ids];n=len(ids);e=np.zeros(n,bool)
    for i,t in enumerate(toks):
        nl='Ċ' in t
        core=t.rstrip('"\')]}”’')
        nxt=toks[i+1] if i+1<n else ' '
        sp=(nxt[:1] in ('Ġ','Ċ',' ','\n')) or i+1==n
        if nl or (core[-1:] in ('.','?','!') and sp):e[i]=True
    if n:e[-1]=True
    return e
def chunk_sents(ends,c,CH=1024):
    """sentences of chunk c as (start,end) token offsets inside the chunk; a sentence never crosses a chunk boundary"""
    seg=ends[c*CH:(c+1)*CH];out=[];s=0
    for i,x in enumerate(seg):
        if x:out.append((s,i+1));s=i+1
    if s<len(seg):out.append((s,len(seg)))
    return out
def window_sents(ends,k,CH=1024,nw=8):
    """-> (sid [nw*CH] int32 (-1 = no sentence: chunk before the stream start), starts, lens) of the window of chunk k"""
    W=nw*CH;sid=np.full(W,-1,np.int32);st=[];ln=[]
    for c in range(max(0,k-nw),k):
        off=(c-k+nw)*CH
        for a,b in chunk_sents(ends,c,CH):sid[off+a:off+b]=len(st);st.append(off+a);ln.append(b-a)
    return sid,np.asarray(st,np.int64),np.asarray(ln,np.int64)
def vote_select(hidx,hval,qsid,wsid,wlen,p,kmin,score='sum'):
    """hidx, hval: [Lg,H,rows,m] top-m window key index / probability (hval<=0 = padding); qsid [rows] query sentence id; wsid [W] window sentence id of every window key (-1 none); wlen [nW].
    -> (hidden sentence mask [nW] bool, info dict)"""
    nQ=int(qsid.max())+1 if len(qsid) else 0;nW=len(wlen)
    if nW==0 or nQ==0:return np.zeros(nW,bool),{'n_query_sents':nQ,'n_window_sents':nW,'sel_per_query':0}
    rows=hidx.shape[2];I=np.moveaxis(hidx,2,0).reshape(rows,-1).astype(np.int64);Vv=np.moveaxis(hval,2,0).reshape(rows,-1)>0
    S=np.where(Vv,wsid[np.clip(I,0,len(wsid)-1)],-1);ok=S>=0
    rr=np.broadcast_to(qsid[:,None],S.shape)[ok];ss=S[ok];V=np.bincount(rr*nW+ss,minlength=nQ*nW).reshape(nQ,nW).astype(np.float64)
    if score=='density':V=V/np.maximum(wlen,1)[None,:]
    V[:,wlen<MIN_TOK]=0;nc=int((wlen>=MIN_TOK).sum())
    if nc==0:return np.zeros(nW,bool),{'n_query_sents':nQ,'n_window_sents':nW,'sel_per_query':0}
    nsel=max(1,int(np.ceil(nc*p/100.0)));order=np.argsort(-V,axis=1,kind='stable')[:,:nsel];sel=np.zeros((nQ,nW),bool);np.put_along_axis(sel,order,True,axis=1);sel&=V>0
    cnt=sel.sum(0);return cnt>=kmin,{'n_query_sents':nQ,'n_window_sents':nW,'sel_per_query':float(sel.sum(1).mean()),'votes_max':int(cnt.max())}
def rands_hidden(wlen,target_tokens,k,seed=RS_SEED):
    """random window sentences (shuffle by (seed, k)) until the hidden-token count reaches target_tokens (overshoot < one sentence)"""
    nW=len(wlen);h=np.zeros(nW,bool);t=0
    if target_tokens<=0 or nW==0:return h
    for j in np.random.default_rng([seed,k]).permutation(nW):
        if t>=target_tokens:break
        h[j]=True;t+=int(wlen[j])
    return h
def label_stats(hid_keys,chunks_meta,k,CH=1024,nw=8):
    """log-only: against the fact table - recall (hidden share of the fact-sentence tokens of the window), precision (fact share of the hidden tokens), background false-mask rate (hidden share of the background tokens), fact share of the window"""
    import e55_diag as DG
    valid,fact,_,_=DG.key_labels(chunks_meta,k,CH,nw);W=nw*CH;h=np.zeros(W+CH,bool);h[:W]=hid_keys;ft=valid&(fact>=0);bg=valid&(fact<0)
    return {'recall':float((h&ft).sum()/max(ft.sum(),1)),'precision':float((h&ft).sum()/max(h.sum(),1)) if h.sum() else None,'bg_false':float((h&bg).sum()/max(bg.sum(),1)),'fact_share':float(ft.sum()/max(valid.sum(),1)),'share':float(h.sum()/max(valid.sum(),1))}
class UV:
    """one object per arm: .keep(k, m, Kc, Vc, ids_k, keep_all) -> (keep 1-D bool [WIN+CH], log dict); the GPU/att pass is in .votes (monkey-patchable for the CPU test)"""
    def __init__(self,arm,ends,chunks_meta=None,L=None,rt=None,s3=None,topm=8,group='g3',score='sum',rs_targets=None,CH=1024,nw=8,log=print):
        import e55_swap as E
        self.arm=arm;self.ends=np.asarray(ends);self.meta=chunks_meta;self.L=L;self.rt=rt;self.s3=s3;self.topm=topm;self.group=group;self.score=score;self.CH=CH;self.nw=nw;self.rs_targets=rs_targets;self.log=log
        if arm in E.UV:self.p,self.kmin=E.UV[arm]
        else:assert arm in E.RS and rs_targets is not None and len(rs_targets)>0,('random-sentence arm needs the per-chunk hidden-token counts of its UV arm',arm)
    def att_pass(self,k,m,Kc,Vc,ids_k,keep):
        """forward-only eager prefix pass with all chunk tokens as queries -> (hidx, hval) [nL,H,CH,topm]"""
        import jax,jax.numpy as jnp;L,rt,s3=self.L,self.rt,self.s3;CH=self.CH;lo=(self.nw-min(k,self.nw))*CH
        s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(keep);s3.S3['att_rows']=jnp.arange(CH,dtype=jnp.int32);s3.S3['att_out']=[];s3.S3['att_noout']=True;s3.S3['att_topk']=self.topm;s3.S3['att_lo']=lo;s3.S3['att_heads']=[]
        try:
            ids=jnp.asarray(ids_k,dtype=jnp.int32)
            with rt.mesh:
                h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
                for l in range(Kc.shape[0]):
                    blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
            heads=s3.S3['att_heads']
        finally:s3.S3.clear()
        nL=len(heads);b=[0,nL//3,2*nL//3,nL];a_,b_=dict(g1=(b[0],b[1]),g2=(b[1],b[2]),g3=(b[2],b[3]),all=(0,nL))[self.group]
        return np.stack([x[0] for x in heads[a_:b_]]),np.stack([x[1] for x in heads[a_:b_]])
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        CH,nw=self.CH,self.nw;W=nw*CH;wsid,wst,wlen=window_sents(self.ends,k,CH,nw);keep=np.ones(W+CH,bool);info={'uv_arm':self.arm,'uv_window_sents':int(len(wlen))}
        if len(wlen)==0:hid=np.zeros(0,bool)
        elif self.arm in __import__('e55_swap').UV:
            hidx,hval=self.att_pass(k,m,Kc,Vc,ids_k,keep_all);qs=[];
            for i,(a,b) in enumerate(chunk_sents(self.ends,k,CH)):qs+= [i]*(b-a)
            hid,vi=vote_select(hidx,hval,np.asarray(qs,np.int64),wsid,wlen,self.p,self.kmin,self.score);info.update(vi)
        else:hid=rands_hidden(wlen,int(self.rs_targets[k]),k)
        hk=np.zeros(W,bool)
        if len(wlen):hk=(wsid>=0)&hid[np.clip(wsid,0,len(wlen)-1)]
        keep[:W]=~hk
        if self.meta is not None:info.update({'lab_'+a:b for a,b in label_stats(hk,self.meta,k,CH,nw).items()})
        info['uv_hidden_tokens']=int(hk.sum());self.log(f'  {self.arm} chunk {k}: hidden window tokens {int(hk.sum())} ({hk.sum()/max((wsid>=0).sum(),1):.3f} of window)'+(f", fact recall {info['lab_recall']:.2f} precision {info['lab_precision'] if info['lab_precision'] is None else round(info['lab_precision'],2)} bg false {info['lab_bg_false']:.2f}" if 'lab_recall' in info else '')) if k%16==0 or k<3 else None
        return keep,info
