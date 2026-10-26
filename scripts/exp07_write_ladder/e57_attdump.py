"""E57 pod code: ONE forward-only attention dump of the prefix layers per write chunk that serves BOTH label-free candidate 1 (token votes, e55_sentsel.select_heads) and the learned head-weighting mask (e55_sentgrid_ext features), and covers the WINDOW keys and the keys of the CURRENT chunk (Hans: the mask must act on the attention of all tokens, not only on earlier chunks).
Dump of one chunk k (weights W0: the prefix layers are frozen in TTT-E2E, so the attention of the prefix does not depend on the write state), unmasked natural window, every token of the chunk as a query:
  AL  float16 [L, H, nQ, nW + nQ]  attention mass of query sentence s (token mean) on window sentence w (columns 0..nW-1, sum over its key tokens) and on chunk sentence c (columns nW..nW+nQ-1, only c < s: a later query attends to an EARLIER sentence of its own chunk; the own sentence and later sentences are 0 by construction)
  T_max, T_key, T_wsid  [L, H, CH]  per query token and head over the WINDOW keys: maximum mass, key index, window sentence of the key (-1 none)       (as the lean pass of e55_selmask)
  T_cmax, T_ckey, T_csid [L, H, CH] per query token and head over the CHUNK keys of EARLIER sentences only: maximum mass, key index inside the chunk, chunk sentence of the key (-1 none)
The GPU wrapper (install / att4_pass) replays the 'att' branch of s3.py in a new mode 'att4' without touching s3.py; att4_reduce is the numpy / jax part and is CPU tested (e57_mask_stage.py --selftest)."""
import numpy as np
CH=1024;NW=8;WIN=NW*CH
def att4_reduce(ph,Mw,Mc,qavg,cstart,xp=np):
    """ph [H, n, WIN+n] attention probabilities (softmax over window + chunk keys, causal / window mask applied); Mw [WIN, nW] and Mc [n, nQ] one-hot key -> sentence; qavg [nQ, n] token-mean weights of the query sentences; cstart [n] start offset (in the chunk) of the sentence of every query token.
    -> dict of arrays of ONE layer: AL [H,nQ,nW+nQ], T_max/T_key [H,n] (window keys), T_cmax/T_ckey [H,n] (keys of earlier chunk sentences)"""
    pw=ph[:,:,:WIN];pc=ph[:,:,WIN:];n=pc.shape[1];nQ=Mc.shape[1]
    al_w=xp.einsum('sq,hqw->hsw',qavg,xp.einsum('hqk,kw->hqw',pw,Mw));al_c=xp.einsum('sq,hqc->hsc',qavg,xp.einsum('hqk,kc->hqc',pc,Mc))
    below=(xp.arange(nQ)[None,:]<xp.arange(nQ)[:,None])           # chunk sentence c strictly before query sentence s
    al_c=al_c*below[None].astype(al_c.dtype)
    valid=(xp.arange(n)[None,:]<xp.asarray(cstart)[:,None])        # keys of earlier sentences than the query token's own
    pcm=xp.where(valid[None],pc,-1.0)
    return dict(AL=xp.concatenate([al_w,al_c],axis=2),T_max=pw.max(-1),T_key=pw.argmax(-1),T_cmax=xp.maximum(pcm.max(-1),0.0),T_ckey=pcm.argmax(-1))
def layout4(ends,k,UVm,CH=CH,nw=NW):
    """-> dict: q = chunk sentences [(a,b)], wsid [WIN], wst, wlen (window sentences), csid [CH] (chunk sentence of every token), cstart [CH]"""
    q=UVm.chunk_sents(ends,k,CH);wsid,wst,wlen=UVm.window_sents(ends,k,CH,nw);csid=np.full(CH,-1,np.int32);cstart=np.zeros(CH,np.int32)
    for i,(a,b) in enumerate(q):csid[a:b]=i;cstart[a:b]=a
    return dict(q=q,wsid=wsid,wst=wst,wlen=wlen,csid=csid,cstart=cstart)
def install(s3):
    """add the mode 'att4' to the sliding-window attention of s3 (the 'att' branch of s3.py with the reduction of att4_reduce); other modes are passed on unchanged"""
    import jax,jax.numpy as jnp
    SWA=s3.SWA;prev=SWA.full_sw_attention;WIN_=s3.WIN
    def _full4(self,hidden_states,seq,state):
        if s3.S3.get('mode')!='att4':return prev(self,hidden_states,seq,state)
        S3=s3.S3;Kc,Vc=state;q0=S3['q0'];keep=S3['keep'];n=hidden_states.shape[0]
        xq,xk,xv=self.get_attention_input(hidden_states,position_ids=q0+jnp.arange(n,dtype=jnp.int32))
        K=jnp.concatenate([Kc.astype(xk.dtype),xk],axis=0);V=jnp.concatenate([Vc.astype(xv.dtype),xv],axis=0)
        qi=(q0+jnp.arange(n,dtype=jnp.int32))[:,None];kj=(q0-WIN_+jnp.arange(WIN_+n,dtype=jnp.int32))[None,:]
        mask=(qi>=kj)&(qi-kj<WIN_)&(kj>=0)&(keep if keep.ndim==2 else keep[None,:])
        q=xq.astype(jnp.float32);Kx=K.astype(jnp.float32)
        if Kx.shape[1]!=q.shape[1]:Kx=jnp.repeat(Kx,q.shape[1]//Kx.shape[1],axis=1)
        sc=jnp.einsum('qhd,khd->hqk',q,Kx)*(q.shape[-1]**-0.5);sc=jnp.where(mask[None],sc,-1e30);ph=jax.nn.softmax(sc,axis=-1)
        r=att4_reduce(ph,S3['att4_Mw'],S3['att4_Mc'],S3['att4_qavg'],S3['att4_cstart'],xp=jnp)
        S3['att4_out'].append({k_:np.asarray(v_,np.float16 if k_ in('AL','T_max','T_cmax') else np.int16) for k_,v_ in r.items()})
        o=jax.nn.dot_product_attention(xq,K,V,mask=mask,implementation='xla');o=self._merge_heads(o)
        return (self.get_attention_output(o),state)
    SWA.full_sw_attention=_full4
def att4_pass(L,rt,s3,UVm,m,Kc,Vc,ids_k,k,ends,all_ids):
    """GPU forward-only pass of the prefix layers for chunk k with the unmasked window -> d (the arrays of the module doc plus the sentence layout and token ids)"""
    import jax,jax.numpy as jnp
    lo=layout4(ends,k,UVm);q=lo['q'];nQ=len(q);nW=len(lo['wlen']);qavg=np.zeros((nQ,CH),np.float32)
    for i,(a,b) in enumerate(q):qavg[i,a:b]=1.0/(b-a)
    Mw=np.zeros((WIN,nW),np.float32);ok=lo['wsid']>=0;Mw[np.nonzero(ok)[0],lo['wsid'][ok]]=1.0
    Mc=np.zeros((CH,nQ),np.float32);okc=lo['csid']>=0;Mc[np.nonzero(okc)[0],lo['csid'][okc]]=1.0
    s3.S3.clear();s3.S3.update(mode='att4',q0=jnp.int32(k*CH),keep=jnp.asarray(np.ones(WIN+CH,bool)),att4_Mw=jnp.asarray(Mw),att4_Mc=jnp.asarray(Mc),att4_qavg=jnp.asarray(qavg),att4_cstart=jnp.asarray(lo['cstart']),att4_out=[])
    try:
        ids=jnp.asarray(ids_k,dtype=jnp.int32)
        with rt.mesh:
            h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
            for l in range(Kc.shape[0]):
                blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
        per=s3.S3['att4_out']
    finally:s3.S3.clear()
    return assemble(per,lo,all_ids,ids_k,k)
def assemble(per,lo,all_ids,ids_k,k,CH=CH,NW=NW):
    """per: list over layers of the att4_reduce arrays -> d (dict of numpy arrays) as the selectors expect"""
    q=lo['q'];nQ=len(q);wsid=lo['wsid'];csid=lo['csid']
    S={k_:np.stack([p[k_] for p in per]) for k_ in per[0]}
    key=S['T_key'].astype(np.int64);ckey=S['T_ckey'].astype(np.int64)
    wid=np.full(WIN,-1,np.int32)
    for c_ in range(max(0,k-NW),k):off=(c_-k+NW)*CH;wid[off:off+CH]=np.asarray(all_ids[c_*CH:(c_+1)*CH],np.int32)
    return {'AL':S['AL'],'T_max':S['T_max'],'T_key':S['T_key'],'T_wsid':wsid[key].astype(np.int16),'T_cmax':S['T_cmax'],'T_ckey':S['T_ckey'],'T_csid':np.where(S['T_cmax']>0,csid[ckey],-1).astype(np.int16),
        'q_start':np.asarray([a for a,_ in q],np.int32),'q_len':np.asarray([b-a for a,b in q],np.int32),'w_start':np.asarray(lo['wst'],np.int32),'w_len':np.asarray(lo['wlen'],np.int32),'c_len':np.asarray([b-a for a,b in q],np.int32),
        'w_chunk':(k-NW+np.asarray(lo['wst'],np.int32)//CH).astype(np.int32),'win_ids':wid,'q_ids':np.asarray(ids_k,np.int32)}
