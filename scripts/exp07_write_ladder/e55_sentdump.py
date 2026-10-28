"""E55 32.10 sentence-level selection dump (forward only, no training step of its own): at the chunks of a running stream (weights BEFORE the write of the chunk, window = the real 8 chunks WITHOUT any mask), the attention mass of every sentence of the write chunk on every sentence of the window.
Sentences come from the token stream only (e55_uv.sentence_ends: end punctuation + space / newline, newline, chunk start); NOTHING of the fact table is used here and this module does not import the label module.
AL[layer,h,q,w]: the same per layer (float16); T_max / T_sum / T_ent / T_wsid [layer,h,CH]: per query token and head over the window keys: max attention mass, summed mass, entropy of the window mass, window sentence of the max key (-1 none).
A[g,h,q,w]: g = layer group g1 / g2 / g3 (thirds of the prefix layers, mean over the layers of the group), h = query head, q = sentence of the write chunk, w = sentence of the window; value = attention mass of the tokens of q on the key tokens of w (summed over the key tokens of w), averaged over the query tokens of q (row sums <= 1; the rest is on the current chunk and the sinks). float16.
Per query sentence: token count, first-token offset in the chunk, mean token CE of the write forward (the chunk loss under the arm's own window and weights before the update; positions with loss mask only). Per window sentence: token count, offset in the window, chunk number. Token ids of the chunk and of the window are stored (for the CPU rare / repeated score). Used with L.run_stream(..., diag_fn=fn, diag_chunks=...) (fn.wants_ctx = True: the stream passes the suffix state, the chunk and the prefix output of the write)."""
import numpy as np
import e55_uv as UVm
CH=1024;NW=8;WIN=NW*CH
def layout(ends,k,CH=CH,nw=NW):
    """-> (query sentences [(a,b)] inside chunk k, window key -> sentence id [WIN] (-1 none), window sentence starts, window sentence lengths)"""
    q=UVm.chunk_sents(ends,k,CH);wsid,wst,wlen=UVm.window_sents(ends,k,CH,nw);return q,wsid,wst,wlen
def groups(nL):
    b=[0,nL//3,2*nL//3,nL];return {'g1':(b[0],b[1]),'g2':(b[1],b[2]),'g3':(b[2],b[3])}
def group_mean(per):
    """per: list of nL arrays [H,nQ,nW] float32 -> float16 [3,H,nQ,nW] (g1, g2, g3; mean over the layers of the group)"""
    g=groups(len(per));return np.stack([np.mean(per[a:b],axis=0) for a,b in g.values()]).astype(np.float16)
def make_sentdump_fn(L,rt,s3,ends,all_ids,log=print):
    """-> fn(k,m,Kc,Vc,keep,ids_k,ctx) for L.run_stream(diag_fn=fn); ends: bool [N] sentence ends of the whole stream (e55_uv.sentence_ends); all_ids: the stream token ids (for the stored window ids)"""
    import jax,jax.numpy as jnp,equinox as eqx
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        q,wsid,wst,wlen=layout(ends,k);nQ,nW=len(q),len(wlen);assert nQ>0 and nW>0,(k,nQ,nW)
        qavg=np.zeros((nQ,CH),np.float32)
        for i,(a,b) in enumerate(q):qavg[i,a:b]=1.0/(b-a)
        M=np.zeros((WIN,nW),np.float32);ok=wsid>=0;M[np.nonzero(ok)[0],wsid[ok]]=1.0
        lo=(NW-min(k,NW))*CH;keep_all=np.ones(WIN+CH,bool)
        s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(keep_all);s3.S3['att_rows']=jnp.arange(CH,dtype=jnp.int32);s3.S3['att_out']=[];s3.S3['att_noout']=True;s3.S3['att_lo']=lo;s3.S3['att_heads']=[]
        s3.S3['att_sentM']=jnp.asarray(M);s3.S3['att_qavg']=jnp.asarray(qavg);s3.S3['att_sent']=[];s3.S3['att_tokstat']=[];s3.S3['att_vnorm']=[];s3.S3['att_sentv']=[]
        try:
            ids=jnp.asarray(ids_k,dtype=jnp.int32)
            with rt.mesh:
                h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
                for l in range(Kc.shape[0]):
                    blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
            per=s3.S3['att_sent'];ts=s3.S3['att_tokstat'];pv=s3.S3['att_sentv'];vnl=s3.S3['att_vnorm']
        finally:s3.S3.clear()
        assert len(per)==Kc.shape[0]==len(ts),(len(per),Kc.shape[0],len(ts))
        A=group_mean(per);AV=np.stack(pv).astype(np.float16);VN=np.stack(vnl);Tk=np.stack([t[3] for t in ts]);Tsk=np.stack([t[4] for t in ts]);Trc=np.stack([t[5] for t in ts]);AL=np.stack(per).astype(np.float16);Tm=np.stack([t[0] for t in ts]);Ts=np.stack([t[1] for t in ts]);Te=np.stack([t[2] for t in ts]);Tw=np.where(wsid[np.stack([t[3] for t in ts]).astype(np.int64)]>=0,wsid[np.stack([t[3] for t in ts]).astype(np.int64)],-1).astype(np.int16);rs=float(np.max(A.astype(np.float32).sum(-1)))
        if ctx is not None:
            with rt.mesh:lp=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],jnp.asarray(keep)),np.float64)
            lm=np.asarray(ctx['c'].loss_masks,bool);ce=-lp
        else:ce=np.full(CH,np.nan);lm=np.zeros(CH,bool)
        qce=np.asarray([float(ce[a:b][lm[a:b]].mean()) if lm[a:b].any() else np.nan for a,b in q],np.float32)
        woff=np.asarray(wst,np.int32);wid=np.full(WIN,-1,np.int32);lo_c=max(0,k-NW)
        for c_ in range(lo_c,k):
            off=(c_-k+NW)*CH;wid[off:off+CH]=np.asarray(all_ids[c_*CH:(c_+1)*CH],np.int32)
        log(f'  sentence dump at chunk {k}: {nQ} query sentences x {nW} window sentences, {A.shape[1]} heads, max row sum {rs:.3f}')
        return {'A':A,'AL':AL,'AV':AV,'VN':VN,'T_key':Tk,'T_sink':Tsk,'T_rec':Trc,'T_max':Tm,'T_sum':Ts,'T_ent':Te,'T_wsid':Tw,'q_start':np.asarray([a for a,_ in q],np.int32),'q_len':np.asarray([b-a for a,b in q],np.int32),'q_ce':qce,'q_tce':np.where(lm,ce,np.nan).astype(np.float16),'w_start':woff,'w_len':np.asarray(wlen,np.int32),'w_chunk':(k-NW+woff//CH).astype(np.int32),'win_ids':wid,'q_ids':np.asarray(ids_k,np.int32),'max_row_sum':np.float32(rs)}
    fn.wants_ctx=True
    return fn
def flat(d,prefix=''):
    out={}
    for k,v in d.items():
        for n,a in v.items():out[f'{prefix}c{k}/{n}']=np.asarray(a)
    return out
