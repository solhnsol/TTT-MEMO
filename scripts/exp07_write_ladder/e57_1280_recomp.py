"""E57 job 1280 RECOMP forward (GPU; new file; GPU-untested at the time of writing; definition (A) of the 1280 runner design: the 24 frozen PREFIX layers are recomputed over the window of chunk k under an S_i key mask; the 8 suffix layers use the carried natural state with the same keys hidden). Nothing in the existing files is changed: the attention is patched here by chaining SWA.full_sw_attention with a new mode 'rctx'.
For a branch chunk k (>= 8) the window of chunk k is the 8 chunks lo..k-1 (lo = k - 8). `window_kv` recomputes, layer by layer and chunk by chunk, the post-RoPE K,V of the window chunks with
  * queries of chunk j attending to the previous WIN = 8192 positions: for positions before lo * CH the NATURAL recorded K,V (exact: they cannot depend on any token at or after lo * CH), for positions inside the window the K,V recomputed here (the layer-l K,V of chunk j depend on the hidden states of chunk j at layer l, which depend on the recomputed K,V of the earlier layers),
  * the keys at the stream positions `hidden` (absolute token positions, S_i) hidden from EVERY query row at every layer (a row at a hidden position keeps its own key; otherwise a row with no visible key is possible), RoPE positions unchanged.
With hidden = empty the result is the recomputed natural window K,V (the c of freeze A); it is compared with the cached natural K,V of the same window (nc), which is what the gate c_minus_nc_abs_max <= 1e-4 measures downstream (numerical agreement of the XLA attention used here with the cuDNN attention used for the recording is checked by the smoke run, not assumed).
  window_kv(L, LD, m, Kh, Vh, ids_stream, k, hidden=(), keep_fn=None)  -> (Kw, Vw) device arrays [24, WIN, nh, d]; keep_fn(j) overrides the hidden-key matrix by an explicit 2-D keep per chunk (used for the band mask)
Needs the lab modules (server only): L = the Ladder55 object (L.embed), LD / s3 for S3 and Batch."""
import numpy as np
CH=1024;NW=8;WIN=NW*CH
_PATCHED={'done':False}
def install_patch(s3,jax,jnp,SWA):
    """chain SWA.full_sw_attention: mode 'rctx' computes the 'ctx' attention of s3 (queries of one chunk over the previous WIN positions plus the chunk, 2-D keep) and returns the fresh K,V of the chunk through S3['rec']; every other mode goes to the previous function unchanged"""
    if _PATCHED['done']:return
    prev=SWA.full_sw_attention;S3=s3.S3
    def full_r(self,hidden_states,seq,state):
        if S3.get('mode')!='rctx':return prev(self,hidden_states,seq,state)
        Kc,Vc=state;q0=S3['q0'];keep=S3['keep'];n=hidden_states.shape[0]
        xq,xk,xv=self.get_attention_input(hidden_states,position_ids=q0+jnp.arange(n,dtype=jnp.int32));S3['rec']=(xk,xv)
        K=jnp.concatenate([Kc.astype(xk.dtype),xk],axis=0);V=jnp.concatenate([Vc.astype(xv.dtype),xv],axis=0)
        qi=(q0+jnp.arange(n,dtype=jnp.int32))[:,None];kj=(q0-WIN+jnp.arange(WIN+n,dtype=jnp.int32))[None,:]
        mask=(qi>=kj)&(qi-kj<WIN)&(kj>=0)&keep
        o=jax.nn.dot_product_attention(xq,K,V,mask=mask,implementation='xla');o=self._merge_heads(o)
        return (self.get_attention_output(o),state)
    SWA.full_sw_attention=full_r;_PATCHED['done']=True
def keep_matrix(j,hidden):
    """2-D keep [CH, WIN + CH] for chunk j: all True except the columns of the hidden absolute positions, which are False for every row except the row of the hidden position itself (its own key stays visible)"""
    keep=np.ones((CH,WIN+CH),bool);base=j*CH-WIN
    for p in hidden:
        c=p-base
        if 0<=c<WIN+CH:
            keep[:,c]=False
            r=p-j*CH
            if 0<=r<CH:keep[r,c]=True
    return keep
_JIT={}
def window_kv(L,LD,m,Kh,Vh,ids_stream,k,hidden=(),keep_fn=None):
    jax,jnp,eqx=LD.jax,LD.jnp,LD.eqx;import s3 as S3M
    from ttt.model.data import Batch
    install_patch(S3M,jax,jnp,S3M.SWA)
    assert k>=NW,'branch chunks have a full window (k >= 8)'
    lo=k-NW;nl=Kh.shape[0];blocks=m.language_model.model.h.prefix_blocks
    if 'blk' not in _JIT:
        @eqx.filter_jit
        def blk_rctx(blk,x,Kc,Vc,q0,keep,ids_):
            S3M.S3['mode']='rctx';S3M.S3['q0']=q0;S3M.S3['keep']=keep
            try:
                seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool));y,_=blk(x,(Kc,Vc),seq,is_prefix=True);K,V=S3M.S3['rec']
            finally:S3M.S3.clear()
            return y,K,V
        _JIT['blk']=blk_rctx
    blk_rctx=_JIT['blk'];ids=np.asarray(ids_stream)
    ids_c={j:jnp.asarray(ids[j*CH:(j+1)*CH],dtype=jnp.int32) for j in range(lo,k)};h={j:L.embed(m,ids_c[j]) for j in range(lo,k)}
    keeps={j:jnp.asarray(keep_fn(j) if keep_fn is not None else keep_matrix(j,hidden)) for j in range(lo,k)};Kout=[];Vout=[]
    for l in range(nl):
        blk=jax.tree.map(lambda a:a[l] if eqx.is_array(a) else a,blocks)
        nat_lo=max(0,(lo-NW)*CH)                                          # natural K,V of the positions before the window: [nat_lo, lo*CH)
        nK=jnp.asarray(Kh[l,nat_lo:lo*CH]);nV=jnp.asarray(Vh[l,nat_lo:lo*CH])
        if nK.shape[0]<WIN:                                               # only for lo < 8 (not used: k >= 16 is needed for a full natural left context; the zero padding mirrors ctx_kv)
            pad=WIN-nK.shape[0];nK=jnp.concatenate([jnp.zeros((pad,)+nK.shape[1:],nK.dtype),nK]);nV=jnp.concatenate([jnp.zeros((pad,)+nV.shape[1:],nV.dtype),nV])
        recK={};recV={}
        for j in range(lo,k):
            # keys context of chunk j: the WIN positions before j*CH = natural tail of [ (j-NW)*CH, lo*CH ) followed by the recomputed chunks lo..j-1
            nat_part=(lo-(j-NW))*CH                                      # number of natural positions in the context of chunk j
            parts_K=[nK[nK.shape[0]-nat_part:]] if nat_part>0 else [];parts_V=[nV[nV.shape[0]-nat_part:]] if nat_part>0 else []
            for c in range(lo,j):parts_K.append(recK[c]);parts_V.append(recV[c])
            Kc=jnp.concatenate(parts_K,axis=0) if parts_K else None;Vc=jnp.concatenate(parts_V,axis=0) if parts_V else None
            assert Kc.shape[0]==WIN,(Kc.shape,j)
            y,K,V=blk_rctx(blk,h[j],Kc,Vc,jnp.int32(j*CH),keeps[j],ids_c[j]);h[j]=y;recK[j]=K;recV[j]=V
        Kout.append(jnp.concatenate([recK[j] for j in range(lo,k)],axis=0));Vout.append(jnp.concatenate([recV[j] for j in range(lo,k)],axis=0))
    return jnp.stack(Kout),jnp.stack(Vout)
def selftest():
    k=keep_matrix(10,[(10-3)*CH+5,10*CH+7]);assert k.shape==(CH,WIN+CH)
    c1=(10-3)*CH+5-(10*CH-WIN);assert not k[:,c1].any()                       # hidden from every row (the position lies before chunk 10)
    c2=10*CH+7-(10*CH-WIN);assert not k[:7,c2].any() and k[7,c2] and not k[8:,c2].any()   # a hidden position inside the chunk: its own row keeps it
    assert keep_matrix(10,[]).all()
    print('e57_1280_recomp selftest ok (CPU part; the GPU forward is verified by the smoke run)')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
