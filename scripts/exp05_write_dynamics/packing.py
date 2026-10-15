"""Packed probe scoring: several short probes share one 1024-token chunk, isolated from each other.
Semantics are meant to equal one-probe-per-chunk scoring exactly (up to floating point):
- attention: a probe token sees the context (neutral tail or nothing) and its OWN earlier tokens only;
  other probes in the same chunk are masked out in every prefix and suffix layer;
- positions: every probe restarts at the positions it would have had alone (RoPE sees the same relative distances);
- everything else in TTT-E2E's forward is per-token (MLPs, norms), so nothing else mixes probes.
Implemented by patching the SWA attention class outside e2e/ (no file in e2e/ is modified). The patches are inert
unless PACK_PRE / PACK_SUF are set, which happens only while the packed jitted functions below are traced; per-call
segment ids, local positions and masks are traced arguments, so there is ONE compilation per function and shape.
Equivalence with single-probe scoring is checked by verify_packing.py before any use."""
import numpy as np
from ttt.model import attention as _att
import jax,jax.numpy as jnp,equinox as eqx
SWA=_att.SWA
PACK_PRE={};PACK_SUF={}
_orig_mask=SWA.sw_causal_mask;_orig_rope=_att.AttentionBase.apply_rope;_orig_full=SWA.full_sw_attention
def _mask(self,chunk_id):
    m=_orig_mask(self,chunk_id)
    if 'seg' not in PACK_SUF:return m
    seg=PACK_SUF['seg'];cur=seg[:,None]==seg[None,:]
    return m&jnp.concatenate([jnp.ones((self.mini_batch_size,self.window_size),bool),cur],axis=1)
def _rope(self,xis,position_ids):
    if 'loc' in PACK_SUF and isinstance(self,SWA):
        W,B=self.window_size,self.mini_batch_size;n=position_ids.shape[0];loc=PACK_SUF['loc']
        if n==B:position_ids=W+loc
        elif n==W+B:position_ids=jnp.concatenate([jnp.arange(W,dtype=jnp.int32),W+loc])
    return _orig_rope(self,xis,position_ids)
def _full(self,hidden_states,seq,state):
    if 'mask' not in PACK_PRE:return _orig_full(self,hidden_states,seq,state)
    xq,xk,xv=self.get_attention_input(hidden_states,position_ids=seq.position_ids)   # packed positions come in seq
    o=jax.nn.dot_product_attention(xq,xk,xv,mask=PACK_PRE['mask'],implementation='xla')
    return (self.get_attention_output(self._merge_heads(o)),state)
SWA.sw_causal_mask=_mask;_att.AttentionBase.apply_rope=_rope;SWA.full_sw_attention=_full

def pack_rows(rows,CH=1024,one_per_chunk=False):
    """rows: list of (ids, (a,b)) with a,b = logp indices of the scored span (as in Readouts.spans).
    Greedy first-fit in order. Returns chunks [(ids[CH], seg[CH], loc[CH])] and per-row (chunk, a, b) in chunk coords."""
    chunks=[];where=[];cur=[];seg=[];loc=[];s=0
    def flush():
        n=len(cur);chunks.append((np.asarray(cur+[cur[-1]]*(CH-n),np.int32),np.asarray(seg+[-1]*(CH-n),np.int32),np.asarray(loc+list(range(CH-n)),np.int32)))
    for ids,(a,b) in rows:
        if cur and (one_per_chunk or len(cur)+len(ids)>CH):flush();cur,seg,loc,s=[],[],[],0
        off=len(cur);where.append((len(chunks),off+a,off+b))
        cur+=list(ids);seg+=[s]*len(ids);loc+=list(range(len(ids)));s+=1
    if cur:flush()
    return chunks,where

def prefix_mask(seg_full,window):
    """seg_full: [T] with -2 for context tokens (visible to everyone, causally) and segment ids for probe tokens."""
    T=len(seg_full);q=np.arange(T)[:,None];k=np.arange(T)[None,:];sq=seg_full[:,None];sk=seg_full[None,:]
    return (q>=k)&(q-k<window)&((sk==-2)|(sk==sq))

class Packed:
    """Packed R0/R1 scorer bound to a Readouts instance T (reuses its tail, rows and runtime)."""
    def __init__(self,T,raw_rows,GP=8,one_per_chunk=False):
        rt=T.rt;self.T=T;self.rt=rt;self.GP=GP;W=int(rt.cfg.model.sliding_window_size);CH=1024;TL=len(T.tail_ids)
        self.chunks,self.where=pack_rows(raw_rows,CH,one_per_chunk);n=len(self.chunks);pad=(-n)%GP
        ids=np.stack([c[0] for c in self.chunks]+[self.chunks[-1][0]]*pad);seg=np.stack([c[1] for c in self.chunks]+[self.chunks[-1][1]]*pad);loc=np.stack([c[2] for c in self.chunks]+[self.chunks[-1][2]]*pad)
        self.nchunk=n;self.ids=jnp.asarray(ids);self.seg=jnp.asarray(seg);self.loc=jnp.asarray(loc)
        from ttt.model.data import Batch
        @eqx.filter_jit
        def prefix_packed(m,ids_,pos_,mask_):
            PACK_PRE['mask']=mask_
            try:
                seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
                return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),_clone(rt.state_prefix),seq).last_hidden_state
            finally:PACK_PRE.clear()
        @eqx.filter_jit
        def lp_packed(m,st,ids_,tg_,mk_,pref_,seg_,loc_):
            def one(i,t,k,p,s,l):
                PACK_SUF['seg']=s;PACK_SUF['loc']=l
                try:
                    r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k))
                    lg=r.logits.astype(jnp.float32)
                    return jnp.take_along_axis(lg,t[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
                finally:PACK_SUF.clear()
            return jax.vmap(one)(ids_,tg_,mk_,pref_,seg_,loc_)
        self.lp_packed=lp_packed
        # frozen prefix outputs, once: R0 (probe chunk alone, positions restart at 0) and R1 (neutral tail + probe chunk)
        p0=[];p1=[]
        with rt.mesh:
            for j in range(ids.shape[0]):
                sj,lj=seg[j],loc[j]
                p0.append(prefix_packed(rt.base,jnp.asarray(ids[j]),jnp.asarray(lj),jnp.asarray(prefix_mask(sj,W))))
                full=np.concatenate([np.asarray(T.tail_ids,np.int32),ids[j]]);pos=np.concatenate([np.arange(TL,dtype=np.int32),TL+lj])
                sf=np.concatenate([np.full(TL,-2,np.int32),sj])
                p1.append(prefix_packed(rt.base,jnp.asarray(full),jnp.asarray(pos),jnp.asarray(prefix_mask(sf,W)))[TL:])
            self.pref0=jnp.stack(p0);self.pref1=jnp.stack(p1);jax.block_until_ready(self.pref1)
            seqs=[rt.seq(list(x)) for x in ids]
            self.tg=jnp.stack([q.target_tokens for q in seqs]);self.mk=jnp.stack([q.loss_masks for q in seqs])
    def _score(self,m,st,pref):
        out=[]
        for g in range(0,self.ids.shape[0],self.GP):
            out.append(np.asarray(self.lp_packed(m,st,self.ids[g:g+self.GP],self.tg[g:g+self.GP],self.mk[g:g+self.GP],pref[g:g+self.GP],self.seg[g:g+self.GP],self.loc[g:g+self.GP])))
        lp=np.concatenate(out)
        return np.asarray([lp[c,a:b].sum() for c,a,b in self.where])
    def r0(self,m):
        with self.rt.mesh:return self._score(m,_clone(self.rt.state_suffix),self.pref0)
    def r1(self,m):
        rt=self.rt;T=self.T
        with rt.mesh:
            st=_clone(rt.state_suffix)
            for j in range(len(T.tail_ids)//1024):
                st,_=rt.chunk_logp(m,st,rt.chunk(T._tail_seq,j),T._tail_pref[j*1024:(j+1)*1024])
            return self._score(m,st,self.pref1)
def _clone(s):
    from core import clone_state;return clone_state(s)
