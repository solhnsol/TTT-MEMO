"""E57 fast IN reads (jobs 480 / 460): the IN context of an item is the neutral 8192-token tail with its last part replaced by the item's paragraph, so everything before the paragraph is IDENTICAL for all items. The reference path (e55_gpu.Reader55.packed -> packing.Packed, new file here does not change it) recomputes for every item (1) the prefix layers over the whole context, (2) the prefix layers over context + each packed 1024-token candidate chunk, (3) the suffix layers over the 8 context chunks (the suffix KV state). Here the shared part is computed ONCE and reused:
  prefix: per-layer (roped) K, V and the output hidden states of the global tail; an item recomputes only the last R = 1024 context positions (paragraph included) plus the candidate chunk, attending to the cached K, V of positions < 8192 - R (same masks / positions as packing.prefix_mask, same patched xla attention path);
  suffix: the suffix KV state after the chunks 0 .. 6 of the global tail is cached (host); an item continues from the state after chunk 7 - ... i.e. recomputes only the last context chunk(s) (chunk 7 for R = 1024) and then scores the packed candidate chunks as r1 does.
Semantics = packing.Packed.r1 up to floating point (checked by e57_kvreuse_check.py against the reference path on the real model). Paragraphs longer than R tokens fall back to the reference path. No file of the lab code is modified: the attention hook wraps the already patched SWA.full_sw_attention and is inert unless KV['mode'] is set during tracing.
  python3 e57_kvreuse.py --selftest   (CPU, index / attention algebra only)"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]/'scripts/exp05_write_dynamics'));sys.path.insert(0,str(HERE.parents[1]/'scripts/exp04_what_ttt_retains'))
sys.path.insert(0,str(HERE.parents[1]/'e2e'))
import packing as PK
import jax,jax.numpy as jnp,equinox as eqx
from functools import partial
SWA=PK.SWA
KV={}
_prev_full=SWA.full_sw_attention
CH=1024
def _attn_mask_np(seg_full,window):return PK.prefix_mask(seg_full,window)
def _full_kv(self,hidden_states,seq,state):
    mode=KV.get('mode')
    if mode is None:return _prev_full(self,hidden_states,seq,state)
    xq,xk,xv=self.get_attention_input(hidden_states,position_ids=seq.position_ids)
    if mode=='collect':
        KV['out']=(xk,xv);K,V=xk,xv
    else:
        ck,cv=KV['in'];K=jnp.concatenate([ck,xk],axis=0);V=jnp.concatenate([cv,xv],axis=0)
    o=jax.nn.dot_product_attention(xq,K,V,mask=KV['mask'],implementation='xla')
    return (self.get_attention_output(self._merge_heads(o)),state)
SWA.full_sw_attention=_full_kv
def make_prefix_fns(rt):
    """-> (collect(m, ids, pos, mask) -> (hidden, K[L,T,nk,d], V), resume(m, ids, pos, mask, ck, cv) -> hidden)  (jitted, run under rt.mesh and highest precision by the caller)"""
    from ttt.model.data import Batch
    sp=rt.state_prefix
    def run(m,ids_,pos_,mask_,kv,mode):
        KV['mode']=mode;KV['mask']=mask_
        try:
            seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
            lm=m.language_model;blocks=lm.model.h.prefix_blocks;fn=partial(blocks.__class__.__call__,is_prefix=True);PK.PACK_PRE.clear()
            def body(x,xs):
                block,kvin=xs;KV['in']=kvin;KV.pop('out',None);x,_=fn(block,x,None,seq);return x,KV.get('out')
            h,ys=jax.lax.scan(body,lm.wte_call(ids_),(blocks,kv))
            return h,ys
        finally:KV.clear()
    @eqx.filter_jit
    def collect(m,ids_,pos_,mask_):
        h,(k,v)=run(m,ids_,pos_,mask_,None,'collect');return h,k,v
    @eqx.filter_jit
    def resume(m,ids_,pos_,mask_,ck,cv):
        h,_=run(m,ids_,pos_,mask_,(ck,cv),'resume');return h
    return collect,resume
def tree_host(t):return jax.tree_util.tree_map(np.asarray,t)
def tree_dev(t):return jax.tree_util.tree_map(jnp.asarray,t)
class FastIN:
    """IN scorer of one fixed weight set m32 (fp32 runtime rt32) and one fixed 8192-token tail."""
    def __init__(self,rt32,m32,tail_ids,raw_template=None,R=1024,GP=2,log=print):
        import time
        self.rt=rt32;self.m=m32;self.tail=list(tail_ids);self.TL=TL=len(tail_ids);assert TL==8192 and R%CH==0 and R<TL
        self.R=R;self.GP=GP;self.W=int(rt32.cfg.model.sliding_window_size);self.b=TL-R;self.jb=self.b//CH
        self.collect,self.resume=make_prefix_fns(rt32);self.chunk_logp=rt32.chunk_logp
        t0=time.monotonic()
        with rt32.mesh,jax.default_matmul_precision('highest'):
            ids=jnp.asarray(self.tail,jnp.int32);pos=jnp.arange(TL,dtype=jnp.int32);mask=jnp.asarray(_attn_mask_np(np.full(TL,-2,np.int32),self.W))
            self.H,self.K,self.V=self.collect(m32,ids,pos,mask);jax.block_until_ready(self.V)
            log(f'  fast IN: global tail prefix K,V,H cached in {time.monotonic()-t0:.0f}s (K {tuple(self.K.shape)})')
            # suffix states after chunks 0 .. jb-1 of the global tail (host copies)
            from core import clone_state
            st=clone_state(rt32.state_suffix);seq=rt32.seq(self.tail+[rt32.eos]*CH)
            for j in range(self.jb):st,_=rt32.chunk_logp(m32,st,rt32.chunk(seq,j),self.H[j*CH:(j+1)*CH])
            self.st_host=tree_host(st)
        log(f'  fast IN: suffix state after {self.jb} tail chunks cached; total {time.monotonic()-t0:.0f}s')
        @eqx.filter_jit
        def lp_packed(m,st,ids_,tg_,mk_,pref_,seg_,loc_):
            from ttt.model.data import Batch
            def one(i,t,k,p,s,l):
                PK.PACK_SUF['seg']=s;PK.PACK_SUF['loc']=l
                try:
                    r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k))
                    lg=r.logits.astype(jnp.float32);return jnp.take_along_axis(lg,t[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
                finally:PK.PACK_SUF.clear()
            return jax.vmap(one)(ids_,tg_,mk_,pref_,seg_,loc_)
        self.lp_packed=lp_packed
    def score(self,raw_rows,pid):
        """summed log-prob of the span of every candidate row for the context tail[:8192-len(pid)] + pid (len(pid) <= R)"""
        rt=self.rt;TL,R,W,CHs=self.TL,self.R,self.W,CH;lp_=len(pid);assert lp_<=R
        ctx=self.tail[:TL-lp_]+list(pid);assert len(ctx)==TL
        chunks,where=PK.pack_rows(raw_rows,CH);n=len(chunks);padn=(-n)%self.GP
        ids=np.stack([c[0] for c in chunks]+[chunks[-1][0]]*padn);seg=np.stack([c[1] for c in chunks]+[chunks[-1][1]]*padn);loc=np.stack([c[2] for c in chunks]+[chunks[-1][2]]*padn)
        b=self.b;pref1=[]
        with rt.mesh,jax.default_matmul_precision('highest'):
            Hnew=None
            for j in range(ids.shape[0]):
                new_ids=np.concatenate([np.asarray(ctx[b:],np.int32),ids[j]]);pos=np.concatenate([np.arange(b,TL,dtype=np.int32),TL+loc[j]])
                sf=np.concatenate([np.full(TL,-2,np.int32),seg[j]]);mask=_attn_mask_np(sf,W)[b:,:]
                h=self.resume(self.m,jnp.asarray(new_ids),jnp.asarray(pos),jnp.asarray(mask),self.K[:,:b],self.V[:,:b])
                if Hnew is None:Hnew=h[:R]
                pref1.append(h[R:])
            pref1=jnp.stack(pref1);Hctx=jnp.concatenate([self.H[:b],Hnew],axis=0)
            from core import clone_state
            st=tree_dev(self.st_host);seq=rt.seq(ctx+[rt.eos]*CH)
            for j in range(self.jb,TL//CH):st,_=self.chunk_logp(self.m,st,rt.chunk(seq,j),Hctx[j*CH:(j+1)*CH])
            qs=[rt.seq(list(x)) for x in ids];tg=jnp.stack([q.target_tokens for q in qs]);mk=jnp.stack([q.loss_masks for q in qs])
            idsj,segj,locj=jnp.asarray(ids),jnp.asarray(seg),jnp.asarray(loc);out=[]
            for g in range(0,ids.shape[0],self.GP):
                out.append(np.asarray(self.lp_packed(self.m,st,idsj[g:g+self.GP],tg[g:g+self.GP],mk[g:g+self.GP],pref1[g:g+self.GP],segj[g:g+self.GP],locj[g:g+self.GP])))
        lp=np.concatenate(out);return np.asarray([lp[c,a:b_].sum() for c,a,b_ in where])
def selftest():
    # attention algebra: full causal windowed attention == (cached K,V of positions < b) + resumed rows with the sliced mask
    rng=np.random.default_rng(0);T,b,nk,d,W=40,30,2,8,16
    q=rng.standard_normal((T,nk,d)).astype(np.float32);k=rng.standard_normal((T,nk,d)).astype(np.float32);v=rng.standard_normal((T,nk,d)).astype(np.float32)
    seg=np.full(T,-2,np.int32);seg[34:]=0;m=_attn_mask_np(seg,W)
    full=np.asarray(jax.nn.dot_product_attention(jnp.asarray(q),jnp.asarray(k),jnp.asarray(v),mask=jnp.asarray(m),implementation='xla'))
    K=np.concatenate([k[:b],k[b:]]);part=np.asarray(jax.nn.dot_product_attention(jnp.asarray(q[b:]),jnp.asarray(K),jnp.asarray(np.concatenate([v[:b],v[b:]])),mask=jnp.asarray(m[b:,:]),implementation='xla'))
    assert np.allclose(full[b:],part,atol=1e-5)
    # the rows before b do not look at positions >= b (causality), so the cached part is valid for every item sharing tokens < b
    assert not m[:b,b:].any()
    print('e57_kvreuse selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
