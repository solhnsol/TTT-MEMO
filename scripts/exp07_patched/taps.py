"""Per-token factors of the inner-parameter gradient at fixed weights (shared by gradfactors.py and the KEYSTAT screen).
make_taps(m, rt32, Batch) -> (deltas, acts):
  deltas(ids, tg, mk, pref, seg, loc) -> d1, d3, d2 [L, T, d]   dL/dz for z1 = x_in W1, z3 = x_in W3, z2 = x_mid W2 of the prime MLP,
                                                               from zero taps added inside e2e's own Block.__call__ (L = mean loss)
  acts(ids, tg, mk, pref, seg, loc)   -> x_in [L,T,2560], x_mid [L,T,5632], max |hidden - suffix_call hidden|, max |hidden|
The e2e/ code is not modified: the 8-block lax.scan is unrolled here and each layer's prime MLP is swapped for a tap module."""
import jax,jax.numpy as jnp,equinox as eqx
import packing
from ttt.model.transformer import Block
from ttt.model.loss import cross_entropy_loss_and_accuracy
from readouts import clone_state
CH=1024
class TapMLP(eqx.Module):
    w1:object;w2:object;w3:object;dropout:object;e1:jax.Array;e3:jax.Array;e2:jax.Array
    def __call__(self,x):
        z1=self.w1(x)+self.e1;z3=self.w3(x)+self.e3;z2=self.w2(jax.nn.silu(z1)*z3)+self.e2
        return self.dropout(z2)
def make_taps(m,rt32,Batch):
    lm=m.language_model;H=lm.model.h;cfg=lm.config;NL=jax.tree_util.tree_leaves(H.suffix_blocks)[0].shape[0]
    dff=int(H.suffix_blocks.feed_forward_prime.w1.weight.shape[-1]);dm=int(H.suffix_blocks.feed_forward_prime.w1.weight.shape[-2])
    def layer(i):return jax.tree.map(lambda a:a[i] if eqx.is_array(a) else a,H.suffix_blocks)
    def st_layer(st,i):return jax.tree.map(lambda a:a[i],st)
    def head(hs):
        hs=jax.vmap(lm.model.ln_f)(hs)
        if cfg.tie_word_embeddings:return hs@lm.model.wte.weight.T.astype(hs.dtype)
        return lm.lm_head(hs)
    def tapped_loss(taps,ids_,tg_,mk_,pref_):
        seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_);st=clone_state(rt32.state_suffix);x=pref_
        for i in range(NL):
            b=layer(i);f=b.feed_forward_prime
            b=eqx.tree_at(lambda q:q.feed_forward_prime,b,TapMLP(f.w1,f.w2,f.w3,f.dropout,*taps[i]))
            x,_=Block.__call__(b,x,st_layer(st,i),seq,is_prefix=False)
        loss,_=cross_entropy_loss_and_accuracy(head(x),seq.target_tokens,seq.loss_masks);return loss
    def block_capture(b,x,st_i,seq,xin,xmid):
        rmsf=lambda n,u:n(u)
        so,_=b.seq_modeling_forward(lambda blk,inp,sq,s_:type(blk).__call__(blk,inp,sq,s_,is_prefix=False),rmsf,b.seq_norm,b.seq_modeling_block,b.seq_post_norm,x,st_i,seq)
        x=x+so;f=b.feed_forward_prime
        inp=jax.vmap(lambda u:b.ffn_prime_norm(u))(x) if b.config.pre_norm else x
        xin.append(inp);z1=f.w1(inp);z3=f.w3(inp);mid=jax.nn.silu(z1)*z3;xmid.append(mid);o=f.w2(mid)
        if b.config.post_norm:o=jax.vmap(lambda u:b.ffn_prime_post_norm(u))(o)
        x=x+o;fo=b.ffn_forward(lambda ff,u:ff(u),rmsf,b.ffn_norm,b.feed_forward,b.ffn_post_norm,x);return x+fo
    @eqx.filter_jit
    def deltas(ids_,tg_,mk_,pref_,seg_,loc_):
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:
            z=[(jnp.zeros((CH,dff),jnp.float32),jnp.zeros((CH,dff),jnp.float32),jnp.zeros((CH,dm),jnp.float32)) for _ in range(NL)]
            g=jax.grad(tapped_loss)(z,ids_,tg_,mk_,pref_)
        finally:packing.PACK_SUF.clear()
        return jnp.stack([a for a,_,_ in g]),jnp.stack([c for _,c,_ in g]),jnp.stack([e for _,_,e in g])
    @eqx.filter_jit
    def acts(ids_,tg_,mk_,pref_,seg_,loc_):
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:
            seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_);st=clone_state(rt32.state_suffix);x=pref_;xin=[];xmid=[]
            for i in range(NL):x=block_capture(layer(i),x,st_layer(st,i),seq,xin,xmid)
            ref=lm.suffix_call(prefix_outputs=pref_,state=clone_state(rt32.state_suffix),seq=seq).last_hidden_states
            return jnp.stack(xin),jnp.stack(xmid),jnp.max(jnp.abs(jax.vmap(lm.model.ln_f)(x)-ref)),jnp.max(jnp.abs(ref))
        finally:packing.PACK_SUF.clear()
    return deltas,acts,NL,dm,dff
