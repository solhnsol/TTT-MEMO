"""Post-hoc per-token gradient-factor extraction at W0 (approved 2026-10-16 22:36 KST, Hans "both"; no judgement; for the
November amortization work). Same deterministic chunk inputs as the suppression screen, chunk condition (chunk alone,
positions from 0), fp32 model, matmul precision 'highest'.
Kinds per world-seed: 'ntp' = NTP on the original chunk, every in-chunk target (all 63 chunks; the masked write's own
loss); 'c2X' / 'c2Y' = place-token loss on the original / span-labelled c2-swapped chunk (suppress.chunk_rows 'X' / 'Y',
the screen's loss positions); optional 'lfX' / 'lfY' (LF-swap pairs, all-token loss).
Per chunk and suffix layer l (8) and SwiGLU matrix of the inner MLP (feed_forward_prime: w1 gate, w3 up, w2 down; y = x @ W):
  x_in  [T, 2560]  input of w1 / w3 (= ffn_prime_norm output);  x_mid [T, 5632]  input of w2 (= silu(z1) * z3);
  d1, d3 [T, 5632], d2 [T, 2560]  dL/dz (L = the chunk's mean loss), so dL/dW1 = x_in^T d1, dL/dW3 = x_in^T d3, dL/dW2 = x_mid^T d2.
delta: gradient of the loss w.r.t. zero 'tap' tensors added to z1, z3, z2 inside e2e's own Block.__call__ (the prime MLP
module is swapped for a tap module with the same weights); x: an explicit forward of the same blocks (the e2e Block methods
in the same order) that records the MLP inputs; checked: its final hidden state equals the model's suffix_call, and per
matrix the reconstructed sum equals the screen's loss_grad (relative error reported per chunk).
Output: DUMP/s{seed}_{w}/{kind}_k{kk}.npz (x_in, x_mid, d1, d3, d2 stacked over layers, fp32) + manifest.json (W0 hash, code
sha, chunk list, loss positions, check errors). Resumable: existing chunk files are skipped.
Usage: SEED=101 KINDS=ntp,c2X,c2Y python gradfactors.py   (DUMP default /home/solhnsol/workspace/TTT-MEMO-dumps/gradfactors)"""
import os,sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
sys.path.insert(0,str(LD.ROOT/'scripts/exp04_what_ttt_retains'))
import suppress as SP
from ttt.model.transformer import Block,SwiGLUMLP
from ttt.model.loss import cross_entropy_loss_and_accuracy
DUMP=Path(os.environ.get('DUMP','/home/solhnsol/workspace/TTT-MEMO-dumps/gradfactors'))
class TapMLP(eqx.Module):
    w1:object;w2:object;w3:object;dropout:object;e1:jax.Array;e3:jax.Array;e2:jax.Array
    def __call__(self,x):
        z1=self.w1(x)+self.e1;z3=self.w3(x)+self.e3;z2=self.w2(jax.nn.silu(z1)*z3)+self.e2
        return self.dropout(z2)
def main():
    seed=int(os.environ['SEED']);kinds=os.environ.get('KINDS','ntp,c2X,c2Y').split(',')
    rt=Runtime();L=LD.Ladder(rt,seed,readout=True);rt32=L.rt32;m0=rt32.base;T=L.T;W=int(rt.cfg.model.sliding_window_size)
    h0=reuse._hash_pytree(rt.base.inner_parameters())[0]
    code_sha={f:hashlib.sha256(open(Path(__file__).resolve().parent/f,'rb').read()).hexdigest() for f in ('gradfactors.py','suppress.py','ladder.py')}
    seg=np.zeros(CH,np.int32);loc=np.arange(CH,dtype=np.int32);pm=jnp.asarray(packing.prefix_mask(seg,W))
    Batch=L.Batch;lm=m0.language_model;H=lm.model.h;NL=jax.tree_util.tree_leaves(H.suffix_blocks)[0].shape[0]
    cfg=lm.config
    @eqx.filter_jit
    def prefix32(m,ids_,pos_,mask_):   # as grad_components.prefix32 (the screen's chunk-alone prefix)
        packing.PACK_PRE['mask']=mask_
        try:
            seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
            return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),clone_state(rt32.state_prefix),seq).last_hidden_state
        finally:packing.PACK_PRE.clear()
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
    def _block_capture(b,x,st_i,seq,xin,xmid):
        cfg_b=b.config;rmsf=lambda n,u:n(u)
        so,_=b.seq_modeling_forward(lambda blk,inp,sq,s_:type(blk).__call__(blk,inp,sq,s_,is_prefix=False),rmsf,b.seq_norm,b.seq_modeling_block,b.seq_post_norm,x,st_i,seq)
        x=x+so;f=b.feed_forward_prime
        inp=jax.vmap(lambda u:b.ffn_prime_norm(u))(x) if cfg_b.pre_norm else x
        xin.append(inp);z1=f.w1(inp);z3=f.w3(inp);mid=jax.nn.silu(z1)*z3;xmid.append(mid);o=f.w2(mid)
        if cfg_b.post_norm:o=jax.vmap(lambda u:b.ffn_prime_post_norm(u))(o)
        x=x+o;fo=b.ffn_forward(lambda ff,u:ff(u),rmsf,b.ffn_norm,b.feed_forward,b.ffn_post_norm,x);return x+fo
    dff=int(H.suffix_blocks.feed_forward_prime.w1.weight.shape[-1]);dm=int(H.suffix_blocks.feed_forward_prime.w1.weight.shape[-2])
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
            for i in range(NL):x=_block_capture(layer(i),x,st_layer(st,i),seq,xin,xmid)
            ref=lm.suffix_call(prefix_outputs=pref_,state=clone_state(rt32.state_suffix),seq=seq).last_hidden_states
            return jnp.stack(xin),jnp.stack(xmid),jnp.max(jnp.abs(jax.vmap(lm.model.ln_f)(x)-ref)),jnp.max(jnp.abs(ref))
        finally:packing.PACK_SUF.clear()
    @eqx.filter_jit
    def ref_grad(m,ids_,tg_,mk_,pref_,seg_,loc_):
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:
            seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_)
            g=eqx.filter_grad(lambda mm:MetaModel.lm_loss(mm,seq,clone_state(rt32.state_suffix),prefix_outputs=pref_)[0])(m)
        finally:packing.PACK_SUF.clear()
        p=g.language_model.model.h.suffix_blocks.feed_forward_prime;return p.w1.weight,p.w3.weight,p.w2.weight
    # inputs
    def rows_for(kind,w):
        sw=w=='B'
        if kind=='ntp':r,_=SP.chunk_rows(T.w,sw,T.tok,rt.bos,rt.eos,'ntp');return [(x[0],x[1],x[3]['k']) for x in r]
        if kind in ('c2X','c2Y'):r,_=SP.chunk_rows(T.w,sw,T.tok,rt.bos,rt.eos,'X' if kind=='c2X' else 'Y');return [(x[0],x[1],x[3]['k']) for x in r]
        ext=SP.external_names(T.w,SP.gw.ntok,T.w['name_len']);sp=[(a,b) for a,b,_ in json.load(open(LD.OUT/f'ner_spans_s{seed}_{w}.json'))['spans']]
        pr,_=SP.lf_chunk_pairs(T.w,sw,T.tok,rt.bos,rt.eos,'lfswap',sp,ext,seed);j=0 if kind=='lfX' else 1
        return [(p[j][0],p[j][1],p[j][3]['k']) for p in pr]
    with rt32.mesh,jax.default_matmul_precision('highest'):
        for w in 'AB':
            D=DUMP/f's{seed}_{w}';D.mkdir(parents=True,exist_ok=True);man_f=D/'manifest.json'
            man=json.load(open(man_f)) if man_f.exists() else {'seed':seed,'world':w,'W0_inner_hash':h0,'code_sha':code_sha,'kinds':{}}
            assert man['W0_inner_hash']==h0
            for kind in kinds:
                ent=man['kinds'].setdefault(kind,{'chunks':{}})
                for ids,mk,k in rows_for(kind,w):
                    fn=D/f'{kind}_k{k:02d}.npz'
                    if fn.exists() and str(k) in ent['chunks']:continue
                    t0=time.monotonic();ids=np.asarray(ids,np.int32);tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32)
                    pref=prefix32(m0,jnp.asarray(ids),jnp.asarray(loc),pm)
                    a=(jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
                    d1,d3,d2=deltas(*a);xin,xmid,hd,hmax=acts(*a);r1,r3,r2=ref_grad(m0,*a)
                    rec=[jnp.einsum('ltd,ltf->ldf',xin,d1),jnp.einsum('ltd,ltf->ldf',xin,d3),jnp.einsum('ltf,ltd->lfd',xmid,d2)]
                    err=[float(jnp.linalg.norm(x-y)/jnp.maximum(jnp.linalg.norm(y),1e-30)) for x,y in zip(rec,(r1,r3,r2))]
                    np.savez(fn,x_in=np.asarray(xin),x_mid=np.asarray(xmid),d1=np.asarray(d1),d3=np.asarray(d3),d2=np.asarray(d2))
                    ent['chunks'][str(k)]={'n_loss':int(mk.sum()),'loss_positions':np.flatnonzero(mk).tolist(),'ids_sha':hashlib.sha256(ids.tobytes()).hexdigest(),
                                           'rel_err_w1_w3_w2':err,'hidden_maxdiff':float(hd),'hidden_max':float(hmax),'sec':time.monotonic()-t0}
                    json.dump(man,open(man_f,'w'),indent=1);print(seed,w,kind,k,'rel err',[f'{e:.2e}' for e in err],'hidden diff',f'{float(hd):.2e}',f'{time.monotonic()-t0:.1f}s',flush=True)
if __name__=='__main__':main()
