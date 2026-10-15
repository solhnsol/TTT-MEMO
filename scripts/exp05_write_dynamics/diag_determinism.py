"""Diagnostic for the failed V1: is the chunk-0 gradient/update reproducible within one process? (no trajectory data produced)"""
import sys,json,os
sys.path.insert(0,'/home/solhnsol/workspace/TTT-MEMO/scripts/exp05_write_dynamics')
from trajectory import *
rt=Runtime();T=Traj(rt,101);P=T.P[False]
ids=P.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
tag=os.environ.get('DIAG_TAG','x');out={'xla_flags':os.environ.get('XLA_FLAGS','')}
with rt.mesh:
    pref=rt.prefix(rt.base,seq);jax.block_until_ready(pref)
    pref2=rt.prefix(rt.base,seq)
    out['prefix_maxabs_diff']=float(jnp.max(jnp.abs(pref.astype(jnp.float32)-pref2.astype(jnp.float32))))
    c=rt.chunk(seq,0);p=pref[:CH];st=(clone_state(rt.state_all),clone_state(rt.state_suffix))
    _,l1=rt.chunk_logp(rt.base,st[1],c,p);_,l2=rt.chunk_logp(rt.base,st[1],c,p)
    out['forward_logp_maxabs_diff']=float(jnp.max(jnp.abs(l1-l2)))
    n1,g1,c1,_=T.own_update(rt.base,st,c,p);n2,g2,c2,_=T.own_update(rt.base,st,c,p)
    old=rt.base.inner_parameters()
    d1=[(a.astype(jnp.float32)-b.astype(jnp.float32)) for a,b in zip(jax.tree.leaves(n1),jax.tree.leaves(old))]
    d2=[(a.astype(jnp.float32)-b.astype(jnp.float32)) for a,b in zip(jax.tree.leaves(n2),jax.tree.leaves(old))]
    num=sum(float(jnp.sum((x-y)**2)) for x,y in zip(d1,d2));den=sum(float(jnp.sum(x*x)) for x in d1)
    out['own_update_rel_diff_same_process']=float(np.sqrt(num/den));out['grad_norm_1_2']=[float(g1),float(g2)]
    opt=rt.base.inner_optimizer(clone_state(rt.state_all)).init(rt.base.inner_parameters())
    r1=rt.update(rt.base,opt,st,c,p);r2=rt.update(rt.base,opt,st,c,p)
    h1=reuse._hash_pytree(r1.new_model.inner_parameters())[0];h2=reuse._hash_pytree(r2.new_model.inner_parameters())[0]
    out['native_update_hash_equal_same_process']=h1==h2
    out['native_update_rel_diff']=float(np.sqrt(sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(r1.new_model.inner_parameters()),jax.tree.leaves(r2.new_model.inner_parameters())))/den))
# cross-process comparison material: sampled elements of the native chunk-0 update (fixed index set)
rng=np.random.default_rng(0);samp=[]
for a,b in zip(jax.tree.leaves(r1.new_model.inner_parameters()),jax.tree.leaves(rt.base.inner_parameters())):
    d=(a.astype(jnp.float32)-b.astype(jnp.float32)).reshape(-1);idx=jnp.asarray(rng.integers(0,d.shape[0],2_000_000));samp.append(np.asarray(d[idx]))
tag=os.environ.get('DIAG_TAG','x');np.savez(OUT/('diag_update_sample_%s.npz'%tag),sample=np.concatenate(samp))
out['native_update_hash']=h1;out['tag']=tag
print('DIAG',json.dumps(out),flush=True)
json.dump(out,open(OUT/('diag_determinism_%s.json'%tag),'w'),indent=1)
