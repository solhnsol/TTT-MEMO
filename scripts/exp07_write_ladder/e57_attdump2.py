"""E57 pod code, token-level attention dump v2 (forward only, W0 weights, no write; post-hoc, exploratory). Same pass as e57_attdump.att4_pass (unmasked natural window, every token of the chunk a query, softmax over window + current-chunk keys) but it saves what the job-278 dump dropped and what the synthetic dump (e55_sentdump_s61..64) had, with the SAME definitions:
  T_max, T_key, T_sum, T_ent  [L,H,CH]  over the WINDOW keys: maximum mass, key index (0..8191, 8191 = key just before the chunk), summed mass, entropy of the unnormalised window mass -sum p log(p+1e-30)   (e55_sentdump / s3.py 'att_tokstat')
  T_sink [L,H,CH]  mass on the first 4 window keys;  T_rec [L,H,CH]  mass on the last 1024 window keys (previous chunk);  T_cur [L,H,CH]  mass on the keys of the CURRENT chunk (the rest of the softmax row)
  POS [L,H,CH,9] float16  window mass by key position: bin 0 = keys 0..3, bin 1 = keys 4..WIN-4097, then keys by distance from the window end: [WIN-4096,WIN-2048), [WIN-2048,WIN-1024), [WIN-1024,WIN-512), [WIN-512,WIN-256), [WIN-256,WIN-128), [WIN-128,WIN-64), [WIN-64,WIN)   (POS_EDGES; the last 5 bins sum to T_rec, bin0 = T_sink)
  VN [L,H,1024] float16  L2 norm of the value vector of the 1024 window keys just before the chunk (the keys of chunk k-1, per query head; kv heads repeated if fewer): the value norm of every stream position is therefore stored once, at the chunk after it; ||v|| of the window key j of chunk k is VN[k - 8 + j // 1024][..., j % 1024] (read through vn_window(...))
Everything is computed from the same ph as e57_attdump.att4_reduce, so rows can be cross-checked with the job-278 dump: mean over the tokens of a query sentence of T_sum[h] == sum_w AL[h, s, w] (e57_attdump2_run.py --check-278).
CPU tested: e57_attdump2_run.py --selftest (reduce5 against a brute-force numpy loop)."""
import numpy as np
CH=1024;NW=8;WIN=NW*CH
POS_EDGES=[0,4,WIN-4096,WIN-2048,WIN-1024,WIN-512,WIN-256,WIN-128,WIN-64,WIN]
def reduce5(ph,vnorm,xp=np):
    """ph [H, n, WIN+n] softmax probabilities (window then chunk keys); vnorm [H, 1024] (norm of V of the last 1024 window keys) -> dict of ONE layer"""
    pw=ph[:,:,:WIN];pc=ph[:,:,WIN:]
    pos=xp.stack([pw[:,:,POS_EDGES[i]:POS_EDGES[i+1]].sum(-1) for i in range(len(POS_EDGES)-1)],axis=-1)
    return dict(T_max=pw.max(-1),T_key=pw.argmax(-1),T_sum=pw.sum(-1),T_ent=-(pw*xp.log(pw+1e-30)).sum(-1),T_sink=pw[:,:,:4].sum(-1),T_rec=pw[:,:,WIN-1024:].sum(-1),T_cur=pc.sum(-1),POS=pos,VN=vnorm)
F16=('T_max','T_sum','T_ent','T_sink','T_rec','T_cur','POS','VN')
def install(s3):
    """add the mode 'att5' to the sliding-window attention of s3 (as e57_attdump.install adds 'att4'); other modes are passed on unchanged"""
    import jax,jax.numpy as jnp
    SWA=s3.SWA;prev=SWA.full_sw_attention;WIN_=s3.WIN
    def _full5(self,hidden_states,seq,state):
        if s3.S3.get('mode')!='att5':return prev(self,hidden_states,seq,state)
        S3=s3.S3;Kc,Vc=state;q0=S3['q0'];n=hidden_states.shape[0]
        xq,xk,xv=self.get_attention_input(hidden_states,position_ids=q0+jnp.arange(n,dtype=jnp.int32))
        K=jnp.concatenate([Kc.astype(xk.dtype),xk],axis=0);V=jnp.concatenate([Vc.astype(xv.dtype),xv],axis=0)
        qi=(q0+jnp.arange(n,dtype=jnp.int32))[:,None];kj=(q0-WIN_+jnp.arange(WIN_+n,dtype=jnp.int32))[None,:]
        mask=(qi>=kj)&(qi-kj<WIN_)&(kj>=0)
        q=xq.astype(jnp.float32);Kx=K.astype(jnp.float32)
        if Kx.shape[1]!=q.shape[1]:Kx=jnp.repeat(Kx,q.shape[1]//Kx.shape[1],axis=1)
        sc=jnp.einsum('qhd,khd->hqk',q,Kx)*(q.shape[-1]**-0.5);sc=jnp.where(mask[None],sc,-1e30);ph=jax.nn.softmax(sc,axis=-1)
        vn=jnp.linalg.norm(V[WIN_-1024:WIN_].astype(jnp.float32),axis=-1)           # [1024, Hkv]
        if vn.shape[1]!=q.shape[1]:vn=jnp.repeat(vn,q.shape[1]//vn.shape[1],axis=1)
        r=reduce5(ph,vn.T,xp=jnp)
        S3['att5_out'].append({k_:np.asarray(v_,np.float16 if k_ in F16 else np.int16) for k_,v_ in r.items()})
        o=jax.nn.dot_product_attention(xq,K,V,mask=mask,implementation='xla');o=self._merge_heads(o)
        return (self.get_attention_output(o),state)
    SWA.full_sw_attention=_full5
def att5_pass(L,rt,s3,m,Kc,Vc,ids_k,k):
    """GPU forward-only pass of the prefix layers for chunk k with the unmasked window -> dict name -> array [L, ...] (the arrays of the module doc)"""
    import jax,jax.numpy as jnp
    s3.S3.clear();s3.S3.update(mode='att5',q0=jnp.int32(k*CH),att5_out=[])
    try:
        ids=jnp.asarray(ids_k,dtype=jnp.int32)
        with rt.mesh:
            h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
            for l in range(Kc.shape[0]):
                blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
        per=s3.S3['att5_out']
    finally:s3.S3.clear()
    return {k_:np.stack([p[k_] for p in per]) for k_ in per[0]}
def vn_window(VNs,k):
    """value norms of the 8 window chunks of chunk k, [L,H,WIN] float32; VNs[j] = the VN array stored at chunk j (= norms of chunk j-1); chunks before 0 are zeros"""
    parts=[]
    for c in range(k-NW,k):
        j=c+1;parts.append(np.asarray(VNs[j],np.float32) if j in VNs and c>=0 else np.zeros_like(np.asarray(next(iter(VNs.values())),np.float32)))
    return np.concatenate(parts,axis=-1)
