"""E55 one-pass runner core (design: E55-controlled-world-binding-design.md 2.1, 2.3). Built on s3.Ladder3 (record / prefix_masked / write_step / read_state / native_step are the lab's unchanged closures); this file adds only
  (1) compose_state: the carried suffix-layer SWA state (stacked over the suffix layers; K and V each [L, 8192, hidden], block of lag j = rows [(8-j)*1024, (9-j)*1024), merged heads, pre-RoPE, after qk_norm, RoPE applied inside the window at window-relative positions; all verified on the public e2e source a4fc4788, ttt/model/attention.py SWA) composed from the real / decoy / background-only chains per e55_swap.plan,
  (2) Ladder55.run_stream: the stream is read ONCE; chunk k is always WRITTEN with its real tokens; the attention window of the write forward is composed per arm (prefix layers: recorded post-RoPE K,V of the three streams via e55_swap.compose_window; suffix layers: compose_state), the three suffix chains are advanced with the SAME current weights, weights are saved at the read chunks.
Every arm goes through the same write path (prefix_masked + write_step); NAT = nothing swapped (U1 shows it equals native_step). Absolute chunk positions are unchanged: a decoy slot holds the K,V the decoy stream has at the same absolute chunk index (RoPE of the prefix layers is absolute, that of the suffix layers window-relative, so both stay aligned).
Arms: NAT, SWD, SWB, SWDS, SWDO, MSK implemented. SWB = every window slot taken from a background-only stream (gen_facts.bg_stream: same kind of text, no fact sentence at all); at p = 1 SWDS is NAT and SWDO is not NAT-equivalent either, so SWD / SWB are the arms that replace the whole window.
STATUS: tested on CPU with a tiny random e2e model (e55_units.py); the 3B GPU run is NOT done."""
import time,hashlib,json
from pathlib import Path
import numpy as np
import e55_swap as E
import s3
jax,jnp,eqx=s3.jax,s3.jnp,s3.eqx
CH=1024;WIN=8192;NW=E.NW
IMPLEMENTED=('NAT','NATM','SWD','SWB','SWDS','SWDO','MSK','SENT','KMASK','SENTO','SENTPQ','OSPAN','PSPAN','NSPAN','OSPANR25','OSPANR50','OSPANR75','SENTP25','SENTP50','SENTP75','MIX25','MIX50','MIX75','ALT','NATADAM','SENTPQADAM','SENTPQADAMH')
CHUNK_META_ARMS=E.MASK_ARMS+tuple(E.MIX)+tuple(E.ADAM)+('ALT',)
def compose_state(chains,pl,counter=None):
    """suffix-layer SWA state with the slot blocks taken from the chain that plan `pl` names ('real' | 'decoy' | 'bg'; 'mask' and absent slots keep the real chain's block, the mask is applied through the key mask). Every other leaf (chunk index) is the real chain's. counter (list) collects the number of KV leaves composed (must be 2: K and V)."""
    def f(*ls):
        l0=ls[0]
        if getattr(l0,'ndim',0)==3 and l0.shape[1]==WIN:
            if counter is not None:counter.append(1)
            blocks=[]
            for j in range(NW,0,-1):
                s=pl.get(j,'real');s='real' if s=='mask' else s
                blocks.append(ls[('real','decoy','bg').index(s)][:,(NW-j)*CH:(NW-j+1)*CH])
            return jnp.concatenate(blocks,axis=1)
        return l0
    return jax.tree.map(f,chains['real'],chains['decoy'],chains['bg'])
def keep_vector(pl):
    """bool [WIN+CH]: True = key kept; 'mask' slots False; same indexing as s3_sets.Sets.keep_vector (index j <-> absolute position k*CH-WIN+j; absent slots stay True, the kj>=0 test of the masks removes them)"""
    keep=np.ones(WIN+CH,bool)
    for j,s in pl.items():
        if s=='mask':keep[(NW-j)*CH:(NW-j+1)*CH]=False
    return keep
class Ladder55(s3.Ladder3):
    def __init__(self,rt,save_dir=None,canary=None):
        """no Readouts / old world: only what build3 and the step functions need"""
        from ttt.model.data import Batch
        self.rt=rt;cfg=rt.cfg.training.optimizer_inner;self.lr=float(cfg.lr);self.clip=float(cfg.clip_gradient);self.lr_scale=1.0;self.Batch=Batch
        self.f=s3.reuse.get_filter_spec(rt.base,rt.cfg.training.spec_inner,'inner parameters');self.save_dir=save_dir;self._nsave=0;self.canary=canary;self.build3();self.build_mix();self.build_adam()
        self.n_masked_keys=0;self.n_masked_pairs=0
    def build_adam(self):
        """E55 29.3 Q1-B: Adam direction on the (clipped) write gradient, same rule as e55_iso (job 192 ISO-ADAM): bias-corrected m / (sqrt(v) + eps), B1 .9, B2 .999, eps 1e-8, fp32; the step SIZE is set by the caller (factor x the NAT update norm of the chunk)"""
        f32=jnp.float32;clip=self.clip;B1,B2,EPS=0.9,0.999,1e-8
        @eqx.filter_jit
        def adam_dir(g,mu,nu,t):
            g=jax.tree.map(lambda x:x.astype(f32),g);gn=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g)));coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));g=jax.tree.map(lambda x:x*coef,g)
            mu=jax.tree.map(lambda a,x:B1*a+(1-B1)*x,mu,g);nu=jax.tree.map(lambda a,x:B2*a+(1-B2)*x*x,nu,g);t1=t+1.0;c1=1-B1**t1;c2=1-B2**t1
            d=jax.tree.map(lambda a,b:(a/c1)/(jnp.sqrt(b/c2)+EPS),mu,nu)
            return d,mu,nu,gn,coef
        self.adam_dir=adam_dir
    def build_mix(self):
        """E55 27.5: one write step on the SUM of two forward/backward passes of the same chunk (unmasked window and masked window), loss = sum_i w_i nll_i with per-token weights ws[pass] (already divided by the number of valid targets, so that w = valid / N in one pass is exactly lm_loss); one clip and one inner SGD step as in write_step. Returns (new inner params, pre-clip gnorm, clip coef, per-pass mean CE over all valid tokens)."""
        lr,clip=self.lr,self.clip;optax=s3.optax;S3SUF=s3.S3SUF
        @eqx.filter_jit
        def write_step_mix(m,st_all,st_suf,c,prefs,kmasks,ws):
            def total(mm):
                tot=0.0;nlls=[]
                for pref_,kmask,w in zip(prefs,kmasks,ws):
                    S3SUF['kmask']=kmask
                    try:out=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
                    finally:S3SUF.clear()
                    lp=jax.nn.log_softmax(out.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];tot=tot+jnp.sum(w*nll);nlls.append(nll)
                return tot,tuple(nlls)
            (_l,nlls),grads=eqx.filter_value_and_grad(total,has_aux=True)(m)
            g=grads.inner_parameters();gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
            newp=jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g)
            valid=c.loss_masks.astype(jnp.float32);ces=tuple(jnp.sum(valid*n_)/jnp.maximum(jnp.sum(valid),1e-10) for n_ in nlls)
            return newp,gn,coef,ces
        self.write_step_mix=write_step_mix
    @staticmethod
    def mix_weights(sents,loss_masks,lam):
        """per-position weights of the two passes of the mix (numpy, [CH]): fact positions = the loss positions that predict a token of a fact sentence (position i predicts token i+1, i.e. i in [start-1, end-1)); unmasked pass: background weight 1, fact weight 1-lam; masked pass: fact weight lam, background 0; both divided by the number of valid targets"""
        valid=np.asarray(loss_masks,np.float32);f=np.zeros(CH,bool)
        for x in sents:f[max(x['start']-1,0):max(x['end']-1,0)]=True
        n=max(float(valid.sum()),1e-10);return (valid*np.where(f,1.0-lam,1.0)/n).astype(np.float32),(valid*np.where(f,lam,0.0)/n).astype(np.float32),int(f.sum())
    def set_inner(self,m,newp):
        inner,outer=eqx.partition(m,self.f);assert jax.tree_util.tree_structure(inner)==jax.tree_util.tree_structure(newp);return eqx.combine(newp,outer)
    def save_state(self,m,name=None):
        leaves=[np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.partition(m,self.f)[0])];Path(self.save_dir).mkdir(parents=True,exist_ok=True)
        fn=Path(self.save_dir)/(name or f'state_{self._nsave:03d}.npz');self._nsave+=1;np.savez(fn,*leaves);return {'_file':str(fn),'_hash':s3.reuse._hash_pytree(m.inner_parameters())[0]}
    def load_state(self,fn):
        z=np.load(fn);leaves=[jnp.asarray(z[f'arr_{i}']) for i in range(len(z.files))]
        inner,outer=eqx.partition(self.rt.base,self.f);td=jax.tree_util.tree_structure(inner);assert td.num_leaves==len(leaves);return eqx.combine(jax.tree_util.tree_unflatten(td,leaves),outer)
    def _stream_seq(self,ids,nch):
        rt=self.rt;ids=np.asarray(ids)[:nch*CH];mask=np.zeros(nch*CH,dtype=bool);mask[:len(ids)-1]=True;return rt.seq(ids,mask=mask)
    def window_inputs(self,arm,k,p,recs,sts,counter=None):
        """what the write forward of chunk k gets: (Kc,Vc,keep,st_suffix,plan). recs[name] = (Kh,Vh) numpy [layers,T,nh,d]; sts[name] = suffix chain state."""
        pl=E.plan(arm,k,p);keep=keep_vector(pl)
        if arm in ('NAT','NATM','MSK','SENT','KMASK','SENTO','SENTPQ','OSPAN','PSPAN','NSPAN','OSPANR25','OSPANR50','OSPANR75','SENTP25','SENTP50','SENTP75'):
            Kc,Vc=self.ctx_kv(*recs['real'],k);st=sts['real']
            if arm in ('SENT','KMASK','SENTO','SENTPQ','OSPAN','PSPAN','NSPAN','OSPANR25','OSPANR50','OSPANR75','SENTP25','SENTP50','SENTP75'):keep,self.n_masked_keys=E.token_masks(arm,k,self.chunks_meta,CH,NW);self.n_masked_pairs=int((~keep).sum()) if keep.ndim==2 else int((~keep).sum())*CH
        else:
            src_k={n:recs[n][0] for n in recs};src_v={n:recs[n][1] for n in recs}
            Kw,_=E.compose_window(src_k,k,pl,CH,axis=1);Vw,_=E.compose_window(src_v,k,pl,CH,axis=1);Kc,Vc=jnp.asarray(Kw),jnp.asarray(Vw)
            st=compose_state({'real':sts['real'],'decoy':sts.get('decoy',sts['real']),'bg':sts.get('bg',sts['real'])},pl,counter)
        return Kc,Vc,keep,st,pl
    def record(self,ids,progress=None):
        """same result as s3.Ladder3.record (bitwise: the same per-layer arrays), but the host arrays [layers, T, nh, d] are allocated once and filled layer by layer: the base version collects two lists and then np.stack()s them, which holds TWICE the memory (60 GiB for one 128-chunk stream, 30 + 30 for K, V) and copies it once more single-threaded. Prints progress every 4 layers (flush) so that a slow recording is visible."""
        rt=self.rt;m=rt.base;ids_=jnp.asarray(ids,dtype=jnp.int32);blocks=m.language_model.model.h.prefix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0]
        h=self.embed(m,ids_);Kh=Vh=None;t0=time.monotonic()
        for l in range(nl):
            blk=jax.tree.map(lambda a:a[l] if eqx.is_array(a) else a,blocks);h,K,V=self.layer_rec(blk,h,ids_);K=np.asarray(jax.device_get(K));V=np.asarray(jax.device_get(V))
            if Kh is None:Kh=np.empty((nl,)+K.shape,K.dtype);Vh=np.empty((nl,)+V.shape,V.dtype)
            Kh[l]=K;Vh[l]=V;del K,V
            if len(ids)>=32*CH and (l%4==3 or l==nl-1):print(f'    record layer {l+1}/{nl}: {time.monotonic()-t0:.0f}s',flush=True)
        return Kh,Vh,h
    def run_stream(self,arm,p,real_ids,decoy_ids=None,bg_ids=None,read_chunks=(),nch=None,tag='',smoke=False,log=print,weights0=None,canary_chunks=None,read_hook=None,hook_chunks=(),norm_targets=None,chunks_meta=None,diag_fn=None,diag_chunks=(),diag_multi=None):
        """-> {'log': per-chunk dicts, 'reads': {chunk: {...state file / hash / canary}}, 'stream_hash', 'clip_bind_frac'}"""
        assert arm in E.ARMS,arm
        if arm not in IMPLEMENTED:raise NotImplementedError(f'arm {arm} is not implemented')
        self.chunks_meta=chunks_meta;self.n_masked_keys=0
        if arm in CHUNK_META_ARMS:assert chunks_meta is not None and len(chunks_meta)>=(nch or len(real_ids)//CH),'SENT / KMASK need the sentence table of the world (chunks_meta)'
        rt=self.rt;n=len(real_ids)//CH;nch=nch or n;assert nch<=n and len(real_ids)%CH==0
        names=['real']+(['decoy'] if arm in ('SWD','SWDS','SWDO') or (diag_multi is not None and decoy_ids is not None) else [])+(['bg'] if arm=='SWB' else [])
        ids_={'real':np.asarray(real_ids),'decoy':None if decoy_ids is None else np.asarray(decoy_ids),'bg':None if bg_ids is None else np.asarray(bg_ids)}
        for nm in names:assert ids_[nm] is not None and len(ids_[nm])>=nch*CH,(nm,'stream missing or too short')
        out={'log':[],'reads':{},'arm':arm,'p':p,'tag':tag,'n_chunks':nch,'stream_hash':hashlib.sha256(np.asarray(real_ids[:nch*CH],np.int64).tobytes()).hexdigest()}
        m=weights0 if weights0 is not None else rt.base;st_all=s3.clone_state(rt.state_all);cum=0.0;adam=None
        if arm in E.ADAM:assert norm_targets is not None and len(norm_targets)>=nch,'Adam arms need the per-chunk update norms of the NAT run of the same seed (norm_targets)'
        with rt.mesh:
            seqs={nm:self._stream_seq(ids_[nm],nch) for nm in names};prefs={nm:rt.prefix(rt.base,seqs[nm]) for nm in names}
            recs={}
            for nm in names:
                t0=time.monotonic();Kh,Vh,_=self.record(ids_[nm][:nch*CH]);recs[nm]=(Kh,Vh);log(f'  recorded natural K,V of {nm} ({Kh.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s')
            sts={nm:s3.clone_state(rt.state_suffix) for nm in names}
            if self.canary:out['canary_w0']=self.canary(m)
            for k in range(nch):
                t0=time.monotonic();old=m;c=rt.chunk(seqs['real'],k);cnt=[]
                self.n_masked_keys=0;self.n_masked_pairs=0;extra_log={}
                if arm in E.MIX:
                    # E55 27.5 two-loss mix: forward/backward twice on the same chunk (unmasked = NAT window, masked = SENTPQ window), gradients summed, ONE clip / step
                    lam=E.MIX[arm];ids_k=jnp.asarray(ids_['real'][k*CH:(k+1)*CH],dtype=jnp.int32)
                    Kc,Vc,keep_u,st_w,pl=self.window_inputs('NAT',k,p,recs,sts,cnt);_,_,keep_m,_,_=self.window_inputs('SENTPQ',k,p,recs,sts,cnt)
                    pref_u=self.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_u));pref_m=self.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_m))
                    w_u,w_m,nf=self.mix_weights(chunks_meta[k]['sents'],np.asarray(c.loss_masks),lam)
                    newp,gn,coef,ces=self.write_step_mix(m,st_all,st_w,c,(pref_u,pref_m),(jnp.asarray(keep_u),jnp.asarray(keep_m)),(jnp.asarray(w_u),jnp.asarray(w_m)));ce=ces[0];extra_log={'ce_masked_pass':float(ces[1]),'mix_lambda':lam,'fact_loss_positions':nf}
                else:
                    arm_k=('SENTPQ' if k%2==0 else 'NAT') if arm=='ALT' else (E.ADAM[arm][0] if arm in E.ADAM else arm)       # ALT: masked loss on the 1st, 3rd, ... chunk (k even), unmasked on the others, one forward each
                    Kc,Vc,keep,st_w,pl=self.window_inputs(arm_k,k,p,recs,sts,cnt)
                    if cnt:assert len(cnt)==2,('expected exactly the K and V leaves of the suffix SWA state, got',len(cnt))
                    pref_w=self.prefix_masked(m,jnp.asarray(ids_['real'][k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep))
                    if arm in E.ADAM:
                        f32=jnp.float32;g_,ce=self.grad_mask(m,st_w,c,pref_w,jnp.asarray(keep))
                        if adam is None:z=jax.tree.map(lambda x:jnp.zeros(x.shape,f32),m.inner_parameters());adam={'mu':z,'nu':jax.tree.map(jnp.zeros_like,z),'t':jnp.asarray(0.0,f32)}
                        d_,mu_,nu_,gn,coef=self.adam_dir(g_,adam['mu'],adam['nu'],adam['t']);adam={'mu':mu_,'nu':nu_,'t':adam['t']+1.0}
                        dn=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(d_))));adam_scale=E.ADAM[arm][1]*float(norm_targets[k])/dn if dn>0 else 1.0
                        newp=jax.tree.map(lambda p_,x:(p_.astype(f32)-adam_scale*x).astype(p_.dtype),m.inner_parameters(),d_)      # fp32 direction, size = factor x NAT update norm of this chunk
                    else:newp,gn,coef,ce=self.write_step(m,st_all,st_w,c,pref_w,jnp.asarray(keep))
                    if arm=='ALT':extra_log={'alt_masked_chunk':k%2==0}
                    if diag_multi is not None and k in diag_chunks:
                        # E55 28.3: forward-only evaluation of the chunk under several window conditions (weights before the write of the chunk): {cond: (Kc, Vc, keep, st_suffix)}
                        conds={}
                        for cn in diag_multi.conds:   # arm names of the window conditions
                            Kx,Vx,kx,sx,_pl=self.window_inputs(cn,k,p,recs,sts,[]);conds[cn]=(Kx,Vx,kx,sx)
                        out.setdefault('diag_multi',{})[k]=diag_multi(k,m,conds,ids_['real'][k*CH:(k+1)*CH],c,out['log'])
                    if diag_fn is not None and k in diag_chunks:out.setdefault('diag',{})[k]=diag_fn(k,m,Kc,Vc,keep,ids_['real'][k*CH:(k+1)*CH])      # E55 27.8: forward-only attention diagnostic with the window inputs of this chunk, weights before its write
                for nm in names:sts[nm]=self.read_state(old,sts[nm],rt.chunk(seqs[nm],k),prefs[nm][k*CH:(k+1)*CH])   # same weights (start of the chunk) for every chain
                scale=adam_scale if arm in E.ADAM else 1.0
                if arm=='NATM':          # size-matched NAT: the NAT update of chunk k is rescaled to the update norm norm_targets[k] (the per-chunk norm of arm SWD of the same seed); direction unchanged
                    assert norm_targets is not None and len(norm_targets)>=nch,'NATM needs the per-chunk update norms of the arm to match'
                    f32=jnp.float32;raw=sum(float(jnp.sum((a.astype(f32)-b.astype(f32))**2)) for a,b in zip(jax.tree.leaves(newp),jax.tree.leaves(old.inner_parameters())))**0.5
                    scale=float(norm_targets[k])/raw if raw>0 else 1.0
                    newp=jax.tree.map(lambda a,b:(b.astype(f32)+(a.astype(f32)-b.astype(f32))*scale).astype(a.dtype),newp,old.inner_parameters())
                m=self.set_inner(m,newp)
                un=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=un
                ent={'scale':scale,'k':k,'gnorm':float(gn),'coef':float(coef),'update_norm':un,'ce':float(ce),'sec':time.monotonic()-t0,'n_swapped_slots':sum(1 for v in pl.values() if v not in ('real',)),'n_masked_slots':sum(1 for v in pl.values() if v=='mask'),'n_masked_keys':int(self.n_masked_keys),'n_masked_pairs':int(getattr(self,'n_masked_pairs',0)) if arm in CHUNK_META_ARMS else 0,**extra_log}
                out['log'].append(ent)
                if read_hook is not None and k in hook_chunks:
                    t1=time.monotonic();out.setdefault('inline',{})[k]=read_hook(k,m);ent['sec_inline_read']=time.monotonic()-t1      # frozen-weight reads: m is only passed to pure scoring functions
                if k in read_chunks:
                    r={**self.save_state(m,f'state_{tag}_{k:03d}.npz')}
                    if self.canary and (canary_chunks is None or k in canary_chunks):r['canary']=self.canary(m)
                    out['reads'][k]=r;log(f'  {arm} chunk {k+1}/{nch}: state saved, hash {r["_hash"]}')
                elif k%8==0:log(f'  {arm} chunk {k+1}/{nch} {ent["sec"]:.1f}s ce {ent["ce"]:.3f}')
            out['final_hash']=s3.reuse._hash_pytree(m.inner_parameters())[0];out['model']=m
        cs=[e['coef'] for e in out['log']];out['clip_bind_frac']=sum(c_<1 for c_ in cs)/len(cs);out['cum_update_norm']=cum
        return out
