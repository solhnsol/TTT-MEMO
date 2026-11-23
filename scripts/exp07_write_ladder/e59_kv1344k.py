"""Key-value decomposition of the pre-clip fast-weight gradient (job 1344k, spec v2 items A-F and G(i); E2E 3B JAX, frozen W0, NO weight update; new file; thread-server 11/23; Hans 11/23 'key-value decomposition', spec reports/server-request-1344k-v2-and-E61.md; EXPLORATORY). Same setting as jobs 1340 / 1344 (U12 world seeds 231..238, 8 chunks 16..127, update-path attention masks of e59_shadow.mk_uv) with the conditions C0 (all other facts hidden, the same fact hidden), SAMEVIS (same fact visible, other facts hidden), NATVIS (no mask), U1vis, U2vis. For every (condition, chunk) and every one of the 8 prime SwiGLU layers (w1, w3: x = ffn_prime_norm(h) [2560]; w2: x2 = silu(z1)*z3 [5632]; e59_kvprobe.py) it takes, for the UPDATE objective (mean NLL of all valid tokens of the chunk, before clipping): the layer inputs x_p, x2_p and the error signals delta_z1, delta_z3, delta_z2 at every position p of the chunk, checks sum_p x2_p (x) delta_z2,p (and the w1 / w3 analogues) against the autodiff gradient of the same objective (relative Frobenius error per layer and matrix), and saves only the scalars / small arrays needed for the analysis (e59_kv_analyze.py):
  X2K [T,8,16]  x2_p . k_f (layer l; k_f = the x2 of the layer at the position that predicts the FIRST answer token of the gold row of fact f in the neutral read prompt of the out-of-window read, W0 fp32; the 16 read facts)
  X2KJ [T,2,16] the same at layer 8 (index 7, = layer 32 of the model) for the 2nd / 3rd answer token keys
  D16 [T,16]    U_c . (gamma * delta_z2,p) at layer 8: the candidate logits of delta_z2 through the final norm weight and the unembedding rows of the first tokens of the 16 candidates (linear approximation: rms scaling of the final norm and the layers after the prime FFN are ignored)
  DN, XN [T,8]  |delta_z2,p|, |x2_p| per layer;  XIN [T,8] |x_p|;  NLL [T] per-position NLL
  meta json: clip-free gradient norms per layer and matrix, global norm, clip coefficient, validation errors, the answer positions of every fact, key norms, the candidate first tokens.
Keys (neutral read prompt, W0): the same 8192-token neutral tail and the gold rows of the out-of-window read (Reader55 / packing.Packed with 1 chunk per call), forward only. Unchanged weights: ws returns the stream weights unchanged (as job 1340 / 1344).
  python3 e59_kv1344k.py --seed 231 --out O [--chunks 16,32,...] [--nch N] [--conds C0,SAMEVIS,...]"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_shadow as SH
CH=1024;NL=8;NREAD=16
CONDS={'C0':'SAME_U1U2','SAMEVIS':('ha','OTHERONLY'),'NATVIS':('nat','NAT'),'U1vis':'SAME_U2','U2vis':'SAME_U1'}
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_kvprobe as KP
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);chunks=[int(x) for x in a.chunks.split(',')];cl=a.conds.split(',');t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w))
    assert all(items[i]['kind']=='fact' and items[i]['fact']==i for i in range(NREAD))
    gold_rows=[raw[i*G.NCAND+items[i]['gold_idx']] for i in range(NREAD)];tok1=np.array([r[0][r[1][0]+1] for r in gold_rows],np.int64);alen=[int(r[1][1]-r[1][0]) for r in gold_rows];NJ=min(3,max(alen))
    # ---- read keys: neutral read prompt, W0 fp32, forward only, probe forward (1 packed chunk per call)
    t0=time.monotonic();keys=None;kf=out/f'kv_s{seed}_keys.npy'
    if os.environ.get('E59_CPU_TEST'):
        _cur=jax.nn.dot_product_attention;jax.nn.dot_product_attention=lambda *a_,**k_:_cur(*a_,**dict(k_,implementation='xla'))       # CPU-only validation run: no cuDNN attention
    if kf.exists():keys=np.load(kf);NJ=keys.shape[1]
    with rt32.mesh,jax.default_matmul_precision('highest'):
      if keys is None:
        T_=LD._TailCtx(rt32,list(out_tail));Pk=LD.packing.Packed(T_,gold_rows,GP=1);m32=rd.to_fp32(rt.base);mp32=KP.install(m32,'fwd');st=LD.packing._clone(rt32.state_suffix)
        for j in range(len(out_tail)//1024):st,_=rt32.chunk_logp(m32,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*1024:(j+1)*1024])
        for c in range(Pk.nchunk):
            KP.CAP['fwd'].clear();lp=Pk.lp_packed(mp32,st,Pk.ids[c:c+1],Pk.tg[c:c+1],Pk.mk[c:c+1],Pk.pref1[c:c+1],Pk.seg[c:c+1],Pk.loc[c:c+1]);jax.block_until_ready(lp);jax.effects_barrier();cap=dict(KP.CAP['fwd']);assert sorted(cap)==list(range(NL)),sorted(cap)
            if keys is None:I=cap[0][1].shape[-1];keys=np.zeros((NREAD,NJ,NL,I),np.float32)
            for r,(cc,a_,b_) in enumerate(Pk.where):
                if cc!=c:continue
                for j in range(min(NJ,b_-a_)):
                    for l in range(NL):keys[r,j,l]=cap[l][1][a_+j]
        del mp32,m32,st
    lg(f'seed {seed}: read keys {keys.shape} in {time.monotonic()-t0:.0f}s (|k| layer 8 first token mean {np.linalg.norm(keys[:,0,7],axis=-1).mean():.2f})')
    base=rt.base;lm=base.language_model;Wte=np.asarray(lm.model.wte.weight,np.float32) if lm.lm_head is None else np.asarray(lm.lm_head.weight,np.float32).T;gam=np.asarray(lm.model.ln_f.weight,np.float32);Uc=(Wte[tok1]*gam[None]).astype(np.float32)   # [16,H] rows of the 16 candidates (first tokens), final-norm weight folded in
    NS=262144;inner=rt.base.inner_parameters();leaves=jax.tree_util.tree_leaves(inner)
    @eqx.filter_jit
    def fwd_probe(mp,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=mp.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        return jnp.sum(o.logits[:8,:8].astype(jnp.float32))
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    @eqx.filter_jit
    def bwd_probe(mp,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(mp,st_suf,c,pref_,kmask,wt);return nll,jnp.sum(KP.WHERE(g).w2.weight.astype(jnp.float32))
    def named_leaves(g):
        d={}
        for pth,leaf in jax.tree_util.tree_flatten_with_path(g)[0]:
            s=jax.tree_util.keystr(pth)
            for nm in ('w1','w2','w3'):
                if f'.{nm}.' in s+'.' or f'{nm}.weight' in s:d[nm]=leaf
        assert sorted(d)==['w1','w2','w3'],list(d);return d
    ARR={};META=dict(seed=seed,chunks=chunks,conds=cl,tok1=tok1.tolist(),alen=alen,NJ=NJ,knorm=np.linalg.norm(keys[:,0],axis=-1).tolist(),rows=[],AP={str(k):[dict(fact=int(d['fact']),second=[int(t)-1 for t in d['second']],first=[int(t)-1 for t in d['first']]) for d in ap[k]] for k in chunks},args=vars(a))
    np.save(out/f'kv_s{seed}_keys.npy',keys)
    for cn in cl:
        arm=CONDS[cn];cnt=[0];L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{cn}',canary=can);L.rec_cache={}
        def ws(m,st_all,st_suf,c,pref,kmask,cnt=cnt,cn=cn):
            k=cnt[0];cnt[0]+=1
            if k in chunks:
                t1=time.monotonic();valid=np.asarray(c.loss_masks,np.float32);Wall=SH.token_weights(ap[k],valid>0)['all'];wt=jnp.asarray(Wall)
                KP.CAP['fwd'].clear();o=fwd_probe(KP.install(m,'fwd'),st_suf,c,pref,kmask);jax.block_until_ready(o);jax.effects_barrier();xs=dict(KP.CAP['fwd']);assert sorted(xs)==list(range(NL))
                g,nll=ref_grad(m,st_suf,c,pref,kmask,wt);nll=np.asarray(nll,np.float32);nl_=named_leaves(g);gm={nm:np.asarray(v,np.float32) for nm,v in nl_.items()}
                KP.CAP['bwd'].clear();nll2,_s=bwd_probe(KP.install(m,'bwd'),st_suf,c,pref,kmask,wt);jax.block_until_ready(nll2);jax.effects_barrier();bw=dict(KP.CAP['bwd']);assert len(bw)==3*NL,len(bw)
                T=nll.shape[0];X2K=np.zeros((T,NL,NREAD),np.float32);X2KJ=np.zeros((T,2,NREAD),np.float32);DN=np.zeros((T,NL),np.float32);XN=np.zeros((T,NL),np.float32);XIN=np.zeros((T,NL),np.float32);val=np.zeros((NL,3));gl=np.zeros((NL,3))
                for l in range(NL):
                    x,x2=xs[l];dz1,dz3,dz2=bw[('z1',l)],bw[('z3',l)],bw[('z2',l)]
                    x=x.astype(np.float32);x2=x2.astype(np.float32);dz1=dz1.astype(np.float32);dz3=dz3.astype(np.float32);dz2=dz2.astype(np.float32)
                    for mi,(nm,rec) in enumerate((('w1',x.T@dz1),('w2',x2.T@dz2),('w3',x.T@dz3))):
                        ref=gm[nm][l];val[l,mi]=float(np.linalg.norm(rec-ref)/(np.linalg.norm(ref)+1e-30));gl[l,mi]=float(np.linalg.norm(ref))
                    X2K[:,l,:]=x2@keys[:,0,l,:].T;DN[:,l]=np.linalg.norm(dz2,axis=1);XN[:,l]=np.linalg.norm(x2,axis=1);XIN[:,l]=np.linalg.norm(x,axis=1)
                    if l==NL-1:
                        D16=dz2@Uc.T
                        for jj in range(2):
                            if NJ>jj+1:X2KJ[:,jj,:]=x2@keys[:,jj+1,l,:].T
                gn=float(np.sqrt((gl**2).sum()));clip=1.0;coef=1.0 if gn<clip else clip/gn
                for nm,v in (('X2K',X2K),('X2KJ',X2KJ),('D16',D16),('DN',DN),('XN',XN),('XIN',XIN),('NLL',nll)):ARR[f'{cn}|{k}|{nm}']=v
                META['rows'].append(dict(cond=cn,k=k,gnorm=gn,coef=coef,layer_gnorm=gl.tolist(),val_relerr=val.tolist(),sec=time.monotonic()-t1))
                lg(f'seed {seed} {cn} k={k}: |g|={gn:.3f} coef={coef:.3f} validation rel.err w1 {val[:,0].max():.2e} w2 {val[:,1].max():.2e} w3 {val[:,2].max():.2e} ({time.monotonic()-t1:.0f}s)')
            return m.inner_parameters(),jnp.asarray(0.0,jnp.float32),jnp.asarray(1.0,jnp.float32),jnp.asarray(0.0,jnp.float32)
        L.write_step=ws;uv=SH.mk_uv(arm,w,seed,lg);t0=time.monotonic()
        r=L.run_stream(NM.ENGINE,1,ids0,read_chunks=[],canary_chunks=[],nch=(a.nch or nch),tag=f'kv_{cn}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=w['chunks'],uv=uv);r.pop('model',None);lg(f'seed {seed} {cn} ({arm}): {a.nch or nch} chunks in {time.monotonic()-t0:.0f}s')
        L.rec_cache=None;del L
    json.dump(META,open(out/f'kv_s{seed}.json','w'),default=float);np.savez(out/f'kv_s{seed}.npz',**ARR);lg('wrote',out/f'kv_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--chunks',default=','.join(map(str,SH.CHUNKS)));ap_.add_argument('--nch',type=int,default=0);ap_.add_argument('--conds',default=','.join(CONDS));a=ap_.parse_args();run(a)
