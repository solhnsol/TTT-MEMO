"""E57 key-mask closure with the query set of the write arms (job 346, post-hoc, exploratory; forward only, W0 weights, NO weight update). For the chunks k % 16 == 0 (16..272) of the one-cycle news stream and every arm of the key-mask files (--keyq-masks F1,F2: e57_keyq_masks.py / e57_keyq_vmasks.py format; qb_<arm>_<k> = the GAINQ512 rows of p=.26, kq_<arm>_<k> = the window keys hidden in those rows) the per-token NLL with the per-query keep of e57_keyq_run.SelMaskKQ (hide only the listed keys in the selected rows, prefix and suffix layers) and the natural window NLL -> <out>/E57_keyq_closure_W0.npz (arms, chunks, nll_win [nc,1024], nll_mask [arms,nc,1024], lm, tgt). Closure = (masked loss - window loss) / (band-512 loss - window loss) is formed offline on the selected rows with the band NLL of e57_band_W0.npz.
  python3 e57_keyq_closure_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --keyq-masks F1,F2 --stage arms"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
import e57_closure_masks as CM
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
WS={'states':None,'masks':[]}
def wnll_stage(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_gpu as RG
    CH=1024;assert N==1 and arms==['NAT'],'one cycle, NAT only'
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_wnll_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    WIN_=8*CH
    import e57_keyq_run as KR
    MK={};armnames=[]
    for f in WS['masks']:
        z={k_:v for k_,v in np.load(f).items()};MK[f]=z;armnames+=[(str(x),f) for x in z['names']]
    objs={an:KR.SelMaskKQ(an,MK[f],log=lambda *x:None) for an,f in armnames}
    def nll_of(m,k,ids_k,Kc,Vc,keep,ctx):
        kp=jnp.asarray(keep)
        if keep is ctx['_natural']:return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],kp),np.float64)
        pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp)
        return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,kp),np.float64)
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        keep=np.asarray(keep,bool);ctx=dict(ctx);ctx['_natural']=keep;lm_=np.asarray(ctx['c'].loss_masks,bool);tg_=np.asarray(ctx['c'].target_tokens,np.int32)
        if k%16!=0 or k<16:return dict(nll=np.zeros(CH,np.float32),nllm=np.zeros((len(armnames),CH),np.float32),lm=lm_,tgt=tg_)
        with rt.mesh:
            r=dict(nll=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32));rows=[]
            for an,f in armnames:
                kp,_=objs[an].keep(k,None,None,None,None,keep);rows.append(nll_of(m,k,ids_k,Kc,Vc,np.asarray(kp,bool),ctx).astype(np.float32))
            r['nllm']=np.stack(rows)
        r['lm']=lm_;r['tgt']=tg_;lg(f'  keyq closure chunk {k}');return r
    fn.wants_ctx=True
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    def dtypes(m):
        import collections as _c
        return dict(_c.Counter(str(x.dtype) for x in LD.jax.tree_util.tree_leaves(LD.eqx.partition(m,rd.f)[0])))
    live=dtypes(rt.base)
    for lab,(fil,hh,bds) in [('W0',('',''  ,[]))]:
        fnj=out/'E57_keyq_closure_W0.json'
        if fnj.exists() and not a.rerun:lg(f'exists, reused: {fnj}');continue
        w0=None
        if fil:
            t1=time.monotonic();w0=rd.load_state(fil,expect_hash=hh);dt=dtypes(w0);assert dt==live,('saved weights have another precision than the training state',dt,live);lg(f'  {lab}: state {fil} loaded, hash checked in {time.monotonic()-t1:.0f}s')
        L=C.Ladder55(rt,save_dir=out/'states'/'E57_keyq_closure',canary=None);L.rec_cache=rc;t0=time.monotonic()
        _ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]      # parameters returned unchanged
        res=L.run_stream('NAT',1,ids_all,nch=nch,tag=f'wnll{lab}',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(nch)),weights0=w0)
        D=res['diag'];ks=sorted(D);assert ks==list(range(nch)),(len(ks),nch);sel=[k for k in ks if k%16==0 and k>=16]
        nll=np.stack([D[k]['nll'] for k in sel]);nm=np.stack([np.stack([D[k]['nllm'][i] for k in sel]) for i in range(len(armnames))]);lm=np.stack([D[k]['lm'] for k in sel]);tgt=np.stack([D[k]['tgt'] for k in sel])
        np.savez_compressed(out/'E57_keyq_closure_W0.npz',arms=np.asarray([a_ for a_,_ in armnames]),chunks=np.asarray(sel),nll_win=nll,nll_mask=nm,lm=lm,tgt=tgt)
        ce=[e['ce'] for e in res['log']];d_=np.array([abs(ce[k]-float(D[k]['nll'][D[k]['lm']].mean())) for k in sel])
        fnj.write_text(json.dumps(dict(arms=[a_ for a_,_ in armnames],chunks=sel,sec_total=time.monotonic()-t0,max_abs_diff=float(d_.max())),default=float));lg(f'wrote E57_keyq_closure_W0.npz, {len(armnames)} arms, {len(sel)} chunks; control max abs diff {float(d_.max()):.4g}')
        del L,res,w0;gc.collect();LD.jax.clear_caches()
def main():
    if '--keyq-masks' in sys.argv:
        i=sys.argv.index('--keyq-masks');WS['masks']=sys.argv[i+1].split(',');del sys.argv[i:i+2]
    elif '--selftest' not in sys.argv:raise SystemExit('--keyq-masks is required')
    import e57_main_run as M
    M.stage_arms=wnll_stage;M.main()
if __name__=='__main__':main()
