"""E57 window-less NLL AFTER writing (job 298, post-hoc, exploratory; forward only, NO weight update): e57_windowless_run.py (job 287a, W0) extended by saved written states. For W0 and for each state given by --wstates LABEL=FILE:HASH,... the per-token NLL of the one-cycle news stream (chunks 0..279) with the natural unmasked window (nll_win) and with the chunk only (all window keys masked, prefix and suffix layers; nll_chunk). A state is loaded with the Reader55 loader (hash and precision checked against the training state) and passed to run_stream as weights0; write_step returns the parameters unchanged, so every chunk is read under the same weights. One process, the natural K,V of the stream recorded once. The label W0 (empty FILE) must reproduce job 287a.
  python3 e57_windowless_state_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --wstates W0=,NAT=F:H,... --stage arms
Output O/E57_windowless_<LABEL>.npz (nll_win, nll_chunk float32 [nch, CH], lm bool, tgt int32) and .json (control: mean masked NLL of the window pass vs the CE of the write forward)."""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
WS={'states':None}
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
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        kc=np.array(keep,bool,copy=True);assert kc.ndim==1 and kc.shape[0]==WIN_+CH,kc.shape;kc[:WIN_]=False      # chunk only
        with rt.mesh:
            lp=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],jnp.asarray(keep)),np.float64)
            pc=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(kc))
            lc=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pc,jnp.asarray(kc)),np.float64)
        return dict(nll=(-lp).astype(np.float32),nllc=(-lc).astype(np.float32),lm=np.asarray(ctx['c'].loss_masks,bool),tgt=np.asarray(ctx['c'].target_tokens,np.int32))
    fn.wants_ctx=True
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    def dtypes(m):
        import collections as _c
        return dict(_c.Counter(str(x.dtype) for x in LD.jax.tree_util.tree_leaves(LD.eqx.partition(m,rd.f)[0])))
    live=dtypes(rt.base)
    for lab,(fil,hh) in WS['states']:
        fnj=out/f'E57_windowless_{lab}.json'
        if fnj.exists() and not a.rerun:lg(f'exists, reused: {fnj}');continue
        w0=None
        if fil:
            t1=time.monotonic();w0=rd.load_state(fil,expect_hash=hh);dt=dtypes(w0);assert dt==live,('saved weights have another precision than the training state',dt,live);lg(f'  {lab}: state {fil} loaded, hash checked in {time.monotonic()-t1:.0f}s')
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_windowless_{lab}',canary=None);L.rec_cache=rc;t0=time.monotonic()
        _ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]      # parameters returned unchanged
        res=L.run_stream('NAT',1,ids_all,nch=nch,tag=f'wnll{lab}',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(nch)),weights0=w0)
        D=res['diag'];ks=sorted(D);assert ks==list(range(nch)),(len(ks),nch)
        nll=np.stack([D[k]['nll'] for k in ks]);nllc=np.stack([D[k]['nllc'] for k in ks]);lm=np.stack([D[k]['lm'] for k in ks]);tgt=np.stack([D[k]['tgt'] for k in ks])
        np.savez_compressed(out/f'E57_windowless_{lab}.npz',nll_win=nll,nll_chunk=nllc,lm=lm,tgt=tgt)
        ce=[e['ce'] for e in res['log']];mm=[float(nll[k][lm[k]].mean()) if lm[k].any() else float('nan') for k in ks];d=np.abs(np.asarray(ce)-np.asarray(mm))
        fnj.write_text(json.dumps(dict(label=lab,state_file=fil or None,state_hash=hh or None,bank=bsrc,n_chunks=nch,sec_total=time.monotonic()-t0,ce_write_forward=ce,mean_masked_nll=mm,max_abs_diff=float(np.nanmax(d)),mean_nll_win=float(nll[lm].mean()),mean_nll_chunk=float(nllc[lm].mean()),frac_tokens_chunk_lower=float((nllc[lm]<nll[lm]).mean()),mean_abs_diff=float(np.nanmean(d)),update_norm=[e['update_norm'] for e in res['log']]),default=float))
        m8=lm[8:280];lg(f'{lab}: chunks 8..279 mean NLL window {float(nll[8:280][m8].mean()):.4f}, chunk only {float(nllc[8:280][m8].mean()):.4f}; control max abs diff {float(np.nanmax(d)):.4g}')
        del L,res,w0;gc.collect();LD.jax.clear_caches()
def main():
    if '--wstates' in sys.argv:
        i=sys.argv.index('--wstates');v=sys.argv[i+1];del sys.argv[i:i+2];WS['states']=[]
        for x in v.split(','):
            lab,_,r=x.partition('=');fil,_,hh=r.partition(':');assert lab and (not fil or hh),('label=FILE:HASH or label= for W0',x);WS['states'].append((lab,(fil,hh)))
    elif '--selftest' not in sys.argv:raise SystemExit('--wstates is required')
    import e57_main_run as M
    M.stage_arms=wnll_stage;M.main()
if __name__=='__main__':main()
