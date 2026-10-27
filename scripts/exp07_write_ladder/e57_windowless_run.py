"""E57 window-less NLL (job 287a, post-hoc, exploratory; forward only, W0 weights, NO weight update): for EVERY chunk k = 0..nch-1 of the one-cycle news stream the per-token NLL under (a) the natural unmasked window (nll_win) and (b) the chunk only: all window keys masked in the prefix AND suffix layers (nll_chunk). Same stream / chunk indices as the attention dump of job 278 (W0). The write step of run_stream is wrapped so that its parameters are returned unchanged (the gradient is still computed and discarded); control: nll_win mean per chunk vs the CE of that write forward.
Code path: copy of e57_wnll_run.py with the second pass added; GPU part NOT exercised before the job.
  python3 e57_wnll_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --stage arms
Output O/E57_windowless_W0.npz (nll_win, nll_chunk float32 [nch, CH], lm bool, tgt int32) and O/E57_windowless_W0.json (per-chunk CE of the run, control: mean masked NLL per chunk vs the CE of the write forward)."""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def wnll_stage(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx
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
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_windowless_W0',canary=None);L.rec_cache=rc;t0=time.monotonic()
    _ws=L.write_step;L.write_step=lambda m,*a,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a,**kw))[1:]      # W0: parameters returned unchanged
    res=L.run_stream('NAT',1,ids_all,nch=nch,tag='wnll',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(nch)))
    D=res['diag'];ks=sorted(D);assert ks==list(range(nch)),(len(ks),nch)
    nll=np.stack([D[k]['nll'] for k in ks]);nllc=np.stack([D[k]['nllc'] for k in ks]);lm=np.stack([D[k]['lm'] for k in ks]);tgt=np.stack([D[k]['tgt'] for k in ks])
    np.savez_compressed(out/'E57_windowless_W0.npz',nll_win=nll,nll_chunk=nllc,lm=lm,tgt=tgt)
    ce=[e['ce'] for e in res['log']];mm=[float(nll[k][lm[k]].mean()) if lm[k].any() else float('nan') for k in ks];d=np.abs(np.asarray(ce)-np.asarray(mm))
    (out/'E57_windowless_W0.json').write_text(json.dumps(dict(bank=bsrc,n_chunks=nch,sec_total=time.monotonic()-t0,ce_write_forward=ce,mean_masked_nll=mm,max_abs_diff=float(np.nanmax(d)),mean_nll_win=float(nll[lm].mean()),mean_nll_chunk=float(nllc[lm].mean()),frac_tokens_chunk_lower=float((nllc[lm]<nll[lm]).mean()),mean_abs_diff=float(np.nanmean(d)),update_norm=[e['update_norm'] for e in res['log']]),default=float))
    lg(f'wrote {out/"E57_windowless_W0.npz"}; control: mean masked NLL vs write-forward CE per chunk, max abs diff {float(np.nanmax(d)):.4g}, mean {float(np.nanmean(d)):.4g}')
def main():
    import e57_main_run as M
    M.stage_arms=wnll_stage;M.main()
if __name__=='__main__':main()
