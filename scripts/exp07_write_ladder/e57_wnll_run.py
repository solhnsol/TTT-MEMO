"""E57 write-time NLL check (post-hoc, exploratory): the NAT arm of ONE cycle (280 chunks) re-run exactly as in job 261 / 262 (same stream, same order, plain SGD write), and for EVERY chunk the per-token NLL of the chunk under the weights BEFORE its write (the NAT window, no mask; the same quantity as the write-forward CE that update_log['ce'] records as a mean), saved per token. The CPU analysis e57_wnll_analysis.py then splits the tokens of answer sentences by 'a copy of the same answer sentence already in the window' / none and by exposure number.
  python3 e57_wnll_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --stage arms
Output O/E57_wnll_NAT.npz (nll float32 [nch, CH], lm bool [nch, CH], tgt int32 [nch, CH]) and O/E57_wnll_NAT.json (per-chunk CE of the run, control: mean masked NLL per chunk vs the CE of the write forward)."""
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
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        with rt.mesh:lp=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],jnp.asarray(keep)),np.float64)
        return dict(nll=(-lp).astype(np.float32),lm=np.asarray(ctx['c'].loss_masks,bool),tgt=np.asarray(ctx['c'].target_tokens,np.int32))
    fn.wants_ctx=True
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_wnll_NAT',canary=None);L.rec_cache=rc;t0=time.monotonic()
    res=L.run_stream('NAT',1,ids_all,nch=nch,tag='wnll',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(nch)))
    D=res['diag'];ks=sorted(D);assert ks==list(range(nch)),(len(ks),nch)
    nll=np.stack([D[k]['nll'] for k in ks]);lm=np.stack([D[k]['lm'] for k in ks]);tgt=np.stack([D[k]['tgt'] for k in ks])
    np.savez_compressed(out/'E57_wnll_NAT.npz',nll=nll,lm=lm,tgt=tgt)
    ce=[e['ce'] for e in res['log']];mm=[float(nll[k][lm[k]].mean()) if lm[k].any() else float('nan') for k in ks];d=np.abs(np.asarray(ce)-np.asarray(mm))
    (out/'E57_wnll_NAT.json').write_text(json.dumps(dict(bank=bsrc,n_chunks=nch,sec_total=time.monotonic()-t0,ce_write_forward=ce,mean_masked_nll=mm,max_abs_diff=float(np.nanmax(d)),mean_abs_diff=float(np.nanmean(d)),update_norm=[e['update_norm'] for e in res['log']]),default=float))
    lg(f'wrote {out/"E57_wnll_NAT.npz"}; control: mean masked NLL vs write-forward CE per chunk, max abs diff {float(np.nanmax(d)):.4g}, mean {float(np.nanmean(d)):.4g}')
def main():
    import e57_main_run as M
    M.stage_arms=wnll_stage;M.main()
if __name__=='__main__':main()
