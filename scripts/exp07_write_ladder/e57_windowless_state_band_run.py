"""E57 band-window NLL AFTER writing (job 345, post-hoc, exploratory; forward only, NO weight update): like e57_windowless_state_run.py (job 298) but with the BAND baseline of the label-free-mask thread. For each saved written state (--wstates LABEL=FILE:HASH@B1+B2,...; W0 = LABEL=@B1+B2) and the chunks 8..279 of the one-cycle news stream: the per-token NLL under the natural unmasked window (nll_win) and under every band window b (each query row sees only the previous b tokens, window AND chunk keys, prefix AND suffix layers; e57_closure_masks.band_keep) -> <out>/E57_band_<LABEL>.npz (nll_win [280,1024], nll_band [nb,280,1024], bands, lm, tgt; rows of chunks < 8 are zero) + .json (control: window NLL vs the CE of the write forward). Band gain of a state = nll_band - nll_win; its ratio to the W0 band gain is formed offline.
  python3 e57_windowless_state_band_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --wstates NAT=F:H@512+1024,... --stage arms"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
import e57_closure_masks as CM
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
    K0=8
    def nll_of(m,k,ids_k,Kc,Vc,keep,ctx):
        kp=jnp.asarray(keep)
        if keep is ctx['_natural']:return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],kp),np.float64)
        pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp)
        return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,kp),np.float64)
    CUR={'bands':()}
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        keep=np.asarray(keep,bool);assert keep.ndim==1 and keep.shape[0]==WIN_+CH,keep.shape;ctx=dict(ctx);ctx['_natural']=keep
        lm_=np.asarray(ctx['c'].loss_masks,bool);tg_=np.asarray(ctx['c'].target_tokens,np.int32)
        if k<K0:return dict(nll=np.zeros(CH,np.float32),nllb=np.zeros((len(CUR['bands']),CH),np.float32),lm=lm_,tgt=tg_)
        with rt.mesh:
            r=dict(nll=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32))
            r['nllb']=np.stack([nll_of(m,k,ids_k,Kc,Vc,CM.band_keep(keep,bk),ctx).astype(np.float32) for bk in CUR['bands']])
        r['lm']=lm_;r['tgt']=tg_;return r
    fn.wants_ctx=True
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    def dtypes(m):
        import collections as _c
        return dict(_c.Counter(str(x.dtype) for x in LD.jax.tree_util.tree_leaves(LD.eqx.partition(m,rd.f)[0])))
    live=dtypes(rt.base)
    for lab,(fil,hh,bds) in WS['states']:
        CUR['bands']=tuple(bds);fnj=out/f'E57_band_{lab}.json'
        if fnj.exists() and not a.rerun:lg(f'exists, reused: {fnj}');continue
        w0=None
        if fil:
            t1=time.monotonic();w0=rd.load_state(fil,expect_hash=hh);dt=dtypes(w0);assert dt==live,('saved weights have another precision than the training state',dt,live);lg(f'  {lab}: state {fil} loaded, hash checked in {time.monotonic()-t1:.0f}s')
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_band_{lab}',canary=None);L.rec_cache=rc;t0=time.monotonic()
        _ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]      # parameters returned unchanged
        res=L.run_stream('NAT',1,ids_all,nch=nch,tag=f'wnll{lab}',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(nch)),weights0=w0)
        D=res['diag'];ks=sorted(D);assert ks==list(range(nch)),(len(ks),nch)
        nll=np.stack([D[k]['nll'] for k in ks]);nb=np.stack([np.stack([D[k]['nllb'][i] for k in ks]) for i in range(len(bds))]);lm=np.stack([D[k]['lm'] for k in ks]);tgt=np.stack([D[k]['tgt'] for k in ks])
        np.savez_compressed(out/f'E57_band_{lab}.npz',nll_win=nll,nll_band=nb,bands=np.asarray(bds,int),lm=lm,tgt=tgt)
        ce=[e['ce'] for e in res['log']];mm=[float(nll[k][lm[k]].mean()) if lm[k].any() else float('nan') for k in ks];d=np.abs(np.asarray(ce)-np.asarray(mm))
        fnj.write_text(json.dumps(dict(label=lab,state_file=fil or None,state_hash=hh or None,bands=list(bds),bank=bsrc,n_chunks=nch,sec_total=time.monotonic()-t0,max_abs_diff=float(np.nanmax(d[K0:])),mean_nll_win=float(nll[K0:][lm[K0:]].mean()),mean_nll_band={str(b):float(nb[i][K0:][lm[K0:]].mean()) for i,b in enumerate(bds)}),default=float))
        lg(f'{lab}: chunks 8..279 mean NLL window {float(nll[K0:][lm[K0:]].mean()):.4f}, '+', '.join(f'band {b} {float(nb[i][K0:][lm[K0:]].mean()):.4f}' for i,b in enumerate(bds))+f'; control max abs diff {float(np.nanmax(d[K0:])):.4g}')
        del L,res,w0;gc.collect();LD.jax.clear_caches()
def main():
    if '--wstates' in sys.argv:
        i=sys.argv.index('--wstates');v=sys.argv[i+1];del sys.argv[i:i+2];WS['states']=[]
        for x in v.split(','):
            lab,_,r=x.partition('=');r,_,bd=r.partition('@');fil,_,hh=r.partition(':');assert lab and bd and (not fil or hh),('label=FILE:HASH@B1+B2 or label=@B1+B2 for W0',x);WS['states'].append((lab,(fil,hh,[int(b) for b in bd.split('+')])))
    elif '--selftest' not in sys.argv:raise SystemExit('--wstates is required')
    import e57_main_run as M
    M.stage_arms=wnll_stage;M.main()
if __name__=='__main__':main()
