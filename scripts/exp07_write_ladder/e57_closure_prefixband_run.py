"""E57 prefix-only band baseline (job 561; GPU forward only, W0 weights, chunks 8..279; post-hoc, exploratory). Same pass as e57_closure_run.py --bands (the stream is walked with the write step returning the parameters unchanged), but the band-k keep (CM.band_keep: every query sees only the previous k tokens, window AND chunk keys) is given to the PREFIX layers only (L.prefix_masked); the suffix layers keep the natural 1-D window keep. nll_win (natural) is recomputed as the control and compared with a reference npz. Output E57_prefixband_W0.npz: nll_win [280,1024], nll_prefixband [n_bands,280,1024], bands, chunks [8,280], lm, tgt.
  python3 e57_closure_prefixband_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 [--bands 512] [--ref E57_closure_W0.npz] --stage arms   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib,os
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
def prefixband_stage(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx
    CH=1024;WIN_=8*CH;assert N==1 and arms==['NAT'],'one cycle, NAT only'
    bands=[int(x) for x in (getattr(a,'bands',None) or '512').split(',') if x];k0,k1=8,nch
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_prefixband_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    tm=[0.0]
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        t1=time.monotonic();keep=np.asarray(keep,bool);assert keep.ndim==1 and keep.shape[0]==WIN_+CH,keep.shape
        if k<k0:return dict(nll=np.zeros(CH,np.float32),lm=np.asarray(ctx['c'].loss_masks,bool),tgt=np.asarray(ctx['c'].target_tokens,np.int32))
        with rt.mesh:
            kn=jnp.asarray(keep);r=dict(nll=-np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],kn),np.float64).astype(np.float32))      # natural control: prefix and suffix natural
            nb=[]
            for bk in bands:
                pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(CM.band_keep(keep,bk)))      # band in the prefix layers only
                nb.append((-np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,kn),np.float64)).astype(np.float32))      # suffix layers: natural 1-D keep
            r['nllb']=np.stack(nb)
        r['lm']=np.asarray(ctx['c'].loss_masks,bool);r['tgt']=np.asarray(ctx['c'].target_tokens,np.int32);tm[0]+=time.monotonic()-t1
        if k%16==0:lg(f'  prefixband chunk {k}: {tm[0]:.0f}s in forwards')
        return r
    fn.wants_ctx=True
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_prefixband_W0',canary=None);L.rec_cache=rc;t0=time.monotonic()
    _ws=L.write_step;L.write_step=lambda m,*a_,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]      # W0: parameters returned unchanged
    res=L.run_stream('NAT',1,ids_all,nch=nch,tag='prefixband',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(nch)))
    D=res['diag'];ks=sorted(D);assert ks==list(range(nch)),(len(ks),nch)
    nw=np.stack([D[k]['nll'] for k in ks]);lm=np.stack([D[k]['lm'] for k in ks]);tgt=np.stack([D[k]['tgt'] for k in ks])
    nb=np.stack([np.stack([D[k]['nllb'][i] if k>=k0 else np.zeros(CH,np.float32) for k in ks]) for i in range(len(bands))])
    np.savez_compressed(out/'E57_prefixband_W0.npz',chunks=np.asarray([k0,k1]),nll_win=nw,nll_prefixband=nb,bands=np.asarray(bands,int),lm=lm,tgt=tgt)
    ctl={}
    if getattr(a,'ref',None) and os.path.exists(a.ref):
        R=np.load(a.ref);sl=slice(k0,k1);ctl['vs_ref_nll_win_max_abs']=float(np.abs(R['nll_win'][sl]-nw[sl]).max())
        if 'nll_band' in R.files and len(R['nll_band']):
            for i,b in enumerate(bands):
                if b in [int(x) for x in R['bands']]:j=[int(x) for x in R['bands']].index(b);ctl[f'mean_nll_band{b}_both_layers']=float(R['nll_band'][j][sl][lm[sl]].mean())
    (out/'E57_prefixband_W0.json').write_text(json.dumps(dict(bank=bsrc,n_chunks=nch,bands=bands,sec_total=time.monotonic()-t0,sec_forwards=tm[0],control=ctl,mean_nll_win=float(nw[k0:k1][lm[k0:k1]].mean()),mean_nll_prefixband={str(b):float(nb[i][k0:k1][lm[k0:k1]].mean()) for i,b in enumerate(bands)}),indent=1,default=float))
    lg(f'wrote {out/"E57_prefixband_W0.npz"}; control {ctl}')
def main():
    import argparse,e57_main_run as M
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--ref');ap.add_argument('--bands');pa,rest=ap.parse_known_args();sys.argv=[sys.argv[0]]+rest
    def stage(a,*r,**kw):a.ref=pa.ref;a.bands=pa.bands;return prefixband_stage(a,*r,**kw)
    M.stage_arms=stage;M.main()
def selftest():
    CM.selftest();nat=np.ones(9216,bool);nat[:100]=False;b=CM.band_keep(nat,512);assert b.shape==(1024,9216) and nat.ndim==1;print('e57_closure_prefixband_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:main()
