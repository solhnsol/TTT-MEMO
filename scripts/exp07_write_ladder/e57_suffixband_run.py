"""E57 suffix-band run (GPU forward only, W0 weights, NO weight update; post-hoc, exploratory; derived from e57_closure_run.py version b29b5ef21a24 = job 289/290 code path). Question: can the band window gain (nll_band_B - nll_win) be estimated more cheaply by masking ONLY the 8 suffix layers' attention to the previous B tokens, reusing the NATURAL prefix output of the chunk (the 24 prefix layers are not recomputed)?
For every chunk k = 8..279 of the one-cycle news stream (280 chunks, full window), per-token NLL
  nll_win     natural unmasked window, nll_chunk  chunk only (both as job 287a; kept as control against the 287a file),
  nll_sufband [n_bands,280,1024]: every query row sees only the previous B tokens (CM.band_keep, window AND chunk keys) in the 8 SUFFIX layers only, prefix output = natural (ctx['pref_w']). Compare with nll_band of e57_band_W0.npz (prefix AND suffix banded).
Cost per band and chunk: one suffix_call (8 of 32 layers), no prefix_masked. The write step is wrapped so the parameters stay unchanged (W0). A partial file O/E57_suffixband_W0.partial.npz is rewritten every 16 chunks.
Output O/E57_suffixband_W0.npz (nll_win, nll_chunk, nll_sufband, bands, lm, tgt, chunks) and O/E57_suffixband_W0.json (control vs 287a if --ref, seconds). Host RAM as job 289/290 (natural K,V record about 66 GiB; use >=128 GB hosts).
  python3 e57_suffixband_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --bands 128,512,1024 [--ref E57_windowless_W0.npz] --stage arms   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib,os
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
def closure_stage(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx
    CH=1024;WIN_=8*CH;assert N==1 and arms==['NAT'],'one cycle, NAT only'
    bands=[int(x) for x in (getattr(a,'bands',None) or '').split(',') if x]
    if a.masks:MK=np.load(a.masks);names=[str(x) for x in MK['names']];k0,k1=[int(x) for x in MK['chunks']];assert k1==nch,(k1,nch)
    else:MK=None;names=[];k0,k1=8,nch
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_suffixband_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    def nll_of(m,k,ids_k,Kc,Vc,keep,ctx):
        kp=jnp.asarray(keep)
        if keep is ctx['_natural']:return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],kp),np.float64)
        pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp)
        return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,kp),np.float64)
    tm=[0.0];PART={}
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        t1=time.monotonic();keep=np.asarray(keep,bool);assert keep.ndim==1 and keep.shape[0]==WIN_+CH,keep.shape;ctx=dict(ctx);ctx['_natural']=keep
        if k<k0:return dict(nll=np.zeros(CH,np.float32),lm=np.asarray(ctx['c'].loss_masks,bool),tgt=np.asarray(ctx['c'].target_tokens,np.int32))
        with rt.mesh:
            r=dict(nll=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32))
            kc=keep.copy();kc[:WIN_]=False;r['nllc']=nll_of(m,k,ids_k,Kc,Vc,np.broadcast_to(kc,(CH,WIN_+CH)).copy(),ctx).astype(np.float32)      # chunk only (2-D copy of the 1-D keep: same result as job 287a)
            r['nllb']=np.stack([(-np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],jnp.asarray(CM.band_keep(keep,bk))),np.float64)).astype(np.float32) for bk in bands]) if bands else np.zeros((0,CH),np.float32)      # suffix-only band: each query sees only the previous bk tokens in the 8 suffix layers, NATURAL prefix output (no prefix_masked)
            nm=[]
            for nmn in names:
                kp=CM.build_keep(keep,CM.up(MK[f'kw_{nmn}_{k}'],WIN_),CM.up(MK[f'qb_{nmn}_{k}'],CH));nm.append(nll_of(m,k,ids_k,Kc,Vc,kp,ctx).astype(np.float32))
            r['nllm']=np.stack(nm) if nm else np.zeros((0,CH),np.float32)
        r['lm']=np.asarray(ctx['c'].loss_masks,bool);r['tgt']=np.asarray(ctx['c'].target_tokens,np.int32);tm[0]+=time.monotonic()-t1
        PART[k]=r
        if k%16==0:
            lg(f'  suffixband chunk {k}: {tm[0]:.0f}s in masked forwards');ks_=sorted(PART)
            np.savez_compressed(out/'E57_suffixband_W0.partial.tmp.npz',chunks_done=np.asarray(ks_),nll_win=np.stack([PART[q]['nll'] for q in ks_]),nll_chunk=np.stack([PART[q]['nllc'] for q in ks_]),nll_sufband=np.stack([PART[q]['nllb'] for q in ks_],axis=1),bands=np.asarray(bands,int),lm=np.stack([PART[q]['lm'] for q in ks_]),tgt=np.stack([PART[q]['tgt'] for q in ks_]))
            os.replace(out/'E57_suffixband_W0.partial.tmp.npz',out/'E57_suffixband_W0.partial.npz')
        return r
    fn.wants_ctx=True
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_suffixband_W0',canary=None);L.rec_cache=rc;t0=time.monotonic()
    _ws=L.write_step;L.write_step=lambda m,*a_,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]      # W0: parameters returned unchanged
    res=L.run_stream('NAT',1,ids_all,nch=nch,tag='closure',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(nch)))
    D=res['diag'];ks=sorted(D);assert ks==list(range(nch)),(len(ks),nch)
    nw=np.stack([D[k]['nll'] for k in ks]);lm=np.stack([D[k]['lm'] for k in ks]);tgt=np.stack([D[k]['tgt'] for k in ks])
    nc=np.stack([D[k]['nllc'] if k>=k0 else np.zeros(CH,np.float32) for k in ks]);nmk=np.stack([np.stack([D[k]['nllm'][i] if k>=k0 else np.zeros(CH,np.float32) for k in ks]) for i in range(len(names))]) if names else np.zeros((0,nch,CH),np.float32)
    nb=np.stack([np.stack([D[k]['nllb'][i] if k>=k0 else np.zeros(CH,np.float32) for k in ks]) for i in range(len(bands))]) if bands else np.zeros((0,nch,CH),np.float32)
    np.savez_compressed(out/'E57_suffixband_W0.npz',names=np.asarray(names),chunks=np.asarray([k0,k1]),nll_win=nw,nll_chunk=nc,nll_mask=nmk,nll_sufband=nb,bands=np.asarray(bands,int),lm=lm,tgt=tgt,amt=(MK['amt'] if MK is not None else np.zeros((0,nch))))
    ce=[e['ce'] for e in res['log']];mm=[float(nw[k][lm[k]].mean()) if lm[k].any() else float('nan') for k in ks];d=np.abs(np.asarray(ce)-np.asarray(mm));ctl=dict(max_abs_diff=float(np.nanmax(d)),mean_abs_diff=float(np.nanmean(d)))
    if getattr(a,'ref',None) and os.path.exists(a.ref):
        R=np.load(a.ref);sl=slice(k0,k1);ctl['vs_287a_nll_win_max_abs']=float(np.abs(R['nll_win'][sl]-nw[sl]).max());ctl['vs_287a_nll_chunk_max_abs']=float(np.abs(R['nll_chunk'][sl]-nc[sl]).max())
    (out/'E57_suffixband_W0.json').write_text(json.dumps(dict(bank=bsrc,n_chunks=nch,names=names,sec_total=time.monotonic()-t0,sec_masked_forwards=tm[0],control=ctl,mean_nll_win=float(nw[k0:k1][lm[k0:k1]].mean()),mean_nll_chunk=float(nc[k0:k1][lm[k0:k1]].mean()),mean_nll_mask={n:float(nmk[i][k0:k1][lm[k0:k1]].mean()) for i,n in enumerate(names)},mean_nll_sufband={str(b):float(nb[i][k0:k1][lm[k0:k1]].mean()) for i,b in enumerate(bands)},amount={n:float(MK['amt'][i][k0:k1].mean()) for i,n in enumerate(names)}),default=float))
    lg(f'wrote {out/"E57_suffixband_W0.npz"}; control {ctl}')
def main():
    import argparse,e57_main_run as M
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--masks');ap.add_argument('--ref');ap.add_argument('--bands');pa,rest=ap.parse_known_args();sys.argv=[sys.argv[0]]+rest
    orig=M.main
    def stage(a,*r,**kw):a.masks=pa.masks;a.ref=pa.ref;a.bands=pa.bands;return closure_stage(a,*r,**kw)
    M.stage_arms=stage;M.main()
def selftest():
    CM.selftest()
    # mask-file round trip and keep construction on the real format (CPU)
    import tempfile
    rng=np.random.default_rng(1);kw=rng.random(8192)<.3;qb=rng.random(1024)<.2;nat=np.ones(9216,bool);nat[:100]=False
    with tempfile.TemporaryDirectory() as t:
        np.savez(pathlib.Path(t)/'m.npz',names=np.asarray(['X']),chunks=np.asarray([8,280]),**{'kw_X_9':CM.pk(kw),'qb_X_9':CM.pk(qb)});Z=np.load(pathlib.Path(t)/'m.npz')
        kp=CM.build_keep(nat,CM.up(Z['kw_X_9'],8192),CM.up(Z['qb_X_9'],1024))
    assert not kp[:,:100].any() and not kp[qb][:,:8192].any() and (kp[~qb][:,100:8192]==~kw[100:]).all() and kp[:,8192:].all()
    print('e57_suffixband_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:main()
