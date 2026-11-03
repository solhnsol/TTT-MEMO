"""E57 decision-token window gain, QA part 1 (GPU, W0 only, forward only, NO weight update; new file; thread 'QA에서 가림이 안 되는 이유 규명', one run approved by Hans): the W0 pass of e57_pqa7_run.score_stage over the 33 paragraph-group episode streams of job 640 (e57_public_qa_stage6/e57_pqa6_group_ids.npz; the stream walked with a write step that returns the parameters unchanged) for EVERY chunk (k >= 0, the news pass starts at chunk 8): per row the natural NLL (full window, nll_win) and the NLL of the band windows b in {128, 512} (each query row sees only the previous b tokens, window AND chunk keys, prefix and suffix layers; CM.band_keep) -> <out>/e57_dwg_qa_nll.npz with g<gid>_win [nch,1024], g<gid>_band [2,nch,1024], g<gid>_lm [nch,1024] (row i of chunk k predicts the token at stream position k*1024 + i + 1). Window gain = nll_band - nll_win (positive: the window helps).
  python3 e57_dwg_qa_pass.py --src640 D --out O [--max-groups N] [--bands 128,512] | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_pqa6_run as P6
CH=1024;WIN_=8*CH
def run(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);import jax,jax.numpy as jnp,equinox as eqx,e55_run as R
    bands=[int(x) for x in (a.get('bands') or '128,512').split(',')]
    LD,s3,C=R.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True);meta,ids_z=P6.load_groups(a['src640']);G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups']
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    res={};t00=time.monotonic();fn_out=out/'e57_dwg_qa_nll.npz'
    for g in G:
        gid=g['gid'];ids=ids_z[f'g{gid}'];nch=len(ids)//CH;rc={};t0=time.monotonic()
        L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc[(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);del L0;gc.collect()
        def nll_of(m,k,ids_k,Kc,Vc,keep,ctx):
            kp=jnp.asarray(keep)
            if keep is ctx['_natural']:return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],kp),np.float64)
            pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp)
            return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,kp),np.float64)
        def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
            keep=np.asarray(keep,bool);assert keep.ndim==1 and keep.shape[0]==WIN_+CH,keep.shape;ctx=dict(ctx);ctx['_natural']=keep;lm=np.asarray(ctx['c'].loss_masks,bool)
            with rt.mesh:
                nw=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32);nb=np.stack([nll_of(m,k,ids_k,Kc,Vc,CM.band_keep(keep,b),ctx).astype(np.float32) for b in bands])
            return dict(nll=nw,nllb=nb,lm=lm)
        fn.wants_ctx=True
        L=C.Ladder55(rt,save_dir=out/'states'/'score',canary=None);L.rec_cache=rc
        _ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]
        r=L.run_stream('NAT',1,ids,nch=nch,tag=f'dwg{gid}',log=lambda *x:None,chunks_meta=[{'sents':[]} for _ in range(nch)],diag_fn=fn,diag_chunks=set(range(nch)))
        D=r['diag'];assert sorted(D)==list(range(nch))
        res[f'g{gid}_win']=np.stack([D[k]['nll'] for k in range(nch)]);res[f'g{gid}_band']=np.stack([np.stack([D[k]['nllb'][i] for k in range(nch)]) for i in range(len(bands))]);res[f'g{gid}_lm']=np.stack([D[k]['lm'] for k in range(nch)])
        del L,r,rc,Kh0,Vh0;gc.collect();lg(f'group {gid}: {nch} chunks in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s); mean window gain band {bands[0]}: {float((res[f"g{gid}_band"][0]-res[f"g{gid}_win"])[res[f"g{gid}_lm"]].mean()):+.3f}')
        if gid%8==7:np.savez_compressed(fn_out,bands=np.asarray(bands),**res)
    np.savez_compressed(fn_out,bands=np.asarray(bands),**res);lg('wrote',fn_out)
def selftest():
    print('e57_dwg_qa_pass selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k,d=None:sys.argv[sys.argv.index(k)+1] if k in sys.argv else d
    run(dict(src640=g('--src640'),out=g('--out'),max_groups=g('--max-groups'),bands=g('--bands')))
if __name__=='__main__':main()
