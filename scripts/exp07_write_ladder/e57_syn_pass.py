"""E57 synthetic-world pass (job 340ff; GPU forward only, W0 weights, NO weight update; post-hoc, exploratory). For ONE seed of the e55 fact world (cell v3_16_1, 128 chunks of 1024 tokens, window 8 chunks) and the chunks k = 8..127:
  (S1) per-token NLL under the natural window (nll_win), the chunk only (nll_chunk) and the band windows b in --bands (each query row sees only the previous b tokens, window AND chunk keys, prefix and suffix layers; e57_closure_masks.band_keep): the GAINQ rows of the write arms are the top rows by nll_band_b - nll_win   (the same technique as e57_closure_run.py of the news stream)
  (S2) the value-corrected key-mass dump of ALL 1024 query rows of the chunk (the pass of e57_valuedump.py, all heads): per row and band b the TOP-128 far keys (window key 4 <= c <= min(8191, i + 8192 - b)) of ALL (plain mass), ALLV (sum of p x ||v||), ALLVn (per-head value-normalised share), with idx / score / far (e57_valuedump_full.reduce_rows)
Output <out>/syn_s<seed>_nll.npz  (nll_win, nll_chunk [128,1024], nll_band [nb,128,1024], bands, lm, tgt; rows of chunks < 8 zero) + .json (control: mean window NLL vs the CE of the write forward of the same pass) and <out>/syn_s<seed>_dump_j<j>.npz (16-chunk shards: keys c<k>/<name>_b<b>_idx|score|far); resumable per shard.
  python3 e57_syn_pass.py --seed 62 --out O [--bands 256,512] [--chunks 8,127] [--check-only] [--selftest]      (the lab environment variables of e55_run.lab(): XLA_FLAGS=--xla_gpu_deterministic_ops=true)"""
import sys,os,json,time,gc,hashlib,argparse,pathlib,types
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024;WIN=8*CH;SHARD=16
NAMES=('ALL','ALLV','ALLVn');TOPM=128
def synth_args(seed,cell='v3_16_1'):
    return types.SimpleNamespace(cell=cell,seed=seed,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT')
def selftest():
    import e57_valuedump_full as VF
    VF.selftest();a=synth_args(62);assert a.cell=='v3_16_1'
    print('e57_syn_pass selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',help='comma list');ap.add_argument('--out');ap.add_argument('--bands',default='256,512');ap.add_argument('--chunks',default='8,127');ap.add_argument('--check-only',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_uv as UVm
    bands=[int(x) for x in a.bands.split(',')];k0,k1=[int(x) for x in a.chunks.split(',')];out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in a.seed.split(',')]
    if not a.check_only:
        LD,s3,C=R55.lab();rt=LD.Runtime();import jax,jax.numpy as jnp,equinox as eqx
        import e57_valuedump as VD,e57_valuedump_full as VF
    def one(seed):
        enc,w,tf=R55.build(synth_args(seed),seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];nch=w['n_chunks'];assert w['p']==1 and nch==128 and len(ids)==nch*CH,(w['p'],nch,len(ids));assert k1<=nch-1
        print(f'seed {seed}: stream sha {sha[:12]}, {nch} chunks, bands {bands}, chunks {k0}..{k1}',flush=True)
        if a.check_only:print('syn pass check-only OK');return
        lg=lambda *x:print(*x,flush=True);dummy_meta=w['chunks']
        L0=C.Ladder55(rt,save_dir=out/'states'/f'syn_rec_s{seed}',canary=None);t0=time.monotonic()
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
        del _h;lg(f'  natural K,V recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s')
        # ---------------- S2: value dump (before S1; shards are resumable)
        sharp=np.ones((24,32),bool);VD.AD.install(s3);chunks=list(range(k0,k1+1));shards=sorted({k//SHARD for k in chunks});t1=time.monotonic()
        for j in shards:
            fn=out/f'syn_s{seed}_dump_j{j:02d}.npz'
            if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
            res={}
            for k in [k for k in chunks if k//SHARD==j]:
                ids_k=np.asarray(ids[k*CH:(k+1)*CH])
                with rt.mesh:Kc,Vc=L0.ctx_kv(Kh0,Vh0,k)
                d=VD.value_pass(L0,rt,s3,rt.base,Kc,Vc,ids_k,k,sharp,np.arange(CH))
                for nm in NAMES:
                    for b in bands:
                        ix,sc,far=VF.reduce_rows(d[nm],np.arange(CH),b);res[f'c{k}/{nm}_b{b}_idx']=ix;res[f'c{k}/{nm}_b{b}_score']=sc;res[f'c{k}/{nm}_b{b}_far']=far
                if k%8==0:lg(f'  dump chunk {k}: {time.monotonic()-t1:.0f}s')
            tmp=out/f'.tmp_{fn.name}';np.savez(tmp,chunks=np.asarray([k for k in chunks if k//SHARD==j]),**res);tmp.rename(fn);lg('wrote shard',fn.name)
        jax.clear_caches() if hasattr(jax,'clear_caches') else None;gc.collect()
        # ---------------- S1: window / chunk-only / band NLLs
        fnn=out/f'syn_s{seed}_nll.npz'
        if fnn.exists() and not a.rerun:lg('exists, reused:',fnn);return
        rc={(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)}
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
        def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
            keep=np.asarray(keep,bool);ctx=dict(ctx);ctx['_natural']=keep
            if k<k0 or k>k1:return dict(nll=np.zeros(CH,np.float32),lm=np.asarray(ctx['c'].loss_masks,bool),tgt=np.asarray(ctx['c'].target_tokens,np.int32))
            with rt.mesh:
                r=dict(nll=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32))
                kc=keep.copy();kc[:WIN]=False;r['nllc']=nll_of(m,k,ids_k,Kc,Vc,np.broadcast_to(kc,(CH,WIN+CH)).copy(),ctx).astype(np.float32)
                r['nllb']=np.stack([nll_of(m,k,ids_k,Kc,Vc,CM.band_keep(keep,bk),ctx).astype(np.float32) for bk in bands])
            r['lm']=np.asarray(ctx['c'].loss_masks,bool);r['tgt']=np.asarray(ctx['c'].target_tokens,np.int32)
            if k%16==0:lg(f'  nll chunk {k}')
            return r
        fn.wants_ctx=True
        L=C.Ladder55(rt,save_dir=out/'states'/f'syn_nll_s{seed}',canary=None);L.rec_cache=rc;t0=time.monotonic()
        _ws=L.write_step;L.write_step=lambda m,*a_,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]      # W0: parameters returned unchanged
        res=L.run_stream('NAT',1,ids,nch=k1+1,tag='synnll',log=lg,chunks_meta=dummy_meta,diag_fn=fn,diag_chunks=set(range(k1+1)))
        D=res['diag'];ks=sorted(D);nn=k1+1;assert ks==list(range(nn))
        z=lambda k,key,sh=(CH,):D[k][key] if k0<=k<=k1 else np.zeros(sh,np.float32)
        nw=np.stack([z(k,'nll') for k in ks]);lm=np.stack([D[k]['lm'] for k in ks]);tgt=np.stack([D[k]['tgt'] for k in ks]);nc=np.stack([z(k,'nllc') for k in ks])
        nb=np.stack([np.stack([D[k]['nllb'][i] if k0<=k<=k1 else np.zeros(CH,np.float32) for k in ks]) for i in range(len(bands))])
        np.savez_compressed(fnn,nll_win=nw,nll_chunk=nc,nll_band=nb,bands=np.asarray(bands,int),lm=lm,tgt=tgt,chunks=np.asarray([k0,k1]))
        ce=[e['ce'] for e in res['log']];mm=[float(nw[k][lm[k]].mean()) if lm[k].any() else float('nan') for k in ks];d_=np.abs(np.asarray(ce)-np.asarray(mm))
        fnn.with_suffix('.json').write_text(json.dumps(dict(seed=seed,stream_sha=sha,n_chunks=nn,sec_total=time.monotonic()-t0,max_abs_diff=float(np.nanmax(d_[k0:])),mean_nll_win=float(nw[k0:][lm[k0:]].mean()),mean_nll_chunk=float(nc[k0:][lm[k0:]].mean()),mean_nll_band={str(b):float(nb[i][k0:][lm[k0:]].mean()) for i,b in enumerate(bands)}),default=float))
        lg(f'wrote {fnn}; control max abs diff {float(np.nanmax(d_[k0:])):.4g}')

    for sd in seeds:one(sd)
if __name__=='__main__':main()
