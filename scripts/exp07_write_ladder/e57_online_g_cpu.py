"""E57 online g versus W0 g on the CPU (new file; CPU only; GPM thread, Hans 11/06 '2번을 CPU로'): g128 = nll_band128 - nll_win of a chunk k computed (a) with the BASE weights (W0 g, as in e57_syn_w0pass.py / the nll128 files) and (b) with the CURRENT fast weights of a written arm = the state saved after chunk k-1 (the weights before the write of chunk k, i.e. what an online rule sees). Method: a mini-stream of the 9 chunks k-8 .. k of the real stream is run through Ladder55.run_stream with the weights fixed (write step stubbed out), so that the window (8 chunks) and the suffix state are those of chunk k computed with the given weights; the natural K,V of the mini-stream are recorded once per chunk. Caveat (stated in the report): the suffix K,V of the window chunks are recomputed with the CURRENT weights (frozen) instead of the weights that were current when each window chunk was written; chunk 0 of the mini-stream starts with the stream start of the model. Check: the base-weight g of the mini-stream is compared with the stored nll128 file (reproduction).
Per chunk and arm: Spearman correlation of online and W0 g over the eligible rows, Jaccard of the ABS2 row sets (g > 2 nats) and of the top 5% sets, share of rows above tau 2, recall (and precision) of the true deciding rows (rows predicting the first content token of the place) for both.
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu XLA_FLAGS=--xla_gpu_deterministic_ops=true python3 e57_online_g_cpu.py --seed 62 --cell v3_16_1 --chunks 32,64,96 --arms M3W1N05:synqrl,M3TA2W16N05:synqrl --out F.json | --selftest"""
import sys,json,time,gc,hashlib,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
import e57_longcell
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;WIN_=8*CH
def spearman(a,b):
    ra=np.argsort(np.argsort(a));rb=np.argsort(np.argsort(b));return float(np.corrcoef(ra,rb)[0,1])
def jaccard(a,b):
    u=(a|b).sum();return float((a&b).sum()/u) if u else float('nan')
def top_set(g,n):
    s=np.zeros(len(g),bool);s[np.argsort(-g,kind='stable')[:n]]=True;return s
def compare(g_on,g_w0,true):
    n=len(g_on);k5=max(1,int(round(.05*n)));a_on=g_on>2;a_w0=g_w0>2;t_on=top_set(g_on,k5);t_w0=top_set(g_w0,k5)
    rec=lambda s:float((s&true).sum()/true.sum()) if true.sum() else float('nan');pre=lambda s:float((s&true).sum()/max(1,s.sum()))
    return dict(n_rows=n,spearman=spearman(g_on,g_w0),jaccard_abs2=jaccard(a_on,a_w0),jaccard_top5=jaccard(t_on,t_w0),share_above2_online=float(a_on.mean()),share_above2_w0=float(a_w0.mean()),recall_abs2_online=rec(a_on),recall_abs2_w0=rec(a_w0),precision_abs2_online=pre(a_on),precision_abs2_w0=pre(a_w0),recall_top5_online=rec(t_on),recall_top5_w0=rec(t_w0),precision_top5_online=pre(t_on),precision_top5_w0=pre(t_w0),mean_g_online=float(g_on.mean()),mean_g_w0=float(g_w0.mean()))
def selftest():
    r=np.random.default_rng(0);g=r.standard_normal(1000)*2;t=np.zeros(1000,bool);t[np.argsort(-g)[:20]]=True;c=compare(g,g,t);assert abs(c['spearman']-1)<1e-9 and c['jaccard_abs2']==1.0 and c['recall_top5_online']==1.0;c2=compare(g+r.standard_normal(1000)*.5,g,t);assert 0.8<c2['spearman']<1;print('e57_online_g_cpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--chunks',default='32,64,96');ap.add_argument('--arms',default='');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import jax,jax.numpy as jnp,equinox as eqx,e55_run as R55,e55_uv as UVm,e57_syn_pass as SP,e55_gpu as RG,e57_dwg_analyze as DW,e57_pqa2_stream as S
    lg=lambda *x:print(*x,flush=True);sd=a.seed;chunks=[int(x) for x in a.chunks.split(',')];arms=[x.split(':') for x in a.arms.split(',') if x]
    enc,w,tf=R55.build(SP.synth_args(sd,a.cell),sd,arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);tk=S.tokenizer();nch=len(ids)//CH
    place=np.zeros(nch*CH+1,bool)
    for r in DW.locate_syn(tk,w,None):place[r['pos'][0]]=True
    LD,s3,C=R55.lab();_dpa=jax.nn.dot_product_attention
    def _dpa_cpu(*a_,**k_):return _dpa(*a_,**dict(k_,implementation='xla'))      # the fused cuDNN attention has no CPU lowering
    jax.nn.dot_product_attention=_dpa_cpu;rt=LD.Runtime();rd=RG.Reader55(rt,rt,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    Z=np.load(EXP/f'e57_syn_s{sd}/syn_s{sd}_nll128.npz');b=list(Z['bands']).index(128)
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    res={'seed':sd,'cell':a.cell,'method':__doc__.split('Caveat')[0].strip(),'per_chunk':{},'timing_sec':{}}
    for k in chunks:
        t0=time.monotonic();mid=ids[(k-8)*CH:(k+1)*CH];L0=C.Ladder55(rt,save_dir=EXP/'e57_online_g_tmp'/f'rec_s{sd}_{k}',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(mid)
        del _h;rc={(hashlib.sha256(np.asarray(mid,np.int64).tobytes()).hexdigest(),9):(Kh0,Vh0)};del L0;gc.collect();lg(f'chunk {k}: natural K,V of the 9-chunk window stream recorded in {time.monotonic()-t0:.0f}s');res['timing_sec'][f'{k}_record']=time.monotonic()-t0
        def g_of(m,tag):
            t1=time.monotonic();L=C.Ladder55(rt,save_dir=EXP/'e57_online_g_tmp'/f'score_s{sd}_{k}_{tag}',canary=None);L.rec_cache=rc
            def nll_of(m_,kk,ids_k,Kc,Vc,keep,ctx):
                kp=jnp.asarray(keep)
                if keep is ctx['_natural']:return -np.asarray(logp_mask(m_,ctx['st_w'],ctx['c'],ctx['pref_w'],kp),np.float64)
                pm=L.prefix_masked(m_,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(kk*CH),kp);return -np.asarray(logp_mask(m_,ctx['st_w'],ctx['c'],pm,kp),np.float64)
            def fn(kk,m_,Kc,Vc,keep,ids_k,ctx=None):
                keep=np.asarray(keep,bool);ctx=dict(ctx);ctx['_natural']=keep
                with rt.mesh:nw=nll_of(m_,kk,ids_k,Kc,Vc,keep,ctx);nb=nll_of(m_,kk,ids_k,Kc,Vc,CM.band_keep(keep,128),ctx)
                return dict(nll=nw,nllb=nb,lm=np.asarray(ctx['c'].loss_masks,bool))
            fn.wants_ctx=True;L.write_step=lambda m_,*a_,**kw:(m_.inner_parameters(),0.0,0.0,0.0)
            r=L.run_stream('NAT',1,mid,nch=9,tag=f'online_{tag}',log=lambda *x:None,weights0=m,chunks_meta=[{'sents':[]} for _ in range(9)],diag_fn=fn,diag_chunks={8});D=r['diag'][8];lg(f'chunk {k} {tag}: g computed in {time.monotonic()-t1:.0f}s');res['timing_sec'][f'{k}_{tag}']=time.monotonic()-t1
            return (D['nllb']-D['nll']).astype(np.float64)
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;true=np.asarray([place[k*CH+i+1] for i in range(CH)])[el]
        g_stored=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[el];g_w0=g_of(rt.base,'w0')[el]
        ent={'reproduction_of_stored_nll128':dict(spearman=spearman(g_w0,g_stored),max_abs_diff=float(np.abs(g_w0-g_stored).max()),mean_abs_diff=float(np.abs(g_w0-g_stored).mean())),'true_deciding_rows':int(true.sum()),'arms':{}}
        for ar,tag in arms:
            sj=json.load(open(EXP/f'e57_synq/E55_{tag}_{ar}_s{sd}.saved.json'))['reads'][str(k-1)];m=rd.load_state(str(ROOT/sj['_file']),expect_hash=sj['_hash']);g_on=g_of(m,ar)[el];del m;gc.collect()
            ent['arms'][ar]=compare(g_on,g_w0,true)
        res['per_chunk'][str(k)]=ent;pathlib.Path(a.out).write_text(json.dumps(res,indent=1));lg(f'chunk {k} done, total {time.monotonic()-t0:.0f}s',{ar:round(v['spearman'],3) for ar,v in ent['arms'].items()},'repro',round(ent['reproduction_of_stored_nll128']['spearman'],4))
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));lg('wrote',a.out)
if __name__=='__main__':main()
