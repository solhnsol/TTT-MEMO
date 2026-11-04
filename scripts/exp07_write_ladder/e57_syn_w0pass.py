"""E57 synthetic world, W0 window pass for any cell / seed (new file; GPU forward only, NO weight update; thread 'ALLQ보다 나은 가림 방법 찾기', job 760 step 1): the same pass as e57_syn_pass.py S1 (without the value dump) and e57_dwg_qa_pass.py for the stream of ONE seed, e.g. seed 61 of the cell v3_64_4 (512 chunks): per row of the chunks k = 8 .. n_chunks-1 the natural NLL (nll_win) and the NLL of the band window b (each query row sees only the previous b tokens, window AND chunk keys, prefix and suffix layers; e57_closure_masks.band_keep) -> <out>/syn_s<seed>_nll.npz (nll_win [n,1024], nll_band [nb,n,1024], bands, lm, tgt, chunks; rows of chunks < 8 zero) + .json (stream sha, mean window gain). The write step is replaced by the identity (like e57_pqa7_run.score_stage). e57_syn_mmasks.py reads the file.
  python3 e57_syn_w0pass.py --seed 61 --cell v3_64_4 --out O [--bands 256] [--smoke N] | --selftest"""
import sys,json,time,gc,hashlib,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
import e57_longcell      # registers the cell v3_16_1_L1024 (long synthetic stream)
CH=1024;WIN_=8*CH;K0=8
def run(a):
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);import jax,jax.numpy as jnp,equinox as eqx,e55_run as R55,e57_syn_pass as SP
    bands=[int(x) for x in a.bands.split(',')];lg=lambda *x:print(*x,flush=True)
    enc,w,tf=R55.build(SP.synth_args(a.seed,a.cell),a.seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];nch=a.smoke or w['n_chunks'];ids=np.asarray(ids)[:nch*CH];assert len(ids)==nch*CH
    lg(f'seed {a.seed} cell {a.cell}: stream sha {sha[:12]}, {nch} chunks, bands {bands}')
    LD,s3,C=R55.lab();rt=LD.Runtime()
    if a.segmented:import e57_pqa2_run as P2;P2.patch_segmented(log=lg)         # 1024 chunks in one call: the K,V record runs at ~300 s per layer; segmented record / prefix as in the 6-cycle news run
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    t0=time.monotonic();L0=C.Ladder55(rt,save_dir=out/'states'/f'rec_s{a.seed}',canary=None)
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
    del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s')
    def nll_of(m,k,ids_k,Kc,Vc,keep,ctx):
        kp=jnp.asarray(keep)
        if keep is ctx['_natural']:return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],kp),np.float64)
        pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp)
        return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,kp),np.float64)
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        keep=np.asarray(keep,bool);assert keep.ndim==1 and keep.shape[0]==WIN_+CH,keep.shape;ctx=dict(ctx);ctx['_natural']=keep;lm=np.asarray(ctx['c'].loss_masks,bool)
        with rt.mesh:
            nw=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32);nb=np.stack([nll_of(m,k,ids_k,Kc,Vc,CM.band_keep(keep,b),ctx).astype(np.float32) for b in bands])
        if k%32==0:lg(f'  chunk {k}: {time.monotonic()-t0:.0f}s, mean window gain {float((nb[0]-nw)[lm].mean()):+.3f}')
        return dict(nll=nw,nllb=nb,lm=lm)
    fn.wants_ctx=True
    L=C.Ladder55(rt,save_dir=out/'states'/f'score_s{a.seed}',canary=None);L.rec_cache=rc
    _ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]
    r=L.run_stream('NAT',1,ids,nch=nch,tag=f'w0pass_s{a.seed}',log=lambda *x:None,chunks_meta=[{'sents':[]} for _ in range(nch)],diag_fn=fn,diag_chunks=set(range(K0,nch)))
    D=r['diag'];assert sorted(D)==list(range(K0,nch))
    nwin=np.zeros((nch,CH),np.float32);nband=np.zeros((len(bands),nch,CH),np.float32);lm=np.zeros((nch,CH),bool)
    for k in range(K0,nch):nwin[k]=D[k]['nll'];nband[:,k]=D[k]['nllb'];lm[k]=D[k]['lm']
    tgt=np.zeros((nch,CH),np.int32)
    for k in range(nch):
        seg=np.asarray(ids[k*CH+1:(k+1)*CH+1]);tgt[k,:len(seg)]=seg
    fn_=out/f'syn_s{a.seed}_nll.npz';np.savez_compressed(fn_,nll_win=nwin,nll_band=nband,bands=np.asarray(bands),lm=lm,tgt=tgt,chunks=np.asarray([K0,nch]))
    g=(nband[0]-nwin)[lm];fn_.with_suffix('.json').write_text(json.dumps(dict(seed=a.seed,cell=a.cell,stream_sha=sha[:16],n_chunks=nch,bands=bands,mean_window_gain_band0=float(g.mean()),mean_nll_win=float(nwin[lm].mean()),sec=time.monotonic()-t0)));lg('wrote',fn_)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--out');ap.add_argument('--bands',default='256');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--segmented',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:print('e57_syn_w0pass selftest ok');return
    run(a)
if __name__=='__main__':main()
