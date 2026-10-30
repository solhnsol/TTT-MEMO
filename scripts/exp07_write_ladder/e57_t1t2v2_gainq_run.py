"""E57 T1/T2 v2 GAINQ128 (job 575; GPU; new file; post-hoc, exploratory): the GAINQ write arm on the SAME frozen v2 setup as job 570 (e57_t1t2v2_run.py: same streams, frozen questions, candidates, OUT reads).
Stream A (the news stream, ids_sha 715077df): the chunk-279 state of GAINQ128 p=.26 of job 292b (E57_main_GAINQ128, SELC1 engine, Adam direction, 0.5 x NAT norm of E57_main_NAT; same engine and norm targets as the ALLQ128 arm of job 570) is read as it is, nothing is rewritten.
Stream B20 (286 chunks, ids_sha 13089ac9): stage score = forward-only W0 pass (the stream walked with the write step returning the parameters unchanged, as e57_closure_run.py): per row natural nll_win and band-128 nll (both layers, CM.band_keep); stage mask = the top p=.26 of the ELIGIBLE rows (CM.rows_eligible; sentences = the fragments of the jsonl) per chunk 8..285 by nll_band128 - nll_win (CM.top_rows) -> qband mask file; stage write = SELC1 engine with e57_qband_run.SelMaskQ of arm GAINQ128 (norm targets of the NAT arm of job 570 on B20). Stage read = e57_t1t2v2_run.read_stage on the manifest.
  python3 e57_t1t2v2_gainq_run.py --stage score|mask|write|read --sentences S.jsonl --out O [--nat-ref NATB20.arms.json] [--p 0.26] [--frozen F --cands C --bank B --manifest M] |  --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_t1t2v2_run as V
CH=1024;WIN_=8*CH;K0=8;BAND=128
def frag_sids(sentences,nch):
    """sid int [nch, CH]: sentence (= fragment) index inside the chunk, -1 where no fragment covers the token"""
    sid=np.full((nch,CH),-1,np.int32);cnt={}
    for l in open(sentences):
        r=json.loads(l)
        if r.get('type')!='frag':continue
        c=r['chunk'];i=cnt.get(c,0);cnt[c]=i+1;a=r['tok_in_chunk'];sid[c,a:min(CH,a+r['ntok'])]=i
    return sid
def score_stage(args):
    out=pathlib.Path(args['out']);out.mkdir(parents=True,exist_ok=True)
    import jax,jax.numpy as jnp,equinox as eqx,e55_run as R
    LD,s3,C=R.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True)
    ids=V.stream_ids(args['sentences'],'B20');nch=len(ids)//CH;rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
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
        keep=np.asarray(keep,bool);assert keep.ndim==1 and keep.shape[0]==WIN_+CH,keep.shape;ctx=dict(ctx);ctx['_natural']=keep
        lm=np.asarray(ctx['c'].loss_masks,bool)
        if k<K0:return dict(nll=np.zeros(CH,np.float32),nllb=np.zeros(CH,np.float32),lm=lm)
        with rt.mesh:
            nw=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32);nb=nll_of(m,k,ids_k,Kc,Vc,CM.band_keep(keep,BAND),ctx).astype(np.float32)
        if k%16==0:lg(f'  score chunk {k}')
        return dict(nll=nw,nllb=nb,lm=lm)
    fn.wants_ctx=True
    L=C.Ladder55(rt,save_dir=out/'states'/'score',canary=None);L.rec_cache=rc;t0=time.monotonic()
    _ws=L.write_step;L.write_step=lambda m,*a_,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]
    res=L.run_stream('NAT',1,ids,nch=nch,tag='v2B20score',log=lg,chunks_meta=[{'sents':[]} for _ in range(nch)],diag_fn=fn,diag_chunks=set(range(nch)))
    D=res['diag'];ks=sorted(D);assert ks==list(range(nch))
    np.savez_compressed(out/'E57_v2B20_score_W0.npz',chunks=np.asarray([K0,nch]),band=np.asarray(BAND),nll_win=np.stack([D[k]['nll'] for k in ks]),nll_band=np.stack([D[k]['nllb'] for k in ks]),lm=np.stack([D[k]['lm'] for k in ks]))
    lg(f'wrote {out/"E57_v2B20_score_W0.npz"} in {time.monotonic()-t0:.0f}s')
def build_masks(Z,sid,p):
    nch=Z['nll_win'].shape[0];out={'names':np.asarray(['GAINQ128']),'band_GAINQ128':np.asarray(BAND),'all_GAINQ128':np.asarray(0),'p':np.asarray(p),'chunks':np.asarray([K0,nch])}
    for k in range(K0,nch):
        el=CM.rows_eligible(Z['lm'][k],sid[k]);n=int(round(p*el.sum()));out[f'qb_GAINQ128_{k}']=CM.pk(CM.top_rows(Z['nll_band'][k]-Z['nll_win'][k],el,n))
    return out
def mask_stage(args):
    out=pathlib.Path(args['out']);Z=np.load(out/'E57_v2B20_score_W0.npz');nch=Z['nll_win'].shape[0]
    M=build_masks(Z,frag_sids(args['sentences'],nch),float(args.get('p') or 0.26));np.savez_compressed(out/'e57_v2B20_gainq_masks.npz',**M)
    rows=[int(CM.up(M[f'qb_GAINQ128_{k}'],CH).sum()) for k in range(K0,nch)];print('GAINQ128 p=%s: mean banded rows per chunk %.1f (min %d max %d)'%(M['p'],np.mean(rows),min(rows),max(rows)),flush=True)
def write_stage(args):
    out=pathlib.Path(args['out']);import e55_swap as E,e55_run as R,e57_qband_run as QR
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    ids=V.stream_ids(args['sentences'],'B20');nch=len(ids)//CH;masks={k:v for k,v in np.load(out/'e57_v2B20_gainq_masks.npz').items()}
    ntg=list(json.load(open(args['nat_ref']))['norm_targets'][:nch]);assert len(ntg)==nch;rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);del L0;gc.collect();LD.jax.clear_caches()
    arm='GAINQ128';fnA=out/f'E57_v2B20_{arm}.arms.json';L=C.Ladder55(rt,save_dir=out/'states'/f'E57_v2B20_{arm}',canary=None);L.rec_cache=rc;traj=[]
    def hook(k,m):
        cn=can(m);traj.append(dict(chunk=int(k),canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
    t0=time.monotonic()
    res=L.run_stream('SELC1',1,ids,nch=nch,tag=f'v2B20{arm}',log=lg,read_hook=hook,hook_chunks={nch-1},read_chunks={nch-1},chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=QR.SelMaskQ(arm,masks,lg));log=res['log']
    fnA.write_text(json.dumps(dict(arm=arm,engine='SELC1',stream='B20',n_chunks=nch,ids_sha=V.STREAM_SHA['B20'],canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
        update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log]),default=float));print('wrote',fnA,flush=True)
def _pop(flag,default=None):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
    return default
def selftest():
    rng=np.random.default_rng(0);nch=10;sid=np.zeros((nch,CH),np.int32);sid[:,:5]=-1
    Z={'nll_win':rng.random((nch,CH)),'lm':np.ones((nch,CH),bool)};Z['nll_band']=Z['nll_win']+rng.random((nch,CH))
    M=build_masks(Z,sid,0.26);el=CM.rows_eligible(Z['lm'][9],sid[9]);q=CM.up(M['qb_GAINQ128_9'],CH);n=int(round(.26*el.sum()))
    assert q.sum()==n and not q[~el].any() and 'qb_GAINQ128_7' not in M and int(M['band_GAINQ128'])==128
    import e57_qband_run as QR;s=QR.SelMaskQ('GAINQ128',M,log=lambda *x:None);nat=np.ones(WIN_+CH,bool);kp,info=s.keep(9,None,None,None,None,nat);assert info['q_banded']==n and kp.shape==(CH,WIN_+CH)
    print('e57_t1t2v2_gainq_run selftest ok')
def main():
    stage=_pop('--stage');a=dict(sentences=_pop('--sentences'),out=_pop('--out'),nat_ref=_pop('--nat-ref'),p=_pop('--p'))
    if stage=='score':score_stage(a)
    elif stage=='mask':mask_stage(a)
    elif stage=='write':write_stage(a)
    else:raise SystemExit('--stage score|mask|write (the read stage is e57_t1t2v2_run.py --stage read on the manifest)')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:main()
