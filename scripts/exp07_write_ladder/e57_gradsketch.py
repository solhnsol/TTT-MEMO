"""E57 gradient-sketch dump (job 490; GPU, one news cycle, NO state kept, NO evaluation, no pass criterion; new file). Same stream, fast-weight layers and NAT write path as job 430 (e55_core Ladder55.run_stream, arm NAT, no mask: the native write_step is called unchanged, so the weight trajectory is the NAT one; the final inner-weight hash is compared with the saved NAT run). Dumps, with ONE fixed count-sketch projection (Rademacher signs, hash buckets from a seeded numpy table, 256 dims, the same table for every layer, chunk and fragment; seed and method are written to the meta file):
  A  per stream chunk k = 0..279: sketch of the (raw, pre-clip) NAT write gradient of the inner down_proj (feed_forward_prime.w2.weight, shape [8 layers, 5632, 2560]) at the weights BEFORE the write of chunk k, per layer [280, 8, 256] (the gradient is recomputed with the same grad function as the write step, so it equals the written one up to compile-level rounding); with the gradient norms and the CE;
  B  the 1,999 strict_gold fragments of e57_stream_sentences.jsonl (job 281; fid list file with tok0 / ntok), each as ONE ISOLATED chunk: its own tokens (stream positions tok0 .. tok0+ntok-1), padded to 1024 with EOS, positions from 0, empty suffix KV, W0 weights, loss = mean CE over the in-fragment next-token targets (ntok-1 positions), same grad function and the same sketch, [1999, 8, 256], with fid, chunk, ntok, CE, gradient norms;
  C  per stream chunk: the token-mean of the input of the LAST suffix layer's down_proj (silu(w1 x) * w3 x of feed_forward_prime, 5632 dims, from an extra forward with the weights before the write) sketched to 256 dims with a count-sketch table of its own size (seed + 1, same method).
Also: the exact cosine matrix of the first 20 chunks' layer-0 gradients against the cosine matrix of their sketches (precision check of the sketch).
  python3 e57_gradsketch.py --frags F.json <the arguments of e57_main_run.py with --stage arms --arms NAT> --nat-ref E57_main_NAT.arms.json   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
CH=1024;NB=256;SEED=490001;IDS_SHA='715077dff4f602c6'
GS={'frags':None,'natref':None}
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def make_table(n,seed,nb=NB):
    """count-sketch table: bucket in [0, nb) and sign +-1 for every coordinate (numpy PCG64, seed); y_b = sum_{i: bucket(i)=b} sign(i) x_i"""
    rng=np.random.default_rng(seed);buckets=rng.integers(0,nb,size=n,dtype=np.int32);signs=(rng.integers(0,2,size=n,dtype=np.int8).astype(np.float32)*2-1);return buckets,signs
def sketch_np(x,buckets,signs,nb=NB):
    out=np.zeros(x.shape[:-1]+(nb,),np.float64)
    for idx in np.ndindex(*x.shape[:-1]):np.add.at(out[idx],buckets,x[idx].astype(np.float64)*signs)
    return out.astype(np.float32)
def selftest():
    b,s=make_table(1000,SEED);assert b.min()>=0 and b.max()<NB and set(np.unique(s))=={-1.0,1.0}
    b2,s2=make_table(1000,SEED);assert (b==b2).all() and (s==s2).all()
    rng=np.random.default_rng(1);x=rng.standard_normal((3,1000)).astype(np.float32);y=rng.standard_normal((3,1000)).astype(np.float32)
    sx,sy=sketch_np(x,b,s),sketch_np(y,b,s);assert sx.shape==(3,NB)
    import jax,jax.numpy as jnp
    f=lambda a,bk,sg:jax.vmap(lambda r:jax.ops.segment_sum(r*sg,bk,num_segments=NB))(a)
    assert np.allclose(np.asarray(f(jnp.asarray(x),jnp.asarray(b),jnp.asarray(s))),sx,atol=1e-4)
    big=rng.standard_normal((2,200000)).astype(np.float32);bb,ss=make_table(200000,SEED);z=sketch_np(big,bb,ss);ex=big[0]@big[1]/np.linalg.norm(big[0])/np.linalg.norm(big[1]);sk=z[0]@z[1]/np.linalg.norm(z[0])/np.linalg.norm(z[1]);assert abs(ex-sk)<0.2
    print('e57_gradsketch selftest ok')
def gs_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx,optax
    assert N==1 and nch==280 and arms==['NAT'],(N,nch,arms)
    sha=hashlib.sha256(np.asarray(ids_all).tobytes()).hexdigest()[:16];assert sha==IDS_SHA,('stream ids sha',sha);assert len(ids_all)==280*CH
    fr=json.load(open(GS['frags']))['frags'];assert len(fr)==1999 and len({r['fid'] for r in fr})==1999
    for r in fr:assert 0<=r['tok0'] and r['tok_in_chunk']+r['ntok']<=CH and r['tok0']//CH==r['chunk'] and r['ntok']>=2
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);f32=jnp.float32
    ip=rt.base.inner_parameters();paths=[(jax.tree_util.keystr(p),l) for p,l in jax.tree_util.tree_flatten_with_path(ip)[0]];w2=[p for p,l in paths if 'feed_forward_prime' in p and '.w2.' in p];assert len(w2)==1,w2
    W2=w2[0];shape=[l.shape for p,l in paths if p==W2][0];assert len(shape)==3,shape;NL,A_,B_=shape;npar=A_*B_;lg(f'inner down_proj leaf {W2} {shape}; {npar} coordinates per layer; sketch {NB} dims, count-sketch seed {SEED}')
    def leaf_of(g):
        for p,l in jax.tree_util.tree_flatten_with_path(g)[0]:
            if jax.tree_util.keystr(p)==W2:return l
        raise KeyError(W2)
    bk,sg=make_table(npar,SEED);bkj,sgj=jnp.asarray(bk),jnp.asarray(sg);FF=B_ if False else A_
    bkc,sgc=make_table(A_,SEED+1);bkcj,sgcj=jnp.asarray(bkc),jnp.asarray(sgc)      # the activation has intermediate_size = A_ dims
    @jax.jit
    def sk_leaf(x):return jax.vmap(lambda r:jax.ops.segment_sum(r*sgj,bkj,num_segments=NB))(x.astype(f32).reshape(NL,npar))
    @jax.jit
    def sk_act(v):return jax.ops.segment_sum(v.astype(f32)*sgcj,bkcj,num_segments=NB)
    @jax.jit
    def gnorms(g_leaf):return jnp.sqrt(jnp.sum(jnp.square(g_leaf.astype(f32).reshape(NL,npar)),axis=1))
    # recorded natural K,V (as job 430)
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    # C hook: the input of the down_proj of every feed_forward_prime call (token mean), identified by the layer sums of the w2 weights
    from ttt.model.transformer import SwiGLUMLP
    cap={'on':False,'rows':[]};orig_call=SwiGLUMLP.__call__
    def _cb(tag,v):cap['rows'].append((float(tag),np.asarray(v)))
    def call(self,x):
        if not cap['on']:return orig_call(self,x)
        x2=jax.nn.silu(self.w1(x))*self.w3(x);jax.debug.callback(_cb,jnp.sum(self.w2.weight.astype(f32)),x2.astype(f32).reshape(-1,x2.shape[-1]).mean(axis=0),ordered=True)
        return self.dropout(self.w2(x2))
    SwiGLUMLP.__call__=call
    w2sum=np.asarray(jnp.sum(leaf_of(ip).astype(f32),axis=(1,2)));lg(f'down_proj layer weight sums {w2sum.tolist()}')
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_gradsketch',canary=None);L.rec_cache=rc;S3SUF=s3.S3SUF
    from ttt.model.data import Batch
    @eqx.filter_jit
    def fwd_c(m,st,c,pref,kmask):
        S3SUF['kmask']=kmask
        try:return m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c).logits.astype(jnp.float32).sum()
        finally:S3SUF.clear()
    A=dict(sk=[],gn=[],gn_w2=[],ce=[],csk=[],cnorm=[]);exact0=[];orig_ws=L.write_step
    def ws(m,st_all,st_suf,c,pref_,kmask):
        res=orig_ws(m,st_all,st_suf,c,pref_,kmask)           # the unchanged native NAT write step (new weights, clipped gradient norm, clip coef, CE)
        g,ce=L.grad_mask(m,st_suf,c,pref_,kmask);gl=leaf_of(g);k=len(A['sk'])
        A['sk'].append(np.asarray(sk_leaf(gl)));A['gn_w2'].append(np.asarray(gnorms(gl)));A['gn'].append(float(optax.global_norm(g)));A['ce'].append(float(ce))
        if k<20:exact0.append(np.asarray(gl[0].astype(f32)).reshape(-1))
        cap['rows']=[];cap['on']=True
        try:fwd_c(m,st_suf,c,pref_,kmask);jax.effects_barrier()
        finally:cap['on']=False
        rows=cap['rows'];assert len(rows)==2*NL,('callbacks per forward',len(rows));tags=np.array([t for t,_ in rows]);hit=list(range(0,2*NL,2))      # feed_forward_prime is called first in every suffix block, the regular feed_forward second (checked on CPU: the prime calls carry the w2 weight sums of layers 0..7); order of the ordered callbacks = layer order
        lay=[int(np.argmin(np.abs(w2sum-tags[i]))) for i in hit]
        if cap.get('n',0)==0:lg(f'  C hook: {len(rows)} callbacks; prime-call tags {tags[hit].round(3).tolist()} vs w2 sums {w2sum.round(3).tolist()}; nearest layers {lay}')
        cap['n']=cap.get('n',0)+1
        last=rows[hit[-1]][1];A['csk'].append(np.asarray(sk_act(jnp.asarray(last))));A['cnorm'].append(float(np.linalg.norm(last)))
        return res
    L.write_step=ws
    t0=time.monotonic()
    res=L.run_stream('NAT',1,ids_all,nch=nch,tag='gs',log=lg,read_hook=lambda k,m:can(m),hook_chunks={nch-1},read_chunks={nch-1},chunks_meta=dummy_meta)
    SwiGLUMLP.__call__=orig_call
    fh=res['final_hash'];lg(f'NAT stream with sketches done in {time.monotonic()-t0:.0f}s; final inner hash {fh}; state hash at chunk {nch-1}: {res["reads"][nch-1]["_hash"]}')
    ref=None
    if GS['natref'] and pathlib.Path(GS['natref']).exists():
        ref=json.load(open(GS['natref'])).get('states',{}).get(str(nch-1),{}).get('hash')
    same=(ref==res['reads'][nch-1]['_hash']) if ref else None;lg(f'saved NAT state hash {ref}; identical to this run: {same}')
    log=res['log'];can279=res.get('inline',{}).get(nch-1);lg(f'canary W0 {cw0}; after chunk {nch-1}: {can279}');del res;gc.collect()
    # precision check of the sketch: exact layer-0 gradient cosines of the first 20 chunks vs sketch cosines
    E=np.stack(exact0).astype(np.float64);En=E/np.linalg.norm(E,axis=1,keepdims=True);Sk=np.stack([x[0] for x in A['sk'][:20]]).astype(np.float64);Sn=Sk/np.linalg.norm(Sk,axis=1,keepdims=True)
    cos_ex=En@En.T;cos_sk=Sn@Sn.T;iu=np.triu_indices(20,1);chk=dict(n_chunks=20,layer=0,max_abs_cos_error=float(np.abs(cos_ex-cos_sk)[iu].max()),mean_abs_cos_error=float(np.abs(cos_ex-cos_sk)[iu].mean()),exact_cos_range=[float(cos_ex[iu].min()),float(cos_ex[iu].max())],norm_ratio_sketch_over_exact=[float(x) for x in (np.linalg.norm(Sk,axis=1)/np.linalg.norm(E,axis=1))[:5]]);lg(f'sketch check: {chk}')
    del E,exact0;gc.collect()
    # B: isolated W0 fragments
    m0=rt.base;eos=rt.eos;B=dict(sk=[],gn=[],gn_w2=[],ce=[]);t0=time.monotonic()
    with rt.mesh:
        for i,r in enumerate(fr):
            ids=[int(x) for x in ids_all[r['tok0']:r['tok0']+r['ntok']]];pad=ids+[eos]*(CH-len(ids));mask=np.zeros(CH,bool);mask[:len(ids)-1]=True
            seq=rt.seq(pad,mask=mask);pref=rt.prefix(m0,seq);g,ce=L.grad_nat(m0,s3.clone_state(rt.state_suffix),seq,pref);gl=leaf_of(g)
            B['sk'].append(np.asarray(sk_leaf(gl)));B['gn_w2'].append(np.asarray(gnorms(gl)));B['gn'].append(float(optax.global_norm(g)));B['ce'].append(float(ce))
            if i%200==0:lg(f'  fragment {i+1}/1999 fid {r["fid"]} ntok {r["ntok"]}: {time.monotonic()-t0:.0f}s ce {B["ce"][-1]:.3f}')
    lg(f'fragments done in {time.monotonic()-t0:.0f}s')
    np.savez(out/'E57_gradsketch.npz',A_sketch=np.stack(A['sk']).astype(np.float32),A_gnorm=np.asarray(A['gn'],np.float32),A_gnorm_w2=np.stack(A['gn_w2']).astype(np.float32),A_ce=np.asarray(A['ce'],np.float32),A_chunk=np.arange(nch),
        C_sketch=np.stack(A['csk']).astype(np.float32),C_norm=np.asarray(A['cnorm'],np.float32),
        B_sketch=np.stack(B['sk']).astype(np.float32),B_gnorm=np.asarray(B['gn'],np.float32),B_gnorm_w2=np.stack(B['gn_w2']).astype(np.float32),B_ce=np.asarray(B['ce'],np.float32),
        B_fid=np.asarray([r['fid'] for r in fr]),B_chunk=np.asarray([r['chunk'] for r in fr]),B_ntok=np.asarray([r['ntok'] for r in fr]),B_tok0=np.asarray([r['tok0'] for r in fr]))
    chunk_fids={}
    for r in fr:chunk_fids.setdefault(r['chunk'],[]).append(r['fid'])
    meta=dict(job=490,leaf=W2,leaf_shape=list(shape),sketch=dict(kind='count-sketch (signed hash buckets)',dims=NB,table='numpy default_rng(seed).integers(0,256) buckets, then integers(0,2)*2-1 signs, one coordinate order = row-major flatten of [5632,2560] of one layer',seed_gradient=SEED,seed_activation=SEED+1,same_table_for_all_layers_chunks_fragments=True,activation_dims=int(A_)),
        A='raw (pre-clip) NAT write gradient at the weights before the write of chunk k, recomputed by the same grad function as the write step',B='isolated chunk: fragment tokens + EOS pad, positions from 0, empty suffix KV, W0, loss mean CE over ntok-1 in-fragment targets, grad_nat',C='token mean of the input of the last suffix layer feed_forward_prime down_proj (silu(w1 x)*w3 x), weights before the write',
        canary_w0=cw0,canary_279=can279,ids_sha=sha,frags_file=GS['frags'],final_inner_hash=fh,nat_state_hash_279=ref,identical_to_saved_NAT=same,sketch_check=chk,write_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','gnorm','sec') if k_ in e} for e in log],chunk_fids={str(k):v for k,v in chunk_fids.items()},fragments=[{k_:r[k_] for k_ in ('fid','chunk','ci','tok0','tok_in_chunk','ntok','event','art','art_sent')} for r in fr])
    (out/'E57_gradsketch_meta.json').write_text(json.dumps(meta,default=float));lg('wrote E57_gradsketch.npz and E57_gradsketch_meta.json')
    import os
    for f in (out/'states'/'E57_gradsketch').glob('*.npz'):os.remove(f)
def main():
    if '--selftest' in sys.argv:selftest();return
    GS['frags']=_pop('--frags');GS['natref']=_pop('--nat-ref');assert GS['frags']
    import e57_main_run as M
    M.stage_arms=gs_arms;M.main()
if __name__=='__main__':main()
