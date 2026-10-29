"""E57 narrow gradient-sketch dump (job 500; GPU, one news cycle, no evaluation, no pass criterion; new file; wraps the pieces of e57_gradsketch.py (job 490)). The weights follow the NAT write path of job 490 (e55_core run_stream arm NAT, native write_step untouched); at every chunk k = 0..279, at the weights BEFORE the write of the chunk, extra gradients of the same loss function (s3 grad_mask, mean CE over the selected target positions) are sketched with the count-sketch of job 490 (same method and seed 490001, 2048 buckets, one table for all layers / chunks / variants), leaf feed_forward_prime.w2.weight [8, 5632, 2560]:
  NAT      whole-chunk loss, natural window (reference; = the NAT write gradient)
  ORACLE   in-context (real chunk, window, suffix state), loss ONLY on the target positions of the strict_gold fragment tokens of this chunk (fids of e57_strict_gold_fids.json: fragment tokens j in [tok_in_chunk, tok_in_chunk + ntok) are targets, i.e. the loss rows are the positions j-1 clipped to >= 0); chunks without such a fragment: zeros + flag
  GAINQ    whole-chunk loss, GAINQ512 p=.26 query-row masking as in the GAINQ arm (e57_qband_run.SelMaskQ: for chunks >= 8 the rows of the mask file see only the previous 512 tokens, window AND chunk keys, prefix AND suffix layers; the same prefix_masked / grad_mask calls as run_stream makes for the SELC1 engine)
  GAINQSEL the same masked forward, loss ONLY on the selected (banded) query rows; chunks < 8 / without selected rows: zeros + flag
Per chunk and variant: sketch [8, 2048] of the raw (pre-clip) gradient, per-layer gradient norm, CE, number of loss tokens (the gradient is the gradient of the MEAN over these tokens; multiply by the count for the sum). Also saved: exact-vs-sketch cosine check of the first 20 NAT gradients (layer 0), canary W0 / after chunk 279, comparison of the per-chunk update norms with the saved NAT run.
  python3 e57_gradsketch_narrow.py --frags F.json --qmasks M.npz --nat-ref E57_main_NAT.arms.json <e57_main_run.py arguments with --stage arms --arms NAT>   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_gradsketch as GSM
CH=1024;NB=2048;SEED=490001;WIN=8192;K0=8;ARM='GAINQ512'
NS={'frags':None,'qmasks':None,'natref':None}
def frag_rows(frags,nch=280):
    """-> {chunk: list of target positions (sorted, unique) of the strict_gold fragment tokens}"""
    out={k:set() for k in range(nch)}
    for r in frags:
        s,n,k=r['tok_in_chunk'],r['ntok'],r['chunk'];assert s+n<=CH
        for j in range(s,s+n):
            i=max(j-1,0)
            if j>=1:out[k].add(i)       # token 0 of a chunk has no in-chunk predictor
    return {k:sorted(v) for k,v in out.items()}
def selftest():
    GSM.selftest()
    fr=[dict(chunk=0,tok_in_chunk=0,ntok=3),dict(chunk=1,tok_in_chunk=10,ntok=2)];r=frag_rows(fr,3);assert r[0]==[0,1] and r[1]==[9,10] and r[2]==[],r
    print('e57_gradsketch_narrow selftest ok')
def narrow_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx,optax
    import e57_qband_run as QR,e57_closure_masks as CM
    assert N==1 and nch==280 and arms==['NAT'],(N,nch,arms)
    sha=hashlib.sha256(np.asarray(ids_all).tobytes()).hexdigest()[:16];assert sha==GSM.IDS_SHA,('stream ids sha',sha)
    fr=json.load(open(NS['frags']))['frags'];assert len(fr)==1999;rows_of=frag_rows(fr,nch)
    masks={k:v for k,v in np.load(NS['qmasks']).items()};assert ARM in [str(x) for x in masks['names']] and int(masks[f'band_{ARM}'])==512 and not int(masks[f'all_{ARM}'])
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);f32=jnp.float32
    ip=rt.base.inner_parameters();paths=[(jax.tree_util.keystr(p),l) for p,l in jax.tree_util.tree_flatten_with_path(ip)[0]];w2=[p for p,l in paths if 'feed_forward_prime' in p and '.w2.' in p];assert len(w2)==1;W2=w2[0]
    NL,A_,B_=[l.shape for p,l in paths if p==W2][0];npar=A_*B_;lg(f'leaf {W2} [{NL},{A_},{B_}]; count-sketch {NB} dims, seed {SEED}')
    def leaf_of(g):
        for p,l in jax.tree_util.tree_flatten_with_path(g)[0]:
            if jax.tree_util.keystr(p)==W2:return l
        raise KeyError(W2)
    bk,sg=GSM.make_table(npar,SEED,NB);bkj,sgj=jnp.asarray(bk),jnp.asarray(sg)
    @jax.jit
    def sk_leaf(x):return jax.vmap(lambda r:jax.ops.segment_sum(r*sgj,bkj,num_segments=NB))(x.astype(f32).reshape(NL,npar))
    @jax.jit
    def gnorms(g_leaf):return jnp.sqrt(jnp.sum(jnp.square(g_leaf.astype(f32).reshape(NL,npar)),axis=1))
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_gradsketch_narrow',canary=None);L.rec_cache=rc
    from ttt.model.data import Batch
    SMq=QR.SelMaskQ(ARM,masks,log=lambda *x:None)
    V=['NAT','ORACLE','GAINQ','GAINQSEL'];R={v:dict(sk=[],gn=[],ce=[],nt=[],flag=[]) for v in V};extra=dict(n_banded=[],exact0=[])
    def rec(v,g,ce,nt):
        if g is None:R[v]['sk'].append(np.zeros((NL,NB),np.float32));R[v]['gn'].append(np.zeros(NL,np.float32));R[v]['ce'].append(0.0);R[v]['nt'].append(0);R[v]['flag'].append(0)
        else:
            gl=leaf_of(g);R[v]['sk'].append(np.asarray(sk_leaf(gl)));R[v]['gn'].append(np.asarray(gnorms(gl)));R[v]['ce'].append(float(ce));R[v]['nt'].append(nt);R[v]['flag'].append(1)
            return gl
    def diag(k,m,Kc,Vc,keep,ids_k,ctx):
        c=ctx['c'];st_w=ctx['st_w'];pref_w=ctx['pref_w'];keep_j=jnp.asarray(keep)
        g,ce=L.grad_mask(m,st_w,c,pref_w,keep_j);gl=rec('NAT',g,ce,int(np.asarray(c.loss_masks).sum()))
        if k<20:extra['exact0'].append(np.asarray(gl[0].astype(f32)).reshape(-1))
        del g
        rows=rows_of[k]
        if rows:
            ma=np.zeros(CH,bool);ma[rows]=True;ca=Batch(input_ids=c.input_ids,target_tokens=c.target_tokens,loss_masks=jnp.asarray(ma))
            g,ce=L.grad_mask(m,st_w,ca,pref_w,keep_j);rec('ORACLE',g,ce,int(ma.sum()));del g
        else:rec('ORACLE',None,0,0)
        kp,info=SMq.keep(k,m,Kc,Vc,ids_k,keep);nb=int(info['q_banded']);extra['n_banded'].append(nb)
        if k>=K0:
            kp_j=jnp.asarray(kp);pref_q=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp_j)
            g,ce=L.grad_mask(m,st_w,c,pref_q,kp_j);rec('GAINQ',g,ce,int(np.asarray(c.loss_masks).sum()));del g
            if nb>0:
                qb=CM.up(masks[f'qb_{ARM}_{k}'],CH);mb=np.asarray(c.loss_masks)&qb;cb=Batch(input_ids=c.input_ids,target_tokens=c.target_tokens,loss_masks=jnp.asarray(mb))
                g,ce=L.grad_mask(m,st_w,cb,pref_q,kp_j);rec('GAINQSEL',g,ce,int(mb.sum()));del g
            else:rec('GAINQSEL',None,0,0)
            del pref_q
        else:
            g,ce=L.grad_mask(m,st_w,c,pref_w,keep_j);rec('GAINQ',g,ce,int(np.asarray(c.loss_masks).sum()));del g      # chunks < 8: GAINQ = natural window (no banded rows)
            rec('GAINQSEL',None,0,0)
        if k%8==0:lg(f'  diag chunk {k}: oracle tokens {len(rows)}, banded rows {nb}')
        return None
    diag.wants_ctx=True
    t0=time.monotonic()
    res=L.run_stream('NAT',1,ids_all,nch=nch,tag='gn',log=lg,read_hook=lambda k,m:can(m),hook_chunks={nch-1},read_chunks={nch-1},chunks_meta=dummy_meta,diag_fn=diag,diag_chunks=set(range(nch)))
    fh=res['final_hash'];log=res['log'];can279=res.get('inline',{}).get(nch-1);lg(f'stream with sketches done in {time.monotonic()-t0:.0f}s; final inner hash {fh}; canary W0 {cw0}; after chunk {nch-1}: {can279}');del res;gc.collect()
    natcmp=None
    if NS['natref'] and pathlib.Path(NS['natref']).exists():
        ul={e['k']:e for e in json.load(open(NS['natref']))['update_log']};dn=[abs(e['update_norm']-ul[e['k']]['update_norm'])/ul[e['k']]['update_norm'] for e in log if e['k'] in ul]
        natcmp=dict(n=len(dn),max_rel_diff_update_norm_chunks_0_278=max(dn[:279]),rel_diff_update_norm_chunk_279=dn[279]);lg(f'vs saved NAT run: {natcmp}')
    E=np.stack(extra['exact0']).astype(np.float64);En=E/np.linalg.norm(E,axis=1,keepdims=True);Sk=np.stack([x[0] for x in R['NAT']['sk'][:20]]).astype(np.float64);Sn=Sk/np.linalg.norm(Sk,axis=1,keepdims=True);iu=np.triu_indices(20,1);ce_=En@En.T;cs_=Sn@Sn.T
    chk=dict(n_chunks=20,layer=0,variant='NAT',max_abs_cos_error=float(np.abs(ce_-cs_)[iu].max()),mean_abs_cos_error=float(np.abs(ce_-cs_)[iu].mean()),exact_cos_range=[float(ce_[iu].min()),float(ce_[iu].max())]);lg(f'sketch check: {chk}')
    arrs={}
    for v in V:
        arrs[f'{v}_sketch']=np.stack(R[v]['sk']).astype(np.float32);arrs[f'{v}_gnorm_w2']=np.stack(R[v]['gn']).astype(np.float32);arrs[f'{v}_ce']=np.asarray(R[v]['ce'],np.float32);arrs[f'{v}_ntok']=np.asarray(R[v]['nt'],np.int32);arrs[f'{v}_flag']=np.asarray(R[v]['flag'],np.int8)
    arrs['chunk']=np.arange(nch);arrs['n_banded_rows']=np.asarray(extra['n_banded'],np.int32);arrs['n_oracle_tokens']=np.asarray([len(rows_of[k]) for k in range(nch)],np.int32)
    np.savez(out/'E57_gradsketch_narrow.npz',**arrs)
    chunk_fids={}
    for r in fr:chunk_fids.setdefault(r['chunk'],[]).append(r['fid'])
    meta=dict(job=500,leaf=W2,leaf_shape=[NL,A_,B_],sketch=dict(kind='count-sketch (signed hash buckets), same method as job 490',dims=NB,seed=SEED,table='numpy default_rng(seed).integers(0,2048) buckets, then integers(0,2)*2-1 signs, row-major flatten of [5632,2560] of one layer; same table for every layer, chunk and variant'),
        variants=dict(NAT='whole-chunk loss, natural window',ORACLE='loss only on the target positions j-1 of the strict_gold fragment tokens j of the chunk (token 0 of a chunk dropped), natural window, same suffix state',GAINQ='GAINQ512 p=.26 query-row masking (e57_qband_run.SelMaskQ, mask file hash in the job log), whole-chunk loss; chunks < 8 = natural',GAINQSEL='same masked forward, loss only on the banded query rows'),
        loss_normalisation='mean CE over the loss rows (gradient of the mean); multiply by *_ntok for the sum',
        weights_path='NAT write path (native write_step, gradients recorded not applied)',canary_w0=cw0,canary_279=can279,final_inner_hash=fh,nat_ref_compare=natcmp,sketch_check=chk,ids_sha=sha,
        write_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','gnorm','sec') if k_ in e} for e in log],chunk_fids={str(k):v for k,v in chunk_fids.items()},qmasks=NS['qmasks'],frags_file=NS['frags'])
    (out/'E57_gradsketch_narrow_meta.json').write_text(json.dumps(meta,default=float));lg('wrote E57_gradsketch_narrow.npz and E57_gradsketch_narrow_meta.json')
    import os
    for f in (out/'states'/'E57_gradsketch_narrow').glob('*.npz'):os.remove(f)
def main():
    if '--selftest' in sys.argv:selftest();return
    NS['frags']=GSM._pop('--frags');NS['qmasks']=GSM._pop('--qmasks');NS['natref']=GSM._pop('--nat-ref');assert NS['frags'] and NS['qmasks']
    import e57_main_run as M
    M.stage_arms=narrow_arms;M.main()
if __name__=='__main__':main()
