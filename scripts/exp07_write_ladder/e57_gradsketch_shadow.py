"""E57 shadow-free oracle gradient sketches (job 530; GPU, one news cycle, no evaluation, no pass criterion; new file; builds on e57_gradsketch_narrow.py of job 500). Same NAT write path (native write_step untouched, gradients recorded not applied), same 2048-dim count-sketch (seed 490001), same leaf feed_forward_prime.w2.weight [8, 5632, 2560], same loss as the ORACLE variant of job 500 (mean CE over the target positions j-1 of the strict_gold fragment tokens j of the chunk), at the weights before each chunk's write. Three variants per chunk:
  A  ORACLE (natural window; recomputed as the check of job 500)
  B  ORACLE_SHADOWFREE: as A, but in the forward (prefix AND suffix layers, the SelMaskQ mechanism: per-query keep [1024, 8192 + 1024]) the query rows of each gold fragment do not attend to the key positions of the fragments listed in mask_same_fact_fids (e57_shadow_mask_table.json); query rows of a fragment = the loss rows j-1 and the rows of its own tokens, i.e. positions tok_in_chunk - 1 .. tok_in_chunk + ntok - 1 (clipped to the chunk); key positions = tok0 .. tok0 + ntok - 1 of the listed fragments (stream positions of e57_stream_sentences.jsonl)
  C  ORACLE_RANDMASK: as B with mask_random_other_event_fids
Chunks where no mask applies (no listed fragment) reuse A for B / C (identical forward). Saved per chunk: sketches [280, 8, 2048], mean CE of A / B / C, the number of loss rows, the number of newly masked (query row, key column) cells for B / C.
  python3 e57_gradsketch_shadow.py --frags F.json --shadow T.json --stream-frags S.jsonl --nat-ref E57_main_NAT.arms.json --oracle-ref E57_gradsketch_narrow.npz <e57_main_run.py arguments with --stage arms --arms NAT>   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_gradsketch as GSM
import e57_gradsketch_narrow as NAR
CH=1024;WIN=8192;NB=2048;SEED=490001
SH={'frags':None,'shadow':None,'stream':None,'natref':None,'oracleref':None}
def query_rows(s,n):
    lo=max(s-1,0);hi=min(s+n-1,CH-1);return list(range(lo,hi+1))
def mask_cells(base,k,frag_list,table_frags,stream_pos):
    """-> (keep [CH, WIN+CH] bool, number of newly masked cells). frag_list = [(s, n, [fids to mask])] for the gold fragments of chunk k"""
    kp=np.broadcast_to(base,(CH,WIN+CH)).copy();before=int(kp.sum())
    for s,n,fids in frag_list:
        rows=query_rows(s,n);cols=[]
        for f in fids:
            t0,nt=stream_pos[f];assert t0+nt<=k*CH+s+1 or t0<k*CH+s,('key fragment not earlier than the query fragment',f)
            c0=t0-k*CH+WIN;c=np.arange(c0,c0+nt);cols.append(c[(c>=0)&(c<WIN+CH)])
        if cols:
            cc=np.unique(np.concatenate(cols));kp[np.ix_(rows,cc)]=False
    return kp,before-int(kp.sum())
def selftest():
    NAR.selftest()
    base=np.ones(WIN+CH,bool);base[:100]=False;sp={7:(1000,10),8:(500,5)}
    kp,nc=mask_cells(base,2,[(3000-2048,4,[7])],None,sp) if False else (None,0)
    sp={7:(2000,10)};kp,nc=mask_cells(base,2,[(900,4,[7])],None,sp)       # chunk 2 starts at 2048; fragment at in-chunk 900, key fragment 2000..2009 -> window cols
    rows=query_rows(900,4);assert rows==list(range(899,904))
    c0=2000-2048+WIN;assert (~kp[np.ix_(rows,range(c0,c0+10))]).all() and nc==5*10 and kp[0].all()==False or True
    assert nc==50 and kp[898,c0] and not kp[899,c0] and not kp[903,c0+9] and kp[904,c0]
    print('e57_gradsketch_shadow selftest ok')
def shadow_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx,optax
    assert N==1 and nch==280 and arms==['NAT'],(N,nch,arms)
    sha=hashlib.sha256(np.asarray(ids_all).tobytes()).hexdigest()[:16];assert sha==GSM.IDS_SHA,('stream ids sha',sha)
    fr=json.load(open(SH['frags']))['frags'];assert len(fr)==1999;rows_of=NAR.frag_rows(fr,nch)
    tab={r['fid']:r for r in json.load(open(SH['shadow']))['frags']};assert set(tab)=={r['fid'] for r in fr}
    sp={}
    for l in open(SH['stream']):
        r=json.loads(l)
        if r.get('type')=='frag':sp[r['fid']]=(r['tok0'],r['ntok'])
    need={f for r in tab.values() for f in r['mask_same_fact_fids']+r['mask_random_other_event_fids']};assert need<=set(sp),sorted(need-set(sp))[:5]
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
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_gradsketch_shadow',canary=None);L.rec_cache=rc
    from ttt.model.data import Batch
    R={v:dict(sk=[],ce=[],cells=[]) for v in 'ABC'};nrows=[]
    def rec(v,gl,ce,cells):
        R[v]['sk'].append(np.asarray(sk_leaf(gl)) if gl is not None else np.zeros((NL,NB),np.float32));R[v]['ce'].append(float(ce));R[v]['cells'].append(cells)
    by_chunk={}
    for r in fr:by_chunk.setdefault(r['chunk'],[]).append(r)
    def diag(k,m,Kc,Vc,keep,ids_k,ctx):
        c=ctx['c'];st_w=ctx['st_w'];pref_w=ctx['pref_w'];keep_j=jnp.asarray(keep);base=np.asarray(keep,bool);rows=rows_of[k];nrows.append(len(rows))
        if not rows:
            for v in 'ABC':rec(v,None,0.0,0)
            return None
        ma=np.zeros(CH,bool);ma[rows]=True;ca=Batch(input_ids=c.input_ids,target_tokens=c.target_tokens,loss_masks=jnp.asarray(ma))
        g,ce=L.grad_mask(m,st_w,ca,pref_w,keep_j);gl=leaf_of(g);rec('A',gl,ce,0);skA=R['A']['sk'][-1];ceA=float(ce);del g
        for v,key in (('B','mask_same_fact_fids'),('C','mask_random_other_event_fids')):
            fl=[(r['tok0']-k*CH,r['ntok'],tab[r['fid']][key]) for r in by_chunk.get(k,[]) if tab[r['fid']][key]]
            kp,nc=mask_cells(base,k,fl,tab,sp) if fl else (None,0)
            if nc==0:R[v]['sk'].append(skA.copy());R[v]['ce'].append(ceA);R[v]['cells'].append(0);continue
            kp_j=jnp.asarray(kp);pref_q=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp_j)
            g,ce=L.grad_mask(m,st_w,ca,pref_q,kp_j);rec(v,leaf_of(g),ce,nc);del g,pref_q
        if k%8==0:lg(f'  diag chunk {k}: loss rows {len(rows)}, masked cells B {R["B"]["cells"][-1]} C {R["C"]["cells"][-1]}')
        return None
    diag.wants_ctx=True
    t0=time.monotonic()
    res=L.run_stream('NAT',1,ids_all,nch=nch,tag='gsh',log=lg,read_hook=lambda k,m:can(m),hook_chunks={nch-1},read_chunks={nch-1},chunks_meta=dummy_meta,diag_fn=diag,diag_chunks=set(range(nch)))
    fh=res['final_hash'];log=res['log'];can279=res.get('inline',{}).get(nch-1);lg(f'stream with sketches done in {time.monotonic()-t0:.0f}s; final inner hash {fh}; canary W0 {cw0}; after chunk {nch-1}: {can279}');del res;gc.collect()
    cmp500=None
    if SH['oracleref'] and pathlib.Path(SH['oracleref']).exists():
        o=np.load(SH['oracleref']);skA=np.stack(R['A']['sk']);cmp500=dict(max_abs_diff_sketch=float(np.abs(o['ORACLE_sketch']-skA).max()),max_abs_diff_ce=float(np.abs(o['ORACLE_ce']-np.asarray(R['A']['ce'],np.float32)).max()),n_chunks_with_rows=int((o['ORACLE_flag']>0).sum()));lg(f'A vs job 500 ORACLE: {cmp500}')
    natcmp=None
    if SH['natref'] and pathlib.Path(SH['natref']).exists():
        ul={e['k']:e for e in json.load(open(SH['natref']))['update_log']};dn=[abs(e['update_norm']-ul[e['k']]['update_norm'])/ul[e['k']]['update_norm'] for e in log if e['k'] in ul]
        natcmp=dict(n=len(dn),max_rel_diff_update_norm_chunks_0_278=max(dn[:279]),rel_diff_update_norm_chunk_279=dn[279]);lg(f'vs saved NAT run: {natcmp}')
    arrs={'chunk':np.arange(nch),'n_loss_rows':np.asarray(nrows,np.int32)}
    for v in 'ABC':arrs[f'{v}_sketch']=np.stack(R[v]['sk']).astype(np.float32);arrs[f'{v}_ce']=np.asarray(R[v]['ce'],np.float32);arrs[f'{v}_masked_cells']=np.asarray(R[v]['cells'],np.int64)
    np.savez(out/'E57_gradsketch_shadow.npz',**arrs)
    nsame=sum(1 for r in tab.values() if r['mask_same_fact_fids']);nrand=sum(1 for r in tab.values() if r['mask_random_other_event_fids'])
    meta=dict(job=530,shadow_table_sha256=hashlib.sha256(open(SH['shadow'],'rb').read()).hexdigest(),leaf=W2,leaf_shape=[NL,A_,B_],sketch=dict(dims=NB,seed=SEED,same_table_as='job 500'),variants=dict(A='ORACLE (recomputed)',B='ORACLE_SHADOWFREE: query rows = positions tok_in_chunk-1 .. tok_in_chunk+ntok-1 of each gold fragment cannot attend to the key positions of the fragments in mask_same_fact_fids (prefix and suffix layers, per-query keep)',C='same with mask_random_other_event_fids'),
        table_note=dict(gold_fragments=len(tab),with_same_fact_mask=nsame,with_random_mask=nrand,mean_len_same=float(np.mean([len(r['mask_same_fact_fids']) for r in tab.values()])),mean_len_random=float(np.mean([len(r['mask_random_other_event_fids']) for r in tab.values()])),total_same=int(sum(len(r['mask_same_fact_fids']) for r in tab.values())),total_random=int(sum(len(r['mask_random_other_event_fids']) for r in tab.values())),equal_counts=int(sum(1 for r in tab.values() if len(r['mask_same_fact_fids'])==len(r['mask_random_other_event_fids'])))),
        loss='mean CE over the loss rows, as job 500 ORACLE',weights_path='NAT write path',canary_w0=cw0,canary_279=can279,final_inner_hash=fh,nat_ref_compare=natcmp,job500_oracle_compare=cmp500,ids_sha=sha,
        write_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','gnorm','sec') if k_ in e} for e in log])
    (out/'E57_gradsketch_shadow_meta.json').write_text(json.dumps(meta,default=float));lg('wrote E57_gradsketch_shadow.npz and E57_gradsketch_shadow_meta.json')
    import os
    for f in (out/'states'/'E57_gradsketch_shadow').glob('*.npz'):os.remove(f)
def main():
    if '--selftest' in sys.argv:selftest();return
    SH['frags']=GSM._pop('--frags');SH['shadow']=GSM._pop('--shadow');SH['stream']=GSM._pop('--stream-frags');SH['natref']=GSM._pop('--nat-ref');SH['oracleref']=GSM._pop('--oracle-ref');assert SH['frags'] and SH['shadow'] and SH['stream']
    import e57_main_run as M
    M.stage_arms=shadow_arms;M.main()
if __name__=='__main__':main()
