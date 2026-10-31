"""E57 public QA stage 2 (E57 122-Q; new file; GPU): stream S (SQuAD dev without Super_Bowl_50 / Warsaw, 2 cycles, e57_pqa2_stream.py) is written into the model by five arms and the 1,000 frozen stage-1 SQuAD items (16 candidates, items file hash d80f9987) are read OUT (neutral 8192-token SQuAD tail, bank-v1 forbid list: the tail of stage 1) with the saved states of the end of cycle 1 and of cycle 2. Raw scores only, no thresholds.
Stages:
  score     forward-only W0 pass over the whole stream (e57_t1t2v2_gainq_run.score_stage with the stream ids of S): per row natural and band-128 NLL -> E57_v2B20_score_W0.npz (file name of the news job, kept)
  mask      GAINQ128P10 (top 10% of the eligible rows per chunk >= 8 by nll_band128 - nll_win) and RANDQ128P10 (random 10%, default_rng(585) over the chunks; e57_t1t2v2_lowp_run.build_lowp, the builder of job 585) -> e57_pqa2_masks.npz
  write     NAT (plain SGD, saves the per-chunk update norms), NATQ (engine NATADAMH, Adam direction, 0.5 x NAT norm, no mask), ALLQ128 (SELC1 + SelMaskQ on all rows), GAINQ128P10, RANDQ128P10 (SELC1 + SelMaskQ): the same engines / norms / settings as jobs 570 / 585, one continuous stream of both cycles; the state is saved and the canary (generation / tail NLL, measured only) is read at the end of cycle 1 and of cycle 2
  validate  FastOUT (KV reuse, below) against the reference scorer (e55_gpu.Reader55.packed -> packing.Packed.r1): W0 on all stored stage-1 OUT rows (all 1,000 items and the first 100) and a fresh reference run on the first 100 items for W0 and for one trained state
  read      W0 + the states of a manifest, FastOUT, all 1,000 items x 16 candidates -> E57_pqa2_reads_<tag>.json (rows) and E57_pqa2_keys_<tag>.json (accuracy / rank ratio by state and answer type, paired differences with bootstrap 95% CI)
FastOUT: the OUT context is the SAME 8192-token tail for every item, and the prefix layers are not trained, so (1) K, V and the hidden states of the tail's prefix layers are computed once; (2) the prefix output of every packed candidate chunk (1024 tokens) is computed once from the cached K, V (e57_kvreuse.make_prefix_fns 'resume', R = 0: the chunk attends to the cached K, V of the tail; mask / positions as packing.Packed.pref1); (3) per state only the suffix state over the 8 tail chunks and the packed suffix pass over the chunks are computed. Same quantities as packing.Packed.r1 up to floating point.
  python3 e57_pqa2_run.py --stage score|mask|write|validate|read --stream-dir D --out O [--nat-ref ...] [--arms ...] [--manifest M --tag T --items-file F --bank B] | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_t1t2v2_run as V
import e57_t1t2v2_gainq_run as G
import e57_t1t2v2_lowp_run as LP
import e57_pqa2_stream as S
import e57_squad_w0_reads as Q
CH=1024;WIN_=8*CH;K0=8
ARMS=['NAT','NATQ','ALLQ128','GAINQ128P10','RANDQ128P10'];ENGINE={'NAT':'NAT','NATQ':'NATADAMH','ALLQ128':'SELC1','GAINQ128P10':'SELC1','RANDQ128P10':'SELC1'}
def _pop(flag,default=None):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
    return default
def use_stream(sdir):
    """route the stream-id lookup of the news code to stream S (the sentences jsonl argument is not used)"""
    ids,sid,meta=S.load(sdir);V.stream_ids=lambda path,stream:ids;return ids,sid,meta
def score_stage(a):
    ids,sid,meta=use_stream(a['stream_dir']);G.score_stage(dict(out=a['out'],sentences=None))
def mask_stage(a):
    out=pathlib.Path(a['out']);ids,sid,meta=S.load(a['stream_dir']);Z=np.load(out/'E57_v2B20_score_W0.npz');nch=Z['nll_win'].shape[0];assert nch==len(ids)//CH==sid.shape[0]
    M=LP.build_lowp(Z,sid,float(a.get('p') or 0.10));np.savez_compressed(out/'e57_pqa2_masks.npz',**M)
    for arm in ('GAINQ128P10','RANDQ128P10'):
        rows=[int(CM.up(M[f'qb_{arm}_{k}'],CH).sum()) for k in range(K0,nch)];print(f'{arm}: mean banded rows per chunk {np.mean(rows):.1f} (min {min(rows)} max {max(rows)})',flush=True)
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_swap as E,e55_run as R,e57_qband_run as QR
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    ids,sid,meta=S.load(a['stream_dir']);nch=len(ids)//CH;n1=meta['chunks_per_cycle'][0];ends={n1-1,nch-1};lg(f'stream S: {len(ids)} tokens, {nch} chunks, cycle ends at chunk {sorted(ends)}, ids_sha {meta["ids_sha"]}')
    masks={k:v for k,v in np.load(out/'e57_pqa2_masks.npz').items()};masks.update({'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)})
    rc={};L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    ntg=None
    for arm in a['arms']:
        fnA=out/f'E57_pqa2_{arm}.arms.json'
        if fnA.exists() and not a.get('rerun'):
            d=json.load(open(fnA))
            if arm=='NAT':ntg=d['norm_targets']
            print('exists, reused:',fnA);continue
        if arm!='NAT':
            if ntg is None:
                f=out/'E57_pqa2_NAT.arms.json';assert f.exists(),'NAT (norm targets) must be available first';ntg=json.load(open(f))['norm_targets']
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_pqa2_{arm}',canary=None);L.rec_cache=rc;traj=[]
        uv=None if arm in ('NAT','NATQ') else QR.SelMaskQ(arm,masks,lg)
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(ENGINE[arm],1,ids,nch=nch,tag=f'pqa2{arm}',log=lg,read_hook=hook,hook_chunks=ends,read_chunks=ends,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv);log=res['log']
        if arm=='NAT':ntg=[e['update_norm'] for e in log]
        fnA.write_text(json.dumps(dict(arm=arm,engine=ENGINE[arm],stream='S',n_chunks=nch,chunks_per_cycle=meta['chunks_per_cycle'],ids_sha=meta['ids_sha'],canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log],norm_targets=ntg if arm=='NAT' else None),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def manifest(out,cycle):
    """-> {state name: {label, cycle, file, hash}} of the cycle-end states (cycle 1 or 2) of the five arms"""
    out=pathlib.Path(out);m={}
    for arm in ARMS:
        d=json.load(open(out/f'E57_pqa2_{arm}.arms.json'));k=str(d['chunks_per_cycle'][0]-1 if cycle==1 else d['n_chunks']-1);s=d['states'][k]
        m[f'{arm}_c{cycle}']=dict(label=arm,cycle=cycle,chunk=int(k),file=s['file'],hash=s['hash'])
    return m
class FastOUT:
    """OUT scorer with KV reuse: one fixed 8192-token tail, one fixed set of packed candidate rows, many weight sets (the inner parameters change, the prefix layers do not)"""
    def __init__(self,rt32,m32,tail_ids,raw_rows,GP=2,log=print,group=640):
        import e57_kvreuse as KR,packing as PK,jax,jax.numpy as jnp,equinox as eqx
        self.KR,self.PK,self.jax,self.jnp=KR,PK,jax,jnp;self.rt=rt32;self.tail=list(tail_ids);self.TL=TL=len(tail_ids);assert TL==8192;self.GP=GP;W=int(rt32.cfg.model.sliding_window_size)
        collect,resume=KR.make_prefix_fns(rt32);t0=time.monotonic()
        # packing in groups of `group` rows (stage 1: 40 items x 16 candidates): the prefix mask limits the window by the INDEX distance, so the tail tokens a row sees depend on its offset inside the packed chunk; the same grouping as stage 1 gives the same packed chunks
        self.chunks=[];self.where=[]
        for g0 in range(0,len(raw_rows),group):
            ch_,wh_=PK.pack_rows(raw_rows[g0:g0+group],CH);o=len(self.chunks);self.chunks+=ch_;self.where+=[(c+o,a,b) for c,a,b in wh_]
        n=len(self.chunks);padn=(-n)%GP
        ids=np.stack([c[0] for c in self.chunks]+[self.chunks[-1][0]]*padn);seg=np.stack([c[1] for c in self.chunks]+[self.chunks[-1][1]]*padn);loc=np.stack([c[2] for c in self.chunks]+[self.chunks[-1][2]]*padn)
        self.ids,self.seg,self.loc=ids,seg,loc;self.nchunk=n
        with rt32.mesh,jax.default_matmul_precision('highest'):
            h,K,V_=collect(m32,jnp.asarray(self.tail,jnp.int32),jnp.arange(TL,dtype=jnp.int32),jnp.asarray(KR._attn_mask_np(np.full(TL,-2,np.int32),W)));self.H=h;jax.block_until_ready(V_)
            pref=[]
            for j in range(ids.shape[0]):
                sf=np.concatenate([np.full(TL,-2,np.int32),seg[j]]);mask=KR._attn_mask_np(sf,W)[TL:,:]
                pref.append(np.asarray(resume(m32,jnp.asarray(ids[j]),jnp.asarray(TL+loc[j]),jnp.asarray(mask),K,V_)))
            self.pref1=np.stack(pref)           # host copy: the device holds only the group being scored
            self.tailseq=rt32.seq(self.tail+[rt32.eos]*CH)
            qs=[rt32.seq(list(x)) for x in ids];self.tg=jnp.stack([q.target_tokens for q in qs]);self.mk=jnp.stack([q.loss_masks for q in qs])
            self.idsj,self.segj,self.locj=jnp.asarray(ids),jnp.asarray(seg),jnp.asarray(loc)
        log(f'  FastOUT: tail K,V,H and {n} packed chunks (pref1) prepared in {time.monotonic()-t0:.0f}s')
        from ttt.model.data import Batch
        @eqx.filter_jit
        def lp_packed(m,st,ids_,tg_,mk_,pref_,seg_,loc_):
            def one(i,t,k,p,s,l):
                PK.PACK_SUF['seg']=s;PK.PACK_SUF['loc']=l
                try:
                    r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k))
                    lg=r.logits.astype(jnp.float32);return jnp.take_along_axis(lg,t[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
                finally:PK.PACK_SUF.clear()
            return jax.vmap(one)(ids_,tg_,mk_,pref_,seg_,loc_)
        self.lp_packed=lp_packed
    def score(self,m32):
        """summed log-prob of the span of every row (same order as the raw rows)"""
        rt=self.rt;jax=self.jax;jnp=self.jnp;from core import clone_state
        with rt.mesh,jax.default_matmul_precision('highest'):
            st=clone_state(rt.state_suffix)
            for j in range(self.TL//CH):st,_=rt.chunk_logp(m32,st,rt.chunk(self.tailseq,j),self.H[j*CH:(j+1)*CH])
            out=[];GP=self.GP
            for g in range(0,self.ids.shape[0],GP):out.append(np.asarray(self.lp_packed(m32,st,self.idsj[g:g+GP],self.tg[g:g+GP],self.mk[g:g+GP],jnp.asarray(self.pref1[g:g+GP]),self.segj[g:g+GP],self.locj[g:g+GP])))
        lp=np.concatenate(out);return np.asarray([lp[c,a:b].sum() for c,a,b in self.where])
def setup_read(a,lg):
    """model, fp32 reader, items, rows, OUT tail (as stage 1)"""
    import e55_run as R,e55_gpu as RG,e57_reads as RD
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg('canary W0',can(rt.base))
    items,h=Q.load_items_file(a['items']);bank=RD.load_items(a['bank'])
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);assert len(out_tail)==8192
    tsha=hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16];lg(f'{len(items)} items, {len(raw)} candidate rows, items file sha {h[:12]}, OUT tail sha {tsha}')
    return LD,rt,rt32,rd,items,h,raw,ntok,out_tail,tsha
def rows_of(items,ntok,flat,cond):
    rows=[];o=0
    for it,nt in zip(items,ntok):nc=len(it['cands']);rows.append(Q.row_of(it,cond,flat[o:o+nc],nt));o+=nc
    assert o==len(flat);return rows
def validate_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True)
    LD,rt,rt32,rd,items,h,raw,ntok,out_tail,tsha=setup_read(a,lg);ref=json.load(open(a['stage1']))['rows'];refo=[r for r in ref if r['condition']=='OUT'];assert [r['id'] for r in refo]==[it['id'] for it in items]
    refflat=np.concatenate([np.asarray(r['cand_lp'],np.float64) for r in refo]);m32w0=rd.to_fp32(rt.base);res={}
    def compare(name,fast,reference,n_items):
        d=np.abs(fast-reference);fr=rows_of(items[:n_items],ntok[:n_items],fast,'F');rr=rows_of(items[:n_items],ntok[:n_items],reference,'R')
        res[name]=dict(n_items=n_items,n_rows=int(len(fast)),max_abs_diff=float(d.max()),mean_abs_diff=float(d.mean()),max_rel_diff=float((d/np.maximum(np.abs(reference),1e-6)).max()),
            acc_sum_flips=int(sum(x['acc_sum']!=y['acc_sum'] for x,y in zip(fr,rr))),acc_mean_flips=int(sum(x['acc_mean']!=y['acc_mean'] for x,y in zip(fr,rr))),rank_sum_changes=int(sum(x['rank_sum']!=y['rank_sum'] for x,y in zip(fr,rr))),rank_mean_changes=int(sum(x['rank_mean']!=y['rank_mean'] for x,y in zip(fr,rr))),
            acc_sum_fast=float(np.mean([x['acc_sum'] for x in fr])),acc_sum_ref=float(np.mean([x['acc_sum'] for x in rr])),acc_mean_fast=float(np.mean([x['acc_mean'] for x in fr])),acc_mean_ref=float(np.mean([x['acc_mean'] for x in rr])))
        lg(f'  {name}: n_items {n_items}, max|diff| {d.max():.3e}, mean|diff| {d.mean():.3e}, acc_sum flips {res[name]["acc_sum_flips"]}, acc_mean flips {res[name]["acc_mean_flips"]}, rank_sum changes {res[name]["rank_sum_changes"]}, rank_mean changes {res[name]["rank_mean_changes"]}')
    t0=time.monotonic();fo=FastOUT(rt32,m32w0,out_tail,raw,log=lg);tprep=time.monotonic()-t0;t0=time.monotonic();f0=fo.score(m32w0);tsc=time.monotonic()-t0;lg(f'  FastOUT W0: prep {tprep:.0f}s, score {tsc:.0f}s')
    compare('W0_all1000_vs_stage1_stored_rows',f0,refflat,len(items));n100=100;nr=sum(len(it['cands']) for it in items[:n100]);compare('W0_first100_vs_stage1_stored_rows',f0[:nr],refflat[:nr],n100)
    # fresh reference run (the reference path of stage 1: packs of 40 items) on the first 100 items, W0 and one trained state
    off=[0]
    for it in items:off.append(off[-1]+len(it['cands']))
    def ref_run(m32):
        flat=[]
        with LD.jax.default_matmul_precision('highest'):
            for g0 in range(0,n100,40):
                g1=min(g0+40,n100);P=rd.packed(raw[off[g0]:off[g1]],out_tail);flat.append(np.asarray(P.r1(m32)));del P
        return np.concatenate(flat)
    t0=time.monotonic();r0=ref_run(m32w0);tref=time.monotonic()-t0;lg(f'  reference scorer W0, first 100 items: {tref:.0f}s');compare('W0_first100_vs_fresh_reference_run',f0[:nr],r0,n100)
    res['W0_fresh_reference_vs_stage1_stored_rows_first100']=dict(max_abs_diff=float(np.abs(r0-refflat[:nr]).max()))
    st=json.load(open(a['state_manifest']))
    for name in a.get('val_states',['GAINQ128P10_A']):
        s=st[name];m=rd.load_state(s['file'],expect_hash=s['hash']);m32=rd.to_fp32(m);t0=time.monotonic();ft=fo.score(m32);t1=time.monotonic()-t0;t0=time.monotonic();rt_=ref_run(m32);t2=time.monotonic()-t0
        compare(f'{name}_first100_vs_fresh_reference_run',ft[:nr],rt_,n100);res[f'{name}_first100_vs_fresh_reference_run'].update(state_hash=s['hash'][:12],sec_fast_all1000=t1,sec_reference_first100=t2);del m,m32
        res[f'{name}_state_effect_on_first100']=dict(max_abs_state_minus_W0_ref=float(np.abs(rt_-r0).max()),mean_abs_state_minus_W0_ref=float(np.abs(rt_-r0).mean()))
    res['meta']=dict(items_sha256=h,out_tail_sha=tsha,fastout_prep_sec=tprep,fastout_score_sec=tsc,reference_first100_sec=tref)
    (out/'E57_pqa2_validate.json').write_text(json.dumps(res,indent=1,default=float));lg('wrote',out/'E57_pqa2_validate.json')
def boot_ci(x,seed=0,B=2000):
    x=np.asarray(x,np.float64);rng=np.random.default_rng(seed);ix=rng.integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return [float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def keys_paired(rows_by_label):
    """per label: accuracy / rank ratio (summed and per-token-mean score) with the bootstrap 95% CI over items, and the paired difference to W0 and to NAT"""
    out={};met=('acc_sum','acc_mean','rank_ratio_sum','rank_ratio_mean')
    def vec(rows,k):return np.asarray([float(r[k]) for r in rows])
    for lab,rows in rows_by_label.items():
        d={'n':len(rows)}
        for k in met:v=vec(rows,k);d[k]=dict(mean=float(v.mean()),ci95=boot_ci(v,1))
        out[lab]=dict(ALL=d,by_type={g:{k:float(np.mean(vec([r for r in rows if r['type']==g],k))) for k in met}|{'n':sum(r['type']==g for r in rows)} for g in sorted({r['type'] for r in rows})})
        for ref in ('W0','NAT'):
            if ref in rows_by_label and lab!=ref:
                rr=rows_by_label[ref];assert [r['id'] for r in rr]==[r['id'] for r in rows];out[lab][f'minus_{ref}']={k:dict(mean=float((vec(rows,k)-vec(rr,k)).mean()),ci95=boot_ci(vec(rows,k)-vec(rr,k),2)) for k in met}
    return out
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True);tag=a['tag']
    LD,rt,rt32,rd,items,h,raw,ntok,out_tail,tsha=setup_read(a,lg);man=json.load(open(a['manifest']));m32w0=rd.to_fp32(rt.base)
    t0=time.monotonic();fo=FastOUT(rt32,m32w0,out_tail,raw,log=lg);tim={'prep':time.monotonic()-t0};rows={};part=out/f'E57_pqa2_reads_{tag}_partial.json'
    t0=time.monotonic();f0=fo.score(m32w0);rows['W0']=rows_of(items,ntok,f0,'W0');tim['W0']=time.monotonic()-t0
    ref1=json.load(open(a.get('stage1') or 'experiments/exp07_write_ladder/e57_squad_w0/E57_squad_w0_reads.json'))['rows'];ref1=np.concatenate([np.asarray(r['cand_lp'],np.float64) for r in ref1 if r['condition']=='OUT']);d1=np.abs(f0-ref1)
    rr=rows_of(items,ntok,ref1,'R');chk=dict(max_abs_diff=float(d1.max()),mean_abs_diff=float(d1.mean()),acc_sum_flips=int(sum(x['acc_sum']!=y['acc_sum'] for x,y in zip(rows['W0'],rr))),acc_mean_flips=int(sum(x['acc_mean']!=y['acc_mean'] for x,y in zip(rows['W0'],rr))),rank_sum_changes=int(sum(x['rank_sum']!=y['rank_sum'] for x,y in zip(rows['W0'],rr))))
    lg('  FastOUT W0 vs stored stage-1 OUT rows (all 1000 items):',json.dumps(chk));lg(f'  W0 read: {tim["W0"]:.0f}s acc_mean {np.mean([r["acc_mean"] for r in rows["W0"]]):.3f}')
    for name,s in man.items():
        t0=time.monotonic();m=rd.load_state(s['file'],expect_hash=s['hash']);rows[s['label']]=rows_of(items,ntok,fo.score(rd.to_fp32(m)),s['label']);del m
        tim[name]=time.monotonic()-t0;lg(f'  {name} read: {tim[name]:.0f}s acc_mean {np.mean([r["acc_mean"] for r in rows[s["label"]]]):.3f}');part.write_text(json.dumps(dict(done=list(tim),rows=rows),default=float))
    kn=keys_paired(rows);meta=dict(items_sha256=h,out_tail_sha=tsha,scorer='FastOUT (KV reuse, packed in groups of 640 rows as stage 1)',w0_vs_stage1_stored=chk,states=man,timings=tim,primary_score='cand_mean = cand_lp / cand_ntok (E49 convention, sum as the second)')
    (out/f'E57_pqa2_reads_{tag}.json').write_text(json.dumps(dict(meta=meta,rows=rows),default=float));(out/f'E57_pqa2_keys_{tag}.json').write_text(json.dumps(dict(meta=meta,keys=kn),indent=1,default=float))
    lg('wrote',out/f'E57_pqa2_reads_{tag}.json');lg(json.dumps({l:(round(v['ALL']['acc_sum']['mean'],3),round(v['ALL']['acc_mean']['mean'],3)) for l,v in kn.items()}))
def selftest():
    sid=np.zeros((12,CH),np.int32);rng=np.random.default_rng(0);Z={'nll_win':rng.random((12,CH)),'lm':np.ones((12,CH),bool)};Z['nll_band']=Z['nll_win']+rng.random((12,CH))
    M=LP.build_lowp(Z,sid,0.10);assert CM.up(M['qb_GAINQ128P10_11'],CH).sum()==int(round(.10*(CH-1)))
    rb={'W0':[dict(id=str(i),type='x',acc_sum=i%2,acc_mean=1,rank_ratio_sum=.5,rank_ratio_mean=.5) for i in range(20)]}
    rb['NAT']=[dict(r,acc_sum=1) for r in rb['W0']];k=keys_paired(rb);assert abs(k['NAT']['minus_W0']['acc_sum']['mean']-0.5)<1e-9 and k['W0']['ALL']['n']==20
    Q.selftest();print('e57_pqa2_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=_pop('--stage');rerun='--rerun' in sys.argv
    if rerun:sys.argv.remove('--rerun')
    a=dict(stream_dir=_pop('--stream-dir'),out=_pop('--out'),arms=(_pop('--arms') or ','.join(ARMS)).split(','),p=_pop('--p'),manifest=_pop('--manifest'),tag=_pop('--tag'),items=_pop('--items-file'),bank=_pop('--bank'),stage1=_pop('--stage1'),state_manifest=_pop('--state-manifest'),rerun=rerun)
    vs=_pop('--val-states')
    if vs:a['val_states']=vs.split(',')
    if stage=='manifest':
        c=int(_pop('--cycle'));m=manifest(a['out'],c);json.dump(m,open(a['manifest'],'w'),indent=1);print({k:(v['chunk'],v['hash'][:12]) for k,v in m.items()});return
    {'score':score_stage,'mask':mask_stage,'write':write_stage,'validate':validate_stage,'read':read_stage}[stage](a)
if __name__=='__main__':main()
