"""E57 oracle / half-oracle arms and distance x visibility cells (GPU; new file; core-claims thread 11/10; the registered e57_natmask_run.py is imported, not edited). Same write run, reads, canary and logs as e57_natmask_run.main (clipped SGD, engine SGDSEL, K0 = 8), with
  NATORACLE : e57_oracle_masks.OracleUV (a sentence row does not see the earlier sentences of its own fact in the window)
  NATHALF   : e57_oracle_masks.HalfOracleUV (every row sees the previous 256 tokens; sentence rows also see the earlier sentences of their own fact)
  plus the arms of e57_natmask_run (NATS, NATMASK, NATMASKC, DIR...). Cells: the registered ones (v3_16_1, v3_64_4, ...) and runtime cells 'v3_<k>_<p>_a<n_app>' (e.g. v3_32_4_a16: 32 facts, period p = 4 = the gap in chunks between consecutive exposures of a fact, 16 exposures, 64 chunks; background books as v3_64_4), registered in gen_facts.CELLS at run time (gen_facts.py is not edited); read points: the registered cell's read_apps, for a runtime cell only the last exposure n_app (out-of-window read + in-window read of group 0), saved state at the same point.
  python3 e57_oracle_run.py --seed S --cell C --out O --arms NATS,NATMASK,NATORACLE,NATHALF [--mask-arm ALLQ256] [--tag T] [--smoke N] [--rerun] | --selftest"""
import sys,os,json,time,gc,re,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_natmask_run as NM
import e57_qband_run as QR
import e57_oracle_masks as OM
from e57_natmask_run import ENGINE,K0,KeyMatchedQ,arm_mode,swap_step,with_base,make_nat_loss_diag,register_engine
def register_cell(cell):
    """runtime cell 'v3_<k>_<p>_a<n_app>' -> gen_facts.CELLS entry (k facts, n = 1, period p, bank 300, n_app exposures, read only at the last exposure)"""
    import gen_facts as G
    m=re.fullmatch(r'v3_(\d+)_(\d+)_a(\d+)',cell)
    if m and cell not in G.CELLS:
        k,p,na=map(int,m.groups());assert k>=p,'facts k >= groups p';G.CELLS[cell]=dict(k=k,n=1,p=p,bank='300',n_app=na,read_apps=(na,));G.CELL_BOOKS[cell]=G.CELL_BOOKS['v3_64_4']
    return G.CELLS[cell]
def parse_arm2(name,mask_arm,rand_arm='RANDQ256P50'):
    if name=='NATORACLE':return 'ORACLE',1.0
    if name=='NATHALF':return 'HALF',1.0
    if name=='NATORACLE_RAND':return 'ORACLE_RAND',1.0
    if name=='NATPLACEBO':return 'PLACEBO',1.0
    return NM.parse_arm(name,mask_arm,rand_arm)
def clip_step(orig,C):
    """clip-threshold sensitivity (Council 11/10): arm suffix _clip<C> (e.g. NATS_clip10): the clipped step of the original rule (coef = min(1, 1.0/gn)) is rescaled to the threshold C: step x min(1, C/gn) / coef; C large = effectively no clip. gn = pre-clip gradient norm returned by the step."""
    import jax,jax.numpy as jnp
    def ws(m,*a,**k):
        newp,gn,coef,ce=orig(m,*a,**k);f=float(jnp.minimum(1.0,C/jnp.maximum(gn,1e-30)))/float(coef);f32=jnp.float32;old=m.inner_parameters()
        newp=jax.tree.map(lambda n,o:(o.astype(f32)+(n.astype(f32)-o.astype(f32))*f).astype(n.dtype),newp,old);return newp,gn,coef,ce
    return ws
def strip_k0(name,default=0):
    """arm suffix _K08 = the mask starts at chunk 8 (the earlier convention) instead of the default chunk K0 (Hans 11/10: mask from the very start, K0 = 0) -> (arm name without the suffix, K0)"""
    return (name[:-4],8) if name.endswith('_K08') else (name,default)
def split_clip(name):
    m=re.search(r'_clip(\d+(?:\.\d+)?)$',name);return (name[:m.start()],float(m.group(1))) if m else (name,None)
def perturb_inner(m,rt,LD,eps,seed):
    """numerical-noise control (Council 11/10): the fast weights of the model are multiplied by (1 + eps x N(0,1)) (a relative perturbation of the order of 1e-7), every leaf with its own key; returns the model with the perturbed inner parameters (weights0 of run_stream)"""
    jax,jnp=LD.jax,LD.jnp;inner=m.inner_parameters();leaves,td=jax.tree_util.tree_flatten(inner);keys=jax.random.split(jax.random.PRNGKey(seed),len(leaves))
    new=[(x*(1+eps*jax.random.normal(k,x.shape,jnp.float32).astype(x.dtype))).astype(x.dtype) for x,k in zip(leaves,keys)]
    return new,td
def make_token_diag(L,LD,s3,rt,holder,chunks,log=print,grad_every=8):
    """Council 11/10 logging: before the write of every diag chunk, per-token NLL (nat) of (a) the FULL natural context and (b) the update path (the arm's mask) split into fact-sentence tokens of the FIRST exposure of their fact (app 0), of LATER exposures (app >= 1) and filler tokens; token t of a sentence is predicted at position t-1. Every `grad_every`-th chunk also the share of the fact tokens in the clipped-gradient direction: ||g_fact|| x (N_fact / N_total) / ||g_total||, g_fact = gradient of the mean loss over the fact tokens only, same state, same keep as the update (2 extra backward passes)."""
    jax,jnp,eqx=LD.jax,LD.jnp,LD.eqx
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    def gnorm(g):return float(jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32)**2) for x in jax.tree.leaves(g))))
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        base=np.asarray(holder['base'],bool);assert base.ndim==1;keep=np.asarray(keep,bool)
        with rt.mesh:
            pref_n=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(base))
            lp=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pref_n,jnp.asarray(base)),np.float64);lpu=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],jnp.asarray(keep)),np.float64)
        lm=np.asarray(ctx['c'].loss_masks,bool);f0=np.zeros(CH_,bool);f1=np.zeros(CH_,bool)
        for x in chunks[k]['sents']:
            seg=slice(max(x['start']-1,0),x['end']-1);(f0 if x['app']==0 else f1)[seg]=True
        f0&=lm;f1&=lm;fl=lm&~(f0|f1);r=dict(app=(int(chunks[k]['sents'][0]['app']) if chunks[k]['sents'] else None),n=int(lm.sum()),n_fact_first=int(f0.sum()),n_fact_later=int(f1.sum()),n_filler=int(fl.sum()))
        for nm,v in (('nat',lp),('upd',lpu)):
            r[f'{nm}_nll_all']=float((-v[lm]).mean())
            for tag,mk in (('fact_first',f0),('fact_later',f1),('filler',fl)):r[f'{nm}_nll_{tag}']=float((-v[mk]).mean()) if mk.any() else None
        if grad_every and k%grad_every==0 and (f0|f1).any():
            fm=jnp.asarray(f0|f1);c2=eqx.tree_at(lambda z:z.loss_masks,ctx['c'],fm)
            with rt.mesh:
                gt,_=L.grad_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],jnp.asarray(keep));gf,_=L.grad_mask(m,ctx['st_w'],c2,ctx['pref_w'],jnp.asarray(keep));ngt=gnorm(gt);ngf=gnorm(gf)
            r['grad_total_norm']=ngt;r['grad_factmean_norm']=ngf;r['fact_grad_share']=ngf*(int((f0|f1).sum())/int(lm.sum()))/ngt if ngt>0 else None
        log(f'  tokdiag chunk {k}: nat nll fact(first/later)/filler {r["nat_nll_fact_first"]}/{r["nat_nll_fact_later"]}/{r["nat_nll_filler"]}, upd {r["upd_nll_fact_first"]}/{r["upd_nll_fact_later"]}/{r["upd_nll_filler"]}');return r
    fn.wants_ctx=True;return fn
CH_=1024
def selftest():
    NM.K0=8;NM.selftest();OM.selftest()
    assert parse_arm2('NATORACLE_RAND','ALLQ256')==('ORACLE_RAND',1.0) and parse_arm2('NATORACLE','ALLQ256')==('ORACLE',1.0) and parse_arm2('NATHALF','ALLQ256')==('HALF',1.0) and parse_arm2('NATMASK','ALLQ256')==('ALLQ256',1.0)
    import gen_facts as G
    c=register_cell('v3_32_16_a16');assert c==dict(k=32,n=1,p=16,bank='300',n_app=16,read_apps=(16,)) and G.CELLS['v3_32_16_a16']['p']==16
    try:register_cell('v3_8_16_a16');raise SystemExit('k<p accepted')
    except AssertionError:pass
    assert strip_k0('NATMASK_K08',0)==('NATMASK',8) and strip_k0('NATMASK',0)==('NATMASK',0)
    import e57_closure_masks as CM
    base0=np.zeros(8192+1024,bool);base0[8192:]=True                           # chunk 0: no window keys, only the chunk keys exist
    d_=np.arange(1024)[:,None]+8192-np.arange(8192+1024)[None,:];nat0=base0[None,:]&(d_>=0);band0=CM.band_keep(base0,256)
    assert np.array_equal(band0[:256],nat0[:256]),'unit test: for rows with <= 256 tokens of context the band mask must equal no mask'     # per-TOKEN band: row i sees the previous 256 tokens
    assert (band0[256:]!=nat0[256:]).any() and np.array_equal(CM.band_keep(base0,10**6),nat0),'band is per token; band = inf is the natural mask'
    assert split_clip('NATS_clip10')==('NATS',10.0) and split_clip('NATMASK')==('NATMASK',None) and parse_arm2('NATPLACEBO','ALLQ256')==('PLACEBO',1.0)
    print('e57_oracle_run selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--save-apps',default='128');ap.add_argument('--out');ap.add_argument('--masks');ap.add_argument('--mask-arm',default='ALLQ256');ap.add_argument('--rand-arm',default='RANDQ256P50');ap.add_argument('--arms');ap.add_argument('--tag',default='synng');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--nat-loss-every',type=int,default=1);ap.add_argument('--k0',type=int,default=0);ap.add_argument('--perturb',type=float,default=0.0);ap.add_argument('--perturb-seed',type=int,default=0);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    cellinfo=register_cell(a.cell);global K0
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G
    import e57_syn_pass as SP
    register_engine()
    seed=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');ro=list(cellinfo['read_apps']);ri=[ro[-1]];sv=[ro[-1]] if a.save_apps=='128' and a.cell!='v3_16_1' else [int(x) for x in a.save_apps.split(',')];CH=1024
    masks={k:v for k,v in np.load(a.masks,allow_pickle=True).items()} if a.masks and os.path.exists(a.masks) else {};masks.setdefault(f'band_{a.mask_arm}',np.asarray(int(re.sub(r'\D','',a.mask_arm))));masks.setdefault(f'all_{a.mask_arm}',np.asarray(1)) if a.mask_arm.startswith('ALLQ') else None;masks['band_INF']=np.asarray(10**6);masks['all_INF']=np.asarray(1);spec={ar:parse_arm2(split_clip(strip_k0(ar)[0])[0],a.mask_arm,a.rand_arm) for ar in arms}
    for ar,(ma,_) in spec.items():assert ma in ('NATQ','ORACLE','ORACLE_RAND','PLACEBO','HALF') or f'band_{ma[5:] if ma.startswith("KEY") else ma}' in masks,('mask arm not in the mask file',ar,ma)
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    lg=lambda *x:print(*x,flush=True)
    enc,w,tf=R55.build(SP.synth_args(seed,a.cell),seed,read_apps=set(ro)|set(ri)|set(sv),arms=['NAT']);S=R55.streams(w,enc);nch=a.smoke or w['n_chunks'];p=w['p']
    ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w)),'stream hash mismatch'
    P=R.plan(w,enc);items=P['items'];reads=[a_*p-1 for a_ in sv] if not a.smoke else [nch-1];canary_chunks=reads
    tok=rt.tok;raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    R_=RG.Reads(rd,raw,len(items),G.NCAND,P['read'],out_tail,rt.base,ro,ri,log=lambda *x:lg(*x));lg(f'seed {seed}: W0 read done')
    for ar in arms:
        fn=OUT/f'E55_{a.tag}_{ar}_s{seed}.saved.json'
        if fn.exists() and not a.rerun:lg('exists, reused:',fn);continue
        K0,_k0=strip_k0(ar,a.k0)[1],None;NM.K0=K0;QR.K0=K0
        ma,mult=spec[ar];uv=(OM.OracleUV(w['chunks'],K0,lg) if ma=='ORACLE' else OM.OracleRandUV(w['chunks'],seed,K0,lg) if ma=='ORACLE_RAND' else OM.PlaceboOracleUV(w['chunks'],K0,lg) if ma=='PLACEBO' else OM.HalfOracleUV(w['chunks'],256,K0,lg) if ma=='HALF' else KeyMatchedQ(ma[3],masks[f'band_{ma[5:]}'],seed,lg) if ma.startswith('KEY') else QR.SelMaskQ(ma,masks,lg));L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_{a.tag}_{ar}_s{seed}',canary=can);got={}
        mode=arm_mode(split_clip(strip_k0(ar)[0])[0]);hold={};logfull=ma!='NATQ';uv=with_base(uv,hold,L if (mode or logfull) else None,LD.jnp,full_every=a.nat_loss_every if logfull else 0,always_full=bool(mode))
        L.write_step=swap_step(L.write_step,mult,mode,hold,every=a.nat_loss_every if logfull else 0);clipv=split_clip(strip_k0(ar)[0])[1]
        if clipv:L.write_step=clip_step(L.write_step,clipv)
        dchunks={c_ for c_ in range(K0,nch,a.nat_loss_every)}|{nch-1} if a.nat_loss_every>0 else set();dfn=make_token_diag(L,LD,s3,rt,hold,w['chunks'],lg) if dchunks else None
        def partial(k,m,ar=ar,got=got):
            h0=LD.reuse._hash_pytree(m.inner_parameters())[0];x=R_.at((k+1)//p,m);h1=LD.reuse._hash_pytree(m.inner_parameters())[0];assert h0==h1,'weights changed during the read';hold.setdefault('read_hash',[]).append((k,h0,h1));got[(k+1)//p]=x
            rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{seed}.read.partial.json','w'))
            return {c_:'ok' for c_ in x}
        hook_chunks=[a_*p-1 for a_ in sorted(set(ro)|set(ri))] if not a.smoke else [nch-1]
        m0=None
        if a.perturb>0:
            newp,td=perturb_inner(rt.base,rt,LD,a.perturb,a.perturb_seed);m0=L.set_inner(rt.base,LD.jax.tree_util.tree_unflatten(td,newp));lg(f'  W0 perturbed (relative {a.perturb:g}, seed {a.perturb_seed})')
        t0=time.monotonic();lg(f'== seed {seed} {ar} (mask {ma}, lr x{mult:g}): {nch} chunks, reads out {ro} in {ri}')
        r=L.run_stream(ENGINE,p,ids,weights0=m0,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{seed}',log=lg,read_hook=partial,hook_chunks=hook_chunks,chunks_meta=w['chunks'],uv=uv,diag_fn=dfn,diag_chunks=dchunks);r.pop('model',None);r['wlog']=hold.get('wlog',[]);r['swap_mode']=mode;r['read_hash']=hold.get('read_hash',[])
        lgs=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lgs])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lgs])),'mean_gnorm':float(np.mean([e['gnorm'] for e in lgs])),'mean_coef':float(np.mean([e['coef'] for e in lgs])),'median_update_norm':float(np.median([e['update_norm'] for e in lgs])),'clip_frac_coef_lt1':float(np.mean([e['coef']<1.0 for e in lgs])),'mean_masked_row_frac':float(np.mean([e.get('q_banded',0) for e in lgs])/CH),'mean_hidden_keys_row':float(np.mean([e['hidden_keys_row_mean'] for e in lgs if 'hidden_keys_row_mean' in e] or [0]))}
        r.update(stream_sha=sha,world_seed=seed,cell=a.cell,read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,k=w['k'],n=w['n'],bank=w['bank_kind'],engine='SGD(clip)',mask_arm=ma,band=(int(masks[f'band_{ma[5:] if ma.startswith("KEY") else ma}']) if ma not in ('NATQ','ORACLE','ORACLE_RAND','PLACEBO','HALF') else None),lr_mult=mult,arm=ar,K0=K0)
        w0=r['canary_w0']
        for k,e in r['reads'].items():
            if 'canary' in e:
                for key in ('_gen_nll','_tail_nll'):lg(f'  canary {key} at chunk {k}: {e["canary"][key]-w0[key]:+.4f}')
        rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');gc.collect();LD.jax.clear_caches()
if __name__=='__main__':main()
