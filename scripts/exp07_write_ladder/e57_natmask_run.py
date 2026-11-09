"""E57 synthetic world: original-NAT (plain clipped SGD) write arms with the ALLQ mask and/or a step-size multiplier (GPU, one seed per call; new file, no existing file edited; design by Hans 11/09 'step size from the original paper'). Same write run, reads, canary and saved state as e57_syn_qrun.py, but the update is the ORIGINAL inner rule (clipped SGD, lr x ilr, NO Adam, NO norm matching) instead of engine SELC1:
  NATS    no mask, multiplier 1 (reproduces the native NAT: control of the write path)      NATSx2 / NATSx4    no mask, lr x 2 / x 4
  NATMASKC / NATMASKD (Council 11/09): same number of hidden keys per row as ALLQ, random keys (C) / most recent keys (D); selected by arm name, band from --mask-arm.
  RANDMASK random-row mask (stored arm --rand-arm, default RANDQ256P50), lr x 1 (Council 11/09: control for the mask effect)
  NATMASK ALLQ mask (band from the mask file, every row from chunk K0), lr x 1               NATMASKx2 / NATMASKx4   ALLQ mask, lr x 2 / x 4
The mask arm is the UV-style SelMaskQ (key `ALLQ<band>` of the mask file); the engine is the arm SGDSEL below, registered here as a UV arm that is NOT in E.ADAM, so e55_core takes the SGD path (write_step) with the masked keep. The multiplier is applied after the clip: new = old + mult x (SGD step). update_norm in the log is measured after the multiplier; 'scale' in the log stays 1.0 and the multiplier is stored as r['lr_mult'].
  python3 e57_natmask_run.py --seed 64 --out O --masks M.npz --mask-arm ALLQ256 --arms NATS,NATMASK,NATMASKx2,... [--tag synng] [--smoke N] [--rerun] | --selftest      (XLA_FLAGS=--xla_gpu_deterministic_ops=true)"""
import sys,os,json,time,gc,re,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_qband_run as QR
ENGINE='SGDSEL'
CH=1024;WIN=8192;K0=8
class KeyMatchedQ:
    """Council 11/09 controls with the SAME number of hidden keys per row as the band arm (ALLQ<band>), chosen differently. kind 'C': random keys among the causal natural-visible keys (distance >= 1); kind 'D': the n_i MOST RECENT causal keys (distance 1..n_i), n_i = number of natural-visible keys at distance >= band (what ALLQ hides). The row's own key (distance 0) stays visible. Applied only to the update-path keep (like SelMaskQ); chunks < K0: natural. Per-row hidden counts are logged."""
    def __init__(self,kind,band,seed=0,log=print):
        assert kind in ('C','D');self.kind=kind;self.band=int(band);self.seed=seed;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<K0:return base,{'uv_arm':'KEY'+self.kind,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        d=np.arange(CH)[:,None]+WIN-np.arange(WIN+CH)[None,:];vis=np.broadcast_to(base,(CH,WIN+CH))
        hidden_allq=vis&(d>=self.band);n=hidden_allq.sum(1)                      # keys ALLQ hides per row
        cand=vis&(d>=1)
        if self.kind=='D':hide=cand&(d<=n[:,None])
        else:
            rng=np.random.default_rng((self.seed,k));sc=np.where(cand,rng.random((CH,WIN+CH),dtype=np.float32),2.0)
            rank=np.argsort(np.argsort(sc,axis=1),axis=1);hide=cand&(rank<n[:,None])
        kp=vis&~hide;nh=hide.sum(1)
        assert (nh<=n).all() and (nh>=np.minimum(n,cand.sum(1))).all() and kp[np.arange(CH),np.arange(CH)+WIN].all()
        if k%16==0:self.log(f'  KEY{self.kind} chunk {k}: hidden keys/row mean {nh.mean():.0f} (ALLQ {n.mean():.0f}), band {self.band}')
        return kp,{'uv_arm':'KEY'+self.kind,'q_banded':CH,'q_band':self.band,'hidden_keys_row_mean':float(nh.mean()),'hidden_keys_row_min':int(nh.min()),'hidden_keys_row_max':int(nh.max()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def paired_item_ci(x,y,level,B=4000,seed=0):
    """Council 11/09 (item-level paired analysis): x, y = accuracy arrays [seeds, items] (same seeds and items), difference d_i = mean over seeds of (x - y) per item, bootstrap over ITEMS -> (mean, lo, hi, p_two_sided of mean<=0 / >=0)"""
    d=(np.asarray(x,float)-np.asarray(y,float)).mean(0);rng=np.random.default_rng(seed);n=len(d);bs=np.array([d[rng.integers(0,n,n)].mean() for _ in range(B)]);al=(1-level)/2
    p=float(2*min((bs<=0).mean(),(bs>=0).mean()));return float(d.mean()),float(np.quantile(bs,al)),float(np.quantile(bs,1-al)),max(p,1.0/B)
def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)"""
    names=sorted(pvals,key=lambda k:pvals[k]);m=len(names);adj={};run=0.0
    for i,k in enumerate(names):run=max(run,min(1.0,(m-i)*pvals[k]));adj[k]=run
    return adj
EQ_MARGIN=0.02
def verdict_rules(contrasts):
    """Council 11/09 (fixed margins, replaces the earlier 2xSD / 0.25|g| rules). contrasts: {name: (a, b)} with a, b = accuracy [seeds, items] in accuracy units (0.02 = 2 pp). For each contrast a - b: EQUIVALENT when the 90% item-bootstrap CI lies inside +-2 pp; SUPERIOR when the 95% CI lower bound > 0 (a > b); the superiority p-values are Holm-corrected across all contrasts (a contrast is called superior only if its Holm-adjusted p < 0.05 as well). Returns {name: dict}."""
    out={};ps={}
    for k,(x,y) in contrasts.items():
        mu,l90,h90,_=paired_item_ci(x,y,0.90);_,l95,h95,p=paired_item_ci(x,y,0.95);out[k]=dict(mean=mu,ci90=(l90,h90),ci95=(l95,h95),p=p,equivalent=bool(l90>=-EQ_MARGIN and h90<=EQ_MARGIN),superior_raw=bool(l95>0));ps[k]=p
    adj=holm(ps)
    for k in out:out[k]['p_holm']=adj[k];out[k]['superior']=bool(out[k]['superior_raw'] and adj[k]<0.05)
    return out
def register_engine():
    """SGDSEL = UV arm on the SGD path: in E.ARMS and E.UVARMS, not in E.ADAM (e55_core then uses write_step, the clipped SGD rule, with the keep from the uv object)."""
    import e55_swap as E
    if ENGINE not in E.ARMS:E.ARMS.append(ENGINE)
    if ENGINE not in E.UVARMS:E.UVARMS=tuple(E.UVARMS)+(ENGINE,)
    assert ENGINE not in E.ADAM
    import e55_core as K
    if ENGINE not in K.IMPLEMENTED:K.IMPLEMENTED=tuple(K.IMPLEMENTED)+(ENGINE,)
    if not getattr(K.Ladder55.run_stream,'_sgdsel',False):
        # e55_core takes arm_k = the arm itself for arms outside E.ADAM, but the UV branch needs arm_k == 'NAT': patch ONE expression of the method source at run time (the file on disk is untouched)
        import inspect,textwrap
        src=textwrap.dedent(inspect.getsource(K.Ladder55.run_stream));old="(E.ADAM[arm][0] if arm in E.ADAM else arm)";assert src.count(old)==1
        src=src.replace(old,f"(E.ADAM[arm][0] if arm in E.ADAM else ('NAT' if arm=='{ENGINE}' else arm))");ns={};exec(compile(src,'<run_stream SGDSEL patch>','exec'),K.__dict__,ns)
        f=ns['run_stream'];f._sgdsel=True;K.Ladder55.run_stream=f
def parse_arm(name,mask_arm,rand_arm='RANDQ256P50'):
    """NATS[xN] -> (no mask 'NATQ', N); NATMASK[xN] -> (mask_arm, N); RANDMASK[xN] -> (rand_arm, N)  (ALLQ masks every row, so a random mask of the SAME ratio is identical; RANDMASK uses the stored random-row arm instead, default 50% of the rows)"""
    m=re.fullmatch(r'(NATS|NATMASK|RANDMASK|NATMASKC|NATMASKD|NATBINF|DIRmaskNORMfull|DIRfullNORMmask)(?:x(\d+(?:\.\d+)?))?',name);assert m,('arm name',name)
    return {'NATS':'NATQ','NATMASK':mask_arm,'RANDMASK':rand_arm,'NATMASKC':'KEYC:'+mask_arm,'NATMASKD':'KEYD:'+mask_arm,'NATBINF':'INF','DIRmaskNORMfull':mask_arm,'DIRfullNORMmask':mask_arm}[m.group(1)],(float(m.group(2)) if m.group(2) else 1.0)
def arm_mode(name):
    """swap mode of the arm: 'DIRM_NF' = direction of the MASKED step, norm of the FULL-context step; 'DIRF_NM' = the reverse; None = ordinary"""
    return {'DIRmaskNORMfull':'DIRM_NF','DIRfullNORMmask':'DIRF_NM'}.get(re.sub(r'x\d+(?:\.\d+)?$','',name))
def swap_step(orig,mult,mode,holder,every=8):
    """write_step wrapper (Council 11/09). The ordinary step is the masked step `orig(m,st_all,st_suf,c,pref,kmask)`. When mode is set (every chunk) or on every `every`-th chunk (logging only), the FULL-context step from the SAME pre-update state is also computed: orig(m,st_all,st_suf,c,holder['pref_n'],holder['base']). Logged per such chunk in holder['wlog']: gnorm and coef (pre-clip norm, clip coefficient) of both passes, the update norms, and cos(update_mask, update_full) (= cos of the two clipped gradients). DIRM_NF applies direction of the masked update with the norm of the full update, DIRF_NM the reverse (update = new - old; then x mult)."""
    import jax,jax.numpy as jnp
    def ws(m,st_all,st_suf,c,pref,kmask):
        newp,gn,coef,ce=orig(m,st_all,st_suf,c,pref,kmask);k=holder.get('k',-1);need=mode is not None or (every and k>=0 and k%every==0 and 'pref_n' in holder)
        f32=jnp.float32;old=m.inner_parameters()
        if need and 'pref_n' in holder:
            newf,gnf,coeff,cef=orig(m,st_all,st_suf,c,holder['pref_n'],jnp.asarray(holder['base']))
            dm=jax.tree.map(lambda n,o:n.astype(f32)-o.astype(f32),newp,old);df=jax.tree.map(lambda n,o:n.astype(f32)-o.astype(f32),newf,old)
            nm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(dm))));nf=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(df))))
            dot=float(sum(jnp.sum(a*b) for a,b in zip(jax.tree.leaves(dm),jax.tree.leaves(df))))
            holder.setdefault('wlog',[]).append(dict(k=k,gnorm_mask=float(gn),coef_mask=float(coef),gnorm_full=float(gnf),coef_full=float(coeff),upd_norm_mask=nm,upd_norm_full=nf,cos_mask_full=dot/(nm*nf) if nm>0 and nf>0 else float('nan'),ce_full=float(cef)))
            if mode=='DIRM_NF':newp=jax.tree.map(lambda o,d:(o.astype(f32)+d*(nf/nm if nm>0 else 1.0)*mult).astype(o.dtype),old,dm)
            elif mode=='DIRF_NM':newp=jax.tree.map(lambda o,d:(o.astype(f32)+d*(nm/nf if nf>0 else 1.0)*mult).astype(o.dtype),old,df)
            elif mult!=1.0:newp=jax.tree.map(lambda n,o:(o.astype(f32)+(n.astype(f32)-o.astype(f32))*mult).astype(n.dtype),newp,old)
            return newp,gn,coef,ce
        assert mode is None or k<K0 or 'pref_n' in holder,'swap arm without the full-context prefix'
        if mult!=1.0:newp=jax.tree.map(lambda n,o:(o.astype(f32)+(n.astype(f32)-o.astype(f32))*mult).astype(n.dtype),newp,old)
        return newp,gn,coef,ce
    return ws
def mult_step(orig,mult):
    """wrap Ladder.write_step: the returned inner parameters old + mult x (new - old) in fp32, cast back; (gnorm, coef, ce) unchanged"""
    import jax,jax.numpy as jnp
    def ws(m,*a,**k):
        newp,gn,coef,ce=orig(m,*a,**k)
        if mult==1.0:return newp,gn,coef,ce
        f32=jnp.float32;old=m.inner_parameters()
        newp=jax.tree.map(lambda n,o:(o.astype(f32)+(n.astype(f32)-o.astype(f32))*mult).astype(n.dtype),newp,old)
        return newp,gn,coef,ce
    return ws
def make_nat_loss_diag(L,LD,s3,rt,holder,log=print):
    """Council 11/09 prediction-path loss: at the chunks of diag_chunks, BEFORE the write of the chunk, the chunk is scored with the FULL natural local context (natural keep in prefix and suffix, current weights, same suffix state), the way e57_closure_suffixband_run does it. holder['base'] is the natural 1-D keep of the chunk, stored by the uv object's keep(). Returns the diag_fn (wants_ctx) -> per-chunk dict {nat_nll_mean, upd_nll_mean (masked update-path loss on the same tokens)}."""
    jax,jnp,eqx=LD.jax,LD.jnp,LD.eqx
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        base=np.asarray(holder['base'],bool);assert base.ndim==1
        with rt.mesh:
            pref_n=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(base))
            lp=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pref_n,jnp.asarray(base)),np.float64)
            lpu=np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],jnp.asarray(np.asarray(keep,bool))),np.float64)
        lm=np.asarray(ctx['c'].loss_masks,bool);r=dict(nat_nll_mean=float((-lp[lm]).mean()),upd_nll_mean=float((-lpu[lm]).mean()),n=int(lm.sum()))
        log(f'  predict-path chunk {k}: natural-context nll {r["nat_nll_mean"]:.4f}, update-path nll {r["upd_nll_mean"]:.4f}');return r
    fn.wants_ctx=True;return fn
def with_base(uv,holder,L=None,jnp=None,full_every=0,always_full=False):
    """wrap uv.keep so that the natural 1-D keep of the chunk (holder['base']) and the chunk index (holder['k']) are stored; with L given, the FULL-context prefix outputs of the chunk (current weights, natural keep) go to holder['pref_n'] on the chunks where swap_step needs them (every chunk if always_full, else every full_every-th chunk from K0)"""
    o=uv.keep
    def keep(k,m,Kc,Vc,ids_k,keep_all):
        holder['base']=np.asarray(keep_all,bool);holder['k']=k;holder.pop('pref_n',None)
        if L is not None and k>=K0 and (always_full or (full_every and k%full_every==0)):
            holder['pref_n']=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_all))
        return o(k,m,Kc,Vc,ids_k,keep_all)
    uv.keep=keep;return uv
def selftest():
    QR.selftest()
    assert parse_arm('NATS','ALLQ256')==('NATQ',1.0) and parse_arm('NATSx4','ALLQ256')==('NATQ',4.0)
    assert parse_arm('RANDMASKx4','ALLQ256')==('RANDQ256P50',4.0) and parse_arm('NATMASK','ALLQ256')==('ALLQ256',1.0) and parse_arm('NATMASKx2','ALLQ128')==('ALLQ128',2.0)
    for bad in ('NATQ','NATMASKx','NATx2'):
        try:parse_arm(bad,'ALLQ256');raise SystemExit('parse_arm accepted '+bad)
        except AssertionError:pass
    register_engine();import e55_swap as E
    assert ENGINE in E.ARMS and ENGINE in E.UVARMS and ENGINE not in E.ADAM
    import e55_core as K;assert ENGINE in K.IMPLEMENTED and K.Ladder55.run_stream._sgdsel
    nat=np.ones(WIN+CH,bool);nat[:50]=False;band=256
    for kind in ('C','D'):
        kp,info=KeyMatchedQ(kind,band,0,lambda *x:None).keep(9,None,None,None,None,nat);assert kp.shape==(CH,WIN+CH)
        d=np.arange(CH)[:,None]+WIN-np.arange(WIN+CH)[None,:];n=(np.broadcast_to(nat,(CH,WIN+CH))&(d>=band)).sum(1);cand=np.broadcast_to(nat,(CH,WIN+CH))&(d>=1)
        assert np.array_equal((cand&~kp).sum(1),n) and kp[np.arange(CH),np.arange(CH)+WIN].all() and not (kp&~np.broadcast_to(nat,(CH,WIN+CH))).any()
        if kind=='D':assert not kp[100,WIN+100-1] and kp[100,WIN+100-n[100]-1] if n[100]>0 and WIN+100-n[100]-1>=50 else True
    k0,_=KeyMatchedQ('C',band,0).keep(3,None,None,None,None,nat);assert k0.ndim==1
    rng=np.random.default_rng(1);n0=rng.random((5,60))<0.3
    big=n0|(rng.random((5,60))<0.5);vr=verdict_rules({'big-vs-nat':(big.astype(float),n0.astype(float)),'same':(n0.astype(float),n0.astype(float))})
    assert vr['big-vs-nat']['superior'] and not vr['big-vs-nat']['equivalent'] and vr['same']['equivalent'] and not vr['same']['superior'],vr
    assert arm_mode('DIRmaskNORMfullx2')=='DIRM_NF' and arm_mode('DIRfullNORMmask')=='DIRF_NM' and arm_mode('NATMASK') is None and parse_arm('DIRmaskNORMfull','ALLQ256')==('ALLQ256',1.0) and parse_arm('NATBINF','ALLQ256')==('INF',1.0)
    assert holm({'a':0.01,'b':0.04})=={'a':0.02,'b':0.04}
    h={};u=with_base(KeyMatchedQ('C',256,0,lambda *x:None),h);nat2=np.ones(WIN+CH,bool);u.keep(3,None,None,None,None,nat2);assert h['base'] is not None and h['base'].shape==(WIN+CH,)
    # multiplier wrapper on a toy 'model'
    import jax.numpy as jnp
    class M_:
        def inner_parameters(self):return {'w':jnp.ones((3,),jnp.float32)}
    orig=lambda m,*a,**k:({'w':jnp.full((3,),0.9,jnp.float32)},1.0,1.0,0.0)
    n2=mult_step(orig,2.0)(M_())[0]['w'];assert np.allclose(np.asarray(n2),0.8,atol=1e-6),n2
    n1=mult_step(orig,1.0)(M_())[0]['w'];assert np.allclose(np.asarray(n1),0.9,atol=1e-6)
    print('e57_natmask_run selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--cell',default='v3_16_1');ap.add_argument('--save-apps',default='128');ap.add_argument('--out');ap.add_argument('--masks');ap.add_argument('--mask-arm',default='ALLQ256');ap.add_argument('--rand-arm',default='RANDQ256P50');ap.add_argument('--arms');ap.add_argument('--tag',default='synng');ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--rerun',action='store_true');ap.add_argument('--nat-loss-every',type=int,default=8);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G
    import e57_syn_pass as SP
    register_engine()
    seed=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True);arms=a.arms.split(',');ro=[16,32,64,128];ri=[128];sv=[int(x) for x in a.save_apps.split(',')];CH=1024
    masks={k:v for k,v in np.load(a.masks,allow_pickle=True).items()} if a.masks and os.path.exists(a.masks) else {};masks.setdefault(f'band_{a.mask_arm}',np.asarray(int(re.sub(r'\D','',a.mask_arm))));masks.setdefault(f'all_{a.mask_arm}',np.asarray(1)) if a.mask_arm.startswith('ALLQ') else None;masks['band_INF']=np.asarray(10**6);masks['all_INF']=np.asarray(1);spec={ar:parse_arm(ar,a.mask_arm,a.rand_arm) for ar in arms}
    for ar,(ma,_) in spec.items():assert ma=='NATQ' or f'band_{ma[5:] if ma.startswith("KEY") else ma}' in masks,('mask arm not in the mask file',ar,ma)
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
        ma,mult=spec[ar];uv=(KeyMatchedQ(ma[3],masks[f'band_{ma[5:]}'],seed,lg) if ma.startswith('KEY') else QR.SelMaskQ(ma,masks,lg));L=C.Ladder55(rt,save_dir=OUT/'states'/f'E55_{a.tag}_{ar}_s{seed}',canary=can);got={}
        mode=arm_mode(ar);hold={};logfull=ma!='NATQ';uv=with_base(uv,hold,L if (mode or logfull) else None,LD.jnp,full_every=a.nat_loss_every if logfull else 0,always_full=bool(mode))
        L.write_step=swap_step(L.write_step,mult,mode,hold,every=a.nat_loss_every if logfull else 0);dchunks={c_ for c_ in range(K0,nch,a.nat_loss_every)}|{nch-1} if a.nat_loss_every>0 else set();dfn=make_nat_loss_diag(L,LD,s3,rt,hold,lg) if dchunks else None
        def partial(k,m,ar=ar,got=got):
            x=R_.at((k+1)//p,m);got[(k+1)//p]=x
            rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':False})[ar];json.dump(rec,open(OUT/f'E55_{a.tag}_{ar}_s{seed}.read.partial.json','w'))
            return {c_:'ok' for c_ in x}
        hook_chunks=[a_*p-1 for a_ in sorted(set(ro)|set(ri))] if not a.smoke else [nch-1]
        t0=time.monotonic();lg(f'== seed {seed} {ar} (mask {ma}, lr x{mult:g}): {nch} chunks, reads out {ro} in {ri}')
        r=L.run_stream(ENGINE,p,ids,decoy_ids=S['decoy'][0] if 'decoy' in S else None,bg_ids=S['nofact'][0] if 'nofact' in S else (S['bg'][0] if 'bg' in S else None),read_chunks=reads,canary_chunks=canary_chunks,nch=nch,tag=ar+f'_s{seed}',log=lg,read_hook=partial,hook_chunks=hook_chunks,chunks_meta=w['chunks'],uv=uv,diag_fn=dfn,diag_chunks=dchunks);r.pop('model',None);r['wlog']=hold.get('wlog',[]);r['swap_mode']=mode
        lgs=r['log'];r['summary']={'mean_update_norm':float(np.mean([e['update_norm'] for e in lgs])),'clip_bind_frac':r['clip_bind_frac'],'cum_update_norm':r['cum_update_norm'],'mean_ce':float(np.mean([e['ce'] for e in lgs])),'mean_gnorm':float(np.mean([e['gnorm'] for e in lgs])),'mean_coef':float(np.mean([e['coef'] for e in lgs])),'median_update_norm':float(np.median([e['update_norm'] for e in lgs])),'clip_frac_coef_lt1':float(np.mean([e['coef']<1.0 for e in lgs])),'mean_masked_row_frac':float(np.mean([e.get('q_banded',0) for e in lgs])/CH),'mean_hidden_keys_row':float(np.mean([e['hidden_keys_row_mean'] for e in lgs if 'hidden_keys_row_mean' in e] or [0]))}
        r.update(stream_sha=sha,world_seed=seed,cell=a.cell,read_apps_out=ro,read_apps_in=ri,save_apps=sv,sec_total=time.monotonic()-t0,k=w['k'],n=w['n'],bank=w['bank_kind'],engine='SGD(clip)',mask_arm=ma,band=(int(masks[f'band_{ma[5:] if ma.startswith("KEY") else ma}']) if ma!='NATQ' else None),lr_mult=mult,arm=ar)
        w0=r['canary_w0']
        for k,e in r['reads'].items():
            if 'canary' in e:
                for key in ('_gen_nll','_tail_nll'):lg(f'  canary {key} at chunk {k}: {e["canary"][key]-w0[key]:+.4f}')
        rec=RG.make_records({ar:got},R_.W0,G.item_meta(items),P['read'],R_.idx,{'cell':a.cell,'seed':seed,'k':w['k'],'n':w['n'],'p':p,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'place_len':w['place_len']})[ar];fr=OUT/f'E55_{a.tag}_{ar}_s{seed}.read.json';json.dump(rec,open(fr,'w'));json.dump(r,open(fn,'w'),default=float)
        lg('wrote',fn,f'{time.monotonic()-t0:.0f}s');gc.collect();LD.jax.clear_caches()
if __name__=='__main__':main()
