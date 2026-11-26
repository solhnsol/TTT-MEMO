"""Cross one-step, STAGE 2 (job 1358b 'xstep'; GPU; NEW file; implementation thread 11/26; design: Hans 11/26 18:12 KST, E59 [S2617-XSTEP-OK]; EXPLORATORY). Uses the stage-1 states of e59_xstep_save.py (weights W_t and suffix carried state S_t of the re-run 1348 streams P1s = C0 stream, P0s = SAMEVIS stream, t = 32 / 64 / 96) and W0 (t = 0, fresh suffix state; one state 'W0').
2x2 = state {C0 stream, SAMEVIS stream} x step mask {C0, SAMEVIS}: for every chunk c = t+j (j = 0..7), ONE step from W_t (every step starts from W_t). Window of chunk c: prefix-layer window K,V = the natural W0 K,V of the stream (as in every stream run); suffix carried state = S_t advanced through chunks t..c-1 with W_t and no update (engine read_state). Same lr, global clip, write layers, loss (NTP over the chunk, token mean) as the stream (e59_kvmulti ws). Steps per chunk: C0 natural, SAMEVIS natural, SIZE-MATCHED = the SAMEVIS update rescaled to the update norm of the C0 natural step (no extra backward).
Check: the step of chunk t with the state's own mask equals the stream weights after t+1 exposures (stage-1 files W_{t+1}: hash and max abs difference; for W0 the stream weights after 1 exposure).
Auxiliary interval runs: from W_t, 8 consecutive steps over chunks t..t+7 with C0 and with SAMEVIS (engine rule: the suffix state is advanced with the weights at the start of each chunk), then R1.
Reads / storage (fam-bind split): R1 16 x 16 candidate scores (12 exposed + 4 never exposed) minus W0 at W_t and after every step; per chunk the list of read facts with a sentence in the chunk (in this U12 world every chunk holds one sentence of EVERY fact, so the list is always all 16); copy side for those facts: R2 and R2m (context = chunk c, as e59_kvwin.py; only for the chunk offsets --r2-offsets, natural steps only; W_t once per chunk) and the NLL of the place tokens under the C0 keep and the SAMEVIS keep before and after every step; per step the gradient norm before clipping, clip coefficient, CE, actual update norm; cosine of the C0 and SAMEVIS natural updates (total and per parameter leaf, per leading index for stacked leaves). Raw per-fact, per-chunk values; no aggregation here (e59_xstep_analyze.py).
  python3 e59_xstep.py --seed 231 --states-dir S1DIR --out O [--ts 0,32,64,96] [--nj 8] [--r2-offsets 0,1,2,3,4,5,6,7] [--check] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
NREAD=16;CH=1024
STATES={'C0':'P1s','SAMEVIS':'P0s'}      # state name -> stage-1 stream arm
MASKS=('C0','SAMEVIS')
def facts_in_chunk(chunks,k):return sorted(int(x['fact']) for x in chunks[k]['sents'] if int(x['fact'])<NREAD)
def cosine_parts(da,db,names):
    """da, db: lists of numpy leaves (updates); -> total cosine, per-leaf cosine, per leading index for leaves with ndim >= 3"""
    dot=0.0;na=0.0;nb=0.0;per={};lead={}
    for nm,a,b in zip(names,da,db):
        a=np.asarray(a,np.float64).ravel() if a.ndim<3 else np.asarray(a,np.float64);b=np.asarray(b,np.float64).ravel() if b.ndim<3 else np.asarray(b,np.float64)
        x=float((a*b).sum());ya=float((a*a).sum());yb=float((b*b).sum());dot+=x;na+=ya;nb+=yb;per[nm]=x/np.sqrt(ya*yb) if ya>0 and yb>0 else None
        if a.ndim>=3:
            aa=a.reshape(a.shape[0],-1);bb=b.reshape(b.shape[0],-1);xx=(aa*bb).sum(1);n1=(aa*aa).sum(1);n2=(bb*bb).sum(1);lead[nm]=[float(xx[i]/np.sqrt(n1[i]*n2[i])) if n1[i]>0 and n2[i]>0 else None for i in range(len(xx))]
    return dict(total=dot/np.sqrt(na*nb) if na>0 and nb>0 else None,per_leaf=per,per_lead=lead)
def selftest():
    chunks=[{'sents':[dict(fact=f) for f in (3,0,20,1)]}];assert facts_in_chunk(chunks,0)==[0,1,3]
    a=[np.ones((2,3,4)),np.array([1.0,0.0])];b=[np.ones((2,3,4)),np.array([0.0,1.0])];c=cosine_parts(a,b,['x','y']);assert abs(c['per_leaf']['x']-1)<1e-12 and abs(c['per_leaf']['y'])<1e-12 and abs(c['total']-24/25)<1e-12 and c['per_lead']['x']==[1.0,1.0]
    b2=[-np.ones((2,3,4)),np.array([1.0,0.0])];c2=cosine_parts(a,b2,['x','y']);assert abs(c2['total']-(-23/25))<1e-12
    print('e59_xstep selftest ok: facts per chunk, cosines')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_shadow as SH,s3
    from e59_kvpos2 import items2,NEXP,CONDS
    from e59_kvmulti import place_masks
    from e59_xstep_save import load_suffix_state
    from ttt.model.data import Batch
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic();SD=pathlib.Path(a.states_dir)
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3_,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();PS=LD.packing
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);NC=G.NCAND;out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w))
    with rt32.mesh,jax.default_matmul_precision('highest'):T0=LD._TailCtx(rt32,list(out_tail));Pout=LD.packing.Packed(T0,raw,GP=2)
    def r1(m):
        with jax.default_matmul_precision('highest'):return np.asarray(Pout.r1(rd.to_fp32(m))).reshape(len(items),NC)[:NREAD]
    W0S=r1(rt.base);s1=json.load(open(SD/f'kvmulti_s{seed}.json'));assert float(np.abs(np.asarray(s1['W0_R1'])-W0S).max())<1e-4,'W0 R1 differs from stage 1';side=json.load(open(SD/f'xstep_save_s{seed}.json'))
    if not side.get('same_as_1348',{}).get('same',False):lg('WARNING: stage-1 re-run is NOT identical to 1348 (see xstep_save json); continuing, flag stored');RES_flag=False
    else:RES_flag=True
    # ---- R2 / R2m with context = chunk c (copy of the e59_kvwin.py read path) ----
    SWA=PS.SWA;_prev=SWA.sw_causal_mask
    def _mask2(self,chunk_id):
        m=_prev(self,chunk_id)
        if 'hide' in PS.PACK_SUF:h=PS.PACK_SUF['hide'];m=m&jnp.concatenate([~h,jnp.ones((self.mini_batch_size,self.mini_batch_size),bool)],axis=1)
        return m
    SWA.sw_causal_mask=_mask2
    Wsl=int(rt32.cfg.model.sliding_window_size)
    @eqx.filter_jit
    def prefix_h(m,ids_,pos_,mask_):
        PS.PACK_PRE['mask']=mask_
        try:
            seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
            return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),PS._clone(rt32.state_prefix),seq).last_hidden_state
        finally:PS.PACK_PRE.clear()
    @eqx.filter_jit
    def lp_hide(m,st,ids_,tg_,mk_,pref_,seg_,loc_,hide_):
        def one(i,t,k,p,s,l,h):
            PS.PACK_SUF['seg']=s;PS.PACK_SUF['loc']=l;PS.PACK_SUF['hide']=h
            try:
                r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k))
                lgt=r.logits.astype(jnp.float32);return jnp.take_along_axis(lgt,t[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lgt,axis=-1)
            finally:PS.PACK_SUF.clear()
        return jax.vmap(one)(ids_,tg_,mk_,pref_,seg_,loc_,hide_)
    class R2Ctx:
        """reads R2 (context = chunk k, nothing hidden) and R2m (each question's own fact sentence of chunk k hidden from the question tokens, prefix and suffix layers), scores of all 16 x 16 candidates"""
        def __init__(self,k):
            self.k=k;ctx=[int(t) for t in ids0[k*CH:(k+1)*CH]];sp={d['fact']:d['span'] for d in ap[k] if d['fact']<NREAD};assert sorted(sp)==list(range(NREAD))
            H={'R2':np.zeros((NREAD,CH),bool),'R2m':np.zeros((NREAD,CH),bool)}
            for q in range(NREAD):H['R2m'][q,sp[q][0]:sp[q][1]]=True
            with rt32.mesh,jax.default_matmul_precision('highest'):self.T=LD._TailCtx(rt32,ctx);self.PK=PS.Packed(self.T,raw,GP=2)
            PK=self.PK;TL=len(self.T.tail_ids);assert TL==CH;npad=PK.ids.shape[0];seg_np=np.asarray(PK.seg);loc_np=np.asarray(PK.loc);ids_np=np.asarray(PK.ids);qmap=[dict() for _ in range(npad)];cnt={}
            for r,(c,_a,_b) in enumerate(PK.where):s=cnt.get(c,0);cnt[c]=s+1;qmap[c][s]=r//NC
            for c in range(PK.nchunk,npad):qmap[c]=qmap[PK.nchunk-1]
            self.VAR={}
            for name,Hv in H.items():
                pref=[];hide=np.zeros((npad,CH,Wsl),bool)
                with rt32.mesh,jax.default_matmul_precision('highest'):
                    for j in range(npad):
                        sj,lj=seg_np[j],loc_np[j];sf=np.concatenate([np.full(TL,-2,np.int32),sj]);mk=PS.prefix_mask(sf,Wsl)
                        for s,q in qmap[j].items():
                            rows=np.where(sj==s)[0]
                            if len(rows)==0:continue
                            mk[np.ix_(TL+rows,np.arange(TL))]&=~Hv[q][None,:];hide[j][np.ix_(rows,Wsl-TL+np.arange(TL))]=Hv[q][None,:]
                        full=np.concatenate([np.asarray(self.T.tail_ids,np.int32),ids_np[j]]);pos=np.concatenate([np.arange(TL,dtype=np.int32),TL+lj])
                        pref.append(prefix_h(rt32.base,jnp.asarray(full),jnp.asarray(pos),jnp.asarray(mk))[TL:])
                    self.VAR[name]=(jnp.stack(pref),jnp.asarray(hide))
        def score(self,m):
            m32=rd.to_fp32(m);T_=self.T;PK=self.PK;npad=PK.ids.shape[0];res={}
            with rt32.mesh:
                st=PS._clone(rt32.state_suffix)
                for j in range(len(T_.tail_ids)//CH):st,_=rt32.chunk_logp(m32,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*CH:(j+1)*CH])
            with rt32.mesh,jax.default_matmul_precision('highest'):
                for nm,(pref,hide) in self.VAR.items():
                    o=[]
                    for g in range(0,npad,PK.GP):o.append(np.asarray(lp_hide(m32,st,PK.ids[g:g+PK.GP],PK.tg[g:g+PK.GP],PK.mk[g:g+PK.GP],pref[g:g+PK.GP],PK.seg[g:g+PK.GP],PK.loc[g:g+PK.GP],hide[g:g+PK.GP])))
                    lp=np.concatenate(o);res[nm]=np.asarray([lp[c,a_:b_].sum() for c,a_,b_ in PK.where]).reshape(len(items),NC)[:NREAD]
            del m32;return res
    # ---- write machinery (as e59_kvmulti ws / e55_core.run_stream) ----
    L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}',canary=can);L.chunks_meta=w['chunks']
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    @eqx.filter_jit
    def nll_all(m,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32);return jax.scipy.special.logsumexp(lgt,axis=-1)-jnp.take_along_axis(lgt,c.target_tokens[:,None],axis=-1)[:,0]
    UV={mk:SH.mk_uv(CONDS[mk],w,seed,lambda *x:None) for mk in MASKS}
    leaf_names=[jax.tree_util.keystr(p_) for p_,_ in jax.tree_util.tree_flatten_with_path(rt.base.inner_parameters())[0]]
    hashf=lambda m:s3.reuse._hash_pytree(m.inner_parameters())[0]
    def delta(newp,m):return [np.asarray(jax.device_get(x.astype(jnp.float32)-y.astype(jnp.float32))) for x,y in zip(jax.tree_util.tree_leaves(newp),jax.tree_util.tree_leaves(m.inner_parameters()))]
    RES=dict(seed=seed,args=vars(a),W0_R1=W0S.tolist(),cand_exposed=[True]*NEXP+[False]*(NC-NEXP),stage1_same_as_1348=RES_flag,facts_per_chunk_note='U12 world: every chunk holds one sentence of every fact',states={},checks={})
    TS=[int(x) for x in a.ts.split(',')];NJ=a.nj;R2OFF=set(int(x) for x in a.r2_offsets.split(',')) if a.r2_offsets else set()
    with rt.mesh:
        seq=L._stream_seq(ids0,nch);prefs=rt.prefix(rt.base,seq);t0=time.monotonic();Kh,Vh,_=L.record(ids0[:nch*CH]);recs={'real':(Kh,Vh)};lg(f'seed {seed}: natural K,V recorded {time.monotonic()-t0:.0f}s')
        st_all=s3.clone_state(rt.state_all)
        def win(k,st):
            Kc,Vc,keep,st_w,pl=L.window_inputs('NAT',k,1,recs,{'real':st},[]);return Kc,Vc,keep,st_w
        def advance(m,st,k):return L.read_state(m,st,rt.chunk(seq,k),prefs[k*CH:(k+1)*CH])
        def step(m,st_w,k,Kc,Vc,keep,mk):
            c=rt.chunk(seq,k);ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);kp,_=UV[mk].keep(k,m,Kc,Vc,ids0[k*CH:(k+1)*CH],keep);km=jnp.asarray(kp);pref=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),km)
            valid=np.asarray(c.loss_masks,np.float32);wt=jnp.asarray(valid/valid.sum());g,nll=ref_grad(m,st_w,c,pref,km,wt);newp,gn,coef=L.sgd_apply(m,st_all,g);del g
            ce=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum());return newp,dict(gnorm=float(gn),coef=float(coef),ce=ce)
        def place_nll(m,st_w,k,Kc,Vc,keep):
            """NLL of the place tokens of the 16 read facts of chunk k (index p-1) under the C0 and the SAMEVIS keep"""
            c=rt.chunk(seq,k);ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);T=CH;pm,od=place_masks(k,T,ap);out_={}
            pos={d['fact']:[t-1 for t in d['place'] if 0<=t-1<T] for d in ap[k] if d['fact']<NREAD}
            for mk in MASKS:
                kp,_=UV[mk].keep(k,m,Kc,Vc,None,keep);km=jnp.asarray(kp);pref=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),km);nl=np.asarray(nll_all(m,st_w,c,pref,km))
                out_[mk]={str(f):[float(nl[i]) for i in pos[f]] for f in range(NREAD)}
            return out_
        for t in TS:
            if t==0:states={'W0':(rt.base,s3.clone_state(rt.state_suffix),None)}
            else:
                states={}
                for sn,arm in STATES.items():
                    info=side['saved'][arm][str(t)];m=L.load_state(ROOT/info['W']['_file']) if not os.path.isabs(info['W']['_file']) else L.load_state(info['W']['_file']);assert hashf(m)==info['W']['_hash'],('W_t hash',sn,t)
                    fn=info['S']['_file'];st=load_suffix_state(fn if os.path.isabs(fn) else ROOT/fn,s3.clone_state(rt.state_suffix));states[sn]=(m,st,arm)
            for sn,(Wt,St,arm) in states.items():
                t1=time.monotonic();rec=dict(t=t,R1_Wt=(r1(Wt)-W0S).tolist(),chunks=[],interval={});RES['states'].setdefault(sn,{})[str(t)]=rec;st=St
                for j in range(NJ):
                    k=t+j;Kc,Vc,keep,st_w=win(k,st);ent=dict(k=k,facts=facts_in_chunk(w['chunks'],k),place_nll_Wt=place_nll(Wt,st_w,k,Kc,Vc,keep),steps={})
                    r2=R2Ctx(k) if j in R2OFF else None
                    if r2 is not None:sc=r2.score(Wt);ent['R2_Wt']={v:(sc[v]).tolist() for v in sc}
                    newp={}
                    for mk in MASKS:
                        newp[mk],d=step(Wt,st_w,k,Kc,Vc,keep,mk);d['update_norm']=L.norm_between(newp[mk],Wt);ent['steps'][mk]=d
                    s_=ent['steps']['C0']['update_norm']/ent['steps']['SAMEVIS']['update_norm'] if ent['steps']['SAMEVIS']['update_norm']>0 else 1.0
                    newp['SAMEVIS_sized']=L.rescale(newp['SAMEVIS'],Wt,s_);ent['steps']['SAMEVIS_sized']=dict(scale=s_,update_norm=L.norm_between(newp['SAMEVIS_sized'],Wt),gnorm=ent['steps']['SAMEVIS']['gnorm'],coef=ent['steps']['SAMEVIS']['coef'],ce=ent['steps']['SAMEVIS']['ce'])
                    ent['cos_C0_SAMEVIS']=cosine_parts(delta(newp['C0'],Wt),delta(newp['SAMEVIS'],Wt),leaf_names)
                    if j==0:
                        own='C0' if sn in ('C0','W0') else 'SAMEVIS';own_arm=arm or 'P1s';ref=side['saved'][own_arm].get(str(t+1)) if t>0 else side['saved'][own_arm].get('1')
                        mo=L.set_inner(Wt,newp[own]);h=hashf(mo);chk=dict(own_mask=own,hash_equal=(ref is not None and h==ref.get('hash')))
                        if ref is not None and 'W' in ref:
                            m_ref=L.load_state(ROOT/ref['W']['_file']);chk['max_abs_diff']=max(float(jnp.abs(x.astype(jnp.float32)-y.astype(jnp.float32)).max()) for x,y in zip(jax.tree_util.tree_leaves(mo.inner_parameters()),jax.tree_util.tree_leaves(m_ref.inner_parameters())));del m_ref
                        if sn=='W0':chk['also_P0s_hash_equal']=side['saved']['P0s'].get('1',{}).get('hash')==h
                        RES['checks'][f'{sn}_t{t}']=chk;lg('CHECK one step = stream',sn,t,chk);del mo
                    for nm_,np_ in newp.items():
                        mm=L.set_inner(Wt,np_);ent['steps'][nm_]['R1']=(r1(mm)-W0S).tolist();ent['steps'][nm_]['place_nll']=place_nll(mm,st_w,k,Kc,Vc,keep)
                        if r2 is not None and nm_ in MASKS:sc=r2.score(mm);ent['steps'][nm_]['R2']={v:sc[v].tolist() for v in sc}
                        del mm
                    del newp,r2;rec['chunks'].append(ent);st=advance(Wt,st,k)
                    lg(f'  seed {seed} state {sn} t {t} chunk {k}: |g| C0 {ent["steps"]["C0"]["gnorm"]:.3f} SV {ent["steps"]["SAMEVIS"]["gnorm"]:.3f}, cos {ent["cos_C0_SAMEVIS"]["total"]:.3f}')
                for mk in MASKS:                     # interval: 8 consecutive steps from W_t, engine rule for the suffix state
                    m=Wt;st=St;steps=[]
                    for j in range(NJ):
                        k=t+j;Kc,Vc,keep,st_w=win(k,st);newp_,d=step(m,st_w,k,Kc,Vc,keep,mk);d['update_norm']=L.norm_between(newp_,m);steps.append(d);st=advance(m,st,k);m=L.set_inner(m,newp_)
                    rec['interval'][mk]=dict(steps=steps,R1=(r1(m)-W0S).tolist());del m
                rec['sec']=time.monotonic()-t1;lg(f'seed {seed} state {sn} t {t}: {rec["sec"]:.0f}s')
                json.dump(RES,open(out/f'xstep_s{seed}.json','w'),default=float)
            del states
    json.dump(RES,open(out/f'xstep_s{seed}.json','w'),default=float);lg('wrote',out/f'xstep_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
    if a.check:
        bad={k_:v for k_,v in RES['checks'].items() if not v['hash_equal']};lg('CHECKS one step = stream:',RES['checks'])
        assert not bad,('one step from W_t with the own mask differs from the stream',bad)
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--states-dir');ap_.add_argument('--ts',default='0,32,64,96');ap_.add_argument('--nj',type=int,default=8);ap_.add_argument('--r2-offsets',default='0,2,4,6');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
