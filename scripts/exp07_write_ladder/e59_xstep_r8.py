"""Cross one-step, 8-chunk WINDOW reads (job 1361 'xstep_r8'; GPU; NEW file; implementation thread 11/27; request: Hans 11/27 13:12 KST "8청크랑 8청크에서 같은 사실 가리기 모든 사실 가리기 이렇게 해서 읽기만 다시 해봐", coordinator reading: re-do the 1358b natural steps from the saved W_t and read before / after; EXPLORATORY).
Same states, arms and chunk offsets as job 1358b (e59_xstep.py): W0 (t = 0) and the 1358a states W_t of the C0 stream (P1s) and the SAMEVIS stream (P0s), t = 32 / 64 / 96; chunk offsets j = 0, 2, 4, 6 (the 1358b R2 offsets); steps C0 natural and SAMEVIS natural from W_t (same code path as 1358b; suffix state S_t advanced with W_t; no size-matched step, no R1 ladder, no interval run).
Reads (Hans 11/27 13:15 KST, two read-only contexts instead of hiding): every question scored on all 16 x 16 candidates (12 exposed + 4 never exposed), fresh suffix state, nothing hidden; context =
  A  : the 8 written chunks as they are (chunks k-7..k ending with the just-written chunk k, 8192 tokens; fewer chunks when k < 7);
  B2 : the same 8 chunks with their filler in place and every fact-sentence slot (all 32 facts) refilled, token for token, with filler (book text) taken in order from far chunks (|d - k| >= 16, starting at chunk (k + 64) mod 128): same length, positions, separators and filler as A, no fact information. Stored per context: share of refilled tokens, hits of any fact name / place / unexposed place token sequence (must be 0), W0 per-token NLL of the context (filler positions and slot positions separately).
Read at W_t (once per state, t and offset) and after each step. --wt-only: only the W_t reads (no steps); --wt-from DIR: a later full run takes the W_t reads from that wt-only output instead of re-scoring them.
Check: R1 after the C0 and SAMEVIS steps of the first offset of every (state, t) is recomputed and compared with the saved 1358b R1 (max abs difference; the step code is the 1358b one).
  python3 e59_xstep_r8.py --seed 231 --states-dir S1DIR --xstep-dir XDIR --out O [--ts 0,32,64,96] [--offsets 0,2,4,6] [--wt-only] [--check] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
NREAD=16;CH=1024;NW=8;SEP=271      # llama3 token '\n\n'
STATES={'C0':'P1s','SAMEVIS':'P0s'};MASKS=('C0','SAMEVIS');CTXS=('A','B2')
def ctx_chunks(k):return list(range(max(0,k-NW+1),k+1))
def hide_sets8(chunks,k):
    """-> {variant: bool [NREAD, TL]} over the context chunks k-7..k (TL = 1024 x number of context chunks)"""
    cc=ctx_chunks(k);TL=len(cc)*CH;own=np.zeros((NREAD,TL),bool);allm=np.zeros(TL,bool)
    for i,c in enumerate(cc):
        for x in chunks[c]['sents']:
            f=int(x['fact'])
            if f<NREAD:a,b=i*CH+int(x['start']),i*CH+int(x['end']);own[f,a:b]=True;allm[a:b]=True
    return {'R8':np.zeros((NREAD,TL),bool),'R8m':own,'R8a':np.broadcast_to(allm,(NREAD,TL)).copy()}
def filler_stream(chunks):
    """all filler (non-sentence) tokens of the stream in order -> (token ids, origin chunk)"""
    t=[];o=[]
    for c,x in enumerate(chunks):
        sm=np.zeros(CH,bool)
        for s_ in x['sents']:sm[int(s_['start']):int(s_['end'])]=True
        ids=np.asarray(x['ids']);keep_=(~sm)&(ids!=SEP);t.append(ids[keep_]);o.append(np.full(int(keep_.sum()),c))      # the '\n\n' separators between inserted sentences are dropped from the donor text (else the refilled slots are runs of separators)
    return np.concatenate(t),np.concatenate(o)
def ctx_ids(chunks,k,kind,FILL=None):
    """-> (context token ids, bool slot mask = refilled fact-sentence positions)"""
    cc=ctx_chunks(k);ids=np.concatenate([np.asarray(chunks[c]['ids']) for c in cc]);slot=np.zeros(len(ids),bool)
    for i,c in enumerate(cc):
        for x in chunks[c]['sents']:slot[i*CH+int(x['start']):i*CH+int(x['end'])]=True
    if kind=='A':return ids,slot
    tok,org=FILL;n=len(org);ok=np.abs(org-k)>=16;start=int(np.searchsorted(org,(k+64)%len(chunks)));order=np.r_[start:n,0:start];order=order[ok[order]];need=int(slot.sum());assert len(order)>=need
    out=ids.copy();out[slot]=tok[order[:need]];return out,slot
def name_hits(ids,seqs):
    ids=list(int(x) for x in ids);h=0
    for q in seqs:
        L=len(q)
        for i in range(len(ids)-L+1):
            if ids[i:i+L]==q:h+=1
    return h
def selftest():
    chunks=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=1,start=30,end=40),dict(fact=20,start=50,end=60)]} for _ in range(12)]
    assert ctx_chunks(9)==list(range(2,10)) and ctx_chunks(3)==[0,1,2,3]
    H=hide_sets8(chunks,9);assert H['R8m'].shape==(16,8192) and H['R8m'][0].sum()==80 and H['R8m'][1].sum()==80 and H['R8m'][2].sum()==0 and H['R8a'][5].sum()==160 and not H['R8'].any()
    assert H['R8m'][0,7*CH+10:7*CH+20].all() and not H['R8a'][0,50:60].any()
    H3=hide_sets8(chunks,3);assert H3['R8m'].shape==(16,4096)
    ch2=[{'ids':list(range(1000+c*CH,1000+(c+1)*CH)),'sents':[dict(fact=0,start=10,end=20)]} for c in range(40)];FILL=filler_stream(ch2);assert len(FILL[0])==40*(CH-10)
    a,sa=ctx_ids(ch2,20,'A');b,sb=ctx_ids(ch2,20,'B2',FILL);assert len(a)==len(b)==8*CH and (sa==sb).all() and sa.sum()==80 and (a[~sa]==b[~sa]).all() and not (a[sa]==b[sa]).any()
    src=(b[sa]-1000)//CH;assert all(abs(int(c)-20)>=16 for c in src) and len(set(b[sa].tolist()))==80
    assert name_hits([1,2,3,1,2],[[1,2],[3]])==3
    print('e59_xstep_r8 selftest ok: context chunks, hide sets R8 / R8m / R8a')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_shadow as SH,s3
    from e59_kvpos2 import items2,NEXP,CONDS
    from e59_xstep_save import load_suffix_state
    from ttt.model.data import Batch
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic();SD=pathlib.Path(a.states_dir)
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3_,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();PS=LD.packing
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0;ch=w['chunks']
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);NC=G.NCAND;out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w))
    with rt32.mesh,jax.default_matmul_precision('highest'):T0=LD._TailCtx(rt32,list(out_tail));Pout=LD.packing.Packed(T0,raw,GP=2)
    def r1(m):
        with jax.default_matmul_precision('highest'):return np.asarray(Pout.r1(rd.to_fp32(m))).reshape(len(items),NC)[:NREAD]
    WTF=json.load(open(pathlib.Path(a.wt_from)/f'xstep_r8_s{seed}.json')) if a.wt_from else None;W0S=r1(rt.base);side=json.load(open(SD/f'xstep_save_s{seed}.json'));XB=json.load(open(pathlib.Path(a.xstep_dir)/f'xstep_s{seed}.json')) if a.check else None
    # ---- 8-chunk context reads (e59_kvwin.py read path, context length generalised) ----
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
    FILL=filler_stream(ch);FORB=[]
    for f_ in w['facts']:
        for nm_ in (f_['X'],f_['P']):FORB+=[tk.encode(' '+nm_,add_special_tokens=False).ids,tk.encode(nm_,add_special_tokens=False).ids]
    for d_ in (UX if isinstance(UX,(list,tuple)) else []):
        nm_=d_ if isinstance(d_,str) else (d_.get('P') if isinstance(d_,dict) else None)
        if nm_:FORB+=[tk.encode(' '+nm_,add_special_tokens=False).ids,tk.encode(nm_,add_special_tokens=False).ids]
    FORB=[q for q in FORB if q]
    class R8Ctx:
        def __init__(self,k,kind):
            self.k=k;cc=ctx_chunks(k);cid,slot=ctx_ids(ch,k,kind,FILL);ctx=[int(t) for t in cid];H={'R':np.zeros((NREAD,len(ctx)),bool)}
            self.stats=dict(kind=kind,n_tokens=len(ctx),slot_share=float(slot.mean()),name_hits=name_hits(cid,FORB) if kind=='B2' else None)
            if kind=='B2':assert self.stats['name_hits']==0,('fact name / place token sequence in the B2 context',k,self.stats)
            with rt32.mesh,jax.default_matmul_precision('highest'):self.T=LD._TailCtx(rt32,ctx);self.PK=PS.Packed(self.T,raw,GP=2)
            PK=self.PK;TL=len(self.T.tail_ids);assert TL==len(cc)*CH and TL<=Wsl;npad=PK.ids.shape[0];seg_np=np.asarray(PK.seg);loc_np=np.asarray(PK.loc);ids_np=np.asarray(PK.ids);qmap=[dict() for _ in range(npad)];cnt={}
            for r,(c,_a,_b) in enumerate(PK.where):s=cnt.get(c,0);cnt[c]=s+1;qmap[c][s]=r//NC
            for c in range(PK.nchunk,npad):qmap[c]=qmap[PK.nchunk-1]
            self.VAR={}
            for name in H:
                Hv=H[name];pref=[];hide=np.zeros((npad,CH,Wsl),bool)
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
        def w0_nll(self):
            """W0 per-token NLL of the context (chunk_logp over the tail chunks), mean over filler positions and over slot positions"""
            T_=self.T;st=PS._clone(rt32.state_suffix);lp=[]
            with rt32.mesh:
                for j in range(len(T_.tail_ids)//CH):st,l_=rt32.chunk_logp(rt32.base,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*CH:(j+1)*CH]);lp.append(np.asarray(l_,np.float64)[:CH])
            return np.concatenate(lp)
        def score(self,m):
            m32=rd.to_fp32(m);T_=self.T;PK=self.PK;npad=PK.ids.shape[0];res={}
            with rt32.mesh:
                st=PS._clone(rt32.state_suffix)
                for j in range(len(T_.tail_ids)//CH):st,_=rt32.chunk_logp(m32,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*CH:(j+1)*CH])
            with rt32.mesh,jax.default_matmul_precision('highest'):
                for nm,(pref,hide) in self.VAR.items():
                    o=[]
                    for g in range(0,npad,PK.GP):o.append(np.asarray(lp_hide(m32,st,PK.ids[g:g+PK.GP],PK.tg[g:g+PK.GP],PK.mk[g:g+PK.GP],pref[g:g+PK.GP],PK.seg[g:g+PK.GP],PK.loc[g:g+PK.GP],hide[g:g+PK.GP])))
                    lp=np.concatenate(o);res=np.asarray([lp[c,a_:b_].sum() for c,a_,b_ in PK.where]).reshape(len(items),NC)[:NREAD].tolist()
            del m32;return res
    # ---- the 1358b step (e59_xstep.py, unchanged logic) ----
    L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}',canary=can);L.chunks_meta=ch
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    UV={mk:SH.mk_uv(CONDS[mk],w,seed,lambda *x:None) for mk in MASKS};hashf=lambda m:s3.reuse._hash_pytree(m.inner_parameters())[0]
    TS=[int(x) for x in a.ts.split(',')];OFF=[int(x) for x in a.offsets.split(',')];NJ=max(OFF)+1
    RES=dict(seed=seed,args=vars(a),W0_R1=W0S.tolist(),cand_exposed=[True]*NEXP+[False]*(NC-NEXP),contexts=list(CTXS),wt_only=a.wt_only,states={},checks={},ctx_stats={},W0_ctx={})
    with rt.mesh:
        seq=L._stream_seq(ids0,nch);prefs=rt.prefix(rt.base,seq);Kh,Vh,_=L.record(ids0[:nch*CH]);recs={'real':(Kh,Vh)};st_all=s3.clone_state(rt.state_all)
        def win(k,st):Kc,Vc,keep,st_w,pl=L.window_inputs('NAT',k,1,recs,{'real':st},[]);return Kc,Vc,keep,st_w
        def advance(m,st,k):return L.read_state(m,st,rt.chunk(seq,k),prefs[k*CH:(k+1)*CH])
        def step(m,st_w,k,Kc,Vc,keep,mk):
            c=rt.chunk(seq,k);ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);kp,_=UV[mk].keep(k,m,Kc,Vc,ids0[k*CH:(k+1)*CH],keep);km=jnp.asarray(kp);pref=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),km)
            valid=np.asarray(c.loss_masks,np.float32);wt=jnp.asarray(valid/valid.sum());g,nll=ref_grad(m,st_w,c,pref,km,wt);newp,gn,coef=L.sgd_apply(m,st_all,g);del g
            return newp,dict(gnorm=float(gn),coef=float(coef),ce=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum()))
        for t in TS:
            if t==0:states={'W0':(rt.base,s3.clone_state(rt.state_suffix))}
            else:
                states={}
                for sn,arm in STATES.items():
                    info=side['saved'][arm][str(t)];m=L.load_state(ROOT/info['W']['_file']);assert hashf(m)==info['W']['_hash'],('W_t hash',sn,t);states[sn]=(m,load_suffix_state(ROOT/info['S']['_file'],s3.clone_state(rt.state_suffix)))
            stc={sn:v[1] for sn,v in states.items()};recs_t={sn:dict(t=t,chunks=[]) for sn in states}
            for j in range(NJ):
                k=t+j
                if j in OFF:
                    t1=time.monotonic();CX={kd:R8Ctx(k,kd) for kd in CTXS}
                    if str(k) not in RES['ctx_stats']:
                        _,slot=ctx_ids(ch,k,'A');st_={}
                        for kd,cx_ in CX.items():
                            nl_=-cx_.w0_nll();sl=np.r_[slot[1:],False][:len(nl_)];st_[kd]=dict(cx_.stats,W0_nll_filler=float(nl_[~sl].mean()),W0_nll_slots=float(nl_[sl].mean()))
                        RES['ctx_stats'][str(k)]=st_;RES['W0_ctx'][str(k)]={kd:cx_.score(rt.base) for kd,cx_ in CX.items()};lg(f'  seed {seed} chunk {k}: contexts',{kd:{q:(round(v,3) if isinstance(v,float) else v) for q,v in d_.items()} for kd,d_ in st_.items()})
                    lg(f'  seed {seed} t {t} chunk {k}: 8-chunk contexts built ({len(ctx_chunks(k))} chunks) {time.monotonic()-t1:.0f}s')
                    for sn,(Wt,_) in states.items():
                        prev=WTF.get('states',{}).get(sn,{}).get(str(t),{}).get('chunks') if WTF else None;pe=[e for e in (prev or []) if e['k']==k]
                        ent=dict(k=k,ctx_chunks=ctx_chunks(k),reads_Wt=pe[0]['reads_Wt'] if pe else {kd:cx_.score(Wt) for kd,cx_ in CX.items()},reads_Wt_from='wt-only run' if pe else 'this run',steps={})
                        if not a.wt_only:
                            Kc,Vc,keep,st_w=win(k,stc[sn])
                            for mk in MASKS:
                                newp,d=step(Wt,st_w,k,Kc,Vc,keep,mk);d['update_norm']=L.norm_between(newp,Wt);mm=L.set_inner(Wt,newp);d['reads']={kd:cx_.score(mm) for kd,cx_ in CX.items()}
                                if a.check and j==OFF[0]:
                                    ref=XB['states'][sn][str(t)]['chunks'][j]['steps'][mk];d['R1_vs_1358b_maxabs']=float(np.abs((r1(mm)-W0S)-np.asarray(ref['R1'])).max());d['gnorm_vs_1358b']=abs(d['gnorm']-ref['gnorm']);RES['checks'][f'{sn}_t{t}_{mk}']=dict(R1_maxabs=d['R1_vs_1358b_maxabs'],gnorm_absdiff=d['gnorm_vs_1358b']);lg('CHECK step = 1358b',sn,t,mk,RES['checks'][f'{sn}_t{t}_{mk}'])
                                ent['steps'][mk]=d;del mm,newp
                        recs_t[sn]['chunks'].append(ent)
                    lg(f'  seed {seed} t {t} chunk {k}: reads done {time.monotonic()-t1:.0f}s');del CX
                if not a.wt_only:
                    for sn,(Wt,_) in states.items():stc[sn]=advance(Wt,stc[sn],k)
            for sn in states:RES['states'].setdefault(sn,{})[str(t)]=recs_t[sn]
            json.dump(RES,open(out/f'xstep_r8_s{seed}.json','w'),default=float);del states,stc
    json.dump(RES,open(out/f'xstep_r8_s{seed}.json','w'),default=float);lg('wrote',out/f'xstep_r8_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
    if a.check and not a.wt_only:
        bad={k_:v for k_,v in RES['checks'].items() if v['R1_maxabs']>1e-3 or v['gnorm_absdiff']>1e-4};assert not bad,('re-done step differs from 1358b',bad)
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--states-dir');ap_.add_argument('--xstep-dir',default='experiments/exp07_write_ladder/xstep');ap_.add_argument('--ts',default='0,32,64,96');ap_.add_argument('--offsets',default='0,2,4,6');ap_.add_argument('--wt-only',action='store_true');ap_.add_argument('--wt-from',default=None);ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
