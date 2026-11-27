"""Relay path closed DURING writing (job 1360 'relay2'; GPU; NEW file; implementation thread 11/26; design: Hans 11/26 23:58 KST, E59 [S2626-RELAY2] / [S2626-RELAY2-OK]; EXPLORATORY). Same structure as job 1358b (e59_xstep.py): stage-1 states of job 1358a (W_t, S_t of the C0 stream P1s and the SAMEVIS stream P0s, t = 32 / 64 / 96), every chunk c = t..t+7 one step from W_t, the suffix state advanced with W_t and no update.
Steps per chunk (current-chunk keep = C0 for all; same lr, clip, loss as the stream):
  C0   : the natural step (carried prefix window = natural W0 record, suffix state = stream S_t advanced) = the 1358b C0 step (check: chunk t = stream W_{t+1} hash).
  D    : prefix window = the D re-record (chunks 0.. re-recorded chunk by chunk, no row sees any fact-sentence key of earlier or the same chunk, positions kept; e59_relay.keep_Drec), suffix state = chain from chunk 0 with W_t over the D prefix outputs with the D keep in the suffix layers.
  C0x  : numerics / chain control for D: prefix window = the chunk-by-chunk re-record with NO mask (same xla attention path as D), suffix state = chain from chunk 0 with W_t over those prefix outputs, no suffix mask. D - C0x isolates the deletion; C0x - C0 shows the numerics (xla chunked re-record vs the cudnn natural record, job 1359 smoke: K max abs 0.77 of scale 12.4 over 10 chunks) plus the W_t-chain approximation.
  D_sized : the D update rescaled to the update norm of the C0 step (auxiliary, no extra backward).
The prefix layers are frozen at W0, so the two re-records do not depend on the state or on t: they are computed once per seed (up to chunk max(t)+7) and shared.
Reads / storage: R1 16 x 16 candidate scores (12 exposed + 4 never exposed) minus W0 at W_t and after every step; facts of the chunk (all 16 in U12); place-token NLL of the 16 read facts at W_t and after every step under the C0, C0x and D conditions; per step |g| before clipping, clip coef, CE, update norm; cosines of the C0, C0x, D updates (total, per leaf); the [S2626-RELAY] A-D measures at W_t = place NLL C0 vs D condition and gradient cosine C0 vs D (these are the step gradients).
Checks (--check): re-record layer-0 K equals the natural record up to bf16 rounding (<= 2% of the K scale; a position / RoPE error is of the order of the scale); the layer-loop prefix output of one chunk with the natural window equals the engine prefix_masked (same xla path); deeper re-record differences are logged (not a gate), beside the engine's own prefix_masked-vs-native difference; attention on the hidden D pairs = 0 is guaranteed by the keep (CPU check); the C0 step of chunk t = stream W_{t+1} hash.
  python3 e59_relay2.py --seed 231 --states-dir S1DIR --out O [--ts 32,64,96] [--nj 8] [--check] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
from e59_relay import keep_Drec,keep_C0,NREAD,CH,NW,WIN
from e59_xstep import facts_in_chunk,cosine_parts,STATES
STEPS=('C0','C0x','D');COND_OF={'C0':'N','C0x':'X','D':'D'}
def selftest():
    import e59_relay as RL;RL.selftest()
    chunks=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=20,start=50,end=60)]} for _ in range(12)];base=np.ones(WIN+CH,bool)
    D=keep_Drec(chunks,3,base);A=keep_C0(chunks,3,base);assert not (D&~np.broadcast_to(base,(CH,WIN+CH))).any() and (~D).sum()>(~A).sum()
    print('e59_relay2 selftest ok (keeps from e59_relay)')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_shadow as SH,s3
    from e59_kvpos2 import items2,NEXP,CONDS as KC
    from e59_xstep_save import load_suffix_state
    from ttt.model.data import Batch
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic();SD=pathlib.Path(a.states_dir)
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3_,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0;ch=w['chunks']
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);NC=G.NCAND;out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w))
    with rt32.mesh,jax.default_matmul_precision('highest'):T0=LD._TailCtx(rt32,list(out_tail));Pout=LD.packing.Packed(T0,raw,GP=2)
    def r1(m):
        with jax.default_matmul_precision('highest'):return np.asarray(Pout.r1(rd.to_fp32(m))).reshape(len(items),NC)[:NREAD]
    W0S=r1(rt.base);side=json.load(open(SD/f'xstep_save_s{seed}.json'));s1same=bool(side.get('same_as_1348',{}).get('same',False))
    L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}',canary=can);L.chunks_meta=ch
    TS=[int(x) for x in a.ts.split(',')];NJ=a.nj;KL=max(TS)+NJ
    # ---- chunk-by-chunk prefix re-recorder (as e59_relay.py) ----
    SWA=s3.SWA;_cur=SWA.full_sw_attention
    def _full_rec(self,hs,seq,state):
        if s3.S3.get('mode')=='ctxrec':
            q0=s3.S3['q0'];n=hs.shape[0];_q,xk,xv=self.get_attention_input(hs,position_ids=q0+jnp.arange(n,dtype=jnp.int32));s3.S3['rec']=(xk,xv);s3.S3['mode']='ctx'
            try:return _cur(self,hs,seq,state)
            finally:s3.S3['mode']='ctxrec'
        return _cur(self,hs,seq,state)
    SWA.full_sw_attention=_full_rec
    @eqx.filter_jit
    def layer_ctx(blk,h,Kc,Vc,q0,keep,ids_):
        s3.S3['mode']='ctxrec';s3.S3['q0']=q0;s3.S3['keep']=keep
        try:
            seq_=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool));o,_=blk(h,(Kc,Vc),seq_,is_prefix=True);K,V=s3.S3['rec']
        finally:s3.S3.clear()
        return o,K,V
    blocks=rt.base.language_model.model.h.prefix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0]      # per-layer slices (BL) dropped with the per-layer loop: the scan recorder needs none
    base_of=lambda k:(k*CH-WIN+np.arange(WIN+CH))>=0
    def keep_nat(chunks,k,base):return np.broadcast_to(np.asarray(base,bool),(CH,WIN+CH)).copy()
    @eqx.filter_jit
    def prefix_rec(m,ids_,Kst,Vst,q0,keep):
        """the engine's s3 prefix_masked (embedding + jax.lax.scan over the prefix blocks, 'ctx' attention), with the post-RoPE K,V of the chunk's own tokens returned per layer as scan outputs (1360 smoke 11/27: a per-layer jit loop differed from the scan by 3 bf16 ulps of the largest activation; the scan keeps the engine's graph)"""
        h=m.language_model.wte_call(ids_);blocks_=m.language_model.model.h.prefix_blocks
        s3.S3['mode']='ctxrec';s3.S3['q0']=q0;s3.S3['keep']=keep
        try:
            seq_=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool))
            def body(x,xs):
                blk,Kc,Vc=xs;y,_=blk(x,(Kc,Vc),seq_,is_prefix=True);K,V=s3.S3['rec'];return y,(K,V)
            h,(Ks,Vs)=jax.lax.scan(body,h,(blocks_,Kst,Vst))
        finally:s3.S3.clear()
        return h,Ks,Vs
    def chunk_forward(k,Kc,Vc,kp):
        ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);h,Ks,Vs=prefix_rec(rt.base,ids_k,Kc,Vc,jnp.int32(k*CH),kp);return h,Ks,Vs
    def masked_record(keepfn,upto,like):
        Kh=np.zeros_like(like[0][:,:upto*CH]);Vh=np.zeros_like(like[1][:,:upto*CH]);P_={};t0=time.monotonic()
        with rt.mesh:
            for k in range(upto):
                Kc,Vc=L.ctx_kv(Kh,Vh,k);h,Ks,Vs=chunk_forward(k,Kc,Vc,jnp.asarray(keepfn(ch,k,base_of(k))))
                Kh[:,k*CH:(k+1)*CH]=np.asarray(jax.device_get(Ks)).reshape(Kh[:,k*CH:(k+1)*CH].shape);Vh[:,k*CH:(k+1)*CH]=np.asarray(jax.device_get(Vs)).reshape(Vh[:,k*CH:(k+1)*CH].shape)
                P_[k]=h
        lg(f'  re-record {keepfn.__name__}: {upto} chunks {time.monotonic()-t0:.0f}s');return Kh,Vh,P_
    RES=dict(seed=seed,args=vars(a),W0_R1=W0S.tolist(),cand_exposed=[True]*NEXP+[False]*(NC-NEXP),stage1_same_as_1348=s1same,steps=list(STEPS)+['D_sized'],states={},checks={},design='S2626-RELAY2; C0x added as the numerics control of D (implementation thread)')
    with rt.mesh:
        seq=L._stream_seq(ids0,nch);prefsN=rt.prefix(rt.base,seq);t0=time.monotonic();KhN,VhN,_=L.record(ids0[:nch*CH]);lg(f'seed {seed}: natural record {time.monotonic()-t0:.0f}s')
    KhX,VhX,PX=masked_record(keep_nat,KL,(KhN,VhN));KhD,VhD,PD=masked_record(keep_Drec,KL,(KhN,VhN))
    f32=lambda x:np.asarray(x,np.float32)
    if a.check:
        with rt.mesh:
            c_={};c_['layer0_K_maxabs_rerecord_vs_natural']=float(np.abs(f32(KhX[0])-f32(KhN[0,:KL*CH])).max());c_['K_maxabs_by_layer_first_mid_last']=[float(np.abs(f32(KhX[l])-f32(KhN[l,:KL*CH])).max()) for l in (0,nl//2,nl-1)];c_['K_scale']=float(np.abs(f32(KhN[:,:KL*CH])).max())
            c_['K_maxabs_by_chunk_last_layer']=[float(np.abs(f32(KhX[nl-1,k*CH:(k+1)*CH])-f32(KhN[nl-1,k*CH:(k+1)*CH])).max()) for k in range(0,KL,8)]
            k=9;Kc,Vc=L.ctx_kv(KhN,VhN,k);kp=jnp.asarray(keep_nat(ch,k,base_of(k)));hL,_,_=chunk_forward(k,Kc,Vc,kp);hE=L.prefix_masked(rt.base,jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp);nat=prefsN[k*CH:(k+1)*CH]
            c_['chunk9_recorder_vs_engine_prefix_masked_maxabs']=float(jnp.abs(hL.astype(jnp.float32)-hE.astype(jnp.float32)).max());c_['chunk9_engine_prefix_masked_vs_native_maxabs']=float(jnp.abs(hE.astype(jnp.float32)-nat.astype(jnp.float32)).max());c_['chunk9_rerecord_vs_native_maxabs']=float(jnp.abs(PX[k].astype(jnp.float32)-nat.astype(jnp.float32)).max());c_['prefix_out_scale']=float(jnp.abs(nat.astype(jnp.float32)).max())
            c_['chunk9_engine_rel_mean']=float(jnp.abs(hE.astype(jnp.float32)-nat.astype(jnp.float32)).mean()/jnp.abs(nat.astype(jnp.float32)).mean());c_['chunk9_rerecord_rel_mean']=float(jnp.abs(PX[k].astype(jnp.float32)-nat.astype(jnp.float32)).mean()/jnp.abs(nat.astype(jnp.float32)).mean())
        RES['checks']['recorder']=c_;lg('CHECK recorder',c_)
        assert c_['layer0_K_maxabs_rerecord_vs_natural']<=0.02*c_['K_scale'],('layer-0 K differs beyond bf16 rounding (matmul tiling): positions / RoPE / ids',c_)      # a position / RoPE error is of the order of the scale
        assert c_['chunk9_recorder_vs_engine_prefix_masked_maxabs']<=1e-2*c_['prefix_out_scale'],('recorder scan differs from the engine prefix_masked',c_)
    # ---- write / read machinery ----
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
    @eqx.filter_jit
    def read_state_m(m,st,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:return m.language_model.suffix_call(prefix_outputs=pref_,state=st,seq=c).new_state
        finally:s3.S3SUF.clear()
    UVC0=SH.mk_uv(KC['C0'],w,seed,lambda *x:None);UVSV=SH.mk_uv(KC['SAMEVIS'],w,seed,lambda *x:None)
    leaf_names=[jax.tree_util.keystr(p_) for p_,_ in jax.tree_util.tree_flatten_with_path(rt.base.inner_parameters())[0]]
    hashf=lambda m:s3.reuse._hash_pytree(m.inner_parameters())[0]
    def delta(newp,m):return [np.asarray(jax.device_get(x.astype(jnp.float32)-y.astype(jnp.float32))) for x,y in zip(jax.tree_util.tree_leaves(newp),jax.tree_util.tree_leaves(m.inner_parameters()))]
    WINS={'N':(KhN,VhN),'X':(KhX,VhX),'D':(KhD,VhD)}
    def pref_of(cond,k,m,Kc,Vc,km):
        return L.prefix_masked(m,jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),km)
    def advance(cond,m,st,k):
        c=rt.chunk(seq,k)
        if cond=='N':return L.read_state(m,st,c,prefsN[k*CH:(k+1)*CH])
        if cond=='X':return L.read_state(m,st,c,PX[k])
        return read_state_m(m,st,c,PD[k],jnp.asarray(keep_Drec(ch,k,base_of(k))))
    with rt.mesh:
        for t in TS:
            for sn,arm in STATES.items():
                t1=time.monotonic();info=side['saved'][arm][str(t)];Wt=L.load_state(ROOT/info['W']['_file']);assert hashf(Wt)==info['W']['_hash'],('W_t hash',sn,t)
                St=load_suffix_state(ROOT/info['S']['_file'],s3.clone_state(rt.state_suffix))
                st={'N':St,'X':s3.clone_state(rt.state_suffix),'D':s3.clone_state(rt.state_suffix)}
                for k in range(t):st['X']=advance('X',Wt,st['X'],k);st['D']=advance('D',Wt,st['D'],k)
                lg(f'  seed {seed} {sn} t {t}: W_t chains X, D rebuilt over chunks 0..{t-1} {time.monotonic()-t1:.0f}s')
                rec=dict(t=t,R1_Wt=(r1(Wt)-W0S).tolist(),chunks=[]);RES['states'].setdefault(sn,{})[str(t)]=rec
                for j in range(NJ):
                    k=t+j;c=rt.chunk(seq,k);valid=np.asarray(c.loss_masks,np.float32);wt=jnp.asarray(valid/valid.sum());pos={d['fact']:[q-1 for q in d['place'] if 0<=q-1<CH] for d in ap[k] if d['fact']<NREAD}
                    ent=dict(k=k,facts=facts_in_chunk(ch,k),place_nll_Wt={},steps={});inp={}
                    for sname in STEPS:
                        cond=COND_OF[sname];Kc,Vc=L.ctx_kv(*WINS[cond],k);kp,_=UVC0.keep(k,Wt,Kc,Vc,ids0[k*CH:(k+1)*CH],np.ones(WIN+CH,bool));km=jnp.asarray(kp);pref=pref_of(cond,k,Wt,Kc,Vc,km);inp[sname]=(st[cond],pref,km,Kc,Vc)      # kp: the stream's NAT keep vector (e55_core.keep_vector: all True; k >= 32 so no absent slots)
                        nl_=np.asarray(nll_all(Wt,st[cond],c,pref,km));ent['place_nll_Wt'][sname]={str(f):[float(nl_[i]) for i in pos[f]] for f in range(NREAD)}
                    newp={}
                    for sname in STEPS:
                        stc,pref,km,_,_=inp[sname];g,nll=ref_grad(Wt,stc,c,pref,km,wt);newp[sname],gn,coef=L.sgd_apply(Wt,s3.clone_state(rt.state_all),g);del g
                        ent['steps'][sname]=dict(gnorm=float(gn),coef=float(coef),ce=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum()),update_norm=L.norm_between(newp[sname],Wt))
                    s_=ent['steps']['C0']['update_norm']/ent['steps']['D']['update_norm'] if ent['steps']['D']['update_norm']>0 else 1.0;newp['D_sized']=L.rescale(newp['D'],Wt,s_);ent['steps']['D_sized']=dict(scale=s_,update_norm=L.norm_between(newp['D_sized'],Wt))
                    dl={sname:delta(newp[sname],Wt) for sname in STEPS};ent['cos']={f'{x}-{y}':cosine_parts(dl[x],dl[y],leaf_names) for x,y in (('C0','D'),('C0x','D'),('C0','C0x'))};del dl
                    if j==0:
                        ref=side['saved'][arm].get(str(t+1),{})      # the stream's own step: C0 mask for the C0 stream (P1s), SAMEVIS mask for the SAMEVIS stream (P0s); 11/27 fix: the SAMEVIS state was compared with its C0 step
                        if sn=='C0':h=hashf(L.set_inner(Wt,newp['C0']))
                        else:
                            stc,_p,_k,Kc,Vc=inp['C0'];kpv,_=UVSV.keep(k,Wt,Kc,Vc,ids0[k*CH:(k+1)*CH],np.ones(WIN+CH,bool));kmv=jnp.asarray(kpv);gv,_n=ref_grad(Wt,stc,c,pref_of('N',k,Wt,Kc,Vc,kmv),kmv,wt);npv,_g,_c=L.sgd_apply(Wt,s3.clone_state(rt.state_all),gv);del gv;h=hashf(L.set_inner(Wt,npv));del npv
                        RES['checks'][f'{sn}_t{t}']=dict(own_mask_step_hash_equal_stream=(h==ref.get('hash')),own_mask=sn);lg('CHECK own-mask step = stream',sn,t,RES['checks'][f'{sn}_t{t}'])
                    for sname,np_ in newp.items():
                        mm=L.set_inner(Wt,np_);ent['steps'][sname]['R1']=(r1(mm)-W0S).tolist();pn={}
                        for s2 in ('C0','D'):
                            stc,pref,km,Kc,Vc=inp[s2];pref2=pref_of(COND_OF[s2],k,mm,Kc,Vc,km);nl_=np.asarray(nll_all(mm,stc,c,pref2,km));pn[s2]={str(f):[float(nl_[i]) for i in pos[f]] for f in range(NREAD)}
                        ent['steps'][sname]['place_nll']=pn;del mm
                    del newp,inp;rec['chunks'].append(ent)
                    for cond in ('N','X','D'):st[cond]=advance(cond,Wt,st[cond],k)
                    lg(f'  seed {seed} {sn} t {t} chunk {k}: |g| C0 {ent["steps"]["C0"]["gnorm"]:.3f} C0x {ent["steps"]["C0x"]["gnorm"]:.3f} D {ent["steps"]["D"]["gnorm"]:.3f}; |dW| C0 {ent["steps"]["C0"]["update_norm"]:.4f} D {ent["steps"]["D"]["update_norm"]:.4f}; cos C0-D {ent["cos"]["C0-D"]["total"]:.4f} C0x-D {ent["cos"]["C0x-D"]["total"]:.4f} C0-C0x {ent["cos"]["C0-C0x"]["total"]:.4f}')
                rec['sec']=time.monotonic()-t1;lg(f'seed {seed} state {sn} t {t}: {rec["sec"]:.0f}s');del Wt,St,st
                json.dump(RES,open(out/f'relay2_s{seed}.json','w'),default=float)
    json.dump(RES,open(out/f'relay2_s{seed}.json','w'),default=float);lg('wrote',out/f'relay2_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
    if a.check:
        bad={k_:v for k_,v in RES['checks'].items() if k_!='recorder' and not v['own_mask_step_hash_equal_stream']};assert not bad,('own-mask step from W_t differs from the stream',bad)
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--states-dir');ap_.add_argument('--ts',default='32,64,96');ap_.add_argument('--nj',type=int,default=8);ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
