"""Diagnosis of L2b - L2 (job 1364: +0.38 8/8) (job 1365 'l2diag', GPU; new file; implementation thread 11/28; Hans 11/28 18:33 / 18:42 KST: run it on the GPU; descriptive). Seeds 231-238, W0, chunks 0 / 9 / 17 / 25 / 31 (chunks > 0 with the 8-chunk D window re-recorded chunk by chunk as in job 1362), write masks L2 and L2b of e59_early1.py (L1 loss = read-fact sentence tokens).
  H2 (copying): head-averaged attention of the place-predicting rows (the rows that predict the place tokens of the 16 read facts) on: other read sentences' place tokens, their name tokens, the rest of the other read sentences, the own sentence, the carried window; prefix layers (all 24, the suffix layers are not instrumented).
  H1 (key mixing): prefix-output vectors at the place-predicting rows (first place token): mean cosine between DIFFERENT facts vs the cosine of a fact's place row to its own name row; per-fact gradients of the place-token loss (6 facts): mean cosine between different facts' gradients.
  Cross influence: one L1 step at chunk 0 from W0 with each mask, then R1 (neutral 8192 tail, 16 x 16): change of the gold score of each fact's own question vs of the same name scored under the other facts' questions (control term) = matched binding parts.
python e59_l2diag.py --seed S --out FILE [--chunks 0,9,17,25,31] [--ngrad 6] [--no-r1]   (CPU smoke: CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu, --chunks 0 --ngrad 2 --no-r1)"""
import sys,os,json,time,pathlib,argparse,random
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
ap_=argparse.ArgumentParser();ap_.add_argument('--out');ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--chunks',default='0,9');ap_.add_argument('--ngrad',type=int,default=6);ap_.add_argument('--no-r1',action='store_true');a=ap_.parse_args()
os.chdir(ROOT);T0=time.time();lg=lambda *x:print(f'[{time.time()-T0:6.0f}s]',*x,flush=True)
import jax,jax.numpy as jnp,equinox as eqx
if jax.default_backend()=='cpu':
    _dpa=jax.nn.dot_product_attention;jax.nn.dot_product_attention=lambda *a_,**k_:_dpa(*a_,**dict(k_,implementation='xla'))      # CPU smoke only (no cudnn); the GPU job keeps the engine kernels
import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_shadow as SH,s3
from e59_kvpos2 import items2,CONDS
from e59_relay import keep_Drec
from e59_early1 import keep_L,read_spans
from ttt.model.data import Batch
NM.register_engine();QR.K0=0;NM.K0=0;U.install();ci=OR.register_cell(U.CELL);LD,s3_,C=R55.lab();rt=LD.Runtime();seed=a.seed;CH=1024;NW=8;WIN=NW*CH;NREAD=16
enc,w,tf=R55.build(SP.synth_args(seed,U.CELL),seed,read_apps=set(ci['read_apps']),arms=['NAT']);S=R55.streams(w,enc);ids0=np.asarray(S['A'][0]);UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);ch=w['chunks'];tk=LP.tk_load();ap,miss,_=LP.answer_positions(w,tk,w['n_chunks']);assert miss==0
L=ISO.Iso55(rt,save_dir=str(pathlib.Path(a.out).parent/'l2diag_states'),canary=None);c0=SH.mk_uv(CONDS['C0'],w,seed,lambda *x:None);lg('runtime and world ready')
SWA=s3.SWA;_cur=SWA.full_sw_attention
def _full_rec(self,hs,seq,state):
    if s3.S3.get('mode')=='ctxrec':
        q0=s3.S3['q0'];n=hs.shape[0];_q,xk,xv=self.get_attention_input(hs,position_ids=q0+jnp.arange(n,dtype=jnp.int32));s3.S3['rec']=(xk,xv);s3.S3['mode']='ctx'
        try:return _cur(self,hs,seq,state)
        finally:s3.S3['mode']='ctxrec'
    return _cur(self,hs,seq,state)
SWA.full_sw_attention=_full_rec
@eqx.filter_jit
def prefix_rec(m,ids_,Kst,Vst,q0,keep):
    h=m.language_model.wte_call(ids_);blocks_=m.language_model.model.h.prefix_blocks
    s3.S3['mode']='ctxrec';s3.S3['q0']=q0;s3.S3['keep']=keep
    try:
        seq_=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool))
        def body(x,xs):
            blk,Kc,Vc=xs;y,_=blk(x,(Kc,Vc),seq_,is_prefix=True);K,V=s3.S3['rec'];return y,(K,V)
        h,(Ks,Vs)=jax.lax.scan(body,h,(blocks_,Kst,Vst))
    finally:s3.S3.clear()
    return h,Ks,Vs
@eqx.filter_jit
def read_state_m(m,st,c,pref_,kmask):
    s3.S3SUF['kmask']=kmask
    try:return m.language_model.suffix_call(prefix_outputs=pref_,state=st,seq=c).new_state
    finally:s3.S3SUF.clear()
def loss_fn(mm,st_suf,c,pref_,kmask,wt):
    s3.S3SUF['kmask']=kmask
    try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
    finally:s3.S3SUF.clear()
    lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
@eqx.filter_jit
def ref_grad(m,st_suf,c,pref_,kmask,wt):
    (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
base_of=lambda k:(k*CH-WIN+np.arange(WIN+CH))>=0
blocks=rt.base.language_model.model.h.prefix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0];BL=[jax.tree.map(lambda x:x[l] if eqx.is_array(x) else x,blocks) for l in range(nl)]
CHUNKS=[int(x) for x in a.chunks.split(',')];KL=max(CHUNKS);RES=dict(seed=seed,chunks=CHUNKS,per_chunk={})
with rt.mesh:
    seq=L._stream_seq(ids0,KL+1);nat=None
    Kh=Vh=None;PD={};st={'D':s3.clone_state(rt.state_suffix)}
    for k in range(KL+1):
        if Kh is None:
            Kn0,Vn0,_=L.record(ids0[:CH]);Kh=np.zeros((Kn0.shape[0],(KL+1)*CH)+Kn0.shape[2:],Kn0.dtype);Vh=np.zeros_like(Kh)
        Kc,Vc=L.ctx_kv(Kh,Vh,k);h,Ks,Vs=prefix_rec(rt.base,jnp.asarray(ids0[k*CH:(k+1)*CH],jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_Drec(ch,k,base_of(k))))
        Kh[:,k*CH:(k+1)*CH]=np.asarray(Ks).reshape(Kh[:,k*CH:(k+1)*CH].shape);Vh[:,k*CH:(k+1)*CH]=np.asarray(Vs).reshape(Vh[:,k*CH:(k+1)*CH].shape);PD[k]=h
        if k in CHUNKS:st[k]=st['D']
        st['D']=read_state_m(rt.base,st['D'],rt.chunk(seq,k),h,jnp.asarray(keep_Drec(ch,k,base_of(k))))
    lg(f'D window recorded over chunks 0..{KL}')
    leafn=lambda g:[np.asarray(x,np.float32).ravel() for x in jax.tree_util.tree_leaves(g)]
    def gcos(ga,gb):
        d=sum(float(np.dot(x,y)) for x,y in zip(ga,gb));na=sum(float(np.dot(x,x)) for x in ga);nb=sum(float(np.dot(y,y)) for y in gb);return d/np.sqrt(na*nb)
    for k in CHUNKS:
        Kc,Vc=L.ctx_kv(Kh,Vh,k);ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],jnp.int32);kp0=np.asarray(c0.keep(k,None,Kc,Vc,None,np.ones(WIN+CH,bool))[0],bool);c=rt.chunk(seq,k)
        sents={int(x['fact']):(int(x['start']),int(x['end'])) for x in ch[k]['sents'] if int(x['fact'])<NREAD};apk={d['fact']:d for d in ap[k] if d['fact']<NREAD}
        prow={f:[p-1 for p in apk[f]['place'] if p>=1] for f in range(NREAD)};nrow={f:[p-1 for p in apk[f]['name'] if p>=1] for f in range(NREAD)}
        rows=np.asarray(sorted(set(r for f in range(NREAD) for r in prow[f])),np.int32);ri={int(r):i for i,r in enumerate(rows)};ent={}
        for mode in ('L2','L2b'):
            kp=keep_L(kp0,ch[k],mode);t1=time.time()
            s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(kp);s3.S3['att_rows']=jnp.asarray(rows);s3.S3['att_out']=[];SWA.full_sw_attention=_cur
            try:
                hh=rt.base.language_model.wte_call(ids_k);sq=Batch(input_ids=ids_k,target_tokens=ids_k,loss_masks=jnp.ones_like(ids_k,dtype=bool))
                for l in range(nl):hh,_=BL[l](hh,(Kc[l],Vc[l]),sq,is_prefix=True)
                per=[np.asarray(x) for x in s3.S3['att_out']]
            finally:s3.S3.clear();SWA.full_sw_attention=_full_rec
            att={nm:[] for nm in ('other_place','other_name','other_rest','own_sentence','window','other_current')}
            for f in range(NREAD):
                a0,b0=sents[f];oth=[g for g in range(NREAD) if g!=f];op=np.asarray([WIN+p for g in oth for p in apk[g]['place']]);on=np.asarray([WIN+p for g in oth for p in apk[g]['name']])
                osent=np.asarray([WIN+t for g in oth for t in range(*sents[g])]);rest=np.setdiff1d(osent,np.concatenate([op,on]));own=np.arange(WIN+a0-1,WIN+b0);cur=np.arange(WIN,WIN+CH);ocur=np.setdiff1d(cur,np.concatenate([own,osent]))
                for r in prow[f]:
                    i=ri[r];v={nm:np.mean([p_[i][cols].sum() for p_ in per]) for nm,cols in (('other_place',op),('other_name',on),('other_rest',rest),('own_sentence',own),('window',np.arange(WIN)),('other_current',ocur))}
                    for nm in att:att[nm].append(float(v[nm]))
            att={nm:float(np.mean(v)) for nm,v in att.items()}
            pref=L.prefix_masked(rt.base,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(kp));H=np.asarray(pref,np.float32)
            P0=np.stack([H[prow[f][0]] for f in range(NREAD)]);N0=np.stack([H[nrow[f][-1]] for f in range(NREAD)]);nz=lambda X:X/np.linalg.norm(X,axis=1,keepdims=True);Pn=nz(P0);Nn=nz(N0)
            cpp=Pn@Pn.T;off=cpp[~np.eye(NREAD,dtype=bool)].mean();own_pn=float(np.mean(np.sum(Pn*Nn,1)));cross_pn=float((Pn@Nn.T)[~np.eye(NREAD,dtype=bool)].mean())
            G_={};
            for f in range(a.ngrad):
                wt=np.zeros(CH,np.float32)
                for p in apk[f]['place']:
                    if p>=1:wt[p-1]=1.0
                wt/=wt.sum();g,_=ref_grad(rt.base,st[k],c,pref,jnp.asarray(kp),jnp.asarray(wt));G_[f]=leafn(g);del g
            gc=[gcos(G_[i],G_[j]) for i in range(a.ngrad) for j in range(i+1,a.ngrad)]
            ent[mode]=dict(att_prefix_mean_over_layers=att,prefix_out_cos_place_rows_between_facts=float(off),cos_place_row_to_own_name_row=own_pn,cos_place_row_to_other_name_rows=cross_pn,grad_cos_between_facts_mean=float(np.mean(gc)),grad_cos_between_facts=gc,sec=time.time()-t1)
            G_.clear();lg(f'chunk {k} {mode}:',json.dumps({kk:(round(v,4) if isinstance(v,float) else v) for kk,v in ent[mode].items() if kk not in ('grad_cos_between_facts',)}))
        RES['per_chunk'][str(k)]=ent;json.dump(RES,open(a.out,'w'),default=float)
    if not a.no_r1:
        rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
        P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w))
        with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));Pout=LD.packing.Packed(T_,raw,GP=2)
        r1=lambda m:np.asarray(Pout.r1(rd.to_fp32(m))).reshape(len(items),G.NCAND)[:NREAD]
        t1=time.time();W0S=r1(rt.base);lg(f'R1 at W0 {time.time()-t1:.0f}s');k=0;Kc,Vc=L.ctx_kv(Kh,Vh,k);ids_k=jnp.asarray(ids0[:CH],jnp.int32);kp0=np.asarray(c0.keep(0,None,Kc,Vc,None,np.ones(WIN+CH,bool))[0],bool);c=rt.chunk(seq,0)
        from e59_early1 import loss_positions
        lm=loss_positions(ch,1).astype(np.float32);wt=jnp.asarray(lm/lm.sum());RES['cross']={}
        def _cands(sd):
            out=[]
            for f in range(16):
                r=random.Random(f"kvpos2|{sd}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
            return out
        cf=_cands(seed)
        for mode in ('L2','L2b'):
            kp=jnp.asarray(keep_L(kp0,ch[0],mode));pref=L.prefix_masked(rt.base,ids_k,Kc,Vc,jnp.int32(0),kp);g,_=ref_grad(rt.base,st[0] if 0 in st else s3.clone_state(rt.state_suffix),c,pref,kp,wt);newp,gn,coef=L.sgd_apply(rt.base,s3.clone_state(rt.state_all),g);del g
            dS=r1(L.set_inner(rt.base,newp))-W0S;gold=[dS[f,0] for f in range(16)];ctl=[np.mean([dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]]) for f in range(16)]
            RES['cross'][mode]=dict(gnorm=float(gn),coef=float(coef),gold_rise=float(np.mean(gold)),control_rise=float(np.mean(ctl)),matched_binding=float(np.mean(np.asarray(gold)-np.asarray(ctl))),exposed_wrong_rise=float(dS[:,1:12].mean()),never_exposed=float(dS[:,12:16].mean()),dS=dS.tolist());lg(f'cross {mode}:',{kk:round(v,4) for kk,v in RES['cross'][mode].items() if kk!='dS'})
            json.dump(RES,open(a.out,'w'),default=float)
lg('done',a.out)
