"""[1344t1 = e59_kvmulti2.py: copy of e59_kvmulti.py with the arms S2As-u / E2As (A-token logit cotangent from the C0 forward, linear surrogate loss); all else as 1344t0] Multi-step real writing, two arms (job 1344t0; GPU; new file derived from e59_kvpos2.py / e59_kvlogit.py, both UNCHANGED; thread-server 11/25; design: Hans / design thread (documents 1/2, 2/2), EXPLORATORY; NOT queued: Hans decides). Question: when the real write is CONTINUED from W0 over 128 chunks with the C0 mask or the SAMEVIS mask in every chunk (original clipped SGD, natural clip, same learning rate; the mask of every chunk is rebuilt from the chunk's own sentences exactly as in the one-step jobs 1344p / 1344p2 / 1344s), do the answer-matched bindings of the two streams separate, and from which exposure on? World = the U12 world of 1344p2 / 1344s (E2E 3B, 32 facts = 16 read facts + 16 extra facts, one sentence per fact per chunk, so exposure e = chunk e-1), read items = the 16 read facts with 12 exposed + 4 never-exposed (fresh_places) candidates (e59_kvpos2.items2). Arms: P0s = SAMEVIS mask in every chunk (('ha','OTHERONLY'): the same fact visible, all other facts hidden), P1s = C0 mask in every chunk (SAME_U1U2: everything earlier hidden). Reads R1 (neutral out-of-window tail + question) right after exposures 8 / 16 / 32 / 64 / 96 / 128, R2 (the just-written chunk, unmasked, as the context + question; W0 of the same context stored) right after the last exposure; stored: dS (16 x 16 candidate scores minus W0) and W0 scores per read, candidates recreated from the seed by items2. [1344t0 diagnostics also store, for the place tokens, cos(e_own, e_other) and the share of the absolute error mass that sits on the first tokens of the 16 read places under each mask] Per chunk: pre-clip gradient norm, applied clip coefficient, chunk loss; every 8th chunk (before the write, weights of that moment): ||e|| (float32, no subtraction), NLL of the 2 place tokens of the 16 read facts under BOTH masks (one forward reused, one extra), sentence order per fact. Divergence flag (not finite, or mean chunk CE of the last 8 chunks > 4.0) is stored, never used to exclude. Final weights (after exposure 128) per arm and seed can be saved (--save-final).
  python3 e59_kvmulti.py --seed 231 --out O [--nch 128] [--expo 8,16,32,64,96,128] [--save-final] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_shadow as SH
import e59_kvlogit as KL
from e59_kvpos2 import items2,NEXP,NREAD,CH,CONDS,NL
ARMS={'S2As-u':'SAMEVIS','E2As':'SAMEVIS','E2All':'SAMEVIS'};MODE={'S2As-u':'S','E2As':'E','E2All':'ALL'};KP=48     # E2All (job 1344t2): the cotangent of EVERY token with a loss is wt*e_C0                           # both arms write with the SAMEVIS graph; only the logit cotangent of the A tokens (place tokens of the 16 read facts, role definition) differs
class UVRec:
    """wraps a mask object: remembers the arguments of the last keep() call (the chunk's natural window K,V and base visibility) so that the OTHER condition's mask can be built for the diagnostics"""
    def __init__(self,u):self.u=u;self.stash=None
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):self.stash=(k,Kc,Vc,keep_all);return self.u.keep(k,m,Kc,Vc,ids_k,keep_all)
    def __getattr__(self,n):return getattr(self.u,n)
def place_masks(k,T,ap):
    """NLL-index mask of the 2 place tokens of the 16 read facts of chunk k and the sentence order (0 place first, 1 name first) per read fact"""
    m=np.zeros(T,bool);od=[None]*NREAD
    for d in ap[k]:
        if d['fact']>=NREAD:continue
        od[d['fact']]=0 if d['order']=='place_first' else 1
        for t in d['place']:
            if 0<=t-1<T:m[t-1]=True
    return m,od
def selftest():
    ap=[[dict(fact=0,place=[8,9],name=[2,3],order='place_first'),dict(fact=1,place=[17],name=[11],order='name_first'),dict(fact=17,place=[28],name=[20],order='name_first')]]
    m,od=place_masks(0,40,ap);assert set(np.where(m)[0])=={7,8,16} and od[:3]==[0,1,None]
    u=UVRec(type('U',(),{'keep':lambda self,*a:(np.ones((2,2),bool),{}),'name':'X'})());u.keep(3,None,'K','V',None,np.ones(4,bool));assert u.stash[0]==3 and u.name=='X'
    import jax,jax.numpy as jnp
    with jax.default_matmul_precision('highest'):      # test (3): linear surrogate sum_p wt_p <c_p, logits_p> (A tokens) + wt NLL (others) == explicit vjp injection of the logit cotangent wt*c (A) / wt*e (others), toy model
        rng=np.random.default_rng(1);Tn,H,V=12,5,7;X=jnp.asarray(rng.normal(size=(Tn,H)),jnp.float32);Wm=jnp.asarray(rng.normal(size=(H,V)),jnp.float32);tg=jnp.asarray(rng.integers(0,V,Tn));wt=jnp.asarray(rng.uniform(0.5,1.5,Tn),jnp.float32);am=np.zeros(Tn,bool);am[[2,3,9]]=True;ix=jnp.asarray(np.where(am)[0]);C=jnp.asarray(rng.normal(size=(3,V)),jnp.float32);amj=jnp.asarray(am)
        def lossS(W):
            lg_=X@W;lp=jax.nn.log_softmax(lg_,axis=-1);nl=-jnp.take_along_axis(lp,tg[:,None],axis=-1)[:,0];return jnp.sum(jnp.where(amj,0.0,wt*nl))+jnp.sum(wt[ix]*jnp.sum(C*lg_[ix],axis=-1))
        g1=jax.grad(lossS)(Wm);lgt,vjp=jax.vjp(lambda W:X@W,Wm);e=jax.nn.softmax(lgt,axis=-1)-jax.nn.one_hot(tg,V);E=wt[:,None]*e;E=E.at[ix].set(wt[ix][:,None]*C);g2=vjp(E)[0]
        assert float(jnp.abs(g1-g2).max())<1e-5*float(jnp.abs(g1).max())+1e-7,'surrogate != vjp injection'
        g3=jax.grad(lossS)(Wm);Ee=wt[:,None]*e;gref=jax.grad(lambda W:jnp.sum(wt*(-jnp.take_along_axis(jax.nn.log_softmax(X@W,axis=-1),tg[:,None],axis=-1)[:,0])))(Wm)
        CC=e[ix];lossS2=lambda W:jnp.sum(jnp.where(amj,0.0,wt*(-jnp.take_along_axis(jax.nn.log_softmax(X@W,axis=-1),tg[:,None],axis=-1)[:,0])))+jnp.sum(wt[ix]*jnp.sum(CC*(X@W)[ix],axis=-1))
        assert float(jnp.abs(jax.grad(lossS2)(Wm)-gref).max())<1e-5*float(jnp.abs(gref).max())+1e-7,'c = e_SV does not reproduce the original gradient (toy)'
    print('e59_kvmulti2 selftest ok: place masks, mask wrapper, linear surrogate == vjp injection, c = e reproduces the original gradient (toy)')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP
    from tokenizers import Tokenizer
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    if os.environ.get('E59_CPU_TEST'):
        _cur=jax.nn.dot_product_attention;jax.nn.dot_product_attention=lambda *a_,**k_:_cur(*a_,**dict(k_,implementation='xla'))
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w));tkz=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));Pout=LD.packing.Packed(T_,raw,GP=2)
    LAST={}
    def margins(S_):                                                                     # per fact (16): gold - best EXPOSED wrong (cands 0..11 exposed, 12..15 never exposed)
        S_=np.asarray(S_)[:NREAD];LAST['S']=S_.copy();g=np.asarray([items[i]['gold_idx'] for i in range(NREAD)]);k=np.arange(NREAD);S_=S_[:,:NEXP];gold=S_[k,g];w_=S_.copy();w_[k,g]=-np.inf;return gold-w_.max(1)
    def read_out(m):
        with jax.default_matmul_precision('highest'):return margins(Pout.r1(rd.to_fp32(m)) .reshape(len(items),G.NCAND))
    W0m=read_out(rt.base);W0S=LAST['S'].copy();lg(f'seed {seed}: W0 margin of the 16 read facts mean {W0m.mean():.3f}')
    base=rt.base;inner0,outer0=eqx.partition(base,rd.f)
    def model_from(newp):return eqx.combine(newp,outer0)
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    EXPO=[int(x) for x in a.expo.split(',')];NCH=a.nch;LASTE=max(EXPO);seed=a.seed
    cf=jnp.asarray(np.unique(np.asarray([int(raw[i*G.NCAND+items[i]['gold_idx']][0][raw[i*G.NCAND+items[i]['gold_idx']][1][0]+1]) for i in range(NREAD)],np.int32)))  # unique first tokens (shared first tokens were counted twice: mass > 1 in the 1348 smoke)
    def loss_s(mm,st_suf,c,pref_,kmask,wt,cA,ixj,kvj,am):
        """original token-weighted NLL outside A; at the A tokens the linear surrogate sum_p wt_p <c_p, f32 logits_p> (logit gradient exactly wt_p c_p, the NLL term of the A token is left out)"""
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32);lp=jax.nn.log_softmax(lgt,axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
        return jnp.sum(jnp.where(am,0.0,wt*nll))+jnp.sum(kvj*wt[ixj]*jnp.sum(cA*lgt[ixj],axis=-1)),nll
    @eqx.filter_jit
    def ref_grad_s(m,st_suf,c,pref_,kmask,wt,cA,ixj,kvj,am):
        (_l,nll),g=eqx.filter_value_and_grad(loss_s,has_aux=True)(m,st_suf,c,pref_,kmask,wt,cA,ixj,kvj,am);return g.inner_parameters(),nll
    @eqx.filter_jit
    def realized_cot(m,st_suf,c,pref_,kmask,wt,cA,ixj,kvj,am):
        """gradient of the surrogate loss with respect to the logits themselves (the realized logit cotangent) """
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32)
        def f(z):
            l_=lgt+z;lp=jax.nn.log_softmax(l_,axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
            return jnp.sum(jnp.where(am,0.0,wt*nll))+jnp.sum(kvj*wt[ixj]*jnp.sum(cA*l_[ixj],axis=-1))
        return jax.grad(f)(jnp.zeros_like(lgt))
    @eqx.filter_jit
    def err2(m,st_suf,c,pref_,kmask,ixj):
        """at the (padded) token indices ixj: e = softmax - onehot (float32, no subtraction), ||e||, NLL, correct-answer logit, correct logit minus the largest wrong logit"""
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32)[ixj];tg=c.target_tokens[ixj];oh=jax.nn.one_hot(tg,lgt.shape[-1],dtype=jnp.bool_)
        lse=jax.scipy.special.logsumexp(lgt,axis=-1);lm=jnp.where(oh,-jnp.inf,lgt);q=jnp.exp(jax.scipy.special.logsumexp(lm,axis=-1)-lse);pm_=jnp.exp(lm-lse[:,None])
        en=jnp.sqrt(q*q+jnp.sum(pm_*pm_,axis=-1));e=pm_-oh.astype(jnp.float32)*q[:,None];gold=jnp.take_along_axis(lgt,tg[:,None],axis=-1)[:,0]
        return e,en,lse-gold,gold,gold-jnp.max(lm,axis=-1)
    @eqx.filter_jit
    def err_fn(m,st_suf,c,pref_,kmask,cf_):
        """per token: e = softmax - onehot in float32 WITHOUT subtraction (e_j = p_j for j != y, e_y = -sum_{j != y} p_j via the logsumexp without the answer), ||e||, NLL, share of |e| on the first tokens of the 16 places"""
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32);tg=c.target_tokens;oh=jax.nn.one_hot(tg,lgt.shape[-1],dtype=jnp.bool_)
        lse=jax.scipy.special.logsumexp(lgt,axis=-1);lm=jnp.where(oh,-jnp.inf,lgt);q=jnp.exp(jax.scipy.special.logsumexp(lm,axis=-1)-lse);pm=jnp.exp(lm-lse[:,None])
        en=jnp.sqrt(q*q+jnp.sum(pm*pm,axis=-1));e=pm-oh.astype(jnp.float32)*q[:,None];nll=lse-jnp.take_along_axis(lgt,tg[:,None],axis=-1)[:,0]
        mass=jnp.sum(jnp.abs(e[:,cf_]),axis=-1)/jnp.maximum(2*q,1e-30);return e,en,nll,mass

    PR2={}
    def read2(m,k):
        if k not in PR2:
            with rt32.mesh,jax.default_matmul_precision('highest'):PR2[k]=LD.packing.Packed(LD._TailCtx(rt32,[int(t) for t in ids0[k*CH:(k+1)*CH]]),raw,GP=2)
        with jax.default_matmul_precision('highest'):return np.asarray(PR2[k].r1(rd.to_fp32(m))).reshape(len(items),G.NCAND)[:NREAD]
    W0R2=read2(rt.base,LASTE-1)
    RES=dict(validation=None,seed=seed,expo=EXPO,nch=NCH,W0_R1=W0S.tolist(),W0_R2=W0R2.tolist(),cand_exposed=[True]*NEXP+[False]*(G.NCAND-NEXP),arms={},args=vars(a));RC={}
    VAL={};VAL8={}
    rel_=lambda x,y:float(jnp.sqrt(sum(jnp.sum((p-q)**2) for p,q in zip(jax.tree_util.tree_leaves(x),jax.tree_util.tree_leaves(y)))/sum(jnp.sum(q**2) for q in jax.tree_util.tree_leaves(y))))
    gnorm_=lambda x:float(jnp.sqrt(sum(jnp.sum(q**2) for q in jax.tree_util.tree_leaves(x))))
    for arm,cn in ARMS.items():
        if arm not in a.arms.split(','):continue
        mode=MODE[arm];ocn='SAMEVIS' if cn=='C0' else 'C0'
        uv=UVRec(SH.mk_uv(CONDS[cn],w,seed,lg));uvo=SH.mk_uv(CONDS[ocn],w,seed,lg);rec=dict(steps=[],diag={},reads={},orders={});cnt=[0]
        L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{arm}',canary=can);L.rec_cache=RC
        def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,uv=uv,uvo=uvo,L=L,cn=cn,ocn=ocn,arm=arm,mode=mode):
            k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=valid/valid.sum();T=len(wt);pm,od=place_masks(k,T,ap);ix=np.where(pm)[0];nA=len(ix);assert 0<nA<=KP,nA
            ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);kk,Kc,Vc,keep_all=uv.stash;assert kk==k
            keep_o,_=uvo.keep(k,m,Kc,Vc,ids_k,keep_all);pref_o=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_o))
            if mode=='ALL':ixp=np.arange(T,dtype=np.int32);kvp=valid.copy();sel=ix                                                              # every token with a loss
            else:ixp=np.zeros(KP,np.int32);ixp[:nA]=ix;kvp=np.zeros(KP,np.float32);kvp[:nA]=1;sel=slice(0,nA)
            ixj=jnp.asarray(ixp);kvj=jnp.asarray(kvp);amj=jnp.ones(T,bool) if mode=='ALL' else jnp.asarray(pm);wtj=jnp.asarray(wt)
            e1,en1,nll1,gl1,mg1=err2(m,st_suf,c,pref,kmask,ixj);e2,en2,nll2,gl2,mg2=err2(m,st_suf,c,pref_o,jnp.asarray(keep_o),ixj)           # e1: SAMEVIS (the weights of this arm now, this arm's own mask), e2: C0
            den=en1*en2;cosv=np.asarray(jnp.where(den>1e-30,jnp.sum(e1*e2,axis=-1)/jnp.maximum(den,1e-30),jnp.nan))[sel];s_=en2/jnp.maximum(en1,1e-30);okk=jnp.isfinite(s_)&(en1>0)&jnp.isfinite(en2);nsmall=0
            if mode in ('E','ALL'):cA=e2;nbad=0
            else:cA=jnp.where(okk[:,None],s_[:,None]*e1,e1);nbad=int((~okk)[sel].sum());nsmall=int(((en1>0)&(en1<1e-30))[sel].sum())               # S2As-u: C0 size, SAMEVIS direction
            if a.validate and not VAL:                                                                                                           # tests at the first chunk of the first arm (the weights are W0; C0 = SAMEVIS there)
                g_ref,_=ref_grad(m,st_suf,c,pref,kmask,wtj);g_s1,_=ref_grad_s(m,st_suf,c,pref,kmask,wtj,e1,ixj,kvj,amj);VAL['test1_cotangent_eSV_vs_original']=rel_(g_s1,g_ref);VAL['mode']=mode
                if mode=='ALL':
                    g_c0,_=ref_grad_s(m,st_suf,c,pref,kmask,wtj,e2,ixj,kvj,amj);VAL['test1b_E2All_cotangent_eC0_vs_original_first_chunk']=rel_(g_c0,g_ref);gs=gnorm_(g_s1);VAL['gnorm_eSV_cotangent']=gs;kvb=json.load(open(pathlib.Path(a.base_dir)/f'kvmulti_s{seed}.json'));VAL['gnorm_P0s_step0_job1344t0']=kvb['arms']['P0s']['steps'][0]['gnorm'];VAL['test2_eSV_step_vs_P0s_step_rel']=abs(gs-VAL['gnorm_P0s_step0_job1344t0'])/VAL['gnorm_P0s_step0_job1344t0'];del g_c0
                    assert VAL['test1b_E2All_cotangent_eC0_vs_original_first_chunk']<5e-3 and VAL['test2_eSV_step_vs_P0s_step_rel']<5e-3,VAL
                else:
                    sv=np.ones(T,np.float32);sv[ix]=np.asarray(jnp.where(okk,s_,1.0))[:nA];g_w,_=ref_grad(m,st_suf,c,pref,kmask,jnp.asarray(wt*sv));g_s2,_=ref_grad_s(m,st_suf,c,pref,kmask,wtj,jnp.where(okk[:,None],s_[:,None]*e1,e1),ixj,kvj,amj)
                    VAL.update(test2_cotangent_sXeSV_vs_weighted_loss=rel_(g_s2,g_w),first_chunk_scale_s_quartiles=np.percentile(np.asarray(s_)[:nA],[25,50,75]).tolist());del g_w,g_s2
                del g_ref,g_s1;lg('VALIDATION',VAL);assert VAL['test1_cotangent_eSV_vs_original']<5e-3 and VAL.get('test2_cotangent_sXeSV_vs_weighted_loss',0)<5e-3,VAL
            if (a.validate or a.validate8) and k==7 and arm not in VAL8:                                                                         # chunk 8: realized logit cotangent vs the target; for S2As-u the s != 1 test against the weighted-loss implementation (tokens with 1e-4 < ||e_SV||, s < 1e5; the rest s = 1)
                Gz=realized_cot(m,st_suf,c,pref,kmask,wtj,cA,ixj,kvj,amj);tgt=(wtj[ixj]*kvj)[:,None]*cA;msk=(kvj>0)[:,None];VAL8[arm]=dict(realized_cotangent_rel_err=float(jnp.linalg.norm(jnp.where(msk,Gz[ixj]-tgt,0.0))/jnp.maximum(jnp.linalg.norm(tgt),1e-30)));del Gz,tgt
                if mode=='S':
                    rs=okk&(en1>1e-4)&(s_<1e5)&(jnp.arange(len(en1))<nA);svr=np.ones(T,np.float32);svr[ix]=np.asarray(jnp.where(rs,s_,1.0))[:nA];cA_r=jnp.where(rs[:,None],s_[:,None]*e1,e1)
                    g_s,_=ref_grad_s(m,st_suf,c,pref,kmask,wtj,cA_r,ixj,kvj,amj);g_w,_=ref_grad(m,st_suf,c,pref,kmask,jnp.asarray(wt*svr));VAL8[arm].update(test2_chunk8_s_ne_1_rel_err=rel_(g_s,g_w),n_restricted_tokens=int(rs.sum()),n_A_tokens=nA,s_quartiles_restricted=np.percentile(np.asarray(s_)[np.asarray(rs)],[25,50,75]).tolist() if int(rs.sum())>0 else None);del g_s,g_w
                lg('VALIDATION chunk 8',arm,VAL8[arm])
            g,nll=ref_grad_s(m,st_suf,c,pref,kmask,wtj,cA,ixj,kvj,amj);newp,gn,coef=L.sgd_apply(m,st_all,g);del g;ce=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum());cot=float(jnp.linalg.norm(cA,axis=-1)[sel].mean())
            rec['steps'].append(dict(k=k,gnorm=float(gn),coef=float(coef),ce=ce,cot_norm=cot,nbad=nbad,nsmall=nsmall,gold_logit=float(np.asarray(gl1)[sel].mean()),margin_sv=float(np.asarray(mg1)[sel].mean())))
            if (k+1)%8==0:
                rec['diag'][str(k+1)]=dict(cond_own=cn,cond_other=ocn,idx=ix.tolist(),en_own=np.asarray(en1)[sel].tolist(),en_other=np.asarray(en2)[sel].tolist(),nll_own=np.asarray(nll1)[sel].tolist(),nll_other=np.asarray(nll2)[sel].tolist(),cos=cosv.tolist(),gold_logit_sv=np.asarray(gl1)[sel].tolist(),gold_logit_c0=np.asarray(gl2)[sel].tolist(),margin_sv=np.asarray(mg1)[sel].tolist(),margin_c0=np.asarray(mg2)[sel].tolist(),order=od)
                if mode=='ALL':
                    ro=KL.entity_masks(k,T,ap,w,'role')['ROLE'];vm=valid>0;E1=np.asarray(en1);E2=np.asarray(en2);kd={}
                    for nm_,v_ in (('place',0),('name',1),('frame',2),('extra',3),('filler',-1)):
                        mk_=vm&(ro==v_);kd[nm_]=dict(n=int(mk_.sum()),en_c0=float(E2[mk_].mean()) if mk_.any() else None,en_sv=float(E1[mk_].mean()) if mk_.any() else None)
                    rec['diag'][str(k+1)]['kinds']=kd
            del e1,e2,cA;return newp,gn,coef,jnp.asarray(ce,jnp.float32)
        def hook(k,m,rec=rec,arm=arm):
            e=k+1;read_out(m);d=dict(dS1=(LAST['S']-W0S).tolist())
            if e==LASTE:
                d['dS2']=(read2(m,k)-W0R2).tolist()
                if a.save_final:d['final_state']=L.save_state(m,name=f'final_{arm}_s{seed}.npz')
            od=[None]*NREAD
            for dd in ap[k]:
                if dd['fact']<NREAD:od[dd['fact']]=0 if dd['order']=='place_first' else 1
            rec['orders'][str(e)]=od;return d
        L.write_step=ws;t0=time.monotonic()
        r=L.run_stream(NM.ENGINE,1,ids0,read_chunks=[],canary_chunks=[],nch=NCH,tag=f'kvmulti_{arm}',log=lg,read_hook=hook,hook_chunks=[e-1 for e in EXPO],chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
        rec['reads']={str(k+1):v for k,v in r.get('inline',{}).items()};rec['sec']=time.monotonic()-t0;st=rec['steps']
        rec['diverged']=bool((not all(np.isfinite([x['gnorm'] for x in st]))) or np.mean([x['ce'] for x in st[-8:]])>4.0);RES['arms'][arm]=rec
        lg(f'seed {seed} {arm} ({cn}): {NCH} chunks in {rec["sec"]:.0f}s, mean |g| {np.mean([x["gnorm"] for x in st]):.3f}, mean coef {np.mean([x["coef"] for x in st]):.3f}, last-8 CE {np.mean([x["ce"] for x in st[-8:]]):.3f}, diverged {rec["diverged"]}, reads at {sorted(rec["reads"])}')
        L.rec_cache=None;del L
    if False:
        d=abs(RES['arms']['P0s']['steps'][0]['gnorm']-RES['arms']['P1s']['steps'][0]['gnorm']);RES['check_chunk0_dgnorm']=d;lg(f'check: chunk 0 has no earlier sentences, so P0s and P1s must give the same step: |d gnorm| {d:.2e}')
    RES['validation']=VAL;RES['validation_chunk8']=VAL8;json.dump(RES,open(out/f'kvmulti2{a.tag}_s{seed}.json','w'),default=float);lg('wrote',out/f'kvmulti2{a.tag}_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=128);ap_.add_argument('--expo',default='8,16,32,48,64,80,96,112,128');ap_.add_argument('--save-final',action='store_true');ap_.add_argument('--validate',action='store_true');ap_.add_argument('--validate8',action='store_true');ap_.add_argument('--arms',default='S2As-u,E2As');ap_.add_argument('--tag',default='');ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');ap_.add_argument('--selftest',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
