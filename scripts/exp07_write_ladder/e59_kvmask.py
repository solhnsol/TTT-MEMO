"""Query-position masking in multi-step real writing (job 1344t2 'kvmask'; GPU; new file derived from e59_kvmulti.py, UNCHANGED; thread-server 11/26; design: Hans / design thread, EXPLORATORY). Every arm writes with the SAMEVIS graph (every sentence row sees the earlier sentences of its OWN fact only, not those of other facts) and the ORIGINAL loss (clipped SGD, natural clip, same learning rate); the arms differ only by an ADDITIONAL attention mask on [query row, key] pairs of the write forward (prefix and suffix layers, the same 2-D keep matrix [CH, WIN+CH]):
  M2As-q : for each of the 16 read facts, the query rows Q = {p-1 : p = the 2 place-token positions of the current sentence} (the rows that predict the place tokens) cannot see the keys K = all tokens of the earlier sentences of the SAME fact inside the 8-chunk window (visible under SAMEVIS).
  M2As-s : the same keys K are hidden from ALL rows of the current sentence of the fact.
  Mrand  : as many query rows as Q (2, or 3 for a 3-token place) of the current sentence that are neither Q rows nor place / name token positions (frame rows), chosen by a fixed random (seed, chunk, fact), cannot see K (the control of M2As-q: same number of hidden query rows).
Chunks without an earlier sentence of the fact in the window (chunk 0) give the SAMEVIS keep for all three arms. Reads / diagnostics as e59_kvmulti.py (+ NLL and ||e|| of the place and name tokens under C0, SAMEVIS and the arm mask every 8 chunks).
  python3 e59_kvmask.py --seed 231 --out O [--nch 128] [--arms M2As-q,M2As-s,Mrand] [--save-final] [--check] | --selftest | --check-masks --seed S"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_shadow as SH
from e59_kvpos2 import items2,NEXP,NREAD,CH,CONDS,NL
WIN=8*CH;NW=8
ARMS={'M2As-q':'q','M2As-s':'s','Mrand':'r'}
def same_fact_keys(chunks,k,fact):
    """key column slices [(a,b)] of the earlier sentences of `fact` inside the window of chunk k (window slot layout of the keep matrix: chunk c at offset (c-k+NW)*CH)"""
    out=[]
    for c in range(max(0,k-NW),k):
        off=(c-k+NW)*CH
        for x in chunks[c]['sents']:
            if int(x['fact'])==int(fact):out.append((off+int(x['start']),off+int(x['end'])))
    return out
def query_rows(d,mode,seed,k):
    """rows (query positions inside the chunk) of fact entry d of answer_positions: 'q' Q = {p-1}; 's' all rows of the sentence; 'r' 2 frame rows (fixed random). -> (rows array, n of rows outside the sentence)"""
    a,b=d['span']
    if mode=='q':rows=[p-1 for p in d['place']]
    elif mode=='s':rows=list(range(a,b))
    else:
        ex=set([p-1 for p in d['place']]+list(d['place'])+list(d['name']));cand=[r for r in range(a,b) if r not in ex];rng=np.random.default_rng((seed,k,int(d['fact'])));rows=sorted(int(r) for r in rng.choice(cand,len(d['place']),replace=False))     # as many rows as the Q rows of the fact (2 for 2-token places, 3 for a 3-token place)
    rows=[r for r in rows if 0<=r<CH];return np.asarray(rows,np.int64),sum(1 for r in rows if not a<=r<b)
class QMaskUV:
    """wraps the SAMEVIS uv object: keep() = the SAMEVIS keep matrix with the extra (query row x key) hiding of the arm; stats in self.stat[k]"""
    def __init__(self,inner,mode,chunks,ap,seed,log=print):self.inner=inner;self.mode=mode;self.chunks=chunks;self.ap=ap;self.seed=seed;self.log=log;self.stat={};self.rows={}
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        kp,info=self.inner.keep(k,m,Kc,Vc,ids_k,keep_all);kp=np.array(kp,bool,copy=True);pairs=0;nrow=0;outside=0;nk=0;rows_all={}
        if self.mode is not None:
            for d in self.ap[k]:
                f=int(d['fact'])
                if f>=NREAD:continue
                keys=same_fact_keys(self.chunks,k,f);rows,out=query_rows(d,self.mode,self.seed,k);rows_all[f]=rows.tolist()
                if not keys or len(rows)==0:continue
                cols=np.concatenate([np.arange(s,e) for s,e in keys]);sub=kp[np.ix_(rows,cols)];pairs+=int(sub.sum());kp[np.ix_(rows,cols)]=False;nrow+=len(rows);outside+=out;nk+=len(cols)
        self.stat[k]=dict(pairs=pairs,rows=nrow,outside=outside,keys=nk);self.rows[k]=rows_all;info=dict(info);info['extra_pairs']=pairs;return kp,info
def selftest():
    chunks=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=1,start=30,end=40)]} for _ in range(12)]
    assert same_fact_keys(chunks,0,0)==[] and same_fact_keys(chunks,1,0)==[(7*CH+10,7*CH+20)] and len(same_fact_keys(chunks,12,0))==8 and same_fact_keys(chunks,9,1)[0]==(30,40)
    d=dict(fact=0,span=(10,20),place=[12,13],name=[17,18]);r,o=query_rows(d,'q',1,3);assert r.tolist()==[11,12] and o==0
    d2=dict(fact=0,span=(10,20),place=[10,11],name=[17,18]);r,o=query_rows(d2,'q',1,3);assert r.tolist()==[9,10] and o==1,(r,o)      # place at the sentence start: row 9 is outside
    r,o=query_rows(d,'s',1,3);assert r.tolist()==list(range(10,20)) and o==0
    r,o=query_rows(d,'r',1,3);assert len(r)==2 and not set(r.tolist())&{11,12,13,17,18} and all(10<=x<20 for x in r.tolist());r2,_=query_rows(d,'r',1,3);assert r.tolist()==r2.tolist()
    class In:
        def keep(self,k,m,Kc,Vc,ids_k,keep_all):kp=np.broadcast_to(np.asarray(keep_all,bool),(CH,WIN+CH)).copy();return kp,{}
    ap=[[dict(fact=0,span=(10,20),place=[12,13],name=[17,18])] for _ in range(12)];base=np.ones(WIN+CH,bool)
    u=QMaskUV(In(),'q',chunks,ap,1);kp,_=u.keep(3,None,None,None,None,base);assert (~kp).sum()==2*3*10 and u.stat[3]['pairs']==60 and not kp[11,(NW-3)*CH+10:(NW-3)*CH+20].any() and kp[14,(NW-3)*CH+10:(NW-3)*CH+20].all()
    u0=QMaskUV(In(),'q',chunks,ap,1);kp0,_=u0.keep(0,None,None,None,None,base);assert kp0.all()
    us=QMaskUV(In(),'s',chunks,ap,1);kps,_=us.keep(3,None,None,None,None,base);assert (~kps).sum()==10*3*10
    print('e59_kvmask selftest ok: key sets, query rows (q, s, rand), mask wrapper')
from e59_kvmulti import UVRec,place_masks
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

    KP=96
    @eqx.filter_jit
    def err3(m,st_suf,c,pref_,kmask,ixj):
        """per selected token: ||e|| (softmax - onehot, float32, no subtraction), NLL, correct logit; NLL of all tokens of the chunk"""
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32);tg=c.target_tokens;lse=jax.scipy.special.logsumexp(lgt,axis=-1);gold_all=jnp.take_along_axis(lgt,tg[:,None],axis=-1)[:,0];nll_all=lse-gold_all
        l1=lgt[ixj];t1=tg[ixj];oh=jax.nn.one_hot(t1,l1.shape[-1],dtype=jnp.bool_);lm=jnp.where(oh,-jnp.inf,l1);q=jnp.exp(jax.scipy.special.logsumexp(lm,axis=-1)-lse[ixj]);pm_=jnp.exp(lm-lse[ixj][:,None])
        return jnp.sqrt(q*q+jnp.sum(pm_*pm_,axis=-1)),nll_all[ixj],gold_all[ixj],nll_all
    def att_check(k,m,stash,qm,VAL,arm):
        """check (2): eager forward-only 'att' pass of the prefix layers of chunk k with the arm keep and with the plain SAMEVIS keep; the attention probability summed over the hidden keys K of every fact, from its hidden query rows, must be 0 with the arm keep (and is > 0 with the SAMEVIS keep)"""
        from ttt.model.data import Batch
        kk,Kc,Vc,keep_all=stash;assert kk==k;kp_arm,_=qm.keep(k,m,Kc,Vc,None,keep_all);kp_sv,_=qm.inner.keep(k,m,Kc,Vc,None,keep_all);rows_by_f=qm.rows[k];facts=[f for f,rr in rows_by_f.items() if len(rr)];allrows=np.asarray(sorted(set(r for f in facts for r in rows_by_f[f])),np.int32);pos={int(r):i for i,r in enumerate(allrows)}
        ids=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);res={}
        for nme,kp in (('arm',kp_arm),('samevis',kp_sv)):
            s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(kp);s3.S3['att_rows']=jnp.asarray(allrows);s3.S3['att_out']=[]
            try:
                with rt.mesh:
                    h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
                    for l in range(Kc.shape[0]):
                        blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
                per=s3.S3['att_out']
            finally:s3.S3.clear()
            tot=[];
            for l in (0,len(per)//2,len(per)-1):
                Aw=np.asarray(per[l]);s_=0.0
                for f in facts:
                    cols=np.concatenate([np.arange(s,e) for s,e in same_fact_keys(w['chunks'],k,f)]);s_+=float(sum(Aw[pos[r],cols].sum() for r in rows_by_f[f]))
                tot.append(s_)
            res[nme]=tot
        VAL['att']=dict(k=k,arm_sum_over_hidden_pairs_layers_first_mid_last=res['arm'],samevis_sum_same_pairs=res['samevis'],n_layers=len(per));lg('CHECK 2 attention',VAL['att']);assert max(res['arm'])<1e-6 and min(res['samevis'])>1e-6,VAL['att']
    @eqx.filter_jit
    def logit_rows(m,st_suf,c,pref_,kmask,rowsj):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        return o.logits.astype(jnp.float32)[rowsj]
    def suf_check(k,m,st_suf,c,stash,qm,L,VAL):
        """check (2b), suffix layers: with the prefix outputs fixed (arm keep), the logits at the hidden query rows change when the suffix-layer key mask is the arm keep instead of the plain SAMEVIS keep (the mask acts in the suffix layers); with identical masks they are identical"""
        kk,Kc,Vc,keep_all=stash;assert kk==k;ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);kp_arm,_=qm.keep(k,m,Kc,Vc,None,keep_all);kp_sv,_=qm.inner.keep(k,m,Kc,Vc,None,keep_all)
        rows=np.asarray(sorted(set(r for rr in qm.rows[k].values() for r in rr)),np.int32);rj=jnp.asarray(rows);pref_a=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(kp_arm))
        la=np.asarray(logit_rows(m,st_suf,c,pref_a,jnp.asarray(kp_arm),rj));ls=np.asarray(logit_rows(m,st_suf,c,pref_a,jnp.asarray(kp_sv),rj));la2=np.asarray(logit_rows(m,st_suf,c,pref_a,jnp.asarray(kp_arm),rj))
        VAL['suffix_effect']=dict(k=k,n_rows=len(rows),max_abs_logit_change_arm_vs_samevis_suffix_mask=float(np.abs(la-ls).max()),max_abs_diff_same_mask_rerun=float(np.abs(la-la2).max()),n_pairs_differ=int((kp_sv&~kp_arm).sum()));lg('CHECK 2b suffix layers',VAL['suffix_effect']);assert VAL['suffix_effect']['max_abs_logit_change_arm_vs_samevis_suffix_mask']>1e-3 and VAL['suffix_effect']['max_abs_diff_same_mask_rerun']<1e-6,VAL['suffix_effect']
    PR2={}
    def read2(m,k):
        if k not in PR2:
            with rt32.mesh,jax.default_matmul_precision('highest'):PR2[k]=LD.packing.Packed(LD._TailCtx(rt32,[int(t) for t in ids0[k*CH:(k+1)*CH]]),raw,GP=2)
        with jax.default_matmul_precision('highest'):return np.asarray(PR2[k].r1(rd.to_fp32(m))).reshape(len(items),G.NCAND)[:NREAD]
    W0R2=read2(rt.base,LASTE-1)
    RES=dict(checks=None,seed=seed,expo=EXPO,nch=NCH,W0_R1=W0S.tolist(),W0_R2=W0R2.tolist(),cand_exposed=[True]*NEXP+[False]*(G.NCAND-NEXP),arms={},args=vars(a));RC={}
    VAL={};CHK={}
    for arm,mode in ARMS.items():
        if arm not in a.arms.split(','):continue
        inner=SH.mk_uv(CONDS['SAMEVIS'],w,seed,lg);qm=QMaskUV(inner,mode,w['chunks'],ap,seed,lg);uv=UVRec(qm);uvc0=SH.mk_uv(CONDS['C0'],w,seed,lg);uvsv=SH.mk_uv(CONDS['SAMEVIS'],w,seed,lg);rec=dict(steps=[],diag={},reads={},orders={},qmask_stat={});cnt=[0]
        L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{arm}',canary=can);L.rec_cache=RC
        def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,uv=uv,qm=qm,uvc0=uvc0,uvsv=uvsv,L=L,arm=arm):
            k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=valid/valid.sum();T=len(wt)
            if a.check and k==8 and arm not in CHK:
                CHK[arm]={};att_check(k,m,uv.stash,qm,CHK[arm],arm);suf_check(k,m,st_suf,c,uv.stash,qm,L,CHK[arm])
            if (k+1)%8==0:
                pm,od=place_masks(k,T,ap);ixp_=np.where(pm)[0];nm=np.zeros(T,bool)
                for d in ap[k]:
                    if d['fact']<NREAD:
                        for t in d['name']:
                            if 0<=t-1<T:nm[t-1]=True
                ixn_=np.where(nm)[0];ix=np.concatenate([ixp_,ixn_]);assert 0<len(ix)<=KP,len(ix);ixa=np.zeros(KP,np.int32);ixa[:len(ix)]=ix;ixj=jnp.asarray(ixa)
                ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);kk,Kc,Vc,keep_all=uv.stash;assert kk==k;dd_={}
                for nme,uvx in (('ARM',None),('SAMEVIS',uvsv),('C0',uvc0)):
                    if uvx is None:pref_x,km_x=pref,kmask
                    else:keep_x,_=uvx.keep(k,m,Kc,Vc,ids_k,keep_all);km_x=jnp.asarray(keep_x);pref_x=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),km_x)
                    en_,nll_,gold_,nllall_=err3(m,st_suf,c,pref_x,km_x,ixj);en_=np.asarray(en_);nll_=np.asarray(nll_);gold_=np.asarray(gold_);npl=len(ixp_)
                    dd_[nme]=dict(en_place=en_[:npl].tolist(),en_name=en_[npl:len(ix)].tolist(),nll_place=nll_[:npl].tolist(),nll_name=nll_[npl:len(ix)].tolist(),gold_place=gold_[:npl].tolist(),gold_name=gold_[npl:len(ix)].tolist(),ce_chunk=float((np.asarray(nllall_)*valid).sum()/valid.sum()))
                dd_['idx_place']=ixp_.tolist();dd_['idx_name']=ixn_.tolist();dd_['order']=od;rec['diag'][str(k+1)]=dd_
            g,nll=ref_grad(m,st_suf,c,pref,kmask,jnp.asarray(wt));newp,gn,coef=L.sgd_apply(m,st_all,g);del g;ce=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum())
            rec['steps'].append(dict(k=k,gnorm=float(gn),coef=float(coef),ce=ce,extra_pairs=qm.stat.get(k,{}).get('pairs'),q_rows=qm.stat.get(k,{}).get('rows')));return newp,gn,coef,jnp.asarray(ce,jnp.float32)
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
        r=L.run_stream(NM.ENGINE,1,ids0,read_chunks=[],canary_chunks=[],nch=NCH,tag=f'kvmask_{arm}',log=lg,read_hook=hook,hook_chunks=[e-1 for e in EXPO],chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
        rec['reads']={str(k+1):v for k,v in r.get('inline',{}).items()};rec['sec']=time.monotonic()-t0;st=rec['steps'];rec['qmask_stat']={str(k):v for k,v in qm.stat.items()};rec['random_rows']={str(k):v for k,v in qm.rows.items()} if mode=='r' else None
        rec['diverged']=bool((not all(np.isfinite([x['gnorm'] for x in st]))) or np.mean([x['ce'] for x in st[-8:]])>4.0);RES['arms'][arm]=rec
        lg(f'seed {seed} {arm} ({mode}): {NCH} chunks in {rec["sec"]:.0f}s, mean |g| {np.mean([x["gnorm"] for x in st]):.3f}, mean coef {np.mean([x["coef"] for x in st]):.3f}, last-8 CE {np.mean([x["ce"] for x in st[-8:]]):.3f}, diverged {rec["diverged"]}, reads at {sorted(rec["reads"])}')
        L.rec_cache=None;del L
    if a.check:
        kvb=json.load(open(pathlib.Path(a.base_dir)/f'kvmulti_s{seed}.json'));g0=kvb['arms']['P0s']['steps'][0]['gnorm'];VAL['gnorm_P0s_step0_job1344t0']=g0
        for arm_,rec_ in RES['arms'].items():VAL[f'chunk0_gnorm_rel_diff_{arm_}']=abs(rec_['steps'][0]['gnorm']-g0)/g0;assert VAL[f'chunk0_gnorm_rel_diff_{arm_}']<5e-3,VAL
        lg('CHECKS',VAL)
    VAL['per_arm']=CHK;RES['checks']=VAL
    json.dump(RES,open(out/f'kvmask_s{seed}.json','w'),default=float);lg('wrote',out/f'kvmask_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=128);ap_.add_argument('--expo',default='8,16,32,48,64,80,96,112,128');ap_.add_argument('--save-final',action='store_true');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');ap_.add_argument('--arms',default='M2As-q,M2As-s,Mrand');ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
