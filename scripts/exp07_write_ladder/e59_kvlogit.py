"""Logit-error SIZE restoration, one step (job 1344s; GPU; new file derived from e59_kvpos2.py; thread-server 11/25; design: Hans / design thread 11/25 (documents 1/3 .. 3/3); EXPLORATORY; NOT queued: Hans decides the position). Question: if in SAMEVIS the logit error of the tokens of a set is made as large as in C0 (direction kept = SAMEVIS), is the binding of the window-free recall written as much as in C0? Same world as 1344p2 (U12, frozen W0, seeds 231..238, measured chunks 16 and 80, conditions C0 = SAME_U1U2 and SAMEVIS = OTHERONLY, one step, same learning rate), e_p = softmax - onehot of token p in float32. Token sets (the NLL index i = position - 1, classes of e59_kvpos class_masks): A = answer tokens (3 tokens) of the 16 read facts' sentences of the current chunk, F = the whole sentences of the 16 read facts (name + frame + answer), O = the sentences of the 16 extra (U2) facts. Tokens outside the set keep weight 1 of their own graph.
IMPLEMENTATION NOTE (identical to the designed logit-cotangent injection): the loss is sum_p wt_p NLL_p, so its logit cotangent is wt_p e_p; a per-token factor s_p on the weight gives the cotangent wt_p s_p e_p. The arm gradients are therefore ref_grad with weights wt x s (same autodiff path as the real step); the selftest checks the equality with an explicit vjp injection on a toy model.
Passes over the stream (weights are frozen at W0, so a pass only provides the graph of one condition): pass 1 'C0' (P1 = real C0 step; stores e_C0 norms, NLL, rows of the sets), pass 2 'SAMEVIS' (P0, S2A, S2F, S2O, S2A-nat), pass 3 'C0x' (X2A = positive control; needs ||e_SV|| of pass 2).
Arms (7): P0 SAMEVIS real step; P1 C0 real step; S2A: SAMEVIS graph, s_p = min(cap, ||e_C0,p|| / ||e_SV,p||) on A, clip coefficient FIXED to the C0 real step's value; S2F: the same on F; S2O: the same on O only (specificity control); X2A: C0 graph, s_p = min(cap, ||e_SV,p|| / ||e_C0,p||) on A, clip coefficient fixed to the SAMEVIS real step's value; S2A-nat: S2A weights, clip rule recomputed in place (natural). cap = 100.
Reads (every arm, W0 of the same context stored): R1 = the neutral out-of-window read (8192-token SQuAD tail, as 1344p2), R2 = the just-written chunk's original text (no mask, common to all arms) as the context. Candidates: 12 exposed (own + 11 other read facts) + 4 never-exposed places (as 1344p2); the full 16 x 16 candidate score change (dS) is stored for every read.
Diagnostics (no extra forward passes beyond one error pass per condition and chunk): per token of A, F, O: ||e_SV||, ||e_C0||, NLL, cos(e_SV, e_C0), the share of the absolute error mass on the first tokens of the 16 read facts' places, token kind; per arm: realized ||cotangent|| / target ||e||, cap hit fraction by kind, set-only gradient norm (arm / C0), pre-clip norm, applied clip coefficient, update norm.
  python3 e59_kvlogit.py --seed 231 --out O [--chunks 16,80] [--nch N] [--cap 100] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_shadow as SH
CH=1024;NL=8;NREAD=16;NEXP=12
CONDS={'C0':'SAME_U1U2','SAMEVIS':('ha','OTHERONLY'),'C0x':'SAME_U1U2'};CLASSES=('ans','name','frame','extra','filler');NFACT=4
ARMS_SV=('P0','S2A','S2F','S2O','S2A-nat')
def class_masks(k,T,ap,w):
    """bool masks over the NLL index i = token position - 1 of chunk k: the 5 classes (a partition) and the fact-level masks f<f>_ans / f<f>_name for facts 0..NFACT-1"""
    m={c:np.zeros(T,bool) for c in CLASSES};byf={d['fact']:d for d in ap[k]};insent=np.zeros(T,bool)
    for x in w['chunks'][k]['sents']:
        f=int(x['fact']);idx=list(range(max(int(x['start'])-1,0),min(int(x['end'])-1,T)));insent[idx]=True
        if f>=NREAD:m['extra'][idx]=True;continue
        d=byf[f];second={t-1 for t in d['second']};first={t-1 for t in d['first']}
        for p in idx:m['ans' if p in second else ('name' if p in first else 'frame')][p]=True
    m['filler']=~insent
    for f in range(NFACT):
        for kind,key in (('ans','second'),('name','first')):
            v=np.zeros(T,bool)
            for t in byf[f][key]:
                if 0<=t-1<T:v[t-1]=True
            m[f'f{f}_{kind}']=v
    return m
def find_sub(ids,sub,end):
    """all start positions of the token subsequence `sub` in ids[:end]"""
    n=len(sub);return [i for i in range(end-n+1) if list(ids[i:i+n])==list(sub)]
def items2(world,UX):
    """read items of the U12 world with 12 exposed places (own at gold_idx 0 + 11 other read facts, seeded per fact) + 4 never-exposed places UX"""
    import random
    facts=[f for f in world['facts'] if f['id']<NREAD];assert len(facts)==NREAD;items=[];QF=__import__('gen_world_ctl').QFRAMES;NQ=len(QF)
    for f in facts:
        r=random.Random(f"kvpos2|{world['seed']}|{f['id']}");oth=[g for g in facts if g['id']!=f['id']];r.shuffle(oth);cl=[f]+oth[:NEXP-1]
        items.append({'kind':'fact','fact':f['id'],'group':f['group'],'name_id':f['id'],'place_id':f['id'],'name':f['X'],'qframe':f['id']%NQ,'prefix':QF[f['id']%NQ].format(X=f['X']),'cands':[' '+x['P'] for x in cl]+[' '+u for u in UX],'gold_idx':0,'cand_fact':[x['id'] for x in cl]+[-1]*len(UX),'cand_group':[x['group'] for x in cl]+[-1]*len(UX),'cand_same':[x['group']==f['group'] for x in cl]+[False]*len(UX),'cand_exposed':[True]*NEXP+[False]*len(UX),'n_rank':NEXP})
    return items

def entity_masks(k,T,ap,w,mode='order'):
    """token sets over the NLL index (position - 1) of chunk k. Two sentence entities (2 tokens each: the PLACE = answer, the person NAME); the U12 templates put the place before the name in ~57% of the sentences. mode 'role': A = place tokens, N = name tokens (independent of the order). mode 'order': A = the LATER mentioned entity (class 'ans' of e59_kvpos / kvpos2: predicted given the earlier one), N = the EARLIER mentioned entity (class 'name' there). F = whole sentences of the read facts (both entities + frame), O = whole sentences of the extra facts. A and N are disjoint and lie inside F in both modes. Also ORD (0 place-first / 1 name-first / -1 none) and ROLE (0 place / 1 name / 2 frame / 3 extra sentence / -1 outside) per token and ORDER[fact] of the read facts."""
    assert mode in ('role','order');m={c:np.zeros(T,bool) for c in 'ANFO'};m['ORD']=np.full(T,-1,np.int8);m['ROLE']=np.full(T,-1,np.int8);m['ORDER']={};byf={d['fact']:d for d in ap[k]}
    for x in w['chunks'][k]['sents']:
        f=int(x['fact']);idx=list(range(max(int(x['start'])-1,0),min(int(x['end'])-1,T)))
        if f>=NREAD:m['O'][idx]=True;m['ROLE'][idx]=3;continue
        m['F'][idx]=True;d=byf[f];o=0 if d['order']=='place_first' else 1;m['ORD'][idx]=o;m['ROLE'][idx]=2;m['ORDER'][f]=o
        for t in d['place']:
            if 0<=t-1<T:m['ROLE'][t-1]=0
        for t in d['name']:
            if 0<=t-1<T:m['ROLE'][t-1]=1
        for tt,key_a,key_n in (('x',('place' if mode=='role' else 'second'),('name' if mode=='role' else 'first')),):
            for t in d[key_a]:
                if 0<=t-1<T:m['A'][t-1]=True
            for t in d[key_n]:
                if 0<=t-1<T:m['N'][t-1]=True
    assert not (m['A']&m['N']).any() and not ((m['A']|m['N'])&~m['F']).any() and not (m['F']&m['O']).any();return m
def sets_of(em):
    """the token sets A, N, F, O (copies)"""
    return {c:em[c].copy() for c in 'ANFO'}
def kind_of(em,idx):
    """token kind per index of idx: ans (place) / name / frame / extra"""
    return ['ans' if em['A'][i] else ('name' if em['N'][i] else ('frame' if em['F'][i] else 'extra')) for i in idx]
def scale_vec(T,mask,e_num,e_den,cap,alpha=1.0):
    """per-token factor s_p: min(cap, e_num/e_den) inside the mask, 1 outside; e_* = ||e|| arrays [T]; returns (s, hit) with hit = bool [T] of the capped tokens"""
    ratio=(e_num/np.maximum(e_den,1e-12))**alpha;s=np.ones(T,np.float32);hit=np.zeros(T,bool)
    s[mask]=np.minimum(cap,ratio[mask]);hit[mask]=ratio[mask]>cap;return s.astype(np.float32),hit
def quart(x):
    x=np.asarray(x,float);return [float(v) for v in np.percentile(x,[25,50,75])] if len(x) else [float('nan')]*3
def selftest():
    import jax,jax.numpy as jnp
    # (1) sets of a toy chunk: A inside F, O disjoint from F; scale_vec outside = 1, cap, ratio
    w={'chunks':[{'sents':[dict(fact=0,start=1,end=10),dict(fact=1,start=10,end=19),dict(fact=17,start=19,end=30)]}]};T=40
    ap=[[dict(fact=0,first=[8,9],second=[2,3],place=[8,9],name=[2,3],order='place_first'),dict(fact=1,first=[11],second=[17],place=[17],name=[11],order='name_first'),dict(fact=2,first=[],second=[],place=[],name=[]),dict(fact=3,first=[],second=[],place=[],name=[]),dict(fact=17,first=[20],second=[28],place=[28],name=[20],order='name_first')]]      # fact 0: place BEFORE name (first = place), fact 1: name before place
    m=entity_masks(0,T,ap,w,'role');S=sets_of(m);mo=entity_masks(0,T,ap,w,'order');So=sets_of(mo);assert set(np.where(So['A'])[0])=={1,2,16} and set(np.where(So['N'])[0])=={7,8,10} and not (So['A']&So['N']).any() and mo['ORDER']=={0:0,1:1} and set(np.where(mo['ORD']==0)[0])==set(range(0,9)) and set(np.where(mo['ROLE']==0)[0])=={7,8,16} and set(np.where(mo['ROLE']==1)[0])=={1,2,10};assert S['A'].sum()==3 and S['N'].sum()==3 and not (S['A']&S['N']).any() and not ((S['A']|S['N'])&~S['F']).any() and not (S['O']&S['F']).any() and S['O'].sum()==11 and S['F'].sum()==18 and set(np.where(S['A'])[0])=={7,8,16} and set(np.where(S['N'])[0])=={1,2,10},'A (place) and N (name) must be disjoint whatever the order of the entities';assert kind_of(m,[7,1,0,20])==['ans','name','frame','extra']
    en=np.full(T,1.0);ed=np.full(T,0.5);ed[7]=1e-9;s,hit=scale_vec(T,S['A'],en,ed,100.);assert np.allclose(s[~S['A']],1) and s[S['A']].max()==100. and hit.sum()==1 and np.isclose(s[S['A']].min(),2.);s5,h5=scale_vec(T,S['A'],en,ed,100.,0.5);assert np.isclose(s5[S['A']].min(),2.**0.5) and s5[S['A']].max()==100. and h5.sum()==1;s15,h15=scale_vec(T,S['A'],en,ed,100.,1.5);assert np.isclose(s15[S['A']].min(),2.**1.5) and s15[S['A']].max()==100. and h15.sum()==1
    # (2) the toy model: the weighted-loss gradient equals the explicit logit-cotangent injection wt s e (the arm construction) and the gradient is linear in the weights
    with jax.default_matmul_precision('highest'):
        rng=np.random.default_rng(0);Tn,H,V=12,5,7;X=jnp.asarray(rng.normal(size=(Tn,H)),jnp.float32);Wm=jnp.asarray(rng.normal(size=(H,V)),jnp.float32);tg=jnp.asarray(rng.integers(0,V,Tn));wt=jnp.asarray(rng.uniform(0.5,1.5,Tn),jnp.float32);sc=jnp.asarray(rng.uniform(0.1,5,Tn),jnp.float32)
        def lossw(W,wv):lp=jax.nn.log_softmax(X@W,axis=-1);return jnp.sum(wv*(-jnp.take_along_axis(lp,tg[:,None],axis=-1)[:,0]))
        g1=jax.grad(lossw)(Wm,wt*sc);lgt,vjp=jax.vjp(lambda W:X@W,Wm);p=jax.nn.softmax(lgt,axis=-1);e=p-jax.nn.one_hot(tg,V);g2=vjp((wt*sc)[:,None]*e)[0]
        assert float(jnp.abs(g1-g2).max())<1e-5*float(jnp.abs(g1).max())+1e-7,'weighted loss gradient != logit-cotangent injection'
        ga=jax.grad(lossw)(Wm,wt*sc*0.3);gb=jax.grad(lossw)(Wm,wt*sc*0.7);assert float(jnp.abs(ga+gb-g1).max())<1e-5*float(jnp.abs(g1).max())+1e-7
        en_f=jnp.linalg.norm(e,axis=-1);oh=jax.nn.one_hot(tg,V,dtype=jnp.bool_);lse=jax.scipy.special.logsumexp(lgt,axis=-1);lm=jnp.where(oh,-jnp.inf,lgt);q=jnp.exp(jax.scipy.special.logsumexp(lm,axis=-1)-lse);pm=jnp.exp(lm-lse[:,None]);en_c=jnp.sqrt(q*q+jnp.sum(pm*pm,axis=-1));assert float(jnp.abs(en_f-en_c).max())<1e-5 and float(jnp.abs((pm-oh*q[:,None])-e).max())<1e-5
        # (3) fixed-coefficient apply: p - s coef g; natural coefficient reproduces the clip rule
        gn=float(jnp.linalg.norm(g1));clip=1.0;coef=1.0 if gn<clip else clip/gn;lr=0.5;newp=Wm-lr*coef*g1;assert abs(float(jnp.linalg.norm(Wm-newp))-lr*coef*gn)<1e-5
    print('e59_kvlogit selftest ok: token sets A in F / O disjoint, scale_vec (cap, outside = 1), weighted loss gradient == explicit logit-cotangent injection (toy), linear in the weights, subtraction-free ||e|| and e, fixed-coefficient apply')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP
    from tokenizers import Tokenizer
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);chunks=[int(x) for x in a.chunks.split(',')];t_all=time.monotonic()
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
    cap=float(a.cap);optax=s3.optax;lr0=float(rt.cfg.training.optimizer_inner.lr)
    gold_tok=[np.asarray(raw[i*G.NCAND+items[i]['gold_idx']][0][raw[i*G.NCAND+items[i]['gold_idx']][1][0]+1:raw[i*G.NCAND+items[i]['gold_idx']][1][1]+1]) for i in range(NREAD)]
    cf=jnp.asarray(np.asarray([int(gold_tok[f][0]) for f in range(NREAD)],np.int32))      # first token of the 16 read facts' places
    DT={};PR2={};W0R2={}
    def pack2(k):
        if k not in PR2:
            with rt32.mesh,jax.default_matmul_precision('highest'):PR2[k]=LD.packing.Packed(LD._TailCtx(rt32,[int(t) for t in ids0[k*CH:(k+1)*CH]]),raw,GP=2)
        return PR2[k]
    def read2(m,k):
        with jax.default_matmul_precision('highest'):return np.asarray(pack2(k).r1(rd.to_fp32(m))).reshape(len(items),G.NCAND)[:NREAD]
    def w0r2(k):
        if k not in W0R2:W0R2[k]=read2(rt.base,k)
        return W0R2[k]
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    @eqx.filter_jit
    def err_fn(m,st_suf,c,pref_,kmask,cf_):
        """per token: e = softmax - onehot in float32 WITHOUT subtraction (e_j = p_j for j != y, e_y = -sum_{j != y} p_j via the logsumexp without the answer), ||e||, NLL, share of |e| on the first tokens of the 16 places"""
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        DT['logits']=str(o.logits.dtype);lgt=o.logits.astype(jnp.float32);tg=c.target_tokens;oh=jax.nn.one_hot(tg,lgt.shape[-1],dtype=jnp.bool_)
        lse=jax.scipy.special.logsumexp(lgt,axis=-1);lm=jnp.where(oh,-jnp.inf,lgt);q=jnp.exp(jax.scipy.special.logsumexp(lm,axis=-1)-lse);pm=jnp.exp(lm-lse[:,None])
        en=jnp.sqrt(q*q+jnp.sum(pm*pm,axis=-1));e=pm-oh.astype(jnp.float32)*q[:,None];nll=lse-jnp.take_along_axis(lgt,tg[:,None],axis=-1)[:,0]
        mass=jnp.sum(jnp.abs(e[:,cf_]),axis=-1)/jnp.maximum(2*q,1e-30);return e,en,nll,mass
    @eqx.filter_jit
    def inj_grad(m,st_suf,c,pref_,kmask,E):
        """explicit logit-cotangent injection (validation only): inner gradient of vjp(logits)(E)"""
        def f(mm):
            s3.S3SUF['kmask']=kmask
            try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
            finally:s3.S3SUF.clear()
            return o.logits
        lgt,vjp=eqx.filter_vjp(f,m);(g,)=vjp(E.astype(lgt.dtype));return g.inner_parameters()
    @eqx.filter_jit
    def apply_fixed(m,st_all,g,coef):
        s=lr0*m.get_ilr_multiplier(st_all.get(m.step_index));return jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g)
    gnorm=eqx.filter_jit(lambda g:optax.global_norm(g))
    tdiff=eqx.filter_jit(lambda g1,g2:optax.global_norm(jax.tree.map(lambda x,y:x.astype(jnp.float32)-y.astype(jnp.float32),g1,g2)))
    RES=dict(seed=seed,chunks=chunks,cap=cap,sets=a.sets,W0_R1=W0S.tolist(),cand_exposed=[True]*NEXP+[False]*(G.NCAND-NEXP),rows=[],validation=[],args=vars(a));ROW={};ST={};cond_list=list(CONDS)
    for cn in cond_list:
        arm=CONDS[cn];cnt=[0];L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{cn}',canary=can);L.rec_cache={}
        def ws(m,st_all,st_suf,c,pref,kmask,cnt=cnt,cn=cn,L=L):
            k=cnt[0];cnt[0]+=1
            if k in chunks:
                t1=time.monotonic();valid=np.asarray(c.loss_masks,np.float32);Wall=SH.token_weights(ap[k],valid>0)['all'].astype(np.float32);T=len(Wall);posm=entity_masks(k,T,ap,w,a.sets);SETS=sets_of(posm);row=ROW.setdefault(k,dict(k=k,arms={},diag={},stats={},check={},sent_order=[posm['ORDER'].get(f_) for f_ in range(NREAD)],sets=a.sets));S_=ST.setdefault(k,{})
                gw=lambda wv:ref_grad(m,st_suf,c,pref,kmask,jnp.asarray(wv,jnp.float32))[0]
                def do_arm(name,g,coef_fixed=None):
                    gn=float(gnorm(g))
                    if coef_fixed is None:newp,_gn,coef=L.sgd_apply(m,st_all,g);coef=float(coef)
                    else:newp=apply_fixed(m,st_all,g,jnp.float32(coef_fixed));coef=float(coef_fixed)
                    if a.no_reads:S1=W0S.copy();S2=np.zeros((NREAD,G.NCAND));w0=np.zeros((NREAD,G.NCAND));del newp                   # debug: gradients and diagnostics only (CPU check), no reads
                    else:mn=model_from(newp);read_out(mn);S1=LAST['S'].copy();S2=read2(mn,k);w0=w0r2(k);del newp,mn
                    row['arms'][name]=dict(gn=gn,coef=coef,upd=coef*gn,dS1=(S1-W0S).tolist(),dS2=(S2-w0).tolist(),W0_R2=w0.tolist());return gn,coef
                union=np.where(SETS['F']|SETS['O'])[0]
                if cn in ('C0','SAMEVIS'):
                    e,en,nll,mass=err_fn(m,st_suf,c,pref,kmask,cf);en_n=np.asarray(en);nll_n=np.asarray(nll);mass_n=np.asarray(mass);rows=np.asarray(e[jnp.asarray(union)]);del e
                if cn=='C0':
                    g=gw(Wall);gn,coef=do_arm('P1',g)
                    setgn={s:float(gnorm(gw(Wall*SETS[s].astype(np.float32)))) for s in 'AFON'}
                    S_['C0']=dict(en=en_n,nll=nll_n,mass=mass_n,rows=rows,union=union,coef=coef,gn=gn,setgn=setgn)
                    # validation (1): the explicit logit-cotangent injection with s = 1 reproduces ref_grad
                    if not RES['validation']:
                        Ee=jnp.asarray(Wall)[:,None]*err_fn(m,st_suf,c,pref,kmask,cf)[0];g2=inj_grad(m,st_suf,c,pref,kmask,Ee);RES['validation'].append(dict(k=k,inject_s1_rel=float(tdiff(g,g2))/float(gnorm(g)),cond='C0'));lg('validation: explicit injection (s=1) vs ref_grad relative error',RES['validation'][-1]['inject_s1_rel']);del Ee,g2
                    row['check']['apply_fixed_vs_sgd_apply']=None
                    newp_a,_g,_c=L.sgd_apply(m,st_all,g);newp_b=apply_fixed(m,st_all,g,jnp.float32(float(_c)));row['check']['apply_fixed_vs_sgd_apply']=float(tdiff(newp_a,newp_b)/jnp.maximum(optax.global_norm(jax.tree.map(lambda x,y:x.astype(jnp.float32)-y.astype(jnp.float32),newp_a,m.inner_parameters())),1e-30));del newp_a,newp_b,g
                elif cn=='SAMEVIS':
                    C=S_['C0'];g=gw(Wall);gn,coef=do_arm('P0',g);S_['SV']=dict(en=en_n,coef=coef,gn=gn);setgn_sv={s:float(gnorm(gw(Wall*SETS[s].astype(np.float32)))) for s in 'AFON'};del g
                    cos=np.einsum('nv,nv->n',rows,C['rows'])/np.maximum(np.linalg.norm(rows,axis=1)*np.linalg.norm(C['rows'],axis=1),1e-30);un=C['union']
                    assert (un==union).all(),'the token sets of the two conditions must be identical'
                    row['diag']=dict(idx=union.tolist(),kind=kind_of(posm,union),ord=[int(v) for v in posm['ORD'][union]],role=[int(v) for v in posm['ROLE'][union]],en_SV=en_n[union].tolist(),en_C0=C['en'][union].tolist(),nll_SV=nll_n[union].tolist(),nll_C0=C['nll'][union].tolist(),cos=cos.tolist(),mass_SV=mass_n[union].tolist(),mass_C0=C['mass'][union].tolist(),setgn_C0=C['setgn'],setgn_SV=setgn_sv)
                    ia=np.where(SETS['A'])[0];lg(f'seed {seed} k={k} write-forward logits dtype {DT.get("logits")}; A tokens (n={len(ia)}): ||e_SV|| min/quartiles {float(en_n[ia].min()):.4g} {quart(en_n[ia])}, ||e_C0|| min/quartiles {float(C["en"][ia].min()):.4g} {quart(C["en"][ia])}, ratio ||e_C0||/||e_SV|| quartiles {quart(C["en"][ia]/np.maximum(en_n[ia],1e-12))}, ratio > cap {float((C["en"][ia]/np.maximum(en_n[ia],1e-12)>cap).mean()):.3f}')
                    for nm,sname,al in (('S2A','A',1.0),('S2F','F',1.0),('S2O','O',1.0),('S2M','N',1.0),('S2A-0.5','A',0.5),('S2A-1.5','A',1.5)):
                        s,hit=scale_vec(T,SETS[sname],C['en'],en_n,cap,al);wv=Wall*s;g=gw(wv);do_arm(nm,g,coef_fixed=C['coef'])
                        mk_=SETS[sname];tgt=en_n**(1-al)*C['en']**al;real=s*en_n;kinds=np.asarray(kind_of(posm,np.where(mk_)[0]));ix=np.where(mk_)[0]
                        row['stats'][nm]=dict(setgn=float(gnorm(gw(wv*mk_.astype(np.float32)))),setgn_C0=C['setgn'][sname],setgn_SV=setgn_sv[sname],**{f'ratio_{kd}':quart(real[ix][kinds==kd]/np.maximum(tgt[ix][kinds==kd],1e-30)) for kd in ('ans','name','frame','extra') if (kinds==kd).any()},**{f'caphit_{kd}':float(hit[ix][kinds==kd].mean()) for kd in ('ans','name','frame','extra') if (kinds==kd).any()},n_tokens=int(mk_.sum()))
                        if nm=='S2A':
                            if len(RES['validation'])<2:                          # validation: the explicit injection (cotangent wt s e of this very arm) equals the weighted-loss gradient on the real model
                                e2=err_fn(m,st_suf,c,pref,kmask,cf)[0];Ee=jnp.asarray(wv)[:,None]*e2;g2=inj_grad(m,st_suf,c,pref,kmask,Ee);RES['validation'].append(dict(k=k,inject_S2A_rel=float(tdiff(g,g2))/float(gnorm(g)),cond='SAMEVIS'));lg('validation: explicit injection (S2A) vs weighted loss relative error',RES['validation'][-1]['inject_S2A_rel']);del e2,Ee,g2
                            gA=g;wA=wv
                        del g
                    gn_n,coef_n=do_arm('S2A-nat',gw(wA));row['stats']['S2A-nat']=dict(coef_natural=coef_n,coef_C0=C['coef'])
                elif cn=='C0x':
                    Cc=S_['C0'];Sv=S_['SV'];g=gw(Wall);row['check']['repro_gn_C0_pass3']=abs(float(gnorm(g))-Cc['gn'])/Cc['gn'];del g
                    s,hit=scale_vec(T,SETS['A'],Sv['en'],Cc['en'],cap);wv=Wall*s;g=gw(wv);do_arm('X2A',g,coef_fixed=Sv['coef']);mk_=SETS['A'];ix=np.where(mk_)[0];kinds=np.asarray(kind_of(posm,ix))
                    row['stats']['X2A']=dict(setgn=float(gnorm(gw(wv*mk_.astype(np.float32)))),setgn_C0=Cc['setgn']['A'],ratio_ans=quart((s*Cc['en'])[ix]/np.maximum(Sv['en'][ix],1e-30)),caphit_ans=float(hit[ix].mean()),n_tokens=int(mk_.sum()));del g
                    s2,hit2=scale_vec(T,SETS['A'],Cc['en'],Sv['en'],cap);wv2=Wall*s2;g=gw(wv2);do_arm('C2A',g,coef_fixed=Cc['coef']);row['stats']['C2A']=dict(setgn=float(gnorm(gw(wv2*mk_.astype(np.float32)))),setgn_C0=Cc['setgn']['A'],ratio_ans=quart(s2[ix]),caphit_ans=float(hit2[ix].mean()),n_tokens=int(mk_.sum()));del g
                lg(f'seed {seed} k={k} pass {cn}: done in {time.monotonic()-t1:.0f}s; arms so far {sorted(row["arms"])}')
            return m.inner_parameters(),jnp.asarray(0.0,jnp.float32),jnp.asarray(1.0,jnp.float32),jnp.asarray(0.0,jnp.float32)
        L.write_step=ws;uv=SH.mk_uv(arm,w,seed,lg);t0=time.monotonic();r=L.run_stream(NM.ENGINE,1,ids0,read_chunks=[],canary_chunks=[],nch=(a.nch or nch),tag=f'kvlogit_{cn}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=w['chunks'],uv=uv);r.pop('model',None);lg(f'seed {seed} {cn} ({arm}): stream done in {time.monotonic()-t0:.0f}s')
        L.rec_cache=None;del L
    RES['rows']=[ROW[k] for k in sorted(ROW)];json.dump(RES,open(out/f'kvlogit_s{seed}.json','w'),default=float);lg('wrote',out/f'kvlogit_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--chunks',default='16,80');ap_.add_argument('--nch',type=int,default=0);ap_.add_argument('--cap',type=float,default=100.);ap_.add_argument('--sets',default='order',choices=('order','role'));ap_.add_argument('--no-reads',action='store_true',help='debug (CPU check): no arm reads');ap_.add_argument('--selftest',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
