"""Multi-step real writing, two arms (job 1344t0; GPU; new file derived from e59_kvpos2.py / e59_kvlogit.py, both UNCHANGED; thread-server 11/25; design: Hans / design thread (documents 1/2, 2/2), EXPLORATORY; NOT queued: Hans decides). Question: when the real write is CONTINUED from W0 over 128 chunks with the C0 mask or the SAMEVIS mask in every chunk (original clipped SGD, natural clip, same learning rate; the mask of every chunk is rebuilt from the chunk's own sentences exactly as in the one-step jobs 1344p / 1344p2 / 1344s), do the answer-matched bindings of the two streams separate, and from which exposure on? World = the U12 world of 1344p2 / 1344s (E2E 3B, 32 facts = 16 read facts + 16 extra facts, one sentence per fact per chunk, so exposure e = chunk e-1), read items = the 16 read facts with 12 exposed + 4 never-exposed (fresh_places) candidates (e59_kvpos2.items2). Arms: P0s = SAMEVIS mask in every chunk (('ha','OTHERONLY'): the same fact visible, all other facts hidden), P1s = C0 mask in every chunk (SAME_U1U2: everything earlier hidden). Reads R1 (neutral out-of-window tail + question) right after exposures 8 / 16 / 32 / 64 / 96 / 128, R2 (the just-written chunk, unmasked, as the context + question; W0 of the same context stored) right after the last exposure; stored: dS (16 x 16 candidate scores minus W0) and W0 scores per read, candidates recreated from the seed by items2. [1344t0 diagnostics also store, for the place tokens, cos(e_own, e_other) and the share of the absolute error mass that sits on the first tokens of the 16 read places under each mask] Per chunk: pre-clip gradient norm, applied clip coefficient, chunk loss; every 8th chunk (before the write, weights of that moment): ||e|| (float32, no subtraction), NLL of the 2 place tokens of the 16 read facts under BOTH masks (one forward reused, one extra), sentence order per fact. Divergence flag (not finite, or mean chunk CE of the last 8 chunks > 4.0) is stored, never used to exclude. Final weights (after exposure 128) per arm and seed can be saved (--save-final).
  python3 e59_kvmulti.py --seed 231 --out O [--nch 128] [--expo 8,16,32,64,96,128] [--save-final] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_shadow as SH
from e59_kvpos2 import items2,NEXP,NREAD,CH,CONDS,NL
ARMS={'P0s':'SAMEVIS','P1s':'C0'}
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
    print('e59_kvmulti selftest ok: place masks and sentence orders, mask wrapper')
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

    PR2={}
    def read2(m,k):
        if k not in PR2:
            with rt32.mesh,jax.default_matmul_precision('highest'):PR2[k]=LD.packing.Packed(LD._TailCtx(rt32,[int(t) for t in ids0[k*CH:(k+1)*CH]]),raw,GP=2)
        with jax.default_matmul_precision('highest'):return np.asarray(PR2[k].r1(rd.to_fp32(m))).reshape(len(items),G.NCAND)[:NREAD]
    W0R2=read2(rt.base,LASTE-1)
    RES=dict(seed=seed,expo=EXPO,nch=NCH,W0_R1=W0S.tolist(),W0_R2=W0R2.tolist(),cand_exposed=[True]*NEXP+[False]*(G.NCAND-NEXP),arms={},args=vars(a));RC={}
    for arm,cn in ARMS.items():
        ocn='SAMEVIS' if cn=='C0' else 'C0'
        uv=UVRec(SH.mk_uv(CONDS[cn],w,seed,lg));uvo=SH.mk_uv(CONDS[ocn],w,seed,lg);rec=dict(steps=[],diag={},reads={},orders={});cnt=[0]
        L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{arm}',canary=can);L.rec_cache=RC
        def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,uv=uv,uvo=uvo,L=L,cn=cn,ocn=ocn):
            k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=valid/valid.sum()
            if (k+1)%8==0:
                T=len(wt);pm,od=place_masks(k,T,ap);ix=np.where(pm)[0];ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);kk,Kc,Vc,keep_all=uv.stash;assert kk==k
                keep_o,_=uvo.keep(k,m,Kc,Vc,ids_k,keep_all);pref_o=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_o))
                e1,en1,nll1,ms1=err_fn(m,st_suf,c,pref,kmask,cf);e2,en2,nll2,ms2=err_fn(m,st_suf,c,pref_o,jnp.asarray(keep_o),cf);ixj=jnp.asarray(ix);cosv=np.asarray(jnp.sum(e1[ixj]*e2[ixj],axis=-1)/jnp.maximum(en1[ixj]*en2[ixj],1e-30));del e1,e2
                rec['diag'][str(k+1)]=dict(cond_own=cn,cond_other=ocn,idx=ix.tolist(),en_own=np.asarray(en1)[ix].tolist(),en_other=np.asarray(en2)[ix].tolist(),nll_own=np.asarray(nll1)[ix].tolist(),nll_other=np.asarray(nll2)[ix].tolist(),cos=cosv.tolist(),mass_own=np.asarray(ms1)[ix].tolist(),mass_other=np.asarray(ms2)[ix].tolist(),order=od)
            g,nll=ref_grad(m,st_suf,c,pref,kmask,jnp.asarray(wt));newp,gn,coef=L.sgd_apply(m,st_all,g);del g;ce=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum())
            rec['steps'].append(dict(k=k,gnorm=float(gn),coef=float(coef),ce=ce));return newp,gn,coef,jnp.asarray(ce,jnp.float32)
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
    if 'P0s' in RES['arms'] and 'P1s' in RES['arms']:
        d=abs(RES['arms']['P0s']['steps'][0]['gnorm']-RES['arms']['P1s']['steps'][0]['gnorm']);RES['check_chunk0_dgnorm']=d;lg(f'check: chunk 0 has no earlier sentences, so P0s and P1s must give the same step: |d gnorm| {d:.2e}')
    json.dump(RES,open(out/f'kvmulti_s{seed}.json','w'),default=float);lg('wrote',out/f'kvmulti_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=128);ap_.add_argument('--expo',default='8,16,32,64,96,128');ap_.add_argument('--save-final',action='store_true');ap_.add_argument('--selftest',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
