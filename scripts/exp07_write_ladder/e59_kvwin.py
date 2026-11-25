"""In-window re-read of the 128-exposure weights of job 1348 (job 1344w2; GPU; READ ONLY, no write; new file; thread-server 11/25; design thread document (A), Hans 11/25 '2번 지금 돌려보고'). Weights: the 16 saved final states (arms P0s, P1s; seeds 231-238; kvmulti/states) and W0. Context = the last chunk (127) of the stream, as read R2 of e59_kvmulti.py. Reads: R2 (no mask), R2m (the question's own fact sentence of chunk 127 is hidden from the question tokens in the read forward: prefix layers (context key columns) and suffix layers (window columns of the carried K,V); token positions unchanged), R2a (the sentences of all 16 read facts hidden; extra-fact sentences and filler visible). Same 16x16 candidates (12 exposed + 4 never exposed, 1344p2 rule), scores of all candidates stored, W0 scores with the same mask stored.
Checks: hidden positions = sentence span of the question's fact (place and name tokens inside; spans of other facts disjoint), R2m with an all-False mask = R2, R2 recomputed = saved R2 (W0 + dS2) of 1348, and for W0 the prefix-only / suffix-only / everything-hidden variants (each channel must change the scores).
  python3 e59_kvwin.py --seed 231 --out O | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
from e59_kvpos2 import items2,NEXP,NREAD,CH
ARMS=('P0s','P1s');VARIANTS=('R2','R2m','R2a')
def hide_sets(ap_last,variant):
    """-> [NREAD,CH] bool: context positions hidden from the tokens of question q (q = read fact q); ap_last = answer_positions of the context chunk"""
    H=np.zeros((NREAD,CH),bool);sp={d['fact']:d['span'] for d in ap_last if d['fact']<NREAD};assert sorted(sp)==list(range(NREAD)),sorted(sp)
    allm=np.zeros(CH,bool)
    for f,(a,b) in sp.items():allm[a:b]=True
    for q in range(NREAD):
        if variant=='R2m':H[q,sp[q][0]:sp[q][1]]=True
        elif variant=='R2a':H[q]=allm
    return H
def selftest():
    ap=[dict(fact=f,span=(10*f,10*f+8),place=[10*f+6],name=[10*f+1],order='place_first') for f in range(16)]+[dict(fact=20,span=(500,508),place=[505],name=[501],order='name_first')]
    Hm=hide_sets(ap,'R2m');Ha=hide_sets(ap,'R2a');H0=hide_sets(ap,'R2')
    assert not H0.any() and Hm.sum(1).tolist()==[8]*16 and Hm[3,30:38].all() and not Hm[3,:30].any() and not Hm[3,38:].any()
    assert (Ha==Ha[0]).all() and Ha[0].sum()==128 and not Ha[0,500:508].any() and not Ha[0,158:].any()
    assert (Hm.sum(0)<=1).all()
    print('e59_kvwin selftest ok: hide sets')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp);PS=LD.packing
    if os.environ.get('E59_CPU_TEST'):
        _cur=jax.nn.dot_product_attention;jax.nn.dot_product_attention=lambda *a_,**k_:_cur(*a_,**dict(k_,implementation='xla'))
    from ttt.model.data import Batch
    SWA=PS.SWA;_prev=SWA.sw_causal_mask
    def _mask2(self,chunk_id):
        m=_prev(self,chunk_id)
        if 'hide' in PS.PACK_SUF:h=PS.PACK_SUF['hide'];m=m&jnp.concatenate([~h,jnp.ones((self.mini_batch_size,self.mini_batch_size),bool)],axis=1)
        return m
    SWA.sw_causal_mask=_mask2
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);NC=G.NCAND;K=127
    ctx=[int(t) for t in ids0[K*CH:(K+1)*CH]];ap_last=ap[K]
    for d in ap_last:
        if d['fact']<NREAD:
            a_,b_=d['span'];assert all(a_<=t<b_ for t in d['place']+d['name']),('fact entities not inside the sentence span',d['fact'])
    spans=sorted(d['span'] for d in ap_last if d['fact']<NREAD);assert all(spans[i][1]<=spans[i+1][0] for i in range(len(spans)-1)),'spans overlap'
    with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,ctx);PK=PS.Packed(T_,raw,GP=2)
    W=int(rt32.cfg.model.sliding_window_size);TL=len(T_.tail_ids);assert TL==CH
    npad=PK.ids.shape[0];seg_np=np.asarray(PK.seg);loc_np=np.asarray(PK.loc);ids_np=np.asarray(PK.ids)
    qmap=[dict() for _ in range(npad)];cnt={}                                           # per chunk: segment id -> question
    for r,(c,_a,_b) in enumerate(PK.where):s=cnt.get(c,0);cnt[c]=s+1;qmap[c][s]=r//NC
    for c in range(PK.nchunk,npad):qmap[c]=qmap[PK.nchunk-1]
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
    VAR={}
    def build_variant(name,H,use_pref=True,use_suf=True):
        """H [NREAD,CH] hidden context positions per question -> (pref [npad,CH,hid], hide [npad,CH,W])"""
        pref=[];hide=np.zeros((npad,CH,W),bool)
        with rt32.mesh,jax.default_matmul_precision('highest'):
            for j in range(npad):
                sj,lj=seg_np[j],loc_np[j];sf=np.concatenate([np.full(TL,-2,np.int32),sj]);mk=PS.prefix_mask(sf,W)
                for s,q in qmap[j].items():
                    rows=np.where(sj==s)[0]
                    if len(rows)==0:continue
                    if use_pref:mk[np.ix_(TL+rows,np.arange(TL))]&=~H[q][None,:]
                    if use_suf:hide[j][np.ix_(rows,W-TL+np.arange(TL))]=H[q][None,:]
                full=np.concatenate([np.asarray(T_.tail_ids,np.int32),ids_np[j]]);pos=np.concatenate([np.arange(TL,dtype=np.int32),TL+lj])
                pref.append(prefix_h(rt32.base,jnp.asarray(full),jnp.asarray(pos),jnp.asarray(mk))[TL:])
            VAR[name]=(jnp.stack(pref),jnp.asarray(hide))
    m_tail=lambda m32:None
    def tail_state(m32):
        with rt32.mesh:
            st=PS._clone(rt32.state_suffix)
            for j in range(len(T_.tail_ids)//CH):st,_=rt32.chunk_logp(m32,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*CH:(j+1)*CH])
        return st
    def score_all(m32,names):
        st=tail_state(m32);res={}
        with rt32.mesh,jax.default_matmul_precision('highest'):
            for nm in names:
                pref,hide=VAR[nm];o=[]
                for g in range(0,npad,PK.GP):o.append(np.asarray(lp_hide(m32,st,PK.ids[g:g+PK.GP],PK.tg[g:g+PK.GP],PK.mk[g:g+PK.GP],pref[g:g+PK.GP],PK.seg[g:g+PK.GP],PK.loc[g:g+PK.GP],hide[g:g+PK.GP])))
                lp=np.concatenate(o);res[nm]=np.asarray([lp[c,a_:b_].sum() for c,a_,b_ in PK.where]).reshape(len(items),NC)[:NREAD]
        return res
    t0=time.monotonic();Hs={v:hide_sets(ap_last,v) for v in VARIANTS}
    for v in VARIANTS:build_variant(v,Hs[v])
    build_variant('R2m0',np.zeros((NREAD,CH),bool));build_variant('R2m_pref',Hs['R2m'],True,False);build_variant('R2m_suf',Hs['R2m'],False,True);build_variant('ALLH',np.ones((NREAD,CH),bool))
    lg(f'seed {seed}: variants built {time.monotonic()-t0:.0f}s')
    m32_0=rd.to_fp32(rt.base);W0=score_all(m32_0,list(VARIANTS)+['R2m0','R2m_pref','R2m_suf','ALLH'])
    with jax.default_matmul_precision('highest'):W0_ref=np.asarray(PK.r1(m32_0)).reshape(len(items),NC)[:NREAD]
    kv=json.load(open(ROOT/'experiments/exp07_write_ladder/kvmulti'/f'kvmulti_s{seed}.json'))
    dd=lambda x,y:float(np.abs(np.asarray(x)-np.asarray(y)).max())
    CHK=dict(W0_R2_vs_Packed_r1=dd(W0['R2'],W0_ref),W0_R2_vs_saved_1348=dd(W0['R2'],kv['W0_R2']),W0_R2m_allfalse_vs_R2=dd(W0['R2m0'],W0['R2']),
             W0_R2m_vs_R2=dd(W0['R2m'],W0['R2']),W0_R2m_prefonly_vs_R2=dd(W0['R2m_pref'],W0['R2']),W0_R2m_sufonly_vs_R2=dd(W0['R2m_suf'],W0['R2']),W0_R2a_vs_R2=dd(W0['R2a'],W0['R2']),W0_allhidden_vs_R2=dd(W0['ALLH'],W0['R2']))
    lg('checks',{k:round(v,4) for k,v in CHK.items()})
    assert CHK['W0_R2_vs_Packed_r1']<0.02 and CHK['W0_R2m_allfalse_vs_R2']<0.02,CHK
    assert CHK['W0_R2m_prefonly_vs_R2']>0.01 and CHK['W0_R2m_sufonly_vs_R2']>0.01 and CHK['W0_allhidden_vs_R2']>0.05,('a mask channel has no effect',CHK)
    RES=dict(seed=seed,variants=list(VARIANTS),W0={v:W0[v].tolist() for v in VARIANTS},checks=CHK,cand_exposed=[True]*NEXP+[False]*(NC-NEXP),arms={},hidden_counts={v:Hs[v].sum(1).tolist() for v in VARIANTS},args=vars(a))
    SRC={x.split('=')[0]:x.split('=')[1] for x in a.srcs.split(',')} if a.srcs else {}
    for arm in (a.arms.split(',') if a.arms else ARMS):
        sj=json.load(open(ROOT/SRC[arm].format(seed=seed))) if arm in SRC else kv;assert abs(dd(sj['W0_R2'],kv['W0_R2']))<1e-6,'the source run has other W0 scores'
        fs=sj['arms'][arm]['reads']['128']['final_state'];m=rd.load_state(ROOT/fs['_file'],fs['_hash']);m32=rd.to_fp32(m);r=score_all(m32,list(VARIANTS));RES['arms'][arm]={v:r[v].tolist() for v in VARIANTS}
        RES['arms'][arm]['R2_vs_saved_1348']=dd(r['R2']-W0['R2'],sj['arms'][arm]['reads']['128']['dS2']);lg(f'seed {seed} {arm}: R2 recomputed vs saved dS2 of 1348: {RES["arms"][arm]["R2_vs_saved_1348"]:.4f}');del m,m32
    json.dump(RES,open(out/f'{a.prefix}_s{seed}.json','w'),default=float);lg('wrote',out/f'{a.prefix}_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--arms',default=None);ap_.add_argument('--srcs',default=None);ap_.add_argument('--prefix',default='kvwin');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
