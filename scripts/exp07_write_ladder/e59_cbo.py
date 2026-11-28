"""Cover-and-rewrite arms CB-O (job 1357 'cbo'; GPU; NEW file derived from e59_kvmask.py, which stays UNCHANGED; implementation thread 11/26; design: Hans 11/26 14:15 / 14:24 KST, E59 [S2611-CBO-OK] [S2611-CBO-FINAL]; EXPLORATORY).
World / write = jobs 1348 / 1354 (E2E 3B, U12 world: 16 read facts + 16 extra facts, one sentence per fact per chunk, 128 chunks, clipped SGD, natural clip, same learning rate). Every stream arm writes TWO steps per chunk:
  write 1 = the C0 write of P1s (SAME_U1U2: every sentence row cannot see any earlier-chunk sentence in the window), weights W -> W1;
  write 2 = from W1, the SAME chunk, the SAME loss (NTP over the whole chunk, the same token weights), the same clip / lr, with the keep matrix of the arm. Write 2 starts from the C0 keep and changes ONLY the rows of the current sentences of the 16 READ facts (extra-fact rows and non-sentence rows keep C0):
    C0x2        : nothing changed (write 2 = C0 again).
    C0+ISO-mask : the read-fact sentence rows cannot see anything outside their own sentence (earlier chunks and every other token of the current chunk).
    CLZ-ramp    : each key token of K (the earlier sentences of the SAME fact inside the 8-chunk window) stays hidden with probability r and is re-opened otherwise; r = 0.2 + 0.8 k/127 (chunk k, fixed to the 128-chunk schedule also in short smokes), drawn anew for every write (rng (seed, k, arm)).
    CLZ-fix     : as CLZ-ramp with r = 0.6.
    NAME-lure   : K is re-opened except its name tokens; the earlier-chunk sentences of the OTHER facts (read and extra) are also re-opened for these rows.
    NAME-nolure : K re-opened except its name tokens; other facts stay hidden.
  ISOx2 (no stream, the target line): per chunk k the 16 read-fact sentences of chunk k as isolated rows ('\\n\\n' + sentence, e55_iso row definition, shuffled with the e55_iso rng), packed (exp05 packing), TWO SGD steps on the packed rows per chunk, same lr, natural clip, NO norm matching (Hans 11/26).
The carried window K,V of earlier chunks are the natural W0 K,V recorded once (as in every e55_core stream); the suffix state carried to the next chunk is computed with the weights at the START of the chunk (e55_core.run_stream), so write 2 changes the weights only.
Saved (fam-bind split): R1 candidate scores 16 x 16 (12 exposed + 4 never exposed) minus W0 at exposures 8..128, R2 at 128, both writes' gradient norm / clip coef / CE / update, divergence flag, per write-2 mask pattern (re-opened / hidden K tokens, place first token and name tokens of K hidden or not, pairs changed from C0), place / name NLL under C0 (weights W) and under the write-2 keep (weights W1) every 8 chunks, final weights (--save-final; reverse read later, read-only job).
  python3 e59_cbo.py --seed 231 --out O [--nch 128] [--arms C0x2,...] [--save-final] [--check] | --selftest"""
import sys,os,json,time,pathlib,argparse,random
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
from e59_kvmask import same_fact_keys
NREAD=16;CH=1024;NW=8;WIN=NW*CH;RAMP_NCH=128
ARMS={'C0x2':('c0',None),'C0+ISO-mask':('iso',None),'CLZ-ramp':('clz','ramp'),'CLZ-fix':('clz',0.6),'NAME-lure':('name',True),'NAME-nolure':('name',False),'ISOx2':('isox2',None),'ISO1':('isox1',None)}      # ISO1 (S2628-REDO step 1, Hans 11/28): the ISOx2 path with ONE isolated step per chunk
ARM_ID={a:i for i,a in enumerate(ARMS)}
def clz_r(spec,k):return 0.2+0.8*min(k,RAMP_NCH-1)/(RAMP_NCH-1) if spec=='ramp' else float(spec)
def other_keys(chunks,k,fact):
    """key column slices of the earlier-chunk sentences of every OTHER fact inside the window of chunk k"""
    out=[]
    for c in range(max(0,k-NW),k):
        off=(c-k+NW)*CH
        for x in chunks[c]['sents']:
            if int(x['fact'])!=int(fact):out.append((off+int(x['start']),off+int(x['end'])))
    return out
def k_parts(chunks,ap,k,fact):
    """-> (K columns, place-first-token columns of K, name-token columns of K) for `fact` in the window of chunk k"""
    cols=[];pf=[];nm=[];byc={}
    for c in range(max(0,k-NW),k):
        off=(c-k+NW)*CH
        for x in chunks[c]['sents']:
            if int(x['fact'])!=int(fact):continue
            cols.append(np.arange(off+int(x['start']),off+int(x['end'])))
            d=[d for d in ap[c] if int(d['fact'])==int(fact)];assert len(d)==1,(c,fact)
            pf.append(off+int(d[0]['place'][0]));nm+= [off+int(t) for t in d[0]['name']]
    cols=np.concatenate(cols) if cols else np.zeros(0,np.int64)
    return cols,np.asarray(pf,np.int64),np.asarray(nm,np.int64)
class Write2Mask:
    """keep matrix of write 2 from the C0 keep of the chunk (kp_c0 [CH, WIN+CH] bool) and the 1-D base visibility keep_all; stat[k] = mask pattern"""
    def __init__(self,arm,chunks,ap,seed):self.arm=arm;self.mode,self.spec=ARMS[arm];self.chunks=chunks;self.ap=ap;self.seed=seed;self.stat={};self.sets={}
    def keep(self,k,kp_c0,keep_all):
        kp=np.array(kp_c0,bool,copy=True);base=np.broadcast_to(np.asarray(keep_all,bool),(CH,WIN+CH));st=dict(K=0,K_hidden=0,pf=0,pf_hidden=0,name=0,name_hidden=0,opened_pairs=0,closed_pairs=0,r=None,per_fact={});sets={}
        if self.mode=='c0':self.stat[k]=st;self.sets[k]=sets;return kp
        rng=np.random.default_rng((self.seed,k,ARM_ID[self.arm],2));r=clz_r(self.spec,k) if self.mode=='clz' else None;st['r']=r
        for x in sorted(self.chunks[k]['sents'],key=lambda s:int(s['fact'])):
            f=int(x['fact'])
            if f>=NREAD:continue
            rows=np.arange(int(x['start']),int(x['end']))
            if self.mode=='iso':
                own=np.zeros(WIN+CH,bool);own[WIN+rows[0]:WIN+rows[-1]+1]=True;new=kp[rows]&own[None,:];st['closed_pairs']+=int((kp[rows]&~new).sum());kp[rows]=new;sets[f]=dict(rows=rows,closed=np.where(~own)[0],opened=np.zeros(0,np.int64));continue
            K,pf,nm=k_parts(self.chunks,self.ap,k,f)
            if len(K)==0:sets[f]=dict(rows=rows,closed=np.zeros(0,np.int64),opened=np.zeros(0,np.int64));continue
            if self.mode=='clz':hid=rng.random(len(K))<r
            else:hid=np.isin(K,nm)
            opened=K[~hid];closed=K[hid]
            if self.mode=='name' and self.spec:
                ok=np.concatenate([np.arange(s,e) for s,e in other_keys(self.chunks,k,f)]) if other_keys(self.chunks,k,f) else np.zeros(0,np.int64);opened=np.concatenate([opened,ok])
            before=kp[np.ix_(rows,opened)].copy();kp[np.ix_(rows,opened)]=base[np.ix_(rows,opened)];st['opened_pairs']+=int((kp[np.ix_(rows,opened)]&~before).sum())
            hset=set(closed.tolist());st['K']+=len(K);st['K_hidden']+=len(closed);st['pf']+=len(pf);st['pf_hidden']+=sum(1 for c in pf if c in hset);st['name']+=len(nm);st['name_hidden']+=sum(1 for c in nm if c in hset)
            st['per_fact'][f]=dict(K=len(K),K_hidden=len(closed),pf_hidden=[bool(c in hset) for c in pf.tolist()],name_hidden=sum(1 for c in nm if c in hset),name=len(nm))
            sets[f]=dict(rows=rows,closed=closed,opened=opened)
        self.stat[k]=st;self.sets[k]=sets;return kp
def iso_rows_chunk(w,enc,seed,k):
    """the 16 read-fact sentences of chunk k as isolated rows (e55_iso.iso_rows row definition and shuffle rng)"""
    s=[x['text'] for x in w['chunks'][k]['sents'] if int(x['fact'])<NREAD];rng=random.Random(f'E55iso|{w["seed"]}|{seed}|{k}');rng.shuffle(s);out=[]
    for t in s:ids=enc('\n\n'+t);out.append((ids,(0,len(ids)-1)))
    return out
def selftest():
    # toy world: 2 read facts (0, 1) and 1 extra fact (20) per chunk
    chunks=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=1,start=30,end=40),dict(fact=20,start=50,end=60)]} for _ in range(12)]
    ap=[[dict(fact=0,place=[12,13],name=[16,17]),dict(fact=1,place=[32,33],name=[36,37]),dict(fact=20,place=[52,53],name=[56,57])] for _ in range(12)]
    def c0(k):
        base=np.ones(WIN+CH,bool);kp=np.broadcast_to(base,(CH,WIN+CH)).copy()
        for x in chunks[k]['sents']:
            for c in range(max(0,k-NW),k):
                off=(c-k+NW)*CH
                for y in chunks[c]['sents']:kp[x['start']:x['end'],off+y['start']:off+y['end']]=False
        return kp,base
    k=3;kp0,base=c0(k)
    m=Write2Mask('C0x2',chunks,ap,1);assert (m.keep(k,kp0,base)==kp0).all()
    m=Write2Mask('C0+ISO-mask',chunks,ap,1);kp=m.keep(k,kp0,base);assert kp[10:20,WIN+10:WIN+20].all() and not kp[10:20,:WIN+10].any() and not kp[10:20,WIN+20:].any() and (kp[50:60]==kp0[50:60]).all() and (kp[:10]==kp0[:10]).all() and not (kp&~kp0).any()
    m=Write2Mask('NAME-nolure',chunks,ap,1);kp=m.keep(k,kp0,base);K=np.concatenate([(NW-3+j)*CH+np.arange(10,20) for j in range(3)]);nm=np.concatenate([(NW-3+j)*CH+np.array([16,17]) for j in range(3)])
    assert kp[10:20][:,np.setdiff1d(K,nm)].all() and not kp[10:20][:,nm].any() and not kp[10:20][:,(NW-1)*CH+30:(NW-1)*CH+40].any() and (kp[50:60]==kp0[50:60]).all() and m.stat[k]['name_hidden']==12 and m.stat[k]['pf_hidden']==0
    m=Write2Mask('NAME-lure',chunks,ap,1);kp=m.keep(k,kp0,base);assert kp[10:20][:,(NW-1)*CH+30:(NW-1)*CH+40].all() and kp[10:20][:,(NW-1)*CH+50:(NW-1)*CH+60].all() and not kp[10:20][:,nm].any() and (kp[50:60]==kp0[50:60]).all()
    m=Write2Mask('CLZ-fix',chunks,ap,1);kp=m.keep(k,kp0,base);s=m.stat[k];assert s['K']==60 and 0<s['K_hidden']<60 and s['r']==0.6;fr=[];
    for kk in range(8,12):kq,_=c0(kk);m.keep(kk,kq,base);fr.append(m.stat[kk]['K_hidden']/m.stat[kk]['K'])
    assert abs(np.mean(fr)-0.6)<0.1,fr
    kp_b=Write2Mask('CLZ-fix',chunks,ap,1).keep(k,kp0,base);assert (kp_b==kp).all()      # deterministic per (seed, k, arm)
    assert abs(clz_r('ramp',0)-0.2)<1e-12 and abs(clz_r('ramp',127)-1.0)<1e-12
    m=Write2Mask('CLZ-ramp',chunks,ap,1);kq,_=c0(0);assert (m.keep(0,kq,base)==kq).all()     # chunk 0: no K, write 2 = C0
    print('e59_cbo selftest ok: write-2 masks C0x2 / ISO-mask / NAME-lure / NAME-nolure / CLZ (rate, determinism, ramp)')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_shadow as SH,s3
    from e59_kvpos2 import items2,NEXP,CONDS
    from e59_kvmulti import UVRec,place_masks
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3_,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,w['n_chunks']);assert miss==0
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w))
    with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));Pout=LD.packing.Packed(T_,raw,GP=2)
    LAST={}
    def read_out(m):
        with jax.default_matmul_precision('highest'):S_=np.asarray(Pout.r1(rd.to_fp32(m))).reshape(len(items),G.NCAND)[:NREAD]
        LAST['S']=S_.copy();return S_
    W0S=read_out(rt.base).copy();lg(f'seed {seed}: W0 R1 scores read ({W0S.shape})')
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    KP=96
    @eqx.filter_jit
    def err3(m,st_suf,c,pref_,kmask,ixj):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32);tg=c.target_tokens;lse=jax.scipy.special.logsumexp(lgt,axis=-1);gold_all=jnp.take_along_axis(lgt,tg[:,None],axis=-1)[:,0];nll_all=lse-gold_all;return nll_all[ixj],nll_all
    def att_sums(k,m,Kc,Vc,kp,rows,col_sets):
        """eager forward-only 'att' pass of the prefix layers of chunk k with keep kp: attention probability from `rows` summed over each column set, layers first / mid / last"""
        from ttt.model.data import Batch
        rows=np.asarray(rows,np.int32);ids=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32)
        s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(kp);s3.S3['att_rows']=jnp.asarray(rows);s3.S3['att_out']=[]
        try:
            with rt.mesh:
                h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
                for l in range(Kc.shape[0]):
                    blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
            per=s3.S3['att_out']
        finally:s3.S3.clear()
        return {nm:[float(np.asarray(per[l])[:,cols].sum()) if len(cols) else 0.0 for l in (0,len(per)//2,len(per)-1)] for nm,cols in col_sets.items()}
    @eqx.filter_jit
    def logit_rows(m,st_suf,c,pref_,kmask,rowsj):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        return o.logits.astype(jnp.float32)[rowsj]
    PR2={}
    def read2(m,k):
        if k not in PR2:
            with rt32.mesh,jax.default_matmul_precision('highest'):PR2[k]=LD.packing.Packed(LD._TailCtx(rt32,[int(t) for t in ids0[k*CH:(k+1)*CH]]),raw,GP=2)
        with jax.default_matmul_precision('highest'):return np.asarray(PR2[k].r1(rd.to_fp32(m))).reshape(len(items),G.NCAND)[:NREAD]
    EXPO=[int(x) for x in a.expo.split(',')];NCH=a.nch;LASTE=max(EXPO);W0R2=read2(rt.base,LASTE-1)
    RES=dict(checks={},seed=seed,expo=EXPO,nch=NCH,W0_R1=W0S.tolist(),W0_R2=W0R2.tolist(),cand_exposed=[True]*NEXP+[False]*(G.NCAND-NEXP),arms={},args=vars(a),design='CB-O S2611-CBO-FINAL');RC={}
    def mk_hook(rec,L,arm):
        def hook(k,m):
            e=k+1;read_out(m);d=dict(dS1=(LAST['S']-W0S).tolist())
            if e==LASTE:
                d['dS2']=(read2(m,k)-W0R2).tolist()
                if a.save_final:d['final_state']=L.save_state(m,name=f'final_{arm}_s{seed}.npz')
            od=[None]*NREAD
            for dd in ap[k]:
                if dd['fact']<NREAD:od[dd['fact']]=0 if dd['order']=='place_first' else 1
            rec['orders'][str(e)]=od;return d
        return hook
    for arm in a.arms.split(','):
        mode,spec=ARMS[arm];rec=dict(steps=[],diag={},reads={},orders={},mask_stat={});t0=time.monotonic()
        L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{arm}',canary=can);L.rec_cache=RC;hook=mk_hook(rec,L,arm)
        if mode in ('isox2','isox1'):
            packing=LD.packing;Wsl=int(rt.cfg.model.sliding_window_size);m=rt.base;st_all=LD.clone_state(rt.state_all);eos=rt.eos
            with rt.mesh:
                for k in range(NCH):
                    rows=iso_rows_chunk(w,enc,seed,k);pch,where=packing.pack_rows(rows);mks=[np.zeros(CH,bool) for _ in pch]
                    for c_,a_,b_ in where:mks[c_][a_:b_]=True
                    ent=dict(k=k,n_rows=len(rows),n_packed=len(pch),steps=[])
                    for rep in range(2 if mode=='isox2' else 1):
                        for (ids,seg,loc),mk in zip(pch,mks):
                            tg=np.concatenate([ids[1:],[eos]]).astype(np.int32);pref=L.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,Wsl)))
                            g,ce=L.iso_grad(m,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc));newp,gn,coef=L.sgd_apply(m,st_all,g);del g;un=L.norm_between(newp,m);m=L.set_inner(m,newp)
                            ent['steps'].append(dict(rep=rep,gnorm=float(gn),coef=float(coef),ce=float(ce),update_norm=un))
                    rec['steps'].append(ent)
                    if k in [e-1 for e in EXPO]:rec['reads'][str(k+1)]=hook(k,m)
                    if k%8==0:lg(f'  ISOx2 chunk {k+1}/{NCH} ce {ent["steps"][0]["ce"]:.3f}/{ent["steps"][-1]["ce"]:.3f} |g| {ent["steps"][0]["gnorm"]:.3f}')
            gs=[s['gnorm'] for e in rec['steps'] for s in e['steps']];rec['diverged']=bool((not all(np.isfinite(gs))) or np.mean([s['ce'] for e in rec['steps'][-8:] for s in e['steps']])>4.0)
        else:
            uvc0=SH.mk_uv(CONDS['C0'],w,seed,lg);uv=UVRec(uvc0);w2=Write2Mask(arm,w['chunks'],ap,seed);cnt=[0]
            def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,uv=uv,w2=w2,L=L,arm=arm):
                k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=jnp.asarray(valid/valid.sum());T=len(valid)
                kk,Kc,Vc,keep_all=uv.stash;assert kk==k;ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32)
                g,nll=ref_grad(m,st_suf,c,pref,kmask,wt);newp1,gn1,coef1=L.sgd_apply(m,st_all,g);del g;ce1=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum());m1=L.set_inner(m,newp1);un1=L.norm_between(newp1,m)
                kp0=np.asarray(kmask,bool);kp2=w2.keep(k,kp0,keep_all);km2=jnp.asarray(kp2)
                pref2=L.prefix_masked(m1,ids_k,Kc,Vc,jnp.int32(k*CH),km2)      # prefix outputs recomputed with the write-1 weights and the write-2 keep (also for C0x2)
                if a.check and k==8 and 'att' not in RES['checks'].get(arm,{}):
                    ch_=RES['checks'].setdefault(arm,{});sets=w2.sets.get(k,{});rows=sorted(set(int(r) for s in sets.values() for r in s['rows']))
                    if mode=='c0':
                        rows=sorted(set(r for x in w['chunks'][k]['sents'] if int(x['fact'])<NREAD for r in range(int(x['start']),int(x['end']))));Kall=np.concatenate([np.arange(s_,e_) for x in w['chunks'][k]['sents'] if int(x['fact'])<NREAD for s_,e_ in same_fact_keys(w['chunks'],k,int(x['fact']))])
                        r_=att_sums(k,m1,Kc,Vc,kp2,rows,{'K':np.unique(Kall)});ch_['att']=r_;assert max(r_['K'])<1e-6,r_
                    else:
                        # per fact sums would be exact; the union check is conservative for 'closed' (a column closed for one fact may be open for another fact's rows), so check fact by fact
                        res={}
                        for f,s in sets.items():
                            if len(s['rows'])==0:continue
                            r2=att_sums(k,m1,Kc,Vc,kp2,s['rows'],{'closed':s['closed'],'opened':s['opened']});r0=att_sums(k,m1,Kc,Vc,kp0,s['rows'],{'opened':s['opened']});res[int(f)]=dict(arm=r2,c0_opened=r0['opened'])
                            assert max(r2['closed'])<1e-6,(f,r2)
                            if len(s['opened']):assert min(r2['opened'])>1e-6 and max(r0['opened'])<1e-6,(f,r2,r0)
                            if len(res)>=3:break
                        ch_['att']=res
                        rj=jnp.asarray(np.asarray(rows,np.int32));la=np.asarray(logit_rows(m1,st_suf,c,pref2,km2,rj));l0=np.asarray(logit_rows(m1,st_suf,c,pref2,jnp.asarray(kp0),rj));la2=np.asarray(logit_rows(m1,st_suf,c,pref2,km2,rj))
                        ch_['suffix_effect']=dict(max_abs_logit_change=float(np.abs(la-l0).max()),rerun=float(np.abs(la-la2).max()));assert ch_['suffix_effect']['max_abs_logit_change']>1e-3 and ch_['suffix_effect']['rerun']<1e-6,ch_['suffix_effect']
                    lg('CHECK write-2 mask',arm,k,ch_)
                if (k+1)%8==0:
                    pm,od=place_masks(k,T,ap);ixp_=np.where(pm)[0];nmk=np.zeros(T,bool)
                    for d in ap[k]:
                        if d['fact']<NREAD:
                            for t in d['name']:
                                if 0<=t-1<T:nmk[t-1]=True
                    ixn_=np.where(nmk)[0];ix=np.concatenate([ixp_,ixn_]);assert 0<len(ix)<=KP;ixa=np.zeros(KP,np.int32);ixa[:len(ix)]=ix;ixj=jnp.asarray(ixa);dd_={};npl=len(ixp_)
                    for nme,(mx,px,kx) in (('C0_W',(m,pref,kmask)),('W2_W1',(m1,pref2,km2))):
                        nll_,nllall_=err3(mx,st_suf,c,px,kx,ixj);nll_=np.asarray(nll_);dd_[nme]=dict(nll_place=nll_[:npl].tolist(),nll_name=nll_[npl:len(ix)].tolist(),ce_chunk=float((np.asarray(nllall_)*valid).sum()/valid.sum()))
                    dd_['idx_place']=ixp_.tolist();dd_['idx_name']=ixn_.tolist();dd_['order']=od;rec['diag'][str(k+1)]=dd_
                g2,nll2=ref_grad(m1,st_suf,c,pref2,km2,wt);newp2,gn2,coef2=L.sgd_apply(m1,st_all,g2);del g2;ce2=float((np.asarray(nll2,np.float32)*valid).sum()/valid.sum());un2=L.norm_between(newp2,m1)
                st_=dict(w2.stat.get(k,{}));st_['per_fact']={str(f_):v for f_,v in st_.get('per_fact',{}).items()}
                rec['steps'].append(dict(k=k,gnorm1=float(gn1),coef1=float(coef1),ce1=ce1,update1=un1,gnorm2=float(gn2),coef2=float(coef2),ce2=ce2,update2=un2));rec['mask_stat'][str(k)]=st_
                return newp2,gn2,coef2,jnp.asarray(ce2,jnp.float32)
            L.write_step=ws
            r=L.run_stream(NM.ENGINE,1,ids0,read_chunks=[],canary_chunks=[],nch=NCH,tag=f'cbo_{arm}',log=lg,read_hook=hook,hook_chunks=[e-1 for e in EXPO],chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
            rec['reads']={str(k+1):v for k,v in r.get('inline',{}).items()};st=rec['steps']
            rec['diverged']=bool((not all(np.isfinite([x['gnorm1'] for x in st]+[x['gnorm2'] for x in st]))) or np.mean([x['ce2'] for x in st[-8:]])>4.0)
            if a.check:
                kvb=json.load(open(pathlib.Path(a.base_dir)/f'kvmulti_s{seed}.json'));g0=kvb['arms']['P1s']['steps'][0]['gnorm'];RES['checks'].setdefault(arm,{})['chunk0_write1_gnorm_rel_diff_vs_P1s']=d0=abs(st[0]['gnorm1']-g0)/g0;assert d0<5e-3,(arm,d0)
        rec['sec']=time.monotonic()-t0;RES['arms'][arm]=rec;lg(f'seed {seed} {arm}: {NCH} chunks in {rec["sec"]:.0f}s, diverged {rec["diverged"]}, reads at {sorted(rec["reads"],key=int)}')
        L.rec_cache=None;del L
    json.dump(RES,open(out/f'cbo_s{seed}.json','w'),default=float);lg('wrote',out/f'cbo_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=128);ap_.add_argument('--expo',default='8,16,32,48,64,80,96,112,128');ap_.add_argument('--save-final',action='store_true');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');ap_.add_argument('--arms',default=','.join(ARMS));ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
