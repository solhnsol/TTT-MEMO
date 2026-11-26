"""Indirect relay path, step 1 (job 1359 'relay'; GPU; NEW file; implementation thread 11/26; design: Hans 11/26 23:01 KST, E59 [S2626-RELAY]; EXPLORATORY; no stream re-run). The saved 128-exposure weights of job 1348 (P0s = SAMEVIS stream, P1s = C0 stream; seeds 231-238) and W0 read the SAME input (chunks 120..127 of the U12 stream) under four conditions:
  A = now (C0): carried prefix window = the natural W0 K,V (as every stream), suffix state = natural chain with the read weights, current-chunk keep = C0 (sentence rows cannot see earlier-chunk fact sentences).
  B = A + the background (non-sentence) rows of the current chunk cannot see the earlier-chunk fact-sentence keys either (keep_B = all rows x earlier sentence keys hidden).
  C = B + the earlier chunks recomputed with that mask and carried: the prefix K,V of chunks 0..127 are re-recorded chunk by chunk with keep_B (chunked prefix forward, attending to the already re-recorded window), the suffix chain uses those prefix outputs and keep_B in the suffix layers; at the measured chunk, the sentence rows of fact f additionally cannot see the earlier-chunk background keys that come AFTER f's sentence in their chunk (those saw f's sentence inside their chunk). Row-specific, so one forward serves all facts.
  D = deletion baseline, positions kept: re-record with every row unable to see any fact-sentence key (earlier chunks AND the same chunk; a sentence's own rows keep their own sentence so that no row is empty: their keys are hidden from everyone else), suffix chain likewise; measured chunk with keep_B (its own sentences present, earlier ones deleted).
Measures per weights x condition x chunk (raw, per fact): place-token NLL (index p-1) of the 16 read facts; write gradient (NTP over the chunk, token mean, as the stream) norm and the cosines A-D, B-D, C-D (also A-B, B-C; total and per parameter leaf). Weight-independent (prefix layers are frozen at W0, computed once per seed): prefix-output difference of the fact-sentence rows to D (relative L2, per fact) and head-averaged attention mass of the fact-sentence rows (and 64 fixed background rows) on: earlier background keys (contaminated keys), earlier background keys after the fact's own sentence, earlier same-fact sentence keys, earlier other-fact sentence keys, current chunk; prefix layers first / mid / last and mean.
Checks (--check): chunked natural re-record (keep = base) equals the stored natural record on the first chunks (max abs diff, bf16 level); attention mass on hidden pairs = 0 (B: background rows -> earlier sentences; C: fact rows -> background after their sentence); loaded weights hash = 1348.
  python3 e59_relay.py --seed 231 --out O [--meas 120..127] [--weights W0,P0s,P1s] [--check] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
NREAD=16;CH=1024;NW=8;WIN=NW*CH;CONDS=('A','B','C','D')
def sent_mask_chunk(chunk):
    z=np.zeros(CH,bool)
    for x in chunk['sents']:z[int(x['start']):int(x['end'])]=True
    return z
def earlier_cols(chunks,k):
    """window columns of chunk k: (sentence-token columns, background columns, fact id per column (-1 background / outside))"""
    fid=np.full(WIN+CH,-2,np.int64)
    for c in range(max(0,k-NW),k):
        off=(c-k+NW)*CH;fid[off:off+CH]=-1
        for x in chunks[c]['sents']:fid[off+int(x['start']):off+int(x['end'])]=int(x['fact'])
    return fid
def bg_after_cols(chunks,k,f):
    """earlier-chunk background columns that come after fact f's sentence in their chunk"""
    out=[]
    for c in range(max(0,k-NW),k):
        off=(c-k+NW)*CH;sm=sent_mask_chunk(chunks[c]);e=[int(x['end']) for x in chunks[c]['sents'] if int(x['fact'])==int(f)]
        if not e:continue
        p=np.arange(e[0],CH);p=p[~sm[p]];out.append(off+p)
    return np.concatenate(out) if out else np.zeros(0,np.int64)
def keep_C0(chunks,k,base):
    kp=np.broadcast_to(np.asarray(base,bool),(CH,WIN+CH)).copy();fid=earlier_cols(chunks,k);es=fid>=0
    for x in chunks[k]['sents']:kp[int(x['start']):int(x['end'])][:,es]=False
    return kp
def keep_B(chunks,k,base):
    kp=np.broadcast_to(np.asarray(base,bool),(CH,WIN+CH)).copy();kp[:,earlier_cols(chunks,k)>=0]=False;return kp
def keep_C(chunks,k,base):
    kp=keep_B(chunks,k,base)
    for x in chunks[k]['sents']:
        cols=bg_after_cols(chunks,k,int(x['fact']))
        if len(cols):kp[int(x['start']):int(x['end'])][:,cols]=False
    return kp
def keep_Drec(chunks,k,base):
    kp=keep_B(chunks,k,base);b2=np.broadcast_to(np.asarray(base,bool),(CH,WIN+CH))
    cur=WIN+np.where(sent_mask_chunk(chunks[k]))[0];kp[:,cur]=False
    for x in chunks[k]['sents']:a,b=int(x['start']),int(x['end']);kp[a:b,WIN+a:WIN+b]=b2[a:b,WIN+a:WIN+b]
    return kp
MEAS_KEEP={'A':keep_C0,'B':keep_B,'C':keep_C,'D':keep_B}
def bg_rows(chunks,k,n=64,seed=0):
    sm=sent_mask_chunk(chunks[k]);c=np.where(~sm)[0];rng=np.random.default_rng((seed,k));return np.sort(rng.choice(c,min(n,len(c)),replace=False))
def selftest():
    chunks=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=1,start=30,end=40),dict(fact=20,start=50,end=60)]} for _ in range(12)];base=np.ones(WIN+CH,bool);k=3
    off=lambda c:(c-k+NW)*CH;S=np.concatenate([off(c)+np.r_[10:20,30:40,50:60] for c in range(3)])
    A=keep_C0(chunks,k,base);assert not A[10:20][:,S].any() and A[0:10][:,S].all() and A[60:][:,S].all()
    B=keep_B(chunks,k,base);assert not B[:,S].any() and (~B).sum()==CH*len(S)
    ba=bg_after_cols(chunks,k,0);assert len(ba)==3*(CH-20-20) and off(0)+25 in ba and off(0)+15 not in ba and off(0)+35 not in ba and off(0)+5 not in ba
    C=keep_C(chunks,k,base);assert not C[10:20][:,ba].any() and C[0:10][:,ba].all() and C[30:40][:,off(0)+25].all() and not C[30:40][:,off(0)+45].any()
    D=keep_Drec(chunks,k,base);cur=WIN+np.r_[10:20,30:40,50:60];assert not D[0:10][:,cur].any() and not D[60:][:,cur].any() and D[10:20][:,WIN+10:WIN+20].all() and not D[10:20][:,WIN+30:WIN+40].any() and not D[:,S].any()
    assert (keep_C0(chunks,0,base)==True).all() and (keep_Drec(chunks,0,base)[:,:WIN]).all()
    r=bg_rows(chunks,k,8);assert len(r)==8 and not set(r.tolist())&set(range(10,20))
    print('e59_relay selftest ok: keeps A / B / C / D-record, background-after columns, background rows')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_shadow as SH,s3
    from e59_kvpos2 import CONDS as KC
    from ttt.model.data import Batch
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    U.install();cell=U.CELL;OR.register_cell(cell);LD,s3_,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps={16,32,64,128},arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0;ch=w['chunks']
    MEAS=[int(x) for x in a.meas.split(',')];KL=max(MEAS)
    L=ISO.Iso55(rt,save_dir=out/'states',canary=can);L.chunks_meta=ch
    # ---- chunked masked prefix recorder ----
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
            seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool));o,_=blk(h,(Kc,Vc),seq,is_prefix=True);K,V=s3.S3['rec']
        finally:s3.S3.clear()
        return o,K,V
    blocks=rt.base.language_model.model.h.prefix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0];BL=[jax.tree.map(lambda x:x[l] if eqx.is_array(x) else x,blocks) for l in range(nl)]
    base_of=lambda k:(k*CH-WIN+np.arange(WIN+CH))>=0
    def masked_record(keepfn,upto,like):
        Kh=np.zeros_like(like[0]);Vh=np.zeros_like(like[1]);P={};t0=time.monotonic()
        with rt.mesh:
            for k in range(upto):
                Kc,Vc=L.ctx_kv(Kh,Vh,k);ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);h=rt.base.language_model.wte_call(ids_k);kp=jnp.asarray(keepfn(ch,k,base_of(k)))
                for l in range(nl):
                    h,K,V=layer_ctx(BL[l],h,Kc[l],Vc[l],jnp.int32(k*CH),kp,ids_k);Kh[l,k*CH:(k+1)*CH]=np.asarray(jax.device_get(K)).reshape(Kh[l,k*CH:(k+1)*CH].shape);Vh[l,k*CH:(k+1)*CH]=np.asarray(jax.device_get(V)).reshape(Vh[l,k*CH:(k+1)*CH].shape)
                P[k]=h
        lg(f'  masked record {keepfn.__name__}: {upto} chunks {time.monotonic()-t0:.0f}s');return Kh,Vh,P
    with rt.mesh:
        seq=L._stream_seq(ids0,nch);prefsN=rt.prefix(rt.base,seq);t0=time.monotonic();KhN,VhN,_=L.record(ids0[:nch*CH]);lg(f'seed {seed}: natural record {time.monotonic()-t0:.0f}s')
    RES=dict(seed=seed,args=vars(a),meas=MEAS,conds=list(CONDS),checks={},weights={},prefix={})
    if a.check:
        def keep_nat(chunks,k,base):return np.broadcast_to(np.asarray(base,bool),(CH,WIN+CH)).copy()
        Kt,Vt,Pt=masked_record(keep_nat,10,(KhN,VhN));dk=float(np.abs(Kt[:,:10*CH].astype(np.float32)-KhN[:,:10*CH].astype(np.float32)).max());sk=float(np.abs(KhN[:,:10*CH].astype(np.float32)).max())
        dp=float(max(jnp.abs(Pt[k].astype(jnp.float32)-prefsN[k*CH:(k+1)*CH].astype(jnp.float32)).max() for k in range(10)));RES['checks']['chunked_natural_record_vs_record']=dict(K_maxabs=dk,K_scale=sk,prefix_out_maxabs=dp);lg('CHECK chunked natural record',RES['checks']['chunked_natural_record_vs_record'])
        assert dk<0.05*sk,RES['checks'];del Kt,Vt,Pt
    KhB,VhB,PB=masked_record(keep_B,KL+1,(KhN,VhN));KhD,VhD,PD=masked_record(keep_Drec,KL+1,(KhN,VhN))
    WINS={'A':(KhN,VhN),'B':(KhN,VhN),'C':(KhB,VhB),'D':(KhD,VhD)};CHAIN={'A':'N','B':'N','C':'C','D':'D'}
    # ---- weight-independent: prefix outputs and attention mass at the measured chunks ----
    @eqx.filter_jit
    def nll_all(m,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lgt=o.logits.astype(jnp.float32);return jax.scipy.special.logsumexp(lgt,axis=-1)-jnp.take_along_axis(lgt,c.target_tokens[:,None],axis=-1)[:,0]
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    @eqx.filter_jit
    def read_state_m(m,st,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:return m.language_model.suffix_call(prefix_outputs=pref_,state=st,seq=c).new_state
        finally:s3.S3SUF.clear()
    def att_pass(k,Kc,Vc,kp,rows):
        ids=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);s3.S3.clear();s3.S3['mode']='att';s3.S3['q0']=jnp.int32(k*CH);s3.S3['keep']=jnp.asarray(kp);s3.S3['att_rows']=jnp.asarray(rows);s3.S3['att_out']=[]
        SWA.full_sw_attention=_cur
        try:
            with rt.mesh:
                h=rt.base.language_model.wte_call(ids);sq=Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
                for l in range(nl):h,_=BL[l](h,(Kc[l],Vc[l]),sq,is_prefix=True)
            per=[np.asarray(x) for x in s3.S3['att_out']]
        finally:s3.S3.clear();SWA.full_sw_attention=_full_rec
        return per
    meas_keep={cn:{k:MEAS_KEEP[cn](ch,k,base_of(k)) for k in MEAS} for cn in CONDS}
    with rt.mesh:
        for k in MEAS:
            ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);fid=earlier_cols(ch,k);sents=sorted([x for x in ch[k]['sents'] if int(x['fact'])<NREAD],key=lambda x:int(x['fact']));pref={};dk={}
            for cn in CONDS:
                Kc,Vc=L.ctx_kv(*WINS[cn],k);pref[cn]=L.prefix_masked(rt.base,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(meas_keep[cn][k]))
            hD=np.asarray(pref['D'].astype(jnp.float32));rep={}
            for cn in ('A','B','C'):
                hX=np.asarray(pref[cn].astype(jnp.float32));rep[cn]={str(int(x['fact'])):float(np.linalg.norm(hX[int(x['start']):int(x['end'])]-hD[int(x['start']):int(x['end'])])/max(np.linalg.norm(hD[int(x['start']):int(x['end'])]),1e-30)) for x in sents}
                rep[cn]['all_rows']=float(np.linalg.norm(hX-hD)/np.linalg.norm(hD))
            srows=np.concatenate([np.arange(int(x['start']),int(x['end'])) for x in sents]);brows=bg_rows(ch,k,64,seed);rows=np.concatenate([srows,brows]).astype(np.int32);att={}
            for cn in CONDS:
                Kc,Vc=L.ctx_kv(*WINS[cn],k);per=att_pass(k,Kc,Vc,meas_keep[cn][k],rows);i0=0;res={}
                for x in sents:
                    f=int(x['fact']);n_=int(x['end'])-int(x['start']);sl=slice(i0,i0+n_);i0+=n_;ba=bg_after_cols(ch,k,f)
                    sets={'earlier_bg':np.where(fid==-1)[0],'earlier_bg_after_own':ba,'earlier_same_fact':np.where(fid==f)[0],'earlier_other_fact':np.where((fid>=0)&(fid!=f))[0],'current_chunk':np.arange(WIN,WIN+CH)}
                    res[str(f)]={nm:[float(p_[sl][:,cols].sum(1).mean()) if len(cols) else 0.0 for p_ in per] for nm,cols in sets.items()}
                bl=slice(len(srows),len(rows));res['bg_rows']={'earlier_sentences':[float(p_[bl][:,fid>=0].sum(1).mean()) for p_ in per],'earlier_bg':[float(p_[bl][:,fid==-1].sum(1).mean()) for p_ in per]}
                att[cn]=res
            if a.check:
                RES['checks'].setdefault('att_zero',{})[str(k)]=dict(B_bg_rows_to_earlier_sentences=max(att['B']['bg_rows']['earlier_sentences']),C_fact_rows_to_bg_after_own=max(max(att['C'][str(int(x['fact']))]['earlier_bg_after_own']) for x in sents),A_bg_rows_to_earlier_sentences=max(att['A']['bg_rows']['earlier_sentences']))
                z=RES['checks']['att_zero'][str(k)];assert z['B_bg_rows_to_earlier_sentences']<1e-5 and z['C_fact_rows_to_bg_after_own']<1e-5 and z['A_bg_rows_to_earlier_sentences']>1e-5,z
            RES['prefix'][str(k)]=dict(rep_rel_to_D=rep,att=att,n_bg_after_own={str(int(x['fact'])):int(len(bg_after_cols(ch,k,int(x['fact'])))) for x in sents},n_earlier_bg=int((fid==-1).sum()),n_earlier_sent=int((fid>=0).sum()))
            del pref
    lg(f'seed {seed}: prefix measures done {time.monotonic()-t_all:.0f}s')
    # ---- weights: chains in lockstep, measures at MEAS ----
    kv=json.load(open(pathlib.Path(a.base_dir)/f'kvmulti_s{seed}.json'));leaf_names=[jax.tree_util.keystr(p_) for p_,_ in jax.tree_util.tree_flatten_with_path(rt.base.inner_parameters())[0]]
    def dots(ga,gb):
        la=jax.tree_util.tree_leaves(ga);lb=jax.tree_util.tree_leaves(gb);d=[float(jnp.vdot(x.astype(jnp.float32),y.astype(jnp.float32))) for x,y in zip(la,lb)];return d
    for wn in a.weights.split(','):
        if wn=='W0':m=rt.base
        else:
            fs=kv['arms'][wn]['reads']['128']['final_state'];m=L.load_state(ROOT/fs['_file']);assert s3.reuse._hash_pytree(m.inner_parameters())[0]==fs['_hash'],('weights hash',wn)
        st={nm:s3.clone_state(rt.state_suffix) for nm in ('N','C','D')};REC={};t1=time.monotonic()
        with rt.mesh:
            for k in range(KL+1):
                c=rt.chunk(seq,k)
                if k in MEAS:
                    ids_k=jnp.asarray(ids0[k*CH:(k+1)*CH],dtype=jnp.int32);valid=np.asarray(c.loss_masks,np.float32);wt=jnp.asarray(valid/valid.sum());G_={};ent={'place_nll':{},'gnorm':{},'ce':{}}
                    pos={d['fact']:[t-1 for t in d['place'] if 0<=t-1<CH] for d in ap[k] if d['fact']<NREAD}
                    for cn in CONDS:
                        Kc,Vc=L.ctx_kv(*WINS[cn],k);km=jnp.asarray(meas_keep[cn][k]);pref=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),km);stc=st[CHAIN[cn]]
                        nl_=np.asarray(nll_all(m,stc,c,pref,km));ent['place_nll'][cn]={str(f):[float(nl_[i]) for i in pos[f]] for f in range(NREAD)}
                        g,nll=ref_grad(m,stc,c,pref,km,wt);G_[cn]=g;ent['gnorm'][cn]=float(np.sqrt(sum(dots(g,g))));ent['ce'][cn]=float((np.asarray(nll,np.float32)*valid).sum()/valid.sum())
                    nn={cn:dots(G_[cn],G_[cn]) for cn in CONDS};ent['cos']={}
                    for x,y in (('A','D'),('B','D'),('C','D'),('A','B'),('B','C')):
                        d=dots(G_[x],G_[y]);ent['cos'][f'{x}-{y}']=dict(total=sum(d)/np.sqrt(sum(nn[x])*sum(nn[y])),per_leaf={leaf_names[i]:(d[i]/np.sqrt(nn[x][i]*nn[y][i]) if nn[x][i]>0 and nn[y][i]>0 else None) for i in range(len(d))})
                    del G_;REC[str(k)]=ent;lg(f'  seed {seed} {wn} chunk {k}: cos A-D {ent["cos"]["A-D"]["total"]:.4f} B-D {ent["cos"]["B-D"]["total"]:.4f} C-D {ent["cos"]["C-D"]["total"]:.4f}')
                st['N']=L.read_state(m,st['N'],c,prefsN[k*CH:(k+1)*CH]);st['C']=read_state_m(m,st['C'],c,PB[k],jnp.asarray(keep_B(ch,k,base_of(k))));st['D']=read_state_m(m,st['D'],c,PD[k],jnp.asarray(keep_Drec(ch,k,base_of(k))))
        RES['weights'][wn]=dict(chunks=REC,sec=time.monotonic()-t1);lg(f'seed {seed} weights {wn}: {time.monotonic()-t1:.0f}s');del m,st
        json.dump(RES,open(out/f'relay_s{seed}.json','w'),default=float)
    json.dump(RES,open(out/f'relay_s{seed}.json','w'),default=float);lg('wrote',out/f'relay_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--meas',default=','.join(str(x) for x in range(120,128)));ap_.add_argument('--weights',default='W0,P0s,P1s');ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
