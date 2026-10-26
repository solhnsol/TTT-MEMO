"""E57 pod code, mask stage + write-mask object (Hans: the masks act on the attention of ALL tokens, window AND current chunk).
Arms of the pod run (one news cycle, 280 chunks, Adam direction x0.5 of the NAT norm of the same chunk = the engine of the ORC / RAND arms of job 261):
  C1   label-free candidate 1 (V6 tok heads50% g3 thr0.1 trigq>=1, e55_sentgrid_ext.CAND1_CFG); window part = the validated rule unchanged (e55_sentsel.select_heads), chunk part = the same rule on the keys of EARLIER sentences of the chunk
  LRN  learned head weighting (frozen seed-61 coefficients NAT/sum/all/100, e55_sentgrid_ext_s61_coef.npz): sentences of window AND chunk ranked by the logit, highest first until --frac (0.33) of the window + chunk tokens is hidden
  RC1, RLRN  random sentences with the SAME hidden token count in the window and in the chunk as C1 / LRN of the same chunk (overshoot < one sentence)
Mask file e57_pod_masks.npz (selmask format extended): h<r>_<k> bool [nW] hidden window sentences, wlen<k>; ch<r>_<k> bool [nQ] hidden chunk sentences, clen<k>; r = 1 C1, 2 LRN, 3 RC1, 4 RLRN.
SelMaskAll.keep gives a PER-QUERY keep [CH, WIN+CH]: hidden WINDOW tokens are invisible to every query (as in all earlier arms); a hidden CHUNK sentence is invisible to the queries of OTHER sentences of the chunk (the query tokens inside it still see its earlier tokens and themselves: a token never loses itself, and the own sentence keeps its syntax); later queries of the chunk are the only ones that could see it anyway (causal).
  python3 e57_mask_stage.py --selftest | --bundle DIR --out DIR [--frac 0.33] [--check-only]   (GPU forward only; weights W0; no write step)"""
import sys,json,time,argparse,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e55_sentsel as SEL,e55_sentgrid_ext as X
CH=1024;NW=8;WIN=NW*CH;MIN_TOK=SEL.MIN_TOK
COEF_KEY='NAT/sum/all/100'
def load_model(coef):
    Z=np.load(coef);return {nm:Z[f'{COEF_KEY}/{nm}'] for nm in ('w','b','mu','sd')}
def cand1_hidden(d,cfg=None):
    """-> (hid_w [nW], hid_c [nQ]) bool. Window part = SEL.select_heads on the window arrays (unchanged rule). Chunk part: the same heads (head_rank of the window statistics), a query token with T_cmax >= thr votes for the chunk sentence of its max key among the EARLIER sentences of its chunk; hidden when >= k distinct query sentences vote (trig 'q'); candidates >= MIN_TOK tokens"""
    cfg=cfg or X.CAND1_CFG;hid_w=SEL.select_heads(d,cfg)
    nQ=len(d['q_len']);c_len=np.asarray(d['c_len']);cand=c_len>=MIN_TOK
    sel=SEL.head_rank(d,cfg.get('hgroup','g3'),cfg.get('hpct',50));li,hi=np.nonzero(sel);V=np.zeros((len(li),nQ,nQ),bool)
    qs=np.full(np.asarray(d['T_max']).shape[-1],-1,np.int64)
    for i,(a,n) in enumerate(zip(np.asarray(d['q_start']),np.asarray(d['q_len']))):qs[a:a+n]=i
    Tm=np.asarray(d['T_cmax'],np.float32)[li,hi];Tc=np.asarray(d['T_csid'])[li,hi].astype(np.int64)
    ok=(Tm>=cfg.get('thr',0.2))&(qs[None,:]>=0)&(Tc>=0);r,t=np.nonzero(ok);V[r,qs[t],Tc[r,t]]=True
    cnt=V.any(0).sum(0);return hid_w,cand&(cnt>=cfg.get('k',2))
def learned_hidden(d,model,frac,seed=0):
    """-> (hid_w, hid_c): the LR logit of every sentence of the window and of the chunk (features X.features on AL with the chunk columns appended: log head-sentence mass summed over the query sentences), highest first until frac of the window + chunk tokens is hidden"""
    w_len=np.asarray(d['w_len']);c_len=np.asarray(d['c_len']);nW=len(w_len);ln=np.concatenate([w_len,c_len])
    Xf,cand=X.features({'AL':d['AL'],'w_len':ln},'sum','all');z=np.full(len(ln),-np.inf);z[cand]=X.predict_lr(model,Xf)
    h=X.hide_to_fraction(z,ln,frac,np.random.default_rng(seed));return h[:nW],h[nW:]
def random_like(hid_w,hid_c,w_len,c_len,k,seed):
    """random sentences with the same hidden token count as (hid_w, hid_c): window part from all window sentences, chunk part from the chunk sentences that have a later sentence (all but the last) and >= MIN_TOK tokens"""
    import e55_uv as UVm
    w_len=np.asarray(w_len);c_len=np.asarray(c_len);rw=UVm.rands_hidden(w_len,int(w_len[hid_w].sum()),k,seed);rc=np.zeros(len(c_len),bool)
    el=np.nonzero(c_len>=MIN_TOK)[0];el=el[el<len(c_len)-1]
    if len(el):rc[el]=UVm.rands_hidden(c_len[el],int(c_len[hid_c].sum()),k,seed+1000003)
    return rw,rc
def oracle_hidden(k,ends,all_ids,tk,items,UVm,memo):
    """ORACLE answer mask of job 261 (a sentence fragment is hidden when it holds the gold of ANY bank question, e57_bank.ans_matcher) extended to the chunk: -> (hid_w [nW], hid_c [nQ]); needs the lengths from the layout of the chunk"""
    import e57_bank as B
    if 'ms' not in memo:memo['ms']=[B.ans_matcher(it['q'])[0] for it in items];memo['txt']={}
    def hit(a0,ln):
        if a0 not in memo['txt']:memo['txt'][a0]=any(m(tk.decode([int(x) for x in all_ids[a0:a0+int(ln)]])) for m in memo['ms'])
        return memo['txt'][a0]
    wsid,wst,wlen=UVm.window_sents(ends,k,CH,NW);base=(k-NW)*CH;hw=np.asarray([hit(int(st)+base,ln) for st,ln in zip(wst,wlen)],bool) if len(wlen) else np.zeros(0,bool)
    q=UVm.chunk_sents(ends,k,CH);hc=np.asarray([hit(k*CH+a_,b_-a_) for a_,b_ in q],bool) if q else np.zeros(0,bool);return hw,hc
def compute(ends,all_ids,model,frac,nch,k0=1,passfn=None,L=None,rt=None,s3=None,UVm=None,log=print,seed=61,oracle=None):
    """-> npz dict (module doc) and per-chunk statistics; passfn(k, ids_k) -> d replaces the GPU pass (CPU test)"""
    import e55_uv as UVm_
    UVm=UVm or UVm_;out={};stats=[]
    if passfn is None:
        import e57_attdump as AD
        with rt.mesh:Kh,Vh,_=L.record(np.asarray(all_ids)[:nch*CH])
        AD.install(s3)
    for k in range(k0,nch):
        ids_k=np.asarray(all_ids[k*CH:(k+1)*CH])
        if passfn is None:
            with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
            d=AD.att4_pass(L,rt,s3,UVm,rt.base,Kc,Vc,ids_k,k,ends,all_ids)
        else:d=passfn(k,ids_k)
        w_len=np.asarray(d['w_len']);c_len=np.asarray(d['c_len'])
        if len(w_len)==0:continue
        h1=cand1_hidden(d);h2=learned_hidden(d,model,frac);h3=random_like(*h1,w_len,c_len,k,seed);h4=random_like(*h2,w_len,c_len,k,seed+1)
        out[f'wlen{k}']=w_len.astype(np.int32);out[f'clen{k}']=c_len.astype(np.int32);hs=[h1,h2,h3,h4]
        if oracle is not None:ho=oracle(k);hs+=[ho,random_like(*ho,w_len,c_len,k,seed+2)]
        for r,(hw,hc) in enumerate(hs,1):out[f'h{r}_{k}']=np.asarray(hw,bool);out[f'ch{r}_{k}']=np.asarray(hc,bool)
        stats.append(dict(k=k,window_tokens=int(w_len.sum()),chunk_tokens=int(c_len.sum()),**{f'r{r}_win_tokens':int(w_len[hw].sum()) for r,(hw,hc) in enumerate(hs,1)},**{f'r{r}_chunk_tokens':int(c_len[hc].sum()) for r,(hw,hc) in enumerate(hs,1)}))
        if k%16==0:log(f'  mask stage chunk {k}: '+', '.join(f'r{r} window {int(w_len[hw].sum())}/{int(w_len.sum())} chunk {int(c_len[hc].sum())}/{int(c_len.sum())}' for r,(hw,hc) in enumerate(hs,1)))
    return out,stats
class SelMaskAll:
    """UV-compatible mask object (.keep) for e55_core (arms SELC1): stored hidden sentences of window AND chunk, rank r; returns a per-query keep [CH, WIN+CH] (module doc)"""
    def __init__(self,arm,rank,masks,ends,UVm,CH=CH,nw=NW,log=print,guard=0):
        self.guard=int(guard);self.vlog={};self.arm=arm;self.rank=rank;self.masks=masks;self.ends=np.asarray(ends);self.UVm=UVm;self.CH=CH;self.nw=nw;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        CH,nw=self.CH,self.nw;W=nw*CH;wsid,wst,wlen=self.UVm.window_sents(self.ends,k,CH,nw);q=self.UVm.chunk_sents(self.ends,k,CH)
        kw=f'h{self.rank}_{k}';kc=f'ch{self.rank}_{k}';hw=np.zeros(len(wlen),bool);hc=np.zeros(len(q),bool)
        if len(wlen):
            assert kw in self.masks and len(self.masks[kw])==len(wlen) and np.array_equal(self.masks[f'wlen{k}'],wlen),('mask file does not match this stream (window) at chunk',k)
            hw=np.asarray(self.masks[kw],bool)
        if len(q) and kc in self.masks:
            assert len(self.masks[kc])==len(q) and np.array_equal(self.masks[f'clen{k}'],[b-a for a,b in q]),('mask file does not match this stream (chunk) at chunk',k)
            hc=np.asarray(self.masks[kc],bool)
        hk=np.zeros(W,bool)
        if len(wlen):hk=(wsid>=0)&hw[np.clip(wsid,0,len(wlen)-1)]
        csid=np.full(CH,-1,np.int32)
        for i,(a,b) in enumerate(q):csid[a:b]=i
        hck=np.zeros(CH,bool)
        if len(q):hck=(csid>=0)&hc[np.clip(csid,0,len(q)-1)]
        keep=np.ones((CH,W+CH),bool);keep[:,:W]=~hk[None,:]
        same=(csid[:,None]==csid[None,:])                       # query row i, chunk key j: the own sentence stays visible (and the token itself)
        keep[:,W:]=~(hck[None,:]&~same)
        keep[np.arange(CH),W+np.arange(CH)]=True
        for dd in range(1,self.guard+1):                         # LOCAL GUARD: the `guard` tokens right before a query token always stay visible (window tail for the first tokens of the chunk, chunk tokens otherwise)
            keep[np.arange(CH),W+np.arange(CH)-dd]=True
        info={'uv_arm':self.arm,'sel_rank':self.rank,'uv_guard':self.guard,'uv_hidden_tokens':int(hk.sum()),'uv_hidden_chunk_tokens':int(hck.sum()),'uv_window_sents':int(len(wlen)),'uv_chunk_sents':int(len(q))}
        if k%16==0:self.log(f'  {self.arm} chunk {k}: hidden window tokens {int(hk.sum())} ({hk.sum()/max((wsid>=0).sum(),1):.3f} of window), hidden chunk tokens {int(hck.sum())} ({hck.sum()/CH:.3f} of chunk)')
        return keep,info
def synth_dump(k,seed=0,nQ=5,nW=12,L=6,H=4):
    """random dump of one chunk for the CPU test: window sentences of 600 tokens, chunk sentences of ~200 tokens"""
    rng=np.random.default_rng([seed,k]);w_len=np.full(nW,WIN//nW,np.int32);w_len[-1]+=WIN-w_len.sum();c_len=np.full(nQ,CH//nQ,np.int32);c_len[-1]+=CH-c_len.sum()
    q_start=np.concatenate([[0],np.cumsum(c_len)[:-1]]).astype(np.int32);w_start=np.concatenate([[0],np.cumsum(w_len)[:-1]]).astype(np.int32)
    qs=np.repeat(np.arange(nQ),c_len);AL=rng.random((L,H,nQ,nW+nQ)).astype(np.float16)*0.05
    for s in range(nQ):AL[:,:,s,nW+s:]=0
    Tm=rng.random((L,H,CH)).astype(np.float16)*0.3;Tk=rng.integers(0,WIN,(L,H,CH)).astype(np.int16);wsid=np.repeat(np.arange(nW),w_len)
    Tc=rng.random((L,H,CH)).astype(np.float16)*0.3;Tcs=np.full((L,H,CH),-1,np.int16)
    for t in range(CH):
        s=qs[t]
        if s>0:Tcs[:,:,t]=rng.integers(0,s,(L,H))
        else:Tc[:,:,t]=0
    return dict(AL=AL,T_max=Tm,T_key=Tk,T_wsid=wsid[Tk.astype(np.int64)].astype(np.int16),T_cmax=Tc,T_ckey=np.zeros((L,H,CH),np.int16),T_csid=Tcs,q_start=q_start,q_len=c_len.copy(),w_start=w_start,w_len=w_len,c_len=c_len,w_chunk=(k-NW+w_start//CH).astype(np.int32),win_ids=np.zeros(WIN,np.int32),q_ids=np.zeros(CH,np.int32))
def selftest():
    import e57_attdump as AD,e55_uv as UVm
    rng=np.random.default_rng(1)
    # 1 att4_reduce on random probabilities: shapes, causality, sentence mass sums
    H,n,nQ,nW=3,64,4,5;WIN_=AD.WIN;ph=rng.random((H,n,WIN_+n)).astype(np.float32);qi=np.arange(n)[:,None];kj=np.arange(n)[None,:];ph[:,:,WIN_:]*=(kj<=qi)[None];ph/=ph.sum(-1,keepdims=True)
    wsid=np.repeat(np.arange(nW),WIN_//nW+1)[:WIN_];csid=np.repeat(np.arange(nQ),n//nQ);Mw=np.zeros((WIN_,nW),np.float32);Mw[np.arange(WIN_),wsid]=1;Mc=np.zeros((n,nQ),np.float32);Mc[np.arange(n),csid]=1
    qavg=np.zeros((nQ,n),np.float32)
    for s in range(nQ):qavg[s,csid==s]=1.0/(csid==s).sum()
    cstart=np.asarray([int(np.nonzero(csid==s)[0][0]) for s in csid],np.int32);r=AD.att4_reduce(ph,Mw,Mc,qavg,cstart)
    assert r['AL'].shape==(H,nQ,nW+nQ) and np.all(r['AL'][:,:,nW:][:,np.triu_indices(nQ)[0],np.triu_indices(nQ)[1]]==0),'chunk columns must be zero for the own and later sentences'
    assert r['AL'][:,:,:nW].sum()>0
    assert np.all(r['T_cmax'][:,cstart==0]==0),'first-sentence tokens have no earlier chunk key'
    for h in range(H):
        for tok in range(n):
            if cstart[tok]>0:assert r['T_ckey'][h,tok]<cstart[tok],'a chunk vote may only point to an EARLIER sentence'
    # 2 selectors on a synthetic dump
    d=synth_dump(5);hw,hc=cand1_hidden(d,dict(X.CAND1_CFG,thr=0.2,k=1));assert hw.shape==(12,) and hc.shape==(5,) and hc.sum()>0 and hw.sum()>0,(hw.sum(),hc.sum())
    # votes for chunk sentence s only come from queries of later sentences: the first sentence can be hidden, the last sentence is never voted for
    assert not hc[-1],'the last chunk sentence has no later query and cannot be voted for'
    # 3 learned: fraction, union
    Z={'w':rng.standard_normal(6*4),'b':0.0,'mu':np.zeros(24),'sd':np.ones(24)};hw2,hc2=learned_hidden(d,Z,0.33);tot=d['w_len'].sum()+d['c_len'].sum();share=(d['w_len'][hw2].sum()+d['c_len'][hc2].sum())/tot
    assert 0.33<=share<=0.33+600/tot+1e-9,share
    # 4 random control: same hidden tokens (within one sentence) in window and chunk
    for (a,b) in ((hw,hc),(hw2,hc2)):
        rw,rc=random_like(a,b,d['w_len'],d['c_len'],5,7);assert abs(d['w_len'][rw].sum()-d['w_len'][a].sum())<=d['w_len'].max() and (b.sum()==0 or abs(d['c_len'][rc].sum()-d['c_len'][b].sum())<=d['c_len'].max()),(rw.sum(),a.sum())
        assert not rc[-1],'random chunk sentences exclude the last sentence'
    # 5 compute() with a stub pass and the keep matrix of SelMaskAll
    N=14;ids=np.arange(N*CH,dtype=np.int64);ends=np.zeros(N*CH,bool);ends[np.arange(N*CH)%205==204]=True;ends[np.arange(1,N+1)*CH-1]=True
    def passfn(k,ids_k):
        wsid_,wst_,wlen_=UVm.window_sents(ends,k,CH,NW);q=UVm.chunk_sents(ends,k,CH);d=synth_dump(k,nQ=len(q),nW=len(wlen_));d['w_len']=np.asarray(wlen_,np.int32);d['w_start']=np.asarray(wst_,np.int32);d['q_len']=d['c_len']=np.asarray([b-a for a,b in q],np.int32);d['q_start']=np.asarray([a for a,_ in q],np.int32)
        d['T_wsid']=np.where(d['T_key']<WIN,wsid_[np.clip(d['T_key'].astype(np.int64),0,WIN-1)],-1).astype(np.int16);d['w_chunk']=(k-NW+d['w_start']//CH).astype(np.int32)
        qs=np.repeat(np.arange(len(q)),d['q_len']);d['T_csid']=np.where(d['T_csid']>=0,np.minimum(d['T_csid'],np.maximum(qs-1,0)[None,None,:]),-1).astype(np.int16)
        d['AL']=np.random.default_rng(k).random((6,4,len(q),len(wlen_)+len(q))).astype(np.float16)*0.05;return d
    out,stats=compute(ends,ids,Z,0.33,N,k0=1,passfn=passfn,log=lambda *x:None);assert any(k.startswith('ch1_') for k in out) and len(stats)==N-1
    arm=SelMaskAll('SELC1',2,out,ends,UVm,log=lambda *x:None);k=9;keep,info=arm.keep(k,None,None,None,None,None);W=WIN
    assert keep.shape==(CH,W+CH) and keep.dtype==bool and np.all(keep[np.arange(CH),W+np.arange(CH)]),'a token always sees itself'
    q=UVm.chunk_sents(ends,k,CH);csid=np.full(CH,-1);[csid.__setitem__(slice(a,b),i) for i,(a,b) in enumerate(q)];hc=np.asarray(out[f'ch2_{k}'],bool)
    for i in range(len(q)):
        if hc[i]:
            cols=W+np.nonzero(csid==i)[0];rows_other=np.nonzero(csid!=i)[0];rows_own=np.nonzero(csid==i)[0]
            assert not keep[np.ix_(rows_other,cols)].any(),'hidden chunk sentence invisible to the queries of other sentences';assert keep[np.ix_(rows_own,cols)].all(),'own sentence stays visible'
    assert info['uv_hidden_chunk_tokens']==int(sum(b-a for i,(a,b) in enumerate(q) if hc[i]))
    # the window columns are 1-D equivalent (same for all rows)
    assert (keep[:,:W]==keep[0:1,:W]).all()
    G_=6;ag=SelMaskAll('SELC1',2,out,ends,UVm,log=lambda *x:None,guard=G_);kg,_=ag.keep(k,None,None,None,None,None)
    for i in range(0,CH,37):
        for dd in range(1,G_+1):assert kg[i,W+i-dd],('guard: the tokens right before a query stay visible',i,dd)
    assert (kg|keep).sum()==kg.sum() and kg.sum()>=keep.sum() and not np.array_equal(kg,keep),'the guard only adds visible keys'
    assert kg[:,:W].sum()>keep[:,:W].sum() or kg[:,W:].sum()>keep[:,W:].sum()
    print('e57_mask_stage selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--frac',type=float,default=0.33);ap.add_argument('--coef',default=None);ap.add_argument('--tokenizer',default=None);ap.add_argument('--oracle-all',action='store_true',help='also write the ORACLE answer mask extended to the chunk (rank 5) and its same-amount random control (rank 6)');ap.add_argument('--check-only',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1];import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    coef=a.coef or str(ROOT/'experiments/exp07_write_ladder/e55_sentgrid_ext_s61_coef.npz');model=load_model(coef);assert model['w'].shape==(768,),model['w'].shape
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);fn=out/'e57_pod_masks.npz';ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    print(f'stream {n} chunks, {int(ends.sum())} sentence ends, coefficients {coef} sha12 {hashlib.sha256(open(coef,"rb").read()).hexdigest()[:12]}, frac {a.frac}; output {fn} ({"exists" if fn.exists() else "to be written"})',flush=True)
    if a.check_only:
        for k in (8,64,279):
            q=UVm.chunk_sents(ends,k,CH);print(f'  chunk {k}: {len(q)} chunk sentences, {len(UVm.window_sents(ends,k,CH,NW)[2])} window sentences, sentences >= {MIN_TOK} tokens {sum(1 for a_,b_ in q if b_-a_>=MIN_TOK)}',flush=True)
        if a.oracle_all:
            from tokenizers import Tokenizer;tk=Tokenizer.from_file(tfile);memo={};tw=tc=nw_=nc_=0
            for k in range(8,n,4):
                hw,hc=oracle_hidden(k,ends,ids,tk,items,UVm,memo);q=UVm.chunk_sents(ends,k,CH);wl=UVm.window_sents(ends,k,CH,NW)[2];tw+=int(np.asarray(wl)[hw].sum());nw_+=int(np.sum(wl));tc+=sum(b_-a_ for (a_,b_),h in zip(q,hc) if h);nc_+=CH
            print(f'  oracle (window + chunk) over chunks 8..{n-1} step 4: hidden window share {tw/nw_:.3f}, hidden chunk share {tc/nc_:.3f}',flush=True)
        print('mask stage check-only OK');return
    if fn.exists() and not a.rerun:print('exists, reused:',fn);return
    import e55_run as R;LD,s3,C=R.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'states'/'E57_pod_masks',canary=None);L.rec_cache={};t0=time.monotonic();lg=lambda *x:print(*x,flush=True)
    from tokenizers import Tokenizer;tk=Tokenizer.from_file(tfile);memo={};orc=(lambda k:oracle_hidden(k,ends,ids,tk,items,UVm,memo)) if a.oracle_all else None
    masks,stats=compute(ends,ids,model,a.frac,n,L=L,rt=rt,s3=s3,UVm=UVm,log=lg,oracle=orc);np.savez_compressed(fn,**masks)
    tot={}
    for s in stats:
        for k_,v in s.items():
            if k_!='k':tot[k_]=tot.get(k_,0)+v
    rep={'chunks':len(stats),'frac':a.frac,'coef_sha12':hashlib.sha256(open(coef,'rb').read()).hexdigest()[:12],'mean_per_chunk':{k_:v/len(stats) for k_,v in tot.items()},'per_chunk':stats,'sec_total':time.monotonic()-t0,'npz_bytes':fn.stat().st_size}
    (out/'e57_pod_masks_report.json').write_text(json.dumps(rep,default=float));print('wrote',fn,fn.stat().st_size,'bytes',json.dumps(rep['mean_per_chunk']))
if __name__=='__main__':main()
