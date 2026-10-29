"""E57 band reads of written states (job 3xx; GPU forward only, NO weight update; post-hoc, exploratory; new file). For every saved written state (--rstates LABEL=FILE:HASH,...; W0 = LABEL=@W0) the 42 questions of the frozen bank are read like the OUT / IN reads of e57_main_run (candidate rows of e57_reads, summed span log-prob of the candidate after the question prefix, fp32, packed scorer packing.Packed) but with a BAND context: every token of the read (neutral OUT tail / IN window / question / candidate) sees only the previous b tokens (b = 512; itself included), in every prefix AND suffix layer. Per state and condition: question rows of e57_reads.log_rows (gold log-prob, gold rank, rank ratio, accuracy all / hard, every candidate log-prob) with condition OUT_B512 / IN_B512; the window reads of the same states (condition OUT / IN) come from the stored arm files unless --recompute-window.
How the band is applied (new code, nothing of packing.py / ladder.py is edited): (1) prefix layers: the packed prefix pass of packing.Packed is repeated for the band with an explicit mask (q >= k) & (0 <= pos_q - pos_k < b) & (key is context or the same probe), where pos = the true position of the token (context 0..8191, probe 8192 + local index; the original Packed uses the index inside the packed chunk); the neutral tail / window context gets its own band prefix pass; (2) suffix layers: SWA.sw_causal_mask is wrapped with the same distance rule (window columns: loc + W - c, chunk columns: loc_q - loc_k, loc = local position of the probe token) while BAND['b'] is set INSIDE new jitted functions (chunk_logp_b for the context chunks, lp_b for the packed probe chunk), so the band is part of the trace and never shared with the ordinary reads. CONTROL (every run): the band-8192 read with the index-distance prefix mask must reproduce packing.Packed.r1 of W0 on OUT and IN (max |diff| of the candidate log-probs, stored; the run stops above 0.05).
  python3 e57_band_read_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --rstates W0=@W0,NAT=FILE:HASH,... [--bands 512] [--refjson LABEL=PATH,...] [--recompute-window] --stage reads   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib,os
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
CH=1024
BAND={'b':None}
RS={'states':None,'bands':[512],'ref':{},'recompute':False}
def band_prefix_mask(pos,seg,b,index_mode=False):
    """bool [T,T]: sequence order (q >= k), distance d = pos_q - pos_k (index_mode: sequence index distance, as packing.prefix_mask), 0 <= d < b, key is a context token (seg -2) or in the same probe"""
    pos=np.asarray(pos);seg=np.asarray(seg);T=len(pos);q=np.arange(T)[:,None];k=np.arange(T)[None,:]
    d=(q-k) if index_mode else (pos[:,None]-pos[None,:])
    return (q>=k)&(d>=0)&(d<b)&((seg[None,:]==-2)|(seg[None,:]==seg[:,None]))
def suffix_band_mask(m,W,B,qloc,kloc,b,xp):
    """m [B, W+B] the ordinary suffix mask; qloc [B] true local positions of the query tokens; kloc [B] those of the chunk keys; window key column c has position c - W -> distance qloc + W - c; chunk key distance qloc - kloc"""
    kpos=xp.concatenate([xp.arange(W)-W,kloc]);d=qloc[:,None]-kpos[None,:];return m&(d<b)
def patch_swa(PK):
    import jax.numpy as jnp
    if getattr(PK,'_band_patched',False):return
    prev=PK.SWA.sw_causal_mask
    def wrapped(self,chunk_id):
        m=prev(self,chunk_id);b=BAND['b']
        if b is None:return m
        W,B=self.window_size,self.mini_batch_size
        if 'loc' in PK.PACK_SUF:loc=PK.PACK_SUF['loc'];return suffix_band_mask(m,W,B,loc,loc,b,jnp)
        ar=jnp.arange(B);return suffix_band_mask(m,W,B,ar,ar,b,jnp)
    PK.SWA.sw_causal_mask=wrapped;PK._band_patched=True
class BandReader:
    def __init__(s,P,PK,jax,jnp,eqx):
        s.P=P;s.T=P.T;s.rt=P.rt;s.PK=PK;s.jax=jax;s.jnp=jnp;s.eqx=eqx;s.cache={};s.fns={}
    def _fns(s,b):
        if b in s.fns:return s.fns[b]
        jax,jnp,eqx,PK,rt=s.jax,s.jnp,s.eqx,s.PK,s.rt
        from ttt.model.data import Batch
        @eqx.filter_jit
        def prefix_b(m,ids_,pos_,mask_):
            PK.PACK_PRE['mask']=mask_
            try:
                seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
                return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),PK._clone(rt.state_prefix),seq).last_hidden_state
            finally:PK.PACK_PRE.clear()
        @eqx.filter_jit
        def chunk_logp_b(m,st,seq,pref):
            BAND['b']=b
            try:
                r=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=seq);lp=jax.nn.log_softmax(r.logits.astype(jnp.float32),axis=-1)
                return r.new_state,jnp.take_along_axis(lp,seq.target_tokens[:,None],axis=-1)[:,0]
            finally:BAND['b']=None
        @eqx.filter_jit
        def lp_b(m,st,ids_,tg_,mk_,pref_,seg_,loc_):
            def one(i,t,k,p,sg,lc):
                PK.PACK_SUF['seg']=sg;PK.PACK_SUF['loc']=lc;BAND['b']=b
                try:
                    r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k));lg=r.logits.astype(jnp.float32)
                    return jnp.take_along_axis(lg,t[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
                finally:PK.PACK_SUF.clear();BAND['b']=None
            return jax.vmap(one)(ids_,tg_,mk_,pref_,seg_,loc_)
        s.fns[b]=(prefix_b,chunk_logp_b,lp_b);return s.fns[b]
    def prep(s,b,index_mode):
        key=(b,index_mode)
        if key in s.cache:return s.cache[key]
        jnp=s.jnp;rt=s.rt;P=s.P;T=s.T;prefix_b=s._fns(b)[0];TL=len(T.tail_ids);tail=np.asarray(T.tail_ids,np.int32)
        ids=np.asarray(P.ids);seg=np.asarray(P.seg);loc=np.asarray(P.loc)
        with rt.mesh:
            pos=np.arange(TL,dtype=np.int32);tp=prefix_b(rt.base,jnp.asarray(tail),jnp.asarray(pos),jnp.asarray(band_prefix_mask(pos,np.full(TL,-2,np.int32),b,index_mode)))
            p1=[]
            for j in range(ids.shape[0]):
                full=np.concatenate([tail,ids[j]]);pj=np.concatenate([pos,TL+loc[j]]);sf=np.concatenate([np.full(TL,-2,np.int32),seg[j]])
                p1.append(prefix_b(rt.base,jnp.asarray(full),jnp.asarray(pj),jnp.asarray(band_prefix_mask(pj,sf,b,index_mode)))[TL:])
            p1=jnp.stack(p1);s.jax.block_until_ready(p1)
        s.cache[key]=(tp,p1);return s.cache[key]
    def score(s,m,b,index_mode=False):
        """summed candidate log-probs, in the order of P.where (like Packed.r1) but with the band b"""
        PK=s.PK;rt=s.rt;P=s.P;T=s.T;tp,p1=s.prep(b,index_mode);_,chunk_logp_b,lp_b=s._fns(b);out=[]
        with rt.mesh:
            st=PK._clone(rt.state_suffix)
            for j in range(len(T.tail_ids)//1024):st,_=chunk_logp_b(m,st,rt.chunk(T._tail_seq,j),tp[j*1024:(j+1)*1024])
            for g in range(0,P.ids.shape[0],P.GP):out.append(np.asarray(lp_b(m,st,P.ids[g:g+P.GP],P.tg[g:g+P.GP],P.mk[g:g+P.GP],p1[g:g+P.GP],P.seg[g:g+P.GP],P.loc[g:g+P.GP])))
        lp=np.concatenate(out);return np.asarray([lp[c,a:b_].sum() for c,a,b_ in P.where])
def band_read_stage(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG,e57_reads as RD,jax,jax.numpy as jnp,equinox as eqx
    PK=LD.packing;patch_swa(PK);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P_out=rd.packed(raw,out_tail);kk=kks[-1];pos_tokens=(kk+1)*CH
    ids2=np.tile(ids_all[:n*CH],2);s_,e_=max(0,(n+kk+1)*CH-8192),(n+kk+1)*CH;win_ids=[int(x) for x in ids2[s_:e_]];P_in=rd.packed(raw,win_ids);wp=RD.window_present(items,tk.decode(win_ids))
    csa={it['id']:it['q'].get('c_SA') for it in items};readers={'OUT':BandReader(P_out,PK,jax,jnp,eqx),'IN':BandReader(P_in,PK,jax,jnp,eqx)};Ps={'OUT':P_out,'IN':P_in}
    def window_scores(P,m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return np.asarray(P.r1(m32))
    # CONTROL: band 8192 (index-distance prefix mask) == the ordinary read of W0
    ctrl={};m32=rd.to_fp32(rt.base)
    with LD.jax.default_matmul_precision('highest'):
        for cond in ('OUT','IN'):
            base=np.asarray(Ps[cond].r1(m32));br=readers[cond].score(m32,8192,index_mode=True);ctrl[cond]=float(np.abs(base-br).max());lg(f'  control W0 {cond}: band 8192 vs Packed.r1 max |diff| of the candidate log-probs {ctrl[cond]:.5f}')
    assert max(ctrl.values())<0.05,('band pipeline does not reproduce the ordinary read',ctrl)
    out.mkdir(parents=True,exist_ok=True);(out/'E57_bandread_control.json').write_text(json.dumps(ctrl))
    for lab,(fil,hh) in RS['states']:
        fn=out/f'E57_bandread_{lab}.json'
        if fn.exists() and not a.rerun:print('exists, reused:',fn);continue
        t0=time.monotonic();w0=(fil=='');m=rt.base if w0 else rd.load_state(fil,expect_hash=hh);cyc=0 if w0 else 1;rc=kk if w0 else n+kk;rows=[]
        with LD.jax.default_matmul_precision('highest'):
            m32=rd.to_fp32(m)
            for b in RS['bands']:
                for cond in ('OUT','IN'):
                    sc=RD.split_scores(readers[cond].score(m32,b),items);rows+=RD.log_rows(items,sc,ntok,cycle=cyc,read_chunk=rc,pos_tokens=pos_tokens,win=None if cond=='OUT' else (s_,e_),arm=lab,condition=f'{cond}_B{b}',iv=iv,hits=hits,winpresent=None if cond=='OUT' else wp,csa=csa,shadow=shtab)
            wrows=[]
            if RS['recompute']:
                for cond in ('OUT','IN'):sc=RD.split_scores(window_scores(Ps[cond],m),items);wrows+=RD.log_rows(items,sc,ntok,cycle=cyc,read_chunk=rc,pos_tokens=pos_tokens,win=None if cond=='OUT' else (s_,e_),arm=lab,condition=cond,iv=iv,hits=hits,winpresent=None if cond=='OUT' else wp,csa=csa,shadow=shtab)
            elif lab in RS['ref']:
                ref=json.load(open(RS['ref'][lab]))['rows'];wrows=[r for r in ref if r['read_chunk']==rc and r['condition'] in ('OUT','IN')]
        fn.write_text(json.dumps(dict(label=lab,state_file=fil or None,state_hash=hh or None,bands=RS['bands'],bank=bsrc,control=ctrl,window_rows_source=('recomputed' if RS['recompute'] else RS['ref'].get(lab)),rows_band=rows,rows_window=wrows,sec=time.monotonic()-t0),default=float));lg(f'  {lab}: band reads written {fn} ({time.monotonic()-t0:.0f}s)')
        if not w0:del m
        gc.collect()
def main():
    def pop(flag):
        if flag in sys.argv:
            i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
    if '--selftest' in sys.argv:selftest();return
    v=pop('--rstates');assert v,'--rstates is required';RS['states']=[]
    for x in v.split(','):
        lab,_,r=x.partition('=');fil,_,hh=r.partition(':');fil=fil.lstrip('@') if r=='@W0' else fil;assert lab and (r=='@W0' or (fil and hh)),('LABEL=FILE:HASH or LABEL=@W0',x);RS['states'].append((lab,('' if r=='@W0' else fil,hh)))
    bd=pop('--bands');RS['bands']=[int(x) for x in bd.split(',')] if bd else [512]
    rj=pop('--refjson');RS['ref']=dict(x.split('=',1) for x in rj.split(',')) if rj else {}
    if '--recompute-window' in sys.argv:sys.argv.remove('--recompute-window');RS['recompute']=True
    import e57_main_run as M
    M.stage_reads=band_read_stage;M.main()
def selftest():
    import jax.numpy as jnp
    rng=np.random.default_rng(0);TL=40;L=30;pos=np.concatenate([np.arange(TL),TL+np.arange(L)]);seg=np.concatenate([np.full(TL,-2),np.zeros(10,int),np.ones(12,int),np.full(8,2)])
    loc=np.concatenate([np.arange(10),np.arange(12),np.arange(8)]);pos=np.concatenate([np.arange(TL),TL+loc])
    for b in (5,16,10**6):
        M=band_prefix_mask(pos,seg,b)
        for i in range(len(pos)):
            for j in range(len(pos)):
                exp=(j<=i) and 0<=pos[i]-pos[j]<b and (seg[j]==-2 or seg[j]==seg[i]);assert M[i,j]==exp,(b,i,j)
    # index mode == the window rule of packing.prefix_mask
    T=len(pos);q=np.arange(T)[:,None];k=np.arange(T)[None,:];ref=(q>=k)&(q-k<20)&((seg[None,:]==-2)|(seg[None,:]==seg[:,None]));assert np.array_equal(band_prefix_mask(pos,seg,20,index_mode=True),ref)
    # suffix mask: window columns loc+W-c < b, chunk columns loc_q-loc_k < b; b >= W+B is the identity of the ordinary mask
    W,B=24,8;loc2=jnp.asarray(np.array([0,1,2,3,0,1,2,3]));m=jnp.ones((B,W+B),bool)
    bm=np.asarray(suffix_band_mask(m,W,B,loc2,loc2,10,jnp))
    for i in range(B):
        for c in range(W+B):
            kp=(c-W) if c<W else int(loc2[c-W]);assert bm[i,c]==((int(loc2[i])-kp)<10),(i,c)
    assert np.asarray(suffix_band_mask(m,W,B,jnp.arange(B),jnp.arange(B),W+B,jnp)).all()
    print('e57_band_read_run selftest ok')
if __name__=='__main__':main()
