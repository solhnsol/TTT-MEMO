"""8-chunk in-window reads R8 / R8m / R8a on saved final weights (job 1366b 'win8read'; GPU; READ ONLY; new file; implementation thread 11/28; Hans rule 11/27: chunk-context reads use the whole 8-chunk window). Context = chunks 120..127 (8192 tokens ending with the last written chunk), fresh suffix state; every question scored on all 16 x 16 candidates (12 exposed + 4 never exposed):
  R8 nothing hidden; R8m each question's own fact sentences in all 8 context chunks hidden from the question tokens (prefix and suffix layers, positions kept); R8a the sentences of all 16 read facts hidden. W0 scored with the same variants.
Sources: --srcs 'ARM=PATTERN,...' with PATTERN a json of the e59_kvmulti / e59_cbo format containing reads['128']['final_state'] ({seed} is replaced).
  python3 e59_win8read.py --seed 231 --out O --srcs 'L2=experiments/.../kvmulti_s{seed}.json,...' [--k 127]"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
from e59_xstep_r8 import hide_sets8,ctx_chunks
NREAD=16;CH=1024;VARIANTS=('R8','R8m','R8a')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP
    from e59_kvpos2 import items2,NEXP
    from ttt.model.data import Batch
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);LD,s3_,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();PS=LD.packing
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ci['read_apps']),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX);ids0=np.asarray(ids0);ch=w['chunks']
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);NC=G.NCAND
    SWA=PS.SWA;_prev=SWA.sw_causal_mask
    def _mask2(self,chunk_id):
        m=_prev(self,chunk_id)
        if 'hide' in PS.PACK_SUF:h=PS.PACK_SUF['hide'];m=m&jnp.concatenate([~h,jnp.ones((self.mini_batch_size,self.mini_batch_size),bool)],axis=1)
        return m
    SWA.sw_causal_mask=_mask2
    Wsl=int(rt32.cfg.model.sliding_window_size)
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
    class R8Ctx:
        def __init__(self,k):
            self.k=k;cc=ctx_chunks(k);ctx=[int(t) for c in cc for t in ids0[c*CH:(c+1)*CH]];H=hide_sets8(ch,k)
            with rt32.mesh,jax.default_matmul_precision('highest'):self.T=LD._TailCtx(rt32,ctx);self.PK=PS.Packed(self.T,raw,GP=2)
            PK=self.PK;TL=len(self.T.tail_ids);assert TL==len(cc)*CH and TL<=Wsl;npad=PK.ids.shape[0];seg_np=np.asarray(PK.seg);loc_np=np.asarray(PK.loc);ids_np=np.asarray(PK.ids);qmap=[dict() for _ in range(npad)];cnt={}
            for r,(c,_a,_b) in enumerate(PK.where):s=cnt.get(c,0);cnt[c]=s+1;qmap[c][s]=r//NC
            for c in range(PK.nchunk,npad):qmap[c]=qmap[PK.nchunk-1]
            self.VAR={};self.n_hidden={v:H[v].sum(1).tolist() for v in VARIANTS}
            for name in VARIANTS:
                Hv=H[name];pref=[];hide=np.zeros((npad,CH,Wsl),bool)
                with rt32.mesh,jax.default_matmul_precision('highest'):
                    for j in range(npad):
                        sj,lj=seg_np[j],loc_np[j];sf=np.concatenate([np.full(TL,-2,np.int32),sj]);mk=PS.prefix_mask(sf,Wsl)
                        for s,q in qmap[j].items():
                            rows=np.where(sj==s)[0]
                            if len(rows)==0:continue
                            mk[np.ix_(TL+rows,np.arange(TL))]&=~Hv[q][None,:];hide[j][np.ix_(rows,Wsl-TL+np.arange(TL))]=Hv[q][None,:]
                        full=np.concatenate([np.asarray(self.T.tail_ids,np.int32),ids_np[j]]);pos=np.concatenate([np.arange(TL,dtype=np.int32),TL+lj])
                        pref.append(prefix_h(rt32.base,jnp.asarray(full),jnp.asarray(pos),jnp.asarray(mk))[TL:])
                    self.VAR[name]=(jnp.stack(pref),jnp.asarray(hide))
        def score(self,m):
            m32=rd.to_fp32(m);T_=self.T;PK=self.PK;npad=PK.ids.shape[0];res={}
            with rt32.mesh:
                st=PS._clone(rt32.state_suffix)
                for j in range(len(T_.tail_ids)//CH):st,_=rt32.chunk_logp(m32,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*CH:(j+1)*CH])
            with rt32.mesh,jax.default_matmul_precision('highest'):
                for nm,(pref,hide) in self.VAR.items():
                    o=[]
                    for g in range(0,npad,PK.GP):o.append(np.asarray(lp_hide(m32,st,PK.ids[g:g+PK.GP],PK.tg[g:g+PK.GP],PK.mk[g:g+PK.GP],pref[g:g+PK.GP],PK.seg[g:g+PK.GP],PK.loc[g:g+PK.GP],hide[g:g+PK.GP])))
                    lp=np.concatenate(o);res[nm]=np.asarray([lp[c,a_:b_].sum() for c,a_,b_ in PK.where]).reshape(len(items),NC)[:NREAD].tolist()
            del m32;return res
    k=a.k;t0=time.monotonic();cx=R8Ctx(k);lg(f'seed {seed}: 8-chunk context {ctx_chunks(k)} built {time.monotonic()-t0:.0f}s')
    RES=dict(seed=seed,k=k,ctx_chunks=ctx_chunks(k),variants=list(VARIANTS),hidden_counts=cx.n_hidden,cand_exposed=[True]*NEXP+[False]*(NC-NEXP),W0=cx.score(rt.base),arms={},args=vars(a))
    SRC={x.split('=')[0]:x.split('=')[1] for x in a.srcs.split(',')}
    for arm,pat in SRC.items():
        sj=json.load(open(ROOT/pat.format(seed=seed)));fs=sj['arms'][arm]['reads']['128']['final_state'];m=rd.load_state(ROOT/fs['_file'],fs['_hash']);t0=time.monotonic();RES['arms'][arm]=cx.score(m);lg(f'seed {seed} {arm}: scored {time.monotonic()-t0:.0f}s');del m
        json.dump(RES,open(out/f'win8_s{seed}.json','w'))
    json.dump(RES,open(out/f'win8_s{seed}.json','w'));lg('wrote',out/f'win8_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--srcs',default='');ap_.add_argument('--k',type=int,default=127);a=ap_.parse_args();run(a)
