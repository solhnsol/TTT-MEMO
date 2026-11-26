"""Key diagnostic of write keys (job 1344t3 'kvkeys'; GPU; FORWARD ONLY, no write; new file; thread-server 11/26; Hans 'S2611-T3-X-OK', state definition A). For the weights W0 and the saved 128-exposure weights of the arms (P0s, P1s from job 1344t0; M2As-q, M2As-s from job 1344t2; M2As-x later) and the chunks e = 16, 64, 128 (chunk index e-1), for the attention masks C0, SAMEVIS and M2As-q: the input x2 = silu(w1 x) * (w3 x) of the 8 write-target (prime SwiGLU) layers at the positions that predict the FIRST place token of each of the 16 read facts (write keys wk[f, layer]) is compared with the READ keys rk[g, layer] of job 1344k (kv1344k/kv_s<seed>_keys.npy: x2 at the position that predicts the first answer token of the gold row of fact g in the neutral read prompt (8192-token neutral tail), W0 fp32):
  (a) cos_own[f, l] = cos(wk_f, rk_f)  and cos_oth[f, l] = mean over the 15 other facts g of cos(wk_f, rk_g)   (the stored difference cos_own - cos_oth is the requested value)
  (b) cos_c0[f, l] = cos(wk_f under the mask, wk_f under C0) at the same weights and chunk.
Values per layer and per fact are stored as they are (lists [16][8]); write-key norms too.
STATE DEFINITION A and its limit (also written into every result file): the forward of chunk k starts from an EMPTY suffix state at the start of chunk k-8, i.e. the 8 preceding chunks are run through the model from scratch with the weights under study (suffix state = their own keys / values with the weights under study), the prefix layers see the 8 chunks plus chunk k as one sequence without anything before chunk k-8. Therefore the chunks of the window are computed WITHOUT their own earlier context (the first chunks of the window do not see their natural predecessors), unlike the carried state of the real write run (which also carried the weights of that moment, not the end weights). The window of chunk k is the same 8 chunks for all weights and masks.
  python3 e59_keydiag.py --seed 231 --out O --weights "P0s=...json,P1s=...json" [--tag T] | --selftest"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
from e59_kvpos2 import NREAD,CH,CONDS
NW=8;WIN=NW*CH;NL=8
LIMIT=("state definition A: empty suffix state at the start of chunk k-8, the 8 preceding chunks run from scratch with the weights under study; prefix layers see the 8 chunks + chunk k as one sequence without earlier context; the first chunks of the window are computed without their own earlier context; the carried state of the real write run (own weights of that moment) is NOT reproduced; read keys = W0 fp32 neutral-read keys of job 1344k")
def cosm(A,B):
    """A [n,I], B [m,I] -> cosine matrix [n,m]"""
    A=A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-30);B=B/np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-30);return A@B.T
def selftest():
    rng=np.random.default_rng(0);A=rng.normal(size=(16,40));B=rng.normal(size=(16,40));c=cosm(A,B);assert c.shape==(16,16) and abs(c[3,3]-float(A[3]@B[3]/np.linalg.norm(A[3])/np.linalg.norm(B[3])))<1e-12
    own=np.diag(c);oth=(c.sum(1)-own)/15;assert np.allclose(own-oth,np.diag(c)-(c.sum(1)-np.diag(c))/15)
    print('e59_keydiag selftest ok: cosine matrix, own minus mean of the 15 others')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_gpu as RG,e57_natmask_run as NM,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_shadow as SH,e59_kvmask as KM,e59_kvprobe as KP
    NM.register_engine();lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp);PS=LD.packing
    if os.environ.get('E59_CPU_TEST'):
        _cur=jax.nn.dot_product_attention;jax.nn.dot_product_attention=lambda *a_,**k_:_cur(*a_,**dict(k_,implementation='xla'))
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];ids0=np.asarray(ids0);tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,w['n_chunks']);assert miss==0;ch=w['chunks']
    keys=np.load(f'{a.keys_dir}/kv_s{seed}_keys.npy');assert keys.shape[0]==NREAD and keys.shape[2]==NL,keys.shape;RK=keys[:,0]            # [16, 8, I] first answer token, W0
    W=int(rt32.cfg.model.sliding_window_size);assert W==WIN
    inner=SH.mk_uv(CONDS['SAMEVIS'],w,seed,lg);UVS={'C0':SH.mk_uv(CONDS['C0'],w,seed,lg),'SAMEVIS':inner,'M2As-q':KM.QMaskUV(inner,'q',ch,ap,seed,lg)};MASKS=('C0','SAMEVIS','M2As-q')
    @eqx.filter_jit
    def prefix_h(m,ids_,pos_,mask_):
        PS.PACK_PRE['mask']=mask_
        try:
            from ttt.model.data import Batch
            seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
            return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),PS._clone(rt32.state_prefix),seq).last_hidden_state
        finally:PS.PACK_PRE.clear()
    @eqx.filter_jit
    def fwd_cap(mp,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=mp.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        return jnp.sum(o.logits[:8,:8].astype(jnp.float32))
    def tail_state(m32,T_):
        with rt32.mesh:
            st=PS._clone(rt32.state_suffix)
            for j in range(len(T_.tail_ids)//CH):st,_=rt32.chunk_logp(m32,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*CH:(j+1)*CH])
        return st
    WT={'W0':None}
    for x in (a.weights.split(',') if a.weights else []):
        nm,f=x.split('=');r=json.load(open(ROOT/f.format(seed=seed)));fs=r['arms'][nm]['reads']['128']['final_state'];WT[nm]=(ROOT/fs['_file'],fs['_hash'])
    RES=dict(seed=seed,limitations=LIMIT,masks=list(MASKS),weights=list(WT),chunks={},read_key_file=f'{a.keys_dir}/kv_s{seed}_keys.npy',args=vars(a));models={}
    for e in [int(x) for x in a.chunks.split(',')]:
        k=e-1;tail=[int(t) for t in ids0[(k-NW)*CH:k*CH]];cur=[int(t) for t in ids0[k*CH:(k+1)*CH]];t0=time.monotonic();rows=np.asarray([min(d['place'])-1 for d in sorted([d for d in ap[k] if d['fact']<NREAD],key=lambda d:d['fact'])],np.int64);assert len(rows)==NREAD
        with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,tail);c=rt32.seq(cur)
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;TL=len(tail);assert TL==WIN;PREF={};KM_={}
        for mk in MASKS:
            kp,_=UVS[mk].keep(k,None,None,None,None,base);kp=np.asarray(kp,bool);KM_[mk]=jnp.asarray(kp);mask=PS.prefix_mask(np.full(TL+CH,-2,np.int32),W).copy();mask[TL:,:]&=kp
            with rt32.mesh,jax.default_matmul_precision('highest'):PREF[mk]=prefix_h(rt32.base,jnp.asarray(np.asarray(tail+cur,np.int32)),jnp.arange(TL+CH,dtype=jnp.int32),jnp.asarray(mask))[TL:]
        RES['chunks'][str(e)]={}
        for wn,wf in WT.items():
            if wn not in models:models[wn]=rd.to_fp32(rt.base) if wf is None else rd.to_fp32(rd.load_state(wf[0],wf[1]))
            m32=models[wn]
            with rt32.mesh,jax.default_matmul_precision('highest'):
                st=tail_state(m32,T_);mp=KP.install(m32,'fwd');wk={}
                for mk in MASKS:
                    KP.CAP['fwd'].clear();v=fwd_cap(mp,st,c,PREF[mk],KM_[mk]);jax.block_until_ready(v);jax.effects_barrier();cap=dict(KP.CAP['fwd']);assert sorted(cap)==list(range(NL)),sorted(cap)
                    wk[mk]=np.stack([cap[l][1][rows] for l in range(NL)],axis=1)          # [16, 8, I]
            rec={}
            for mk in MASKS:
                own=np.zeros((NREAD,NL));oth=np.zeros((NREAD,NL));c0=np.zeros((NREAD,NL));kn=np.zeros((NREAD,NL))
                for l in range(NL):
                    cm=cosm(wk[mk][:,l],RK[:,l]);own[:,l]=np.diag(cm);oth[:,l]=(cm.sum(1)-np.diag(cm))/(NREAD-1);c0[:,l]=np.diag(cosm(wk[mk][:,l],wk['C0'][:,l]));kn[:,l]=np.linalg.norm(wk[mk][:,l],axis=1)
                rec[mk]=dict(cos_own=own.tolist(),cos_oth_mean=oth.tolist(),cos_own_minus_oth=(own-oth).tolist(),cos_to_C0_write_key=c0.tolist(),write_key_norm=kn.tolist())
            RES['chunks'][str(e)][wn]=rec;lg(f'seed {seed} chunk {e} weights {wn}: own-oth layer 8 mean ({", ".join(f"{mk} {np.mean(rec[mk]["cos_own_minus_oth"],axis=0)[7]:.4f}" for mk in MASKS)}) {time.monotonic()-t0:.0f}s')
        del T_
    json.dump(RES,open(out/f'kvkeys{a.tag}_s{seed}.json','w'),default=float);lg('wrote',out/f'kvkeys{a.tag}_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--chunks',default='16,64,128');ap_.add_argument('--weights',default='');ap_.add_argument('--tag',default='');ap_.add_argument('--keys-dir',default='experiments/exp07_write_ladder/kv1344k');ap_.add_argument('--selftest',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
