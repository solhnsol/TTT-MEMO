"""Diagnosis of the two-path disagreement of job 1347 (generation path vs score path, first-token log p; max 0.12 on the GPU) (CPU only; new file; thread-server 11/25). Same code objects as e62w_run (Packed, lp_packed, tail state, info_packed) on a small input: 2 questions x 16 candidates (invented words, neutral tail), the SAME frozen W0 fp32 model. Variants: A = score path, Packed of the 2 x 16 candidate rows; B = generation path, Packed of the 2 question prefixes; C = score path with ONLY the gold row of question 0; D = generation path with the prefix of question 0 alone; E = generation path with the prefixes in reverse order. Prints the first-token log p of the 16 candidates of question 0 for each variant and the pairwise max differences. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu."""
import sys,os,time,pathlib,random,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import jax,jax.numpy as jnp,equinox as eqx
import e55_run as R55,e55_gpu as RG,e62_templates as TP
lg=lambda *x:print(*x,flush=True);t0=time.monotonic()
_cur=jax.nn.dot_product_attention;jax.nn.dot_product_attention=lambda *a_,**k_:_cur(*a_,**dict(k_,implementation='xla'))
LD,s3,C=R55.lab();from ttt.model.data import Batch;rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp);PS=LD.packing
out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,[]);NC=16
rng=random.Random(1);words=['Reamaeton','Limeworth','Piapiwick','Wisomere','Taramere','Shashaeton','Mutialey','Zuliworth','Shapesumere','Latrialey','Pedatestead','Higiwick','Tetoworth','Letomere','Nuketon','Buluwick','Kovowick','Movumere']
items=[dict(prefix=TP.FRAMES['birth'][0].replace('{F}',nm),cands=[' '+w for w in words[i*2:i*2+NC]]) for i,nm in enumerate(['Josiah Stitiwell','Hosea Tekoworth'])]
raw=R55.rows_for(LD,rt,items);lg('rows',len(raw),f'{time.monotonic()-t0:.0f}s')
with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail))
lg(f'tail ctx {time.monotonic()-t0:.0f}s');m32=rd.to_fp32(rt.base)
def tail_state(m32):
    with rt32.mesh:
        st=PS._clone(rt32.state_suffix)
        for j in range(len(T_.tail_ids)//1024):st,_=rt32.chunk_logp(m32,st,rt32.chunk(T_._tail_seq,j),T_._tail_pref[j*1024:(j+1)*1024])
    return st
@eqx.filter_jit
def info_packed(m,st,ids_,pref_,seg_,loc_,pos_,tok_):
    def one(i,pp,sg,lc,ps,tk_):
        PS.PACK_SUF['seg']=sg;PS.PACK_SUF['loc']=lc
        try:
            r=m.language_model.suffix_call(prefix_outputs=pp,state=st,seq=Batch(input_ids=i,target_tokens=i,loss_masks=jnp.ones_like(i,dtype=bool)))
            lg_=r.logits.astype(jnp.float32)[ps];val=jnp.take_along_axis(lg_,tk_,axis=-1);return val-jax.nn.logsumexp(lg_,axis=-1)[:,None]
        finally:PS.PACK_SUF.clear()
    return jax.vmap(one)(ids_,pref_,seg_,loc_,pos_,tok_)
def gen_path(rows_ids,tm,st):
    P_=PS.Packed(T_,[(list(x),(len(x)-1,len(x))) for x in rows_ids],GP=2);cnt=collections.Counter();nr=max(collections.Counter(w_[0] for w_ in P_.where).values());npad=P_.ids.shape[0]
    pos=np.zeros((npad,nr),np.int32);tok=np.zeros((npad,nr,NC),np.int32);slot=[]
    for i,(c,a_,b_) in enumerate(P_.where):j=cnt[c];cnt[c]+=1;pos[c,j]=a_;tok[c,j]=tm[i];slot.append((c,j))
    with rt32.mesh,jax.default_matmul_precision('highest'):lpc=np.asarray(info_packed(m32,st,P_.ids,P_.pref1,P_.seg,P_.loc,jnp.asarray(pos),jnp.asarray(tok)))
    return np.asarray([lpc[c,j] for c,j in slot]),P_
def score_path(rows,st):
    P_=PS.Packed(T_,rows,GP=2);out=[]
    with rt32.mesh,jax.default_matmul_precision('highest'):
        for g in range(0,P_.ids.shape[0],P_.GP):out.append(np.asarray(P_.lp_packed(m32,st,P_.ids[g:g+P_.GP],P_.tg[g:g+P_.GP],P_.mk[g:g+P_.GP],P_.pref1[g:g+P_.GP],P_.seg[g:g+P_.GP],P_.loc[g:g+P_.GP])))
    lp=np.concatenate(out);return np.asarray([lp[c,a_] for c,a_,b_ in P_.where]),P_
st=tail_state(m32);lg(f'tail state {time.monotonic()-t0:.0f}s')
pref=[list(raw[i*NC][0][:raw[i*NC][1][0]+1]) for i in range(2)];tokmat=np.asarray([[int(raw[i*NC+c][0][raw[i*NC+c][1][0]+1]) for c in range(NC)] for i in range(2)],np.int32)
A,PA=score_path(raw,st);A=A.reshape(2,NC);lg(f'A score path (2x16 rows) {time.monotonic()-t0:.0f}s, chunks {PA.nchunk}')
B,PB=gen_path(pref,tokmat,st);lg(f'B generation path (2 prefixes) {time.monotonic()-t0:.0f}s, chunks {PB.nchunk}')
Cc,_=score_path([raw[0]],st);lg(f'C score path, gold row of q0 only {time.monotonic()-t0:.0f}s')
D,_=gen_path(pref[:1],tokmat[:1],st);lg(f'D generation path, q0 alone {time.monotonic()-t0:.0f}s')
E,_=gen_path(pref[::-1],tokmat[::-1],st);E=E[::-1];lg(f'E generation path, reversed order {time.monotonic()-t0:.0f}s')
np.set_printoptions(precision=3,suppress=True,linewidth=200)
lg('first-token log p of the 16 candidates of question 0:');lg(' A',A[0]);lg(' B',B[0]);lg(' D',D[0]);lg(' C (gold only)',Cc[0],'| A gold',A[0,0],'B gold',B[0,0],'D gold',D[0,0])
for nm,x,y in (('A vs B',A,B),('B vs D (q0)',B[:1],D),('B vs E',B,E),('A[0,0] vs C',A[:1,:1],Cc[:1].reshape(1,1)),('A vs D (q0)',A[:1],D)):lg(f'max |{nm}| = {np.abs(x-y).max():.4f}, mean {np.abs(x-y).mean():.4f}')
