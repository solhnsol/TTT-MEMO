"""E57 value-corrected key-mass dump (GPU forward only, W0, no write; post-hoc, exploratory; not yet run). Same pass as e57_attdump3 (unmasked natural window, prefix layers, sharp heads of the chunk from e57_sharp_heads.npz) on the chunks of a mask file (e57_shadowkey_masks.npz: the 17 chunks k % 16 == 0), but for the SELECTED rows only (rows_<k> of the mask file) it keeps the FULL window vectors (all 8192 window keys), float32:
  SHV, ALLV    sum over heads and layers of p x ||v_key||  (raw value-corrected mass; ||v|| = norm of the key's value vector of that head, as in e57_attdump3; SHV over the sharp heads, ALLV over all 768 heads)
  SHVn, ALLVn  per-head value-normalised share, summed over heads and layers: p x ||v_key|| / sum over the WINDOW keys of p x ||v_key|| of the same head and query row (the chunk keys are not part of the normaliser: their value norms are not passed to the dump)
  SH, ALL      the plain masses again (control; equal to dense_SH / dense_ALL of job 295 up to fp16 rounding)
Output <out>/E57_valuedump_W0.npz: keys 'c<k>/<name>' float32 [R, 8192] (R = rows of the mask file, 102), chunks, rows_<k>. Row r of the arrays = rows_<k>[r]. About 4 x 3.3 MB x 2 per chunk = 28 MB, 0.23 GB for the 17 chunks (float32).
  python3 e57_valuedump.py --bundle B --out O --masks M.npz --sharp-heads S.npz   (GPU) | --selftest   (CPU, brute force check against numpy)"""
import sys,json,time,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_attdump3 as AD
CH=1024
class Dump3V(AD.Dump3):
    def __init__(s,cfg=None):
        super().__init__(cfg);jnp=s.jnp;W,N=s.c['WIN'],s.c['N'];s.SHVn=jnp.zeros((N,W),jnp.float32);s.ALLVn=jnp.zeros((N,W),jnp.float32)
    def layer(s,ph,vnw,sharp,l):
        super().layer(ph,vnw,sharp,l)
        jnp=s.jnp;W=s.c['WIN'];pw=ph[:,:,:W];shf=jnp.asarray(np.asarray(sharp,np.float32));vh=pw*vnw[:,None,:];un=vh/jnp.maximum(vh.sum(-1,keepdims=True),1e-30)
        s.SHVn=s.SHVn+jnp.einsum('h,hqk->qk',shf,un);s.ALLVn=s.ALLVn+un.sum(0)
    def finalize_rows(s,rows):
        jnp=s.jnp;r=jnp.asarray(np.asarray(rows,np.int32));ALL=s.G[0]+s.G[1]+s.G[2]
        return {'SH':np.asarray(s.SH[r],np.float32),'ALL':np.asarray(ALL[r],np.float32),'SHV':np.asarray(s.SHV[r],np.float32),'ALLV':np.asarray(s.ALLV[r],np.float32),'SHVn':np.asarray(s.SHVn[r],np.float32),'ALLVn':np.asarray(s.ALLVn[r],np.float32)}
def value_pass(L,rt,s3,m,Kc,Vc,ids_k,k,sharp,rows):
    """GPU pass of the prefix layers of chunk k -> dict of float32 [R, 8192] (see the module doc)"""
    import jax,jax.numpy as jnp
    D=Dump3V();s3.S3.clear();s3.S3.update(mode='att6',q0=jnp.int32(k*1024),att6=D,att6_l=0,att6_sharp=np.asarray(sharp,bool))
    try:
        ids=jnp.asarray(ids_k,dtype=jnp.int32)
        with rt.mesh:
            h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
            for l in range(Kc.shape[0]):
                blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
        assert s3.S3['att6_l']==AD.CFG['L'],s3.S3['att6_l']
    finally:s3.S3.clear()
    return D.finalize_rows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--masks');ap.add_argument('--sharp-heads');ap.add_argument('--bank');ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    ROOT=HERE.parents[1]
    import gen_world_ctl as G,e57_reads as RD
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    M=np.load(a.masks);chunks=[int(x) for x in M['chunks']];SH=np.load(a.sharp_heads)['sharp'];assert SH.shape==(280,24,32);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);f=out/'E57_valuedump_W0.npz'
    assert not f.exists(),f'{f} exists'
    print(f'value dump: {len(chunks)} chunks {chunks[0]}..{chunks[-1]}, rows {[len(M["rows_%d"%k]) for k in chunks][:3]}..., about {len(chunks)*6*len(M["rows_%d"%chunks[0]])*8192*4/2**20:.0f} MiB',flush=True)
    import e55_run as R;LD,s3,C=R.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'states'/'E57_valuedump',canary=None);lg=lambda *x:print(*x,flush=True)
    with rt.mesh:Kh,Vh,_=L.record(np.asarray(ids)[:n*CH])
    AD.install(s3);res={};t0=time.monotonic()
    for k in chunks:
        ids_k=np.asarray(ids[k*CH:(k+1)*CH])
        with rt.mesh:Kc,Vc=L.ctx_kv(Kh,Vh,k)
        d=value_pass(L,rt,s3,rt.base,Kc,Vc,ids_k,k,SH[k],M[f'rows_{k}'].astype(int))
        for kk,v in d.items():res[f'c{k}/{kk}']=v
        res[f'rows_{k}']=M[f'rows_{k}'];lg(f'  value dump chunk {k}: {time.monotonic()-t0:.0f}s')
    np.savez(f,chunks=np.asarray(chunks),**res);lg('wrote',f)
def selftest():
    import jax.numpy as jnp
    cfg=dict(WIN=128,N=16,M=40,MV=4,MG=4,KS=(8,24,40),RK_ALL=(1,2,4),RK_SH=(1,2,4,8),L=3,H=4);W,N,H,L=cfg['WIN'],cfg['N'],cfg['H'],cfg['L']
    rng=np.random.default_rng(0);D=Dump3V(cfg);PHs=[];VNs=[];SHs=[]
    for l in range(L):
        sc=rng.standard_normal((H,N,W+N)).astype(np.float32)*2.5;causal=np.arange(W+N)[None,:]<=(np.arange(N)[:,None]+W);sc=np.where(causal[None],sc,-1e30)
        ph=np.exp(sc-sc.max(-1,keepdims=True));ph/=ph.sum(-1,keepdims=True);vn=(rng.random((H,W))*3+.2).astype(np.float32);sh=rng.random(H)<.5;sh[0]=True
        PHs.append(ph);VNs.append(vn);SHs.append(sh);D.layer(jnp.asarray(ph),jnp.asarray(vn),sh,l)
    rows=np.array([3,9,15]);o=D.finalize_rows(rows)
    PH=np.stack(PHs);PW=PH[:,:,:,:W];SHm=np.stack(SHs)[:,:,None,None].astype(np.float32);VN=np.stack(VNs)[:,:,None,:]
    ref={'SH':(PW*SHm).sum((0,1)),'ALL':PW.sum((0,1)),'SHV':(PW*SHm*VN).sum((0,1)),'ALLV':(PW*VN).sum((0,1))}
    vh=PW*VN;un=vh/np.maximum(vh.sum(-1,keepdims=True),1e-30);ref['SHVn']=(un*SHm).sum((0,1));ref['ALLVn']=un.sum((0,1))
    for k,v in ref.items():assert o[k].shape==(3,W) and np.allclose(o[k],v[rows],rtol=1e-4,atol=1e-6),k
    assert np.allclose(o['ALLVn'].sum(-1),L*H,rtol=1e-4) and (o['ALLVn'].sum(-1)>0).all()
    print('e57_valuedump selftest ok')
if __name__=='__main__':main()
