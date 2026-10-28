"""E57 key-mass dump v3 (GPU forward only, W0 weights, no write; post-hoc, exploratory). Same pass as e57_attdump2 (unmasked natural window, every token of the chunk a query, prefix layers) but it keeps, per query row, the attention mass on the FAR window keys summed over head sets, the top-M keys of it, and per-key head-vote counts. All masses are post-softmax probabilities over window + current-chunk keys (not renormalised), summed over the layers and heads of a head set; far keys of band k = window key c with distance i + WIN - c >= k and c >= 4 (the first 4 window keys are excluded); k = 128 / 512 / 1024.
Head sets: SH = the sharp heads of the chunk (g3 top 50 % by chunk mean of the raw T_max, label-free, a [24, 32] bool input per chunk from e57_attdump3_heads.py), ALL = all 768 heads; G1 / G2 / G3 = all heads of the layers 0-7 / 8-15 / 16-23 (ALL = G1 + G2 + G3).
Per chunk and per k (names, n = 1024 rows):
  SH<k>_idx int16 [n,128], SH<k>_mass float16 [n,128], SH<k>_cnt uint8 [n,128,10]   top-128 far keys of the SH mass; counts of heads (saturating at 255, SH<k>_sat = number of saturated entries): rank cutoffs 1 / 4 / 16 / 64 (the head ranks its far keys by its own probability and votes for the keys within its top c), then the log-spaced probability histogram: for each edge e in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2) the number of heads whose probability for the key is >= e (cumulative; any threshold in this list, or a bin count by difference, is chosen offline)
  ALL<k>_idx, ALL<k>_mass, ALL<k>_cnt uint8 [n,128,9]   same for ALL: rank cutoffs 1 / 4 / 16 then the 6 edge counts;  ALL<k>_grp float16 [n,32,3] = G1 / G2 / G3 mass at the top-32 keys of ALL
  SHV<k>_idx/_mass, ALLV<k>_idx/_mass [n,64]   top-64 far keys of mass x ||v_key|| (raw ||v|| of the key per head, summed)
  G3<k>_idx/_mass [n,64]   top-64 far keys of the G3 mass
  SH<k>_far, ALL<k>_far float32 [n] (far mass), SH<k>_ent, ALL<k>_ent float32 [n] (entropy of the normalised far mass), SH<k>_cum, ALL<k>_cum float16 [n,5] (share of the far mass held by the top 1 / 4 / 8 / 16 / 32 keys)
  win_SH, win_ALL float32 [n] (mass on all window keys), chunk_SH, chunk_ALL float16 [n,5] (mass on chunk keys by distance (0,64], (64,128], (128,256], (256,512], (512,1024]), headscore float16 [24,32,5] (per layer / head, mean over rows: window mass, top-1 window mass, far mass of the middle band, its top-1 share, entropy of the window mass), dense_SH / dense_ALL float16 [n,8192] on the chunks k % 16 == 0 (validation).
CPU tested: e57_attdump3_run.py --selftest (all outputs against a brute-force numpy loop on a small stream, jax on CPU)."""
import numpy as np
CFG=dict(WIN=8192,N=1024,M=128,MV=64,MG=32,KS=(128,512,1024),RK_ALL=(1,4,16),RK_SH=(1,4,16,64),EDGE=(.005,.01,.02,.05,.1,.2),L=24,H=32)
BINS=(0,64,128,256,512,1024)
class Dump3:
    def __init__(s,cfg=None):
        import jax.numpy as jnp
        s.c=c=dict(CFG,**(cfg or {}));s.jnp=jnp;W,N=c['WIN'],c['N']
        z=lambda *sh:jnp.zeros(sh,jnp.float32)
        s.SH=z(N,W);s.G=[z(N,W) for _ in range(3)];s.SHV=z(N,W);s.ALLV=z(N,W);s.SHc=z(N,N);s.ALLc=z(N,N)
        s.tauSH=[jnp.zeros((N,W),jnp.int16) for _ in c['EDGE']];s.tauALL=[jnp.zeros((N,W),jnp.int16) for _ in c['EDGE']]
        s.rkSH={k:[jnp.zeros((N,W),jnp.int16) for _ in c['RK_SH']] for k in c['KS']};s.rkALL={k:[jnp.zeros((N,W),jnp.int16) for _ in c['RK_ALL']] for k in c['KS']}
        s.hs=[];s.far={k:s.far_mask(k) for k in c['KS']};s.kmid=c['KS'][len(c['KS'])//2]
    def far_mask(s,k):
        jnp=s.jnp;W,N=s.c['WIN'],s.c['N'];return ((jnp.arange(N)[:,None]+W-jnp.arange(W)[None,:])>=k)&(jnp.arange(W)[None,:]>=4)
    def layer(s,ph,vnw,sharp,l):
        """ph [H, N, WIN+N] float32; vnw [H, WIN] value norms of the window keys; sharp bool numpy [H]; l layer index"""
        jnp=s.jnp;c=s.c;W,N=c['WIN'],c['N'];Hh=ph.shape[0];pw=ph[:,:,:W];pc=ph[:,:,W:];shf=jnp.asarray(np.asarray(sharp,np.float32))
        g=min(2,l*3//c['L']);s.G[g]=s.G[g]+pw.sum(0);s.SH=s.SH+jnp.einsum('h,hqk->qk',shf,pw)
        s.SHV=s.SHV+jnp.einsum('hqk,hk->qk',pw,vnw*shf[:,None]);s.ALLV=s.ALLV+jnp.einsum('hqk,hk->qk',pw,vnw)
        s.SHc=s.SHc+jnp.einsum('h,hqj->qj',shf,pc);s.ALLc=s.ALLc+pc.sum(0)
        for i,t in enumerate(c['EDGE']):
            m=(pw>=t);s.tauALL[i]=s.tauALL[i]+m.sum(0).astype(jnp.int16);s.tauSH[i]=s.tauSH[i]+jnp.einsum('h,hqk->qk',shf,m.astype(jnp.float32)).astype(jnp.int16)
        rows=jnp.arange(N)[None,:,None];sidx=np.nonzero(np.asarray(sharp))[0]
        for k in c['KS']:
            pf=jnp.where(s.far[k][None],pw,-1.0)
            v,ix=__import__('jax').lax.top_k(pf,max(c['RK_ALL']))
            for j,r in enumerate(c['RK_ALL']):
                s.rkALL[k][j]=s.rkALL[k][j].at[jnp.broadcast_to(rows,(Hh,N,r)),ix[:,:,:r]].add((v[:,:,:r]>0).astype(jnp.int16))
            if len(sidx):
                v,ix=__import__('jax').lax.top_k(pf[sidx],max(c['RK_SH']))
                for j,r in enumerate(c['RK_SH']):
                    s.rkSH[k][j]=s.rkSH[k][j].at[jnp.broadcast_to(rows,(len(sidx),N,r)),ix[:,:,:r]].add((v[:,:,:r]>0).astype(jnp.int16))
        f5=s.far[s.kmid][None];wsum=pw.sum(-1);ent=-(pw*jnp.log(pw+1e-30)).sum(-1);f512=(pw*f5).sum(-1);t1f=jnp.where(f5,pw,0.0).max(-1)
        s.hs.append(np.asarray(jnp.stack([wsum.mean(-1),pw.max(-1).mean(-1),f512.mean(-1),(t1f/(f512+1e-9)).mean(-1),ent.mean(-1)],-1),np.float16))
    def finalize(s,dense=False):
        jnp=s.jnp;import jax;c=s.c;W,N,M=c['WIN'],c['N'],c['M'];out={};ALL=s.G[0]+s.G[1]+s.G[2];rows=jnp.arange(N)[:,None]
        sets={'SH':s.SH,'ALL':ALL}
        out['win_SH']=np.asarray(s.SH.sum(-1),np.float32);out['win_ALL']=np.asarray(ALL.sum(-1),np.float32)
        d=jnp.arange(N)[:,None]-jnp.arange(N)[None,:];cb=[]
        for nm,Cm in (('SH',s.SHc),('ALL',s.ALLc)):
            cols=[]
            for b in range(5):
                lo,hi=BINS[b],BINS[b+1];msk=(d>=0)&(d<=hi)&((d>lo) if b>0 else (d>=0));cols.append((Cm*msk).sum(-1))
            out['chunk_'+nm]=np.asarray(jnp.stack(cols,-1),np.float16)
        for k in c['KS']:
            far=s.far[k]
            for nm,mass in sets.items():
                mf=jnp.where(far,mass,0.0);tot=mf.sum(-1);v,ix=jax.lax.top_k(mf,M);p=mf/jnp.maximum(tot[:,None],1e-30)
                out[f'{nm}{k}_far']=np.asarray(tot,np.float32);out[f'{nm}{k}_ent']=np.asarray(-(p*jnp.log(p+1e-30)).sum(-1),np.float32)
                cum=jnp.cumsum(v,-1)/jnp.maximum(tot[:,None],1e-30);out[f'{nm}{k}_cum']=np.asarray(jnp.stack([cum[:,j-1] for j in (1,4,8,16,32)],-1),np.float16)
                out[f'{nm}{k}_idx']=np.asarray(ix,np.int16);out[f'{nm}{k}_mass']=np.asarray(v,np.float16)
                accs=(s.rkSH[k]+s.tauSH) if nm=='SH' else (s.rkALL[k]+s.tauALL)
                cnt=jnp.stack([a[rows,ix] for a in accs],-1).astype(jnp.int32);out[f'{nm}{k}_sat']=np.asarray((cnt>=255).sum(),np.int64);out[f'{nm}{k}_cnt']=np.asarray(jnp.minimum(cnt,255),np.uint8)
                if nm=='ALL':out[f'ALL{k}_grp']=np.asarray(jnp.stack([s.G[g][rows,ix[:,:c['MG']]] for g in range(3)],-1),np.float16)
            for nm,mass in (('SHV',s.SHV),('ALLV',s.ALLV),('G3',s.G[2])):
                v,ix=jax.lax.top_k(jnp.where(far,mass,0.0),c['MV']);out[f'{nm}{k}_idx']=np.asarray(ix,np.int16);out[f'{nm}{k}_mass']=np.asarray(v,np.float16)
        out['headscore']=np.stack(s.hs,0)
        if dense:out['dense_SH']=np.asarray(s.SH,np.float16);out['dense_ALL']=np.asarray(ALL,np.float16)
        return out
def install(s3):
    """add the mode 'att6' to the sliding-window attention of s3 (other modes passed on unchanged)"""
    import jax,jax.numpy as jnp
    SWA=s3.SWA;prev=SWA.full_sw_attention;WIN_=s3.WIN
    def _full6(self,hidden_states,seq,state):
        if s3.S3.get('mode')!='att6':return prev(self,hidden_states,seq,state)
        S3=s3.S3;Kc,Vc=state;q0=S3['q0'];n=hidden_states.shape[0]
        xq,xk,xv=self.get_attention_input(hidden_states,position_ids=q0+jnp.arange(n,dtype=jnp.int32))
        K=jnp.concatenate([Kc.astype(xk.dtype),xk],axis=0);V=jnp.concatenate([Vc.astype(xv.dtype),xv],axis=0)
        qi=(q0+jnp.arange(n,dtype=jnp.int32))[:,None];kj=(q0-WIN_+jnp.arange(WIN_+n,dtype=jnp.int32))[None,:]
        mask=(qi>=kj)&(qi-kj<WIN_)&(kj>=0)
        q=xq.astype(jnp.float32);Kx=K.astype(jnp.float32)
        if Kx.shape[1]!=q.shape[1]:Kx=jnp.repeat(Kx,q.shape[1]//Kx.shape[1],axis=1)
        sc=jnp.einsum('qhd,khd->hqk',q,Kx)*(q.shape[-1]**-0.5);sc=jnp.where(mask[None],sc,-1e30);ph=jax.nn.softmax(sc,axis=-1)
        vn=jnp.linalg.norm(V[:WIN_].astype(jnp.float32),axis=-1)
        if vn.shape[1]!=q.shape[1]:vn=jnp.repeat(vn,q.shape[1]//vn.shape[1],axis=1)
        l=S3['att6_l'];S3['att6'].layer(ph,vn.T,S3['att6_sharp'][l],l);S3['att6_l']=l+1
        o=jax.nn.dot_product_attention(xq,K,V,mask=mask,implementation='xla');o=self._merge_heads(o)
        return (self.get_attention_output(o),state)
    SWA.full_sw_attention=_full6
def att6_pass(L,rt,s3,m,Kc,Vc,ids_k,k,sharp,dense=False):
    """GPU forward-only pass of the prefix layers for chunk k -> dict of the arrays of the module doc; sharp bool [24, 32]"""
    import jax,jax.numpy as jnp
    D=Dump3();s3.S3.clear();s3.S3.update(mode='att6',q0=jnp.int32(k*1024),att6=D,att6_l=0,att6_sharp=np.asarray(sharp,bool))
    try:
        ids=jnp.asarray(ids_k,dtype=jnp.int32)
        with rt.mesh:
            h=m.language_model.wte_call(ids);blocks=m.language_model.model.h.prefix_blocks;seq=L.Batch(input_ids=ids,target_tokens=ids,loss_masks=jnp.ones_like(ids,dtype=bool))
            for l in range(Kc.shape[0]):
                blk=jax.tree.map(lambda a_:a_[l],blocks);h,_=blk(h,(Kc[l],Vc[l]),seq,is_prefix=True)
        assert s3.S3['att6_l']==CFG['L'],s3.S3['att6_l']
    finally:s3.S3.clear()
    return D.finalize(dense=dense)
