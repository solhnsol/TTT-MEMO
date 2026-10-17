"""S3 stage 1: forward diagnostic at W0 (no D, no weight update), PROTOCOL "S3 selective masking, registered spec v1.1" + the 10/17 score definitions (registered 17:09 KST).
Per (seed, world) on the original stream (pass 1), chunks processed in order with the natural read chain at W0:
  (1) attention statistics of the read forward at the 8 suffix layers (QK^T recomputed from the same q/k, softmax over the same mask), targets = all loss-position queries of the chunk (label-free):
      A score (mean over layers/heads/queries of the probabilities), B score (mean logit per key over valid (layer, head, query), then softmax over keys), the head-max per-target scores M[q,k]
      (max over heads of the layer-mean probability) and their per-target size-rule unions (k in {16, 64, 256}, p in {0.5, 0.8, 0.9});
  (2) after all chunks: the distance profile f(d) (mean A over the keys of a 256-token distance bin, over all chunks of the stream) -> C = A - f(d); the ensemble = rank-mean of A, B, C;
  (3) masks for every (aggregation, size rule) with the same candidate rule (window before the chunk, sinks = the first 4 positions and the last 128 tokens before the chunk excluded, ties to the
      earlier position), plus count-matched rand-dist and recency, plus the oracle set O; for every eligible chunk (a chunk with an exposure sentence that has a prior same-triple exposure in the window)
      the output distributions at the answer tokens (place tokens of such sentences) under natural / chunk-alone / masked write forwards -> epsilon numerators per (world, pair) cluster;
      PR-AUC (average precision of the key scores against the O labels) per chunk.
Phase 1 (s101): the full grid. Phase 2 (s102, s103): only the policy selected on s101 (OUT/s3_stage1_selection.json) and its baselines.
Usage: s3_stage1.py --world-seed 101 --phase 1 | --world-seed 102 --phase 2 ; --unit-smoke N: only the first N eligible chunks (no files)"""
import os,sys,json,time,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3 as S3M
from s3 import *
import s3_sets as SETS
import ladder as LD
SW=SWA
S3REC={}
_prev_core=SW.core_attention_op
def _core3(self,xq,xk,xv,attention_mask):
    if S3REC.get('on'):
        d=xq.shape[-1];lg=jnp.einsum('qhd,khd->hqk',xq.astype(jnp.float32),xk.astype(jnp.float32))/jnp.sqrt(jnp.float32(d))
        ok=attention_mask[None];lgm=jnp.where(ok,lg,-jnp.inf);p=jax.nn.softmax(lgm,axis=-1)
        S3REC['P']=S3REC['P']+p;S3REC['LS']=S3REC['LS']+jnp.sum(jnp.where(ok,lg,0.0),axis=(0,1));S3REC['CNT']=S3REC['CNT']+jnp.sum(jnp.broadcast_to(ok,lg.shape),axis=(0,1)).astype(jnp.float32)
    return _prev_core(self,xq,xk,xv,attention_mask)
SW.core_attention_op=_core3
SIZE_RULES=[('k',16),('k',64),('k',256),('p',0.5),('p',0.8),('p',0.9)]
AGGS=['A','Ahmax','B','C','ens']
def rule_name(r):return f'{r[0]}{r[1]:g}'
class Stage1(Ladder3):
    def build1(self):
        rt=self.rt;Batch=self.Batch;nh=32
        @eqx.filter_jit
        def capture(m,st_suf,c,pref_c):
            blocks=m.language_model.model.h.suffix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0]
            S3REC['on']=True;S3REC['P']=jnp.zeros((nh,CH,WIN+CH),jnp.float32);S3REC['LS']=jnp.zeros((WIN+CH,),jnp.float32);S3REC['CNT']=jnp.zeros((WIN+CH,),jnp.float32)
            try:
                h=pref_c
                for l in range(nl):
                    blk=jax.tree.map(lambda a:a[l] if eqx.is_array(a) else a,blocks);stl=jax.tree.map(lambda s:s[l],st_suf);h,_=blk(h,stl,c,is_prefix=False)
                Pm=S3REC['P']/nl;LS=S3REC['LS'];CNT=S3REC['CNT']
            finally:S3REC.clear()
            A=jnp.mean(Pm,axis=(0,1));Bl=jnp.where(CNT>0,LS/jnp.maximum(CNT,1.0),-jnp.inf);M=jnp.max(Pm,axis=0)
            return A,Bl,M
        @eqx.filter_jit
        def logits_nat(m,st_suf,c,pref_c,rows):
            lg=m.language_model.suffix_call(prefix_outputs=pref_c,state=st_suf,seq=c).logits.astype(jnp.float32);return jax.nn.softmax(lg[rows],axis=-1)
        @eqx.filter_jit
        def logits_masked(m,st_suf,c,pref_w,kmask,rows):
            S3SUF['kmask']=kmask
            try:lg=m.language_model.suffix_call(prefix_outputs=pref_w,state=st_suf,seq=c).logits.astype(jnp.float32)
            finally:S3SUF.clear()
            return jax.nn.softmax(lg[rows],axis=-1)
        @eqx.filter_jit
        def logits_alone(m,ids_,tg_,mk_,pref_,seg_,loc_,rows):
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_);lg=m.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt.state_suffix),seq=seq).logits.astype(jnp.float32)
            finally:packing.PACK_SUF.clear()
            return jax.nn.softmax(lg[rows],axis=-1)
        self.capture=capture;self.logits_nat=logits_nat;self.logits_masked=logits_masked;self.logits_alone=logits_alone
    def run(self,swap,seed,phase,sel=None,nsmoke=None):
        rt=self.rt;m=rt.base;st_all=clone_state(rt.state_all);ids=np.asarray(self.T.P[swap].ids)[:NCH*CH];W=WIN
        mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
        sets=SETS.Sets(SETS.sentence_table(self.T.w,swap,self.T.tok,gw,SP),seed,1);tab=sets.t
        self.seed_=seed;self.swap_=swap;res={'swap':bool(swap),'seed':seed,'phase':phase,'chunks':{}}
        seg0=np.zeros(CH,np.int32);loc0=np.arange(CH,dtype=np.int32);pm0=jnp.asarray(packing.prefix_mask(seg0,W))
        with rt.mesh:
            tp=time.monotonic();pref_nat=rt.prefix(rt.base,seq);print(f'  stage1 natural prefix {time.monotonic()-tp:.0f}s',flush=True);tp=time.monotonic();Kh,Vh,_=self.record(ids);print(f'  stage1 recorded natural K,V {time.monotonic()-tp:.0f}s',flush=True);st=clone_state(rt.state_suffix);Asc={};Bsc={};t0=time.monotonic()
            # ---- pass 1: attention statistics along the natural chain (needed for every chunk: f(d)) ----
            stash={}
            for k in range(NCH):
                c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH]
                A,Bl,M=self.capture(m,st,c,pref_c);Asc[k]=np.asarray(A);Bsc[k]=np.asarray(Bl)
                eligible=bool(self.answers(sets,k))
                if eligible:stash[k]=np.asarray(M)   # head-max scores [CH, W+CH] kept only for eligible chunks (needed for the unions)
                st=self.read_state(m,st,c,pref_c)
                if k%8==0:print(f'  stage1 pass-1 chunk {k}/{NCH} {time.monotonic()-t0:.0f}s',flush=True)
            # ---- distance profile f(d) and C ----
            fsum={};fcnt={}
            for k in range(NCH):
                k0=k*CH
                for j in range(W):
                    p=k0-W+j
                    if p<0:continue
                    b=(k0-p)//SETS.BIN;fsum[b]=fsum.get(b,0.0)+Asc[k][j];fcnt[b]=fcnt.get(b,0)+1
            f={b:fsum[b]/fcnt[b] for b in fsum}
            # ---- pass 2: masks and epsilon numerators on eligible chunks ----
            st=clone_state(rt.state_suffix)
            for k in range(NCH):
                c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH]
                if k in stash:
                    if nsmoke is not None and len(res['chunks'])>=nsmoke:break
                    te=time.monotonic();res['chunks'][k]=self.eval_chunk(m,st,c,pref_c,k,ids,sets,Asc[k],Bsc[k],stash[k],f,Kh,Vh,phase,sel,pm0,seg0,loc0);print(f'  stage1 eval chunk {k} ({len(res["chunks"])}/{len(stash)}) {time.monotonic()-te:.0f}s',flush=True)
                st=self.read_state(m,st,c,pref_c)
        return res
    def answers(self,sets,k):
        """[(stream index j of the answer token, pair)] : place tokens of current-chunk exposure sentences with a prior same-triple exposure inside the window (row = j - 1 - k0 >= 0)."""
        out=[];lo=max(0,k*CH-WIN)
        for s in sets.cur_sentences(k):
            prior=[u for u in sets.t if u['t']==s['t'] and u['round']<s['round'] and any(lo<=p<s['place'][0] for p in u['names']+u['place'])]
            if prior:
                for j in s['place']:
                    if j-1-k*CH>=0:out.append((j,s['pair']))
        return out
    def cand(self,k):
        W=WIN;pos=k*CH-W+np.arange(W);ok=(pos>=4)&(pos<k*CH-128)&(pos>=0);return ok
    def select(self,score,k,rule,candm):
        """top-k / cumulative-mass set on the window part (index j <-> position k*CH - W + j); ties -> earlier position."""
        sc=np.where(candm,score,-np.inf);idx=np.lexsort((np.arange(len(sc)),-sc));idx=idx[np.isfinite(sc[idx])]
        if rule[0]=='k':sel=idx[:int(rule[1])]
        else:
            v=np.maximum(sc[idx],0);tot=v.sum()
            if tot<=0:sel=idx[:0]
            else:cs=np.cumsum(v)/tot;n=int(np.searchsorted(cs,rule[1])+1);sel=idx[:n]
        return k*CH-WIN+np.sort(sel)
    def eval_chunk(self,m,st,c,pref_c,k,ids,sets,A,Bl,M,f,Kh,Vh,phase,sel,pm0,seg0,loc0):
        rt=self.rt;W=WIN;k0=k*CH;candm=self.cand(k);ans=self.answers(sets,k);rows=jnp.asarray([j-1-k0 for j,_ in ans],dtype=jnp.int32);ys=np.asarray([ids[j] for j,_ in ans]);pairs=[p for _,p in ans]
        pos=k0-W+np.arange(W);dbin=np.where(pos>=0,(k0-pos)//SETS.BIN,0)
        Aw=A[:W].copy();Bw=Bl[:W].copy()
        Bsm=np.where(np.isfinite(Bw)&candm,np.exp(Bw-np.max(Bw[candm&np.isfinite(Bw)])),0.0);Bsm=Bsm/max(Bsm.sum(),1e-30)
        fd=np.array([f.get(int(b),0.0) for b in dbin]);Cw=Aw-fd
        rk=lambda s:np.argsort(np.argsort(-np.where(candm,s,-np.inf)))
        ens=-(rk(Aw)+rk(Bsm)+rk(Cw))/3.0
        scores={'A':Aw,'B':Bsm,'C':Cw,'ens':ens-ens[candm].min()+1e-9}
        O=sets.oracle(k);Oset=set(O);labels=np.asarray([ (k0-W+j) in Oset for j in range(W)])
        out={'n_O':len(O),'answers':[(int(j),pairs[i]) for i,(j,_) in enumerate(ans)],'pr':{},'policies':{}}
        for a in ['A','B','C','ens']:out['pr'][a]=self.avg_prec(scores[a],labels,candm)
        Mw=M[:,:W].copy();Mw[:,~candm]=0.0
        # head-max unions
        order=np.argsort(-Mw,axis=1,kind='stable');
        hm={}
        for r in SIZE_RULES:
            if r[0]=='k':n=np.full(CH,int(r[1]))
            else:
                v=np.take_along_axis(Mw,order,axis=1);tot=v.sum(1,keepdims=True);cs=np.cumsum(v,axis=1)/np.maximum(tot,1e-30);n=(cs<r[1]).sum(1)+1;n=np.where(tot[:,0]>0,n,0)
            u=np.zeros(W,bool)
            for q in range(CH):u[order[q,:n[q]]]=True
            hm[rule_name(r)]=k0-W+np.nonzero(u)[0]
        out['pr']['Ahmax']=self.avg_prec(Mw.max(0),labels,candm)
        # forwards
        Kc,Vc=self.ctx_kv(Kh,Vh,k)   # the window K,V of this chunk, transferred once per chunk (not per policy)
        def prefw(S):
            keep=sets.keep_vector(k,S);pw=self.prefix_masked(rt.base,jnp.asarray(ids[k0:k0+CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k0),jnp.asarray(keep));return pw,jnp.asarray(keep)
        def dist(p):   # ||p - e_y||_2 per answer token
            p=np.asarray(p);py=p[np.arange(len(ys)),ys];return np.sqrt(np.maximum((p**2).sum(1)-2*py+1,0))
        nat=dist(self.logits_nat(rt.base,st,c,pref_c,rows));out['nat']=nat.tolist()
        tg=np.concatenate([ids[k0+1:k0+CH+1],[rt.eos]])[:CH].astype(np.int32);pfa=self.prefix_iso(rt.base,jnp.asarray(ids[k0:k0+CH]),jnp.asarray(loc0),pm0)
        out['alone']=dist(self.logits_alone(rt.base,jnp.asarray(ids[k0:k0+CH]),jnp.asarray(tg),jnp.asarray(np.concatenate([np.ones(CH-1,bool),[False]])),pfa,jnp.asarray(seg0),jnp.asarray(loc0),rows)).tolist()
        pw,kp=prefw(O);out['oracle']=dist(self.logits_masked(rt.base,st,c,pw,kp,rows)).tolist()
        rng=np.random.default_rng([self.seed_,k,int(self.swap_)])
        grid=[(a,r) for a in AGGS for r in SIZE_RULES] if phase==1 else [tuple(sel)] if sel else []
        for a,rr in grid:
            rn=rule_name(rr) if not isinstance(rr,str) else rr
            if a=='Ahmax':S=hm[rn]
            else:S=self.select(scores[a],k,(rr[0],rr[1]) if not isinstance(rr,str) else (rr[0],float(rr[1:])),candm)
            n=len(S);pol={'n':int(n)}
            for nm,SS in (('policy',S),('rand',self.rand_like(sets,k,S,Oset,rng)),('recency',np.arange(max(0,k0-n),k0))):
                if len(SS)==0:pol[nm]=None;continue
                pw,kp=prefw(list(SS));pol[nm]=dist(self.logits_masked(rt.base,st,c,pw,kp,rows)).tolist()
            out['policies'][f'{a}|{rn}']=pol
        return out
    def rand_like(self,sets,k,S,Oset,rng):
        k0=k*CH;lo=max(0,k0-WIN);cand=[p for p in range(lo,k0) if p not in Oset];need=np.asarray([(k0-p)//SETS.BIN for p in S if p<k0],int);pool={}
        for p in cand:pool.setdefault((k0-p)//SETS.BIN,[]).append(p)
        import random;ch,_=SETS._fill(list(need),pool,random.Random(int(rng.integers(1<<31))));return ch
    def avg_prec(self,score,labels,candm):
        sc=score[candm];lab=labels[candm]
        if lab.sum()==0:return None
        o=np.lexsort((np.arange(len(sc)),-sc));l=lab[o];tp=np.cumsum(l);prec=tp/np.arange(1,len(l)+1);return float((prec*l).sum()/l.sum())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--world-seed',type=int,required=True);ap.add_argument('--phase',type=int,choices=[1,2],required=True);ap.add_argument('--worlds',default='AB');ap.add_argument('--unit-smoke',type=int,default=None)
    a=ap.parse_args();assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    sel=None
    if a.phase==2:sel=json.load(open(OUT/'s3_stage1_selection.json'))['policy']
    rt=Runtime();L=Stage1(rt,a.world_seed,readout=False);L.build3();L.build1()
    for w in a.worlds:
        f=OUT/f's3_stage1_s{a.world_seed}_{w}.json'
        if f.exists() and not a.unit_smoke:print('exists',f);continue
        t0=time.monotonic();res=L.run(w=='B',a.world_seed,a.phase,sel,a.unit_smoke)
        if a.unit_smoke:print('smoke ok',w,len(res['chunks']),'eligible chunks',round(time.monotonic()-t0),'s');k=sorted(res['chunks'])[0];print({kk:(vv if kk!='policies' else list(vv)[:3]) for kk,vv in res['chunks'][k].items() if kk in('n_O','answers','pr','policies')});continue
        json.dump(res,open(f,'w'));print('saved',f,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
