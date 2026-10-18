"""S3 stage-1 exploratory aggregation "logit-max" (Hans, 10/18 ~01:16 UTC; design relay): selection only, W0, no weight update, no D. Separate file: nothing the running jobs import is changed.
Per eligible chunk of the natural read chain at W0 (suffix layers 0..7, targets = all 1023 loss positions): s_j = the maximum, over layers, heads and target queries that can attend to j, of the PRE-softmax attention logit q.k/sqrt(d)
(the same causal+window mask as the read forward); candidates = the window before the chunk minus the first 4 stream positions and the last 128 tokens. Global ranking by s_j (ties -> earlier position), no per-target top-k.
Sizes: (a) N = |k64 head-max union| of the same chunk (count-matched main comparison); (b) descriptive N = |k16 union| and |k256 union|.
Outputs per world (s101 A, B): PR-AUC of s_j against the O labels (selectable O tokens), epsilon at the answer tokens of the logit-max sets with count-matched rand-dist and recency baselines, masked counts, and the key-norm check:
the share of the top-N held by the 1% of candidates with the largest key norm (key norm = mean over layers and heads of ||k||), also logged for the k64 head-max union.
Usage: s3_logitmax.py --world-seed 101 [--worlds AB] ; analyze: s3_logitmax.py --analyze"""
import os,sys,json,time,argparse,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_stage1 as ST
from s3_stage1 import *
LM={}
_prev_core2=SWA.core_attention_op
def _core4(self,xq,xk,xv,attention_mask):
    if LM.get('on'):
        d=xq.shape[-1];lg=jnp.einsum('qhd,khd->hqk',xq.astype(jnp.float32),xk.astype(jnp.float32))/jnp.sqrt(jnp.float32(d))
        lgm=jnp.where(attention_mask[None],lg,-jnp.inf);LM['lmax']=jnp.maximum(LM['lmax'],jnp.max(lgm,axis=(0,1)))
        LM['knorm']=LM['knorm']+jnp.mean(jnp.linalg.norm(xk.astype(jnp.float32),axis=-1),axis=-1)
    return _prev_core2(self,xq,xk,xv,attention_mask)
SWA.core_attention_op=_core4
class LogitMax(ST.Stage1):
    def build_lm(self):
        nh=32
        def unions(Mx,cand):
            Mx=jnp.where(cand[None,:],Mx,-1.0);out=[]
            for KA in (16,64,256):
                _,idx=jax.lax.top_k(Mx,KA);valid=jnp.take_along_axis(jnp.broadcast_to(cand[None,:],Mx.shape),idx,axis=1);out.append(jnp.zeros((WIN,),bool).at[idx.reshape(-1)].max(valid.reshape(-1)))
            return out
        @eqx.filter_jit
        def capture_lm(m,st_suf,c,pref_c,cand):
            blocks=m.language_model.model.h.suffix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0]
            ST.S3REC['on']=True;ST.S3REC['P']=jnp.zeros((nh,CH,WIN+CH),jnp.float32);ST.S3REC['LS']=jnp.zeros((WIN+CH,),jnp.float32);ST.S3REC['CNT']=jnp.zeros((WIN+CH,),jnp.float32)
            LM['on']=True;LM['lmax']=jnp.full((WIN+CH,),-jnp.inf,jnp.float32);LM['knorm']=jnp.zeros((WIN+CH,),jnp.float32)
            try:
                h=pref_c
                for l in range(nl):
                    blk=jax.tree.map(lambda a:a[l] if eqx.is_array(a) else a,blocks);stl=jax.tree.map(lambda s:s[l],st_suf);h,_=blk(h,stl,c,is_prefix=False)
                Pm=ST.S3REC['P']/nl;lmax=LM['lmax'][:WIN];knorm=LM['knorm'][:WIN]/nl
            finally:ST.S3REC.clear();LM.clear()
            u16,u64,u256=unions(jnp.max(Pm,axis=0)[:,:WIN],cand)
            return lmax,knorm,u16,u64,u256
        self.capture_lm=capture_lm
    def run_lm(self,swap,seed):
        rt=self.rt;m=rt.base;ids=np.asarray(self.T.P[swap].ids)[:NCH*CH];W=WIN
        mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);sets=SETS.Sets(SETS.sentence_table(self.T.w,swap,self.T.tok,gw,SP),seed,1)
        seg0=np.zeros(CH,np.int32);loc0=np.arange(CH,dtype=np.int32);pm0=jnp.asarray(packing.prefix_mask(seg0,W));res={'swap':bool(swap),'seed':seed,'chunks':{}};self.seed_=seed;self.swap_=swap;t0=time.monotonic()
        with rt.mesh:
            pref_nat=rt.prefix(rt.base,seq);Kh,Vh,_=self.record(ids);print(f'  logitmax setup {time.monotonic()-t0:.0f}s',flush=True);st=clone_state(rt.state_suffix)
            for k in range(NCH):
                c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH];ans=self.answers(sets,k)
                if ans:
                    k0=k*CH;candm=self.cand(k);lmax,knorm,u16,u64,u256=self.capture_lm(m,st,c,pref_c,jnp.asarray(candm))
                    lmax=np.asarray(lmax);knorm=np.asarray(knorm);u={16:np.asarray(u16),64:np.asarray(u64),256:np.asarray(u256)}
                    O=sets.oracle(k);Oset=set(O);labels=np.asarray([(k0-W+j) in Oset for j in range(W)]);out={'n_O':len(O),'pr_lmax':self.avg_prec(np.where(candm,lmax,-np.inf),labels,candm),'sizes':{},'answers':[(int(j),p) for j,p in ans]}
                    rows=jnp.asarray([j-1-k0 for j,_ in ans],dtype=jnp.int32);ys=np.asarray([ids[j] for j,_ in ans])
                    out['pr_hmax_k64']=None
                    sc=np.where(candm,lmax,-np.inf);order=np.lexsort((np.arange(W),-sc));order=order[np.isfinite(sc[order])]
                    kn=np.where(candm,knorm,-np.inf);top1=set(np.argsort(-kn)[:max(1,int(candm.sum()*0.01))].tolist())
                    Kc,Vc=self.ctx_kv(Kh,Vh,k)
                    def prefw(S):
                        keep=sets.keep_vector(k,S);pw=self.prefix_masked(rt.base,jnp.asarray(ids[k0:k0+CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k0),jnp.asarray(keep));return pw,jnp.asarray(keep)
                    def dist(p):
                        p=np.asarray(p);py=p[np.arange(len(ys)),ys];return np.sqrt(np.maximum((p**2).sum(1)-2*py+1,0))
                    nat=dist(self.logits_nat(rt.base,st,c,pref_c,rows));tg=np.concatenate([ids[k0+1:k0+CH+1],[rt.eos]])[:CH].astype(np.int32);pfa=self.prefix_iso(rt.base,jnp.asarray(ids[k0:k0+CH]),jnp.asarray(loc0),pm0)
                    alone=dist(self.logits_alone(rt.base,jnp.asarray(ids[k0:k0+CH]),jnp.asarray(tg),jnp.asarray(np.concatenate([np.ones(CH-1,bool),[False]])),pfa,jnp.asarray(seg0),jnp.asarray(loc0),rows))
                    out['nat']=nat.tolist();out['alone']=alone.tolist();rng=np.random.default_rng([seed,k,int(swap),7])
                    Oc=set(p for p in O if k0-W<=p<k0 and candm[p-(k0-W)])
                    for tag,KA in (('k64',64),('k16',16),('k256',256)):
                        N=int(u[KA].sum());Slm=k0-W+np.sort(order[:N]);Shm=k0-W+np.nonzero(u[KA])[0];rec=np.arange(max(0,k0-N),k0)
                        e={'N':N,'knorm_top1pct_share_lmax':len(set((Slm-(k0-W)).tolist())&top1)/max(N,1),'knorm_top1pct_share_hmax':len(set((Shm-(k0-W)).tolist())&top1)/max(N,1)}
                        for nm,SS in (('lmax',Slm),('rand',self.rand_like(sets,k,Slm,Oset,rng)),('recency',rec)):
                            S_=set(int(x) for x in SS);tp=len(S_&Oc);e[nm+'_prf']={'tp':tp,'fp':len(S_)-tp,'fn':len(Oc)-tp}
                            if len(SS)==0:e[nm]=None;continue
                            pw,kp=prefw(list(SS));e[nm]=dist(self.logits_masked(rt.base,st,c,pw,kp,rows)).tolist()
                        e['overlap_lmax_hmax']=len(set(Slm.tolist())&set(Shm.tolist()))
                        out['sizes'][tag]=e
                    res['chunks'][k]=out
                    print(f'  logitmax chunk {k} ({len(res["chunks"])}) {time.monotonic()-t0:.0f}s',flush=True)
                st=self.read_state(m,st,c,pref_c)
        return res
def analyze():
    import numpy as np
    d={w:json.load(open(OUT/f's3_logitmax_s101_{w}.json')) for w in 'AB' if (OUT/f's3_logitmax_s101_{w}.json').exists()};ch=[(w,k,c) for w,r in d.items() for k,c in r['chunks'].items()]
    pr=[c['pr_lmax'] for _,_,c in ch if c['pr_lmax'] is not None];print(f'{len(ch)} eligible chunk-worlds; PR-AUC(logit-max) mean over chunks with labels: {np.mean(pr):.4f}  (A head-max 0.1988, A 0.1204, B 0.0567, C 0.0795, ens 0.0915 from stage 1)')
    out={'pr_lmax':float(np.mean(pr)),'sizes':{}}
    for tag in ('k64','k16','k256'):
        num={'lmax':0.0,'rand':0.0,'recency':0.0};den=0.0;Ns=[];sh_l=[];sh_h=[];tp=fp=fn=0;ov=[]
        for w,k,c in ch:
            e=c['sizes'][tag];nat=np.array(c['nat']);alone=np.array(c['alone']);den+=float((alone-nat).sum());Ns.append(e['N']);sh_l.append(e['knorm_top1pct_share_lmax']);sh_h.append(e['knorm_top1pct_share_hmax']);ov.append(e['overlap_lmax_hmax']/max(e['N'],1))
            for nm in num:
                p=e[nm];p=nat if p is None else np.array(p);num[nm]+=float((p-nat).sum())
            q=e['lmax_prf'];tp+=q['tp'];fp+=q['fp'];fn+=q['fn']
        pr_=tp/max(tp+fp,1);rc=tp/max(tp+fn,1);f1=2*pr_*rc/(pr_+rc) if pr_+rc>0 else 0.0
        eps={nm:num[nm]/den for nm in num};out['sizes'][tag]=dict(eps=eps,mean_N=float(np.mean(Ns)),F1=f1,P=pr_,R=rc,knorm_top1pct_share_lmax_median=float(np.median(sh_l)),knorm_top1pct_share_hmax_median=float(np.median(sh_h)),overlap_with_hmax_median=float(np.median(ov)))
        print(f'{tag}: N mean {np.mean(Ns):.0f}; eps logit-max {eps["lmax"]:.3f}, rand-dist {eps["rand"]:.3f}, recency {eps["recency"]:.3f}; F1 {f1:.3f} (P {pr_:.3f}, R {rc:.3f}); key-norm top-1% share of the top-N: logit-max median {np.median(sh_l):.3f} [10th {np.percentile(sh_l,10):.3f}, 90th {np.percentile(sh_l,90):.3f}] vs head-max {np.median(sh_h):.3f}; overlap with the head-max union (median share of N) {np.median(ov):.3f}')
    json.dump(out,open(OUT/'s3_logitmax_result.json','w'),indent=1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--worlds',default='AB');ap.add_argument('--analyze',action='store_true');a=ap.parse_args()
    if a.analyze:analyze();return
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime();L=LogitMax(rt,a.world_seed,readout=False);L.build3();L.build1();L.build_lm()
    for w in a.worlds:
        f=OUT/f's3_logitmax_s{a.world_seed}_{w}.json'
        if f.exists():print('exists',f);continue
        t0=time.monotonic();res=L.run_lm(w=='B',a.world_seed);json.dump(res,open(f,'w'));print('saved',f,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
