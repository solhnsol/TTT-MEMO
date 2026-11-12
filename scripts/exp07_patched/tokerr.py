"""tokerr (PROTOCOL "Token-level error-ratio measurement", registered 17:51 KST 10/17): W0, pass-1 original stream, no update.
For every chunk with a place token (480 exposures per world) the output distribution at the place-token rows under (1) the native read (natural chain, carried KV) and (2) the chunk-alone forward
(positions from 0, empty KV); and at the first suffix fast-weight layer the MLP input x (ffn_prime_norm output) of the same rows under both forwards.
Records per place-token piece j: p_nat(y), p_alone(y), ||x_nat||, ||x_alone||, cos(x_nat, x_alone). Interval of the previous same-triple exposure from the CPU counts (definition T).
Usage: tokerr.py --world-seed 101 [--worlds AB]   ->  OUT/tokerr_s{seed}_{w}.json ;  analyze_tokerr.py"""
import os,sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_stage1 as ST
from s3_stage1 import *
class TokErr(ST.Stage1):
    def build_tok(self):
        rt=self.rt;Batch=self.Batch
        def l0_xin(m,st_i,seq,x):
            b=jax.tree.map(lambda a:a[0] if eqx.is_array(a) else a,m.language_model.model.h.suffix_blocks)
            so,_=b.seq_modeling_forward(lambda blk,inp,sq,s_:type(blk).__call__(blk,inp,sq,s_,is_prefix=False),lambda n,u:n(u),b.seq_norm,b.seq_modeling_block,b.seq_post_norm,x,st_i,seq)
            x=x+so;return jax.vmap(lambda u:b.ffn_prime_norm(u))(x) if b.config.pre_norm else x
        @eqx.filter_jit
        def xin_nat(m,st_suf,c,pref_c,rows):
            st0=jax.tree.map(lambda s:s[0],st_suf);return l0_xin(m,st0,c,pref_c)[rows].astype(jnp.float32)
        @eqx.filter_jit
        def xin_alone(m,ids_,tg_,mk_,pref_,seg_,loc_,rows):
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_);st0=jax.tree.map(lambda s:s[0],clone_state(rt.state_suffix));out=l0_xin(m,st0,seq,pref_)[rows].astype(jnp.float32)
            finally:packing.PACK_SUF.clear()
            return out
        self.xin_nat=xin_nat;self.xin_alone=xin_alone
    def run_tok(self,swap,seed):
        rt=self.rt;m=rt.base;ids=np.asarray(self.T.P[swap].ids)[:NCH*CH];W=WIN
        mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
        tab=SETS.sentence_table(self.T.w,swap,self.T.tok,gw,SP)
        seg0=np.zeros(CH,np.int32);loc0=np.arange(CH,dtype=np.int32);pm0=jnp.asarray(packing.prefix_mask(seg0,W));recs=[];t0=time.monotonic()
        # interval of the previous same-triple exposure (definition T)
        interval={}
        for s in tab:
            k=s['place'][0]//CH;k0=k*CH;prev=[u for u in tab if u['t']==s['t'] and u['place'][0]<s['place'][0]]
            if not prev:interval[s['idx']]='none'
            else:
                p=max(u['place'][0] for u in prev);interval[s['idx']]='same' if p//CH==k else ('in8k' if p>=k0-W else 'out8k')
        bychunk={}
        for s in tab:
            for j in s['place']:
                k=j//CH;row=j-1-k*CH
                if row>=0:bychunk.setdefault(k,[]).append((s['idx'],s['pair'],j,row))
        with rt.mesh:
            pref_nat=rt.prefix(rt.base,seq);st=clone_state(rt.state_suffix)
            for k in range(NCH):
                c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH]
                if k in bychunk:
                    items=bychunk[k];rows=jnp.asarray([r for *_,r in items],dtype=jnp.int32);ys=np.asarray([ids[j] for _,_,j,_ in items])
                    pn=np.asarray(self.logits_nat(m,st,c,pref_c,rows));tg=np.concatenate([ids[k*CH+1:(k+1)*CH+1],[rt.eos]])[:CH].astype(np.int32)
                    ii=jnp.asarray(ids[k*CH:(k+1)*CH]);pfa=self.prefix_iso(m,ii,jnp.asarray(loc0),pm0);mk=jnp.asarray(np.concatenate([np.ones(CH-1,bool),[False]]))
                    pa=np.asarray(self.logits_alone(m,ii,jnp.asarray(tg),mk,pfa,jnp.asarray(seg0),jnp.asarray(loc0),rows))
                    xn=np.asarray(self.xin_nat(m,st,c,pref_c,rows));xa=np.asarray(self.xin_alone(m,ii,jnp.asarray(tg),mk,pfa,jnp.asarray(seg0),jnp.asarray(loc0),rows))
                    for i,(idx,pair,j,row) in enumerate(items):
                        nn_=float(np.linalg.norm(xn[i]));na_=float(np.linalg.norm(xa[i]))
                        recs.append(dict(idx=int(idx),pair=pair,j=int(j),k=int(k),y=int(ys[i]),p_nat=float(pn[i,ys[i]]),p_alone=float(pa[i,ys[i]]),xn=nn_,xa=na_,cos=float(xn[i]@xa[i]/max(nn_*na_,1e-30)),interval=interval[idx]))
                st=self.read_state(m,st,c,pref_c)
                if k%8==0:print(f'  tokerr chunk {k}/{NCH} {time.monotonic()-t0:.0f}s',flush=True)
        return {'seed':seed,'swap':bool(swap),'records':recs,'intervals':{str(k):v for k,v in interval.items()}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--world-seed',type=int,required=True);ap.add_argument('--worlds',default='AB');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime();L=TokErr(rt,a.world_seed,readout=False);L.build3();L.build1();L.build_tok()
    for w in a.worlds:
        f=OUT/f'tokerr_s{a.world_seed}_{w}.json'
        if f.exists() and not a.smoke:print('exists',f);continue
        t0=time.monotonic();res=L.run_tok(w=='B',a.world_seed)
        if a.smoke:print('smoke ok',len(res['records']),'records',round(time.monotonic()-t0),'s',res['records'][:2]);return
        json.dump(res,open(f,'w'));print('saved',f,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
