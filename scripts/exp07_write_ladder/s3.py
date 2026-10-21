"""S3 selective masking in the WRITE forward (3B TTT-E2E), PROTOCOL "S3 selective masking, registered spec v1.1" + 10/17 amendments.
Per chunk k: (1) READ forward = the native one (rt.prefix over the stream + suffix with the carried KV state): only advances the suffix state; (2) WRITE forward: the same chunk
with attention to the key positions in a set S removed at ALL layers (24 frozen prefix layers and 8 suffix layers), RoPE positions kept, loss = the native chunk NLL (all positions),
gradient -> the native clipped SGD step (lr x1, clip 1.0). Weights at the start of the chunk are the same for both forwards.
Prefix layers: the natural per-layer post-RoPE K,V of the whole stream are recorded once per (world, pass) (layer-by-layer, kept on host); the masked prefix of a chunk recomputes only the
chunk's queries, attending to the cached natural window keys (minus S) and its own fresh keys, so the effect of S propagates through the chunk's own hidden states at every layer.
Suffix layers: the carried KV window (natural read states) plus the chunk, with the same S through a key-mask AND-ed into SWA.sw_causal_mask.
Cells (--cell): native3 (empty S, flag on; numerical sanity only), oracle (O), othent, randdist, recency, prefixall (see s3_sets.py). Modes as stream2x2.py: --mode save, then --mode eval.
Multi-pass: fresh filler (ladder.fs_streams), weights carry over, KV restarts each pass, readout after every pass.
--unit: GPU unit checks (no D, no state files). --smoke-chunks N: only the first N chunks per pass (canary-only, tag must contain 'smoke').
Usage: s3.py --cell oracle --world-seed 101 --passes 4 --mode save|eval [--worlds AB] [--tag T] [--keep]"""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
import margins_split as MS
from ladder import *
sys.path.insert(0,str(LD.ROOT/'scripts/exp04_what_ttt_retains'))
import suppress as SP
import s3_sets as SETS
from ttt.model import attention as _att
OUT=LD.OUT;SWA=_att.SWA;WIN=SETS.W
S3={};S3SUF={}
_prev_full=SWA.full_sw_attention;_prev_mask=SWA.sw_causal_mask
def _full3(self,hidden_states,seq,state):
    mode=S3.get('mode')
    if mode is None:return _prev_full(self,hidden_states,seq,state)
    if mode=='rec':
        n=hidden_states.shape[0];xq,xk,xv=self.get_attention_input(hidden_states,position_ids=jnp.arange(n));S3['rec']=(xk,xv)
        return _prev_full(self,hidden_states,seq,state)
    if mode=='ctx':
        Kc,Vc=state;q0=S3['q0'];keep=S3['keep'];n=hidden_states.shape[0]
        xq,xk,xv=self.get_attention_input(hidden_states,position_ids=q0+jnp.arange(n,dtype=jnp.int32))
        K=jnp.concatenate([Kc.astype(xk.dtype),xk],axis=0);V=jnp.concatenate([Vc.astype(xv.dtype),xv],axis=0)
        qi=(q0+jnp.arange(n,dtype=jnp.int32))[:,None];kj=(q0-WIN+jnp.arange(WIN+n,dtype=jnp.int32))[None,:]
        mask=(qi>=kj)&(qi-kj<WIN)&(kj>=0)&keep[None,:]
        o=jax.nn.dot_product_attention(xq,K,V,mask=mask,implementation='xla');o=self._merge_heads(o)
        return (self.get_attention_output(o),state)
    raise ValueError(mode)
def _mask3(self,chunk_id):
    m=_prev_mask(self,chunk_id)
    if 'kmask' in S3SUF:m=m&S3SUF['kmask'][None,:]
    return m
SWA.full_sw_attention=_full3;SWA.sw_causal_mask=_mask3
class Ladder3(LD.Ladder):
    def build3(self):
        rt=self.rt;lr,clip=self.lr,self.clip;Batch=self.Batch;lr_scale=float(getattr(self,'lr_scale',1.0))   # E38: python constant in the closure; 1.0 adds no op, so the compiled x1 graph is unchanged
        @eqx.filter_jit
        def layer_rec(blk,h,ids_):
            S3['mode']='rec'
            try:
                seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool))
                out,_=blk(h,None,seq,is_prefix=True);K,V=S3['rec']
            finally:S3.clear()
            return out,K,V
        @eqx.filter_jit
        def embed(m,ids_):return m.language_model.wte_call(ids_)
        @eqx.filter_jit
        def prefix_masked(m,ids_,Kst,Vst,q0,keep):
            h=m.language_model.wte_call(ids_);blocks=m.language_model.model.h.prefix_blocks
            S3['mode']='ctx';S3['q0']=q0;S3['keep']=keep
            try:
                seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool))
                def body(x,xs):
                    blk,Kc,Vc=xs;y,_=blk(x,(Kc,Vc),seq,is_prefix=True);return y,None
                h,_=jax.lax.scan(body,h,(blocks,Kst,Vst))
            finally:S3.clear()
            return h
        @eqx.filter_jit
        def write_step(m,st_all,st_suf,c,pref_,kmask):
            S3SUF['kmask']=kmask
            try:
                (_l,(ce,_n,_s)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,c,st_suf,prefix_outputs=pref_)
            finally:S3SUF.clear()
            g=grads.inner_parameters();gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30))
            s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
            if lr_scale!=1.0:s=s*lr_scale   # clip threshold and coef untouched
            newp=jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g)
            return newp,gn,coef,ce
        @eqx.filter_jit
        def read_state(m,st_suf,c,pref_):
            return m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c).new_state
        @eqx.filter_jit
        def native_step(m,st_all,st_suf,c,pref_):
            (_l,(ce,_n,new_st)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,c,st_suf,prefix_outputs=pref_)
            g=grads.inner_parameters();gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
            if lr_scale!=1.0:s=s*lr_scale
            newp=jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g)
            return newp,new_st,gn,coef,ce
        @eqx.filter_jit
        def grad_nat(m,st_suf,c,pref_):
            (_l,(ce,_n,_s)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,c,st_suf,prefix_outputs=pref_);return grads.inner_parameters(),ce
        @eqx.filter_jit
        def grad_mask(m,st_suf,c,pref_,kmask):
            S3SUF['kmask']=kmask
            try:(_l,(ce,_n,_s)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,c,st_suf,prefix_outputs=pref_)
            finally:S3SUF.clear()
            return grads.inner_parameters(),ce
        self.grad_nat=grad_nat;self.grad_mask=grad_mask
        self.layer_rec=layer_rec;self.embed=embed;self.prefix_masked=prefix_masked;self.write_step=write_step;self.read_state=read_state;self.native_step=native_step
    def record(self,ids):
        """natural per-layer post-RoPE K,V of the whole stream (host, [24,T,nh,d]) and the layer-by-layer natural prefix output (device, for the sanity check)."""
        rt=self.rt;m=rt.base;ids_=jnp.asarray(ids,dtype=jnp.int32);blocks=m.language_model.model.h.prefix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0]
        h=self.embed(m,ids_);Ks=[];Vs=[]
        for l in range(nl):
            blk=jax.tree.map(lambda a:a[l] if eqx.is_array(a) else a,blocks);h,K,V=self.layer_rec(blk,h,ids_);Ks.append(np.asarray(jax.device_get(K)));Vs.append(np.asarray(jax.device_get(V)))
        return np.stack(Ks),np.stack(Vs),h
    def ctx_kv(self,Kh,Vh,k):
        """window slice [24,W,nh,d] of the recorded natural K,V for chunk k (left-padded with zeros for k < 8)."""
        lo=max(0,k*CH-WIN);n=k*CH-lo;K=Kh[:,lo:k*CH];V=Vh[:,lo:k*CH]
        if n<WIN:
            pad=[(0,0),(WIN-n,0)]+[(0,0)]*(K.ndim-2);K=np.pad(K,pad);V=np.pad(V,pad)
        return jnp.asarray(K),jnp.asarray(V)
    def streams_worlds(self,swap,passes):
        if passes>1:return self.fs_streams(swap,passes),self._fs_worlds[swap]
        return [self.T.P[swap].ids],[self.T.w]
    def ev(self,m):
        if getattr(self,'canary_only',False):return {'_gen_nll':self.gen_nll(m),'_tail_nll':self.tail_nll(m),'_canary_only':True}
        return self.evaluate(m)
    def ctx_deltas_w0(self,seq,pref_nat,ids):
        """k6ctx: per chunk (nll with the native window context - nll of the chunk alone), both at W0 (the ctx_gain.py read: native chain over the pass stream; chunk alone with a fresh state and its own prefix). Must run inside rt.mesh."""
        rt=self.rt;m=rt.base;st=clone_state(rt.state_suffix);out=[]
        for k in range(NCH):
            c=rt.chunk(seq,k);st,lp=rt.chunk_logp(m,st,c,pref_nat[k*CH:(k+1)*CH]);ctx=float(-np.asarray(lp,np.float64)[:CH-1].mean())
            sq=rt.seq([int(t) for t in ids[k*CH:(k+1)*CH]]);pf=rt.prefix(m,sq);_,lp2=rt.chunk_logp(m,clone_state(rt.state_suffix),rt.chunk(sq,0),pf[:CH]);out.append(ctx-float(-np.asarray(lp2,np.float64)[:CH-1].mean()))
        return out
    def run_cell(self,swap,cell,passes=1,nch=None,seed=0):
        rt=self.rt;log=[];m=rt.base;st_all=clone_state(rt.state_all);streams,worlds=self.streams_worlds(swap,passes);nch=nch or NCH;cum=0.0
        with rt.mesh:
            log.append({'epoch':0,'margins':self.ev(m)})
            for ps in range(1,passes+1):
                ids=np.asarray(streams[ps-1])[:NCH*CH];stats=[];t_pass=time.monotonic();cum_pass=0.0
                mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref_nat=rt.prefix(rt.base,seq);st_suf=clone_state(rt.state_suffix)
                if cell!='native0':
                    t0=time.monotonic();Kh,Vh,_=self.record(ids);print(f'  recorded natural K,V ({Kh.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s',flush=True)
                    sets=SETS.Sets(SETS.sentence_table(worlds[ps-1],swap,self.T.tok,gw,SP),seed,ps,getattr(self,'fill_frac',0.25))
                    sets.fill_attn_cands=bool(getattr(self,'fill_attn_cands',False));sets.fill_no_o=bool(getattr(self,'fill_no_o',False))
                    if getattr(self,'fill_size_from',None):
                        rf=json.load(open(OUT/f'S3_{self.fill_size_from}_s{seed}_{"B" if swap else "A"}.saved.json'))['log'][ps]['chunks'];sets.fill_total={x['k']:int(x['n_masked']) for x in rf}
                if cell in ('k6ctx','k6ctxq'):t0=time.monotonic();dl=self.ctx_deltas_w0(seq,pref_nat,ids);sel=SETS.select_by_ctx(dl,self.ctx_tau) if cell=='k6ctx' else SETS.select_by_ctx_quantile(dl,self.ctx_frac);print(f'  {cell} deltas at W0 in {time.monotonic()-t0:.0f}s, chunks masked whole {sum(sel)} ({"tau "+str(self.ctx_tau) if cell=="k6ctx" else "lowest "+str(self.ctx_frac)})',flush=True)
                for k in range(nch):
                    t0=time.monotonic();old=m;c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH];ex={}
                    if cell=='native0':   # flag off: the original native step (state and gradient from one forward)
                        newp,st_suf,gn,coef,ce=self.native_step(m,st_all,st_suf,c,pref_c)
                    else:
                        S,fr=(sets.prefixall(k) if sel[k] else [],1.0) if cell in ('k6ctx','k6ctxq') else sets.chunk_set(cell,k);keep=sets.keep_vector(k,S);Kc,Vc=self.ctx_kv(Kh,Vh,k)
                        pref_w=self.prefix_masked(m,jnp.asarray(ids[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep))
                        newp,gn,coef,ce=self.write_step(m,st_all,st_suf,c,pref_w,jnp.asarray(keep));st_suf=self.read_state(m,st_suf,c,pref_c)
                        ex={'n_masked':int(len(S)),'match_frac':float(fr)}
                        if cell in ('k6','k6p'):nO=len(sets.oracle(k));ex.update(n_O=nO,has_O=bool(nO))
                        if cell in ('k6ctx','k6ctxq'):nO=len(sets.oracle(k));ex.update(n_O=nO,has_O=bool(nO),ctx_delta=dl[k],selected=bool(sel[k]))
                        if cell in ('sent','bg','noo'):nO=len(sets.oracle(k));ex.update(n_O=nO,has_O=bool(nO),types=sets.type_counts(k,S))
                        if cell=='k3all':nO=len(sets.oracle(k));ex.update(n_O=nO,has_O=bool(nO),n_K3=len(sets.sentencecover(k)),**{'n_'+a_:b_ for a_,b_ in sets.k3all_parts(k)[1].items()})
                        if cell in ('k3h','k3hc','k3r'):nO=len(sets.oracle(k));ex.update(n_O=nO,has_O=bool(nO),n_K3=len(sets.sentencecover(k)))
                        if cell in ('op','kp','kx','kp_u','kx_u','kx_s','kx_o'):nO=len(sets.oracle(k));P_,X_=sets.pxsplit(k);ex.update(n_O=nO,has_O=bool(nO),n_K3=len(sets.sentencecover(k)),n_P=len(P_),n_X=len(X_))
                        if cell in ('occ','frm','unif_occ'):O_,C_,F_=sets.kindsplit(k);ex.update(n_O=len(O_),has_O=bool(O_),n_OCC=len(C_),n_FRM=len(F_),n_K3=len(O_)+len(C_)+len(F_))
                        if cell in ('k3','k3b','k3f'):nO=len(sets.oracle(k));ex.update(n_O=nO,has_O=bool(nO),n_sent=int(getattr(sets,'last_nsent',0)))
                        if cell in ('oraclefill','randfill'):nO=len(sets.oracle(k));ex.update(n_O=nO,n_F=int(len(S)-nO) if cell=='oraclefill' else int(len(S)))
                    m=self.set_inner(m,newp)
                    un=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=un;cum_pass+=un
                    stats.append({'k':k,'gnorm':float(gn),'coef':float(coef),'update_norm':un,'ce':float(ce),'sec':time.monotonic()-t0,**ex})
                    if k%8==0:print(f'  {cell} {"B" if swap else "A"} pass {ps} chunk {k+1}/{nch} {stats[-1]["sec"]:.1f}s',flush=True)
                cs=[s_['coef'] for s_ in stats]
                log.append({'epoch':ps,'lr_scale':float(getattr(self,'lr_scale',1.0)),'first_chunk_ce':stats[0]['ce'],'mean_update_norm':cum_pass/len(stats),'cum_update_norm':cum,'chunks':stats,'clip_bind_frac':sum(c_<1 for c_ in cs)/len(cs),'mean_coef':sum(cs)/len(cs),'n_masked_total':sum(s_.get('n_masked',0) for s_ in stats),
                            'stream_hash':hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),'inner_hash':reuse._hash_pytree(m.inner_parameters())[0],'sec_pass':time.monotonic()-t_pass,'margins':self.ev(m)})
                print(f'  {cell} {"B" if swap else "A"} pass {ps}: cum upd {cum:.3f}, clip binds {log[-1]["clip_bind_frac"]:.3f}, masked tokens {log[-1]["n_masked_total"]}, {log[-1]["sec_pass"]:.0f}s',flush=True)
                if 'Kh' in dir():del Kh,Vh
        return log
    def unit(self,world_seed):
        """GPU unit checks on pass 1, world A: (U-a) flag on with an EMPTY set: masked prefix vs the natural prefix, write-forward CE vs the native CE; (U-b) an oracle chunk: CE and gradient change;
        (U-c) prefix-all vs chunk-alone sanity: CE with all prior context masked is above the native CE on chunks with context."""
        rt=self.rt;m=rt.base;st_all=clone_state(rt.state_all);ids=np.asarray(self.T.P[False].ids)[:NCH*CH];res={}
        with rt.mesh:
            mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref_nat=rt.prefix(rt.base,seq)
            Kh,Vh,hrec=self.record(ids);res['rec_vs_scan_prefix_maxabs']=float(jnp.max(jnp.abs(hrec.astype(jnp.float32)-pref_nat.astype(jnp.float32))));res['prefix_rms']=float(jnp.sqrt(jnp.mean(pref_nat.astype(jnp.float32)**2)))
            sets=SETS.Sets(SETS.sentence_table(self.T.w,False,self.T.tok,gw,SP),world_seed,1);st=clone_state(rt.state_suffix);rows=[]
            ks=[0,1,5,9,12,18,19,20]
            for k in range(0,21):
                c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH];Kc,Vc=self.ctx_kv(Kh,Vh,k);ids_k=jnp.asarray(ids[k*CH:(k+1)*CH],dtype=jnp.int32)
                if k in ks:
                    full=jnp.ones(WIN+CH,bool);pw=self.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),full)
                    d=float(jnp.max(jnp.abs(pw.astype(jnp.float32)-pref_c.astype(jnp.float32))));rel=d/res['prefix_rms']
                    g0,ce0=self.grad_nat(m,st,c,pref_c);g1,ce1=self.grad_mask(m,st,c,pw,full);gn0=optax.global_norm(g0);gn1=optax.global_norm(g1)
                    dg=optax.global_norm(jax.tree.map(lambda a,b:a-b,g1,g0));dot=sum(jnp.vdot(x,y) for x,y in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)))
                    O=sets.oracle(k);row={'k':k,'prefix_maxabs_vs_natural':d,'rel_to_rms':rel,'ce_native':float(ce0),'ce_empty_mask':float(ce1),'gnorm_native':float(gn0),'gnorm_empty_mask':float(gn1),'grad_rel_diff':float(dg/gn0),'grad_cos':float(dot/(gn0*gn1)),'n_O':len(O)}
                    if O:
                        keep=jnp.asarray(sets.keep_vector(k,O));pw2=self.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),keep);g2,ce2=self.grad_mask(m,st,c,pw2,keep);dot2=sum(jnp.vdot(x,y) for x,y in zip(jax.tree.leaves(g0),jax.tree.leaves(g2)));gn2=optax.global_norm(g2);row.update(ce_oracle=float(ce2),gnorm_oracle=float(gn2),grad_cos_oracle_vs_native=float(dot2/(gn0*gn2)))
                    keepa=jnp.asarray(sets.keep_vector(k,sets.prefixall(k)));pw3=self.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),keepa);g3,ce3=self.grad_mask(m,st,c,pw3,keepa);row.update(ce_prefixall=float(ce3),gnorm_prefixall=float(optax.global_norm(g3)))
                    rows.append(row);print(row,flush=True)
                st=self.read_state(m,st,c,pref_c)
        res['rows']=rows;return res
def unit_diag(L,world_seed):
    """diagnostic for the prefix max-abs criterion (design 17:58): where is the largest difference, how many bf16 ulps, what is the noise floor between two natural kernel paths, and the metric without the outlier dims."""
    rt=L.rt;m=rt.base;ids=np.asarray(L.T.P[False].ids)[:NCH*CH];res={'chunks':[]}
    def ulp(a):
        a=abs(float(a));return 2.0**(np.floor(np.log2(a))-7) if a>0 else 0.0
    with rt.mesh:
        mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref_nat=rt.prefix(rt.base,seq)
        Kh,Vh,hrec=L.record(ids);full=jnp.ones(WIN+CH,bool)
        for k in (0,1,9,18):
            sl=slice(k*CH,(k+1)*CH);scan=np.asarray(pref_nat[sl].astype(jnp.float32));rec=np.asarray(hrec[sl].astype(jnp.float32));Kc,Vc=L.ctx_kv(Kh,Vh,k)
            s3=np.asarray(L.prefix_masked(m,jnp.asarray(ids[sl],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),full).astype(jnp.float32))
            rms=float(np.sqrt(np.mean(scan**2)));dim_rms=np.sqrt(np.mean(scan**2,axis=0));out_dims=np.argsort(-dim_rms)[:8];keep=np.ones(scan.shape[1],bool);keep[out_dims]=False
            row={'k':k,'prefix_rms':rms,'top_rms_dims':[int(d) for d in out_dims],'top_rms_dim_values':[float(dim_rms[d]) for d in out_dims]}
            for nm,(a,b) in (('S3_vs_scan',(s3,scan)),('S3_vs_rec',(s3,rec)),('rec_vs_scan',(rec,scan))):
                d=np.abs(a-b);i,j=np.unravel_index(np.argmax(d),d.shape);ref=float(b[i,j])
                row[nm]=dict(maxabs=float(d.max()),maxabs_over_rms=float(d.max()/rms),rms_diff_over_rms=float(np.sqrt(np.mean((a-b)**2))/rms),at_token=int(i),at_dim=int(j),value=ref,ulps=float(d[i,j]/max(ulp(ref),1e-30)),
                             maxabs_without_top8_dims=float(d[:,keep].max()),maxabs_without_top8_over_rms=float(d[:,keep].max()/rms),
                             median_ulps_at_largest_1pct=float(np.median(d.ravel()[np.argsort(-d.ravel())[:max(1,d.size//100)]]/np.maximum(np.array([ulp(x) for x in b.ravel()[np.argsort(-d.ravel())[:max(1,d.size//100)]]]),1e-30))))
            res['chunks'].append(row);print(json.dumps(row),flush=True)
    return res
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=['native0','native3','oracle','othent','randdist','recency','prefixall','oraclesub','othentsub','oraclefill','randfill','k6','k6p','k3','k3b','k3f','k6ctx','k6ctxq','occ','frm','unif_occ','k3all','k3h','k3hc','k3r','op','kp','kx','kp_u','kx_u','kx_s','kx_o','sent','bg','noo'],default='oracle');ap.add_argument('--world-seed',type=int,default=101)
    ap.add_argument('--worlds',default='AB');ap.add_argument('--mode',choices=['save','eval'],default='save');ap.add_argument('--tag',default='');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--keep',action='store_true');ap.add_argument('--regold',action='store_true',help='E47: --mode eval only: re-score the saved states with the current ladder.evaluate (gold / foil span log-probs stored), keep the states, write <result>.regold.json next to the untouched original and compare the margins with it')
    ap.add_argument('--canary-only',action='store_true');ap.add_argument('--smoke-chunks',type=int,default=None);ap.add_argument('--unit',action='store_true');ap.add_argument('--unit-diag',action='store_true');ap.add_argument('--fill-size-from',default=None,help='E41 U49: oraclefill total per chunk = n_masked of OUT/S3_<TAG>_s<seed>_<world>.saved.json (e.g. attnsel_p4), instead of fill-frac x window');ap.add_argument('--fill-no-o',action='store_true',help='oraclefill: chunks WITHOUT an oracle set are also filled uniformly (total = fill-size-from n_masked of that chunk, else fill-frac x window); needed to pair with attn-sel, which masks 40-50% in every chunk');ap.add_argument('--fill-attn-cands',action='store_true',help='oraclefill: draw the fill only from the attention cells candidate set (positions >= 4 and < chunk start - 128)');ap.add_argument('--fill-frac',type=float,default=0.25,help='E39: oraclefill/randfill target masked fraction of the write window');ap.add_argument('--lr-scale',type=float,default=1.0,help='E38: multiply the inner step size (after ilr multiplier, clip untouched); 1.0 = bit-identical to the old code')
    ap.add_argument('--ctx-frac',type=float,default=0.30,help='k6ctxq: the fraction of chunks 1..62 with the lowest delta that are masked whole (n = round(frac x 62), ties -> lower chunk index)');ap.add_argument('--ctx-tau',type=float,default=-0.38,help='k6ctx: a chunk is masked whole (prefix-all, as K6\') when its window context gain delta (mean NLL with the 8K window - mean NLL alone, read at W0 on the pass stream) is below tau');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime()
    if a.unit_diag:
        L=Ladder3(rt,a.world_seed,readout=False);L.lr_scale=a.lr_scale;L.build3();res=unit_diag(L,a.world_seed);json.dump(res,open(OUT/f's3_unit_diag_s{a.world_seed}.json','w'),indent=1);print('unit diag done');return
    if a.unit:
        L=Ladder3(rt,a.world_seed,readout=False);L.lr_scale=a.lr_scale;L.build3();res=L.unit(a.world_seed);json.dump(res,open(OUT/f's3_unit_s{a.world_seed}.json','w'),indent=1);print('unit done');return
    tag=a.tag or f'{a.cell}_p{a.passes}';lev='S3';SD=OUT/'states'/f'{lev}_{tag}_s{a.world_seed}'
    L=Ladder3(rt,a.world_seed,readout=(a.mode!='save'),save_dir=SD);L.lr_scale=a.lr_scale;L.ctx_tau=a.ctx_tau;L.ctx_frac=a.ctx_frac;L.fill_frac=a.fill_frac;L.fill_size_from=a.fill_size_from;L.fill_attn_cands=a.fill_attn_cands;L.fill_no_o=a.fill_no_o;L.build3();L.canary_only=a.canary_only
    if a.smoke_chunks:assert a.canary_only and 'smoke' in tag
    if not a.smoke_chunks:json.dump({'items':L.items,'fam_sizes':{k:len(v) for k,v in L.fam.items()},'lr':L.lr,'lr_scale':a.lr_scale,'fill_frac':a.fill_frac,'clip':L.clip,'cell':a.cell,'passes':a.passes,'xla_flags':os.environ.get('XLA_FLAGS')},open(OUT/f'meta_{lev}_{tag}_s{a.world_seed}.json','w'),indent=1)
    for w in a.worlds:
        f=OUT/f'{lev}_{tag}_s{a.world_seed}_{w}.json';fs=OUT/f'{lev}_{tag}_s{a.world_seed}_{w}.saved.json'
        if a.regold:
            assert a.mode=='eval','--regold needs --mode eval';f_orig=f;f=f.with_name(f.stem+'.regold.json')   # E47: re-score saved states (never removed)
        if f.exists():print('exists',f);continue
        if a.mode=='eval':
            d=json.load(open(fs));t0=time.monotonic()
            for e in d['log']:
                if isinstance(e.get('margins'),dict) and '_file' in e['margins']:
                    fn=e['margins']['_file'];h=e['margins']['_hash'];mm=L.load_state(fn)
                    assert reuse._hash_pytree(mm.inner_parameters())[0]==h;e['margins']=L.evaluate(mm);e['hash']=h
                    if not (a.keep or a.regold):os.remove(fn)
            if a.regold:
                d['regold_check']=MS.regold_check(json.load(open(f_orig))['log'],d['log']) if f_orig.exists() else None;print('regold check vs the original result (max |diff| of the margins per epoch):',d['regold_check'],flush=True)
            d['sec_eval']=time.monotonic()-t0;json.dump(d,open(f,'w'));print('scored',f,round(d['sec_eval']),'s',flush=True);continue
        L.save_dir=SD/w;L._nsave=0;(SD/w).mkdir(parents=True,exist_ok=True)
        if fs.exists():print('exists',fs);continue
        t0=time.monotonic();log=L.run_cell(w=='B',a.cell,a.passes,a.smoke_chunks,seed=a.world_seed)
        json.dump({'log':log,'sec':time.monotonic()-t0,'cell':a.cell,'passes':a.passes,'lr_scale':a.lr_scale},open(fs,'w'));print('saved',fs,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
