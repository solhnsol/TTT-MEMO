"""E29 input x error factor exchange at W0 (3B TTT-E2E, one step). NOT RUN YET: written from the autosnap/a100 sources (gradfactors.py, s3.py,
s3_stage1.py, ladder.py, packing.py) without GPU access; the 'compare' and '--verify-step' stages exist to catch a wrong assumption before any real run.
Factors per chunk k and suffix layer l (8), SwiGLU matrix of the inner MLP (w1, w3 from x_in; w2 from x_mid; y = x @ W), same definitions as gradfactors.py:
  dL/dW1 = x_in^T d1, dL/dW3 = x_in^T d3, dL/dW2 = x_mid^T d2, L = the chunk's mean loss (all in-chunk targets, native loss mask).
Conditions (all at W0, fp32, matmul 'highest', the same chunk k of the stream, the natural suffix KV state chain at W0):
  A = native: natural prefix over the whole stream, suffix with the carried KV (every token sees exactly 8192 keys).
  M = prefix-all-pos: every window position before the chunk removed at all 32 layers, RoPE absolute positions kept (s3.prefix_masked with keep = chunk part only,
      suffix key-mask S3SUF['kmask'] = [zeros(WIN), ones(CH)]).
Exchange cell (x<-P, d<-Q): g = sum_k w_k * sum_t d^Q_t (x^P_t)^T per layer/matrix with w_k = n_loss_k / sum n_loss (same token position t on both sides).
Realisation: W = W0 - eta * coef * s * g (eta = lr * ilr multiplier, coef = min(1, clip/||g||) on the cell's own aggregated g over the three matrices, s in --scales),
then the full R1-alt readout (ladder.evaluate) -> D_alt per pair = (M_L3(A) - M_L3(B))/2 - (M_XP(A) - M_XP(B))/2 with the A / B world streams of the same seed,
canary rise = max over the two canaries (gen, tail), over worlds. Decomposition: s_x = [D(x<-M,d<-A) - D(A,A)]/[D(M,M) - D(A,A)], s_d = [D(x<-A,d<-M) - D(A,A)]/[...], s_int = 1 - s_x - s_d.
Not implemented here: first-order values <grad_W D, g> and the in-chunk position profile (need the weight-space gradient of the binding margin; margin_grad is a closure inside
grad_components.main and is not importable; separate request after this stage).
Modes:
  compare  --seed 101 --world A --chunks 0,1,2   M extracted here vs the job-90 chunk-alone 'ntp' dump (rel. errors of x_in, x_mid, d1, d3, d2)
  swap     --seed 101 --worlds AB --chunks all --scales 1,2,4,8 [--save-dir DIR] [--verify-step]   online extraction + exchange + readout (no dump needed)
Output: OUT/e29_{mode}_s{seed}.json (+ .md summary). Never modifies existing result files."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3 as S3M
from s3 import *
import ladder as LD
from ladder import *
sys.path.insert(0,str(LD.ROOT/'scripts/exp04_what_ttt_retains'))
import suppress as SP
from ttt.model.transformer import Block
from ttt.model.loss import cross_entropy_loss_and_accuracy
OUT=LD.OUT;DUMP=Path(os.environ.get('DUMP','/home/solhnsol/workspace/TTT-MEMO-dumps/gradfactors'))
KV_SHAPE=tuple(int(x) for x in os.environ.get('E29_KV_SHAPE',f'24,{WIN},32,80').split(','))   # [prefix layers, WIN, heads, head dim]; content is irrelevant for M (all window keys masked)
class TapMLP(eqx.Module):
    w1:object;w2:object;w3:object;dropout:object;e1:jax.Array;e3:jax.Array;e2:jax.Array
    def __call__(self,x):
        z1=self.w1(x)+self.e1;z3=self.w3(x)+self.e3;z2=self.w2(jax.nn.silu(z1)*z3)+self.e2
        return self.dropout(z2)
def build_ops(L,rt32):
    m0=rt32.base;lm=m0.language_model;H=lm.model.h;NL=jax.tree_util.tree_leaves(H.suffix_blocks)[0].shape[0];cfg=lm.config
    dff=int(H.suffix_blocks.feed_forward_prime.w1.weight.shape[-1]);dm=int(H.suffix_blocks.feed_forward_prime.w1.weight.shape[-2])
    def layer(i):return jax.tree.map(lambda a:a[i] if eqx.is_array(a) else a,H.suffix_blocks)
    def st_layer(st,i):return jax.tree.map(lambda a:a[i],st)
    def head(hs):
        hs=jax.vmap(lm.model.ln_f)(hs)
        if cfg.tie_word_embeddings:return hs@lm.model.wte.weight.T.astype(hs.dtype)
        return lm.lm_head(hs)
    def tapped_loss(taps,c,st,pref_):
        x=pref_
        for i in range(NL):
            b=layer(i);f=b.feed_forward_prime
            b=eqx.tree_at(lambda q:q.feed_forward_prime,b,TapMLP(f.w1,f.w2,f.w3,f.dropout,*taps[i]))
            x,_=Block.__call__(b,x,st_layer(st,i),c,is_prefix=False)
        loss,_=cross_entropy_loss_and_accuracy(head(x),c.target_tokens,c.loss_masks);return loss
    def _block_capture(b,x,st_i,seq,xin,xmid):
        cfg_b=b.config;rmsf=lambda n,u:n(u)
        so,_=b.seq_modeling_forward(lambda blk,inp,sq,s_:type(blk).__call__(blk,inp,sq,s_,is_prefix=False),rmsf,b.seq_norm,b.seq_modeling_block,b.seq_post_norm,x,st_i,seq)
        x=x+so;f=b.feed_forward_prime
        inp=jax.vmap(lambda u:b.ffn_prime_norm(u))(x) if cfg_b.pre_norm else x
        xin.append(inp);z1=f.w1(inp);z3=f.w3(inp);mid=jax.nn.silu(z1)*z3;xmid.append(mid);o=f.w2(mid)
        if cfg_b.post_norm:o=jax.vmap(lambda u:b.ffn_prime_post_norm(u))(o)
        x=x+o;fo=b.ffn_forward(lambda ff,u:ff(u),rmsf,b.ffn_norm,b.feed_forward,b.ffn_post_norm,x);return x+fo
    @eqx.filter_jit
    def deltas(c,st,pref_,kmask):
        S3SUF['kmask']=kmask
        try:
            z=[(jnp.zeros((CH,dff),jnp.float32),jnp.zeros((CH,dff),jnp.float32),jnp.zeros((CH,dm),jnp.float32)) for _ in range(NL)]
            g=jax.grad(tapped_loss)(z,c,st,pref_)
        finally:S3SUF.clear()
        return jnp.stack([a for a,_,_ in g]),jnp.stack([b_ for _,b_,_ in g]),jnp.stack([e for _,_,e in g])
    @eqx.filter_jit
    def acts(c,st,pref_,kmask):
        S3SUF['kmask']=kmask
        try:
            x=pref_;xin=[];xmid=[]
            for i in range(NL):x=_block_capture(layer(i),x,st_layer(st,i),c,xin,xmid)
            ref=lm.suffix_call(prefix_outputs=pref_,state=st,seq=c).last_hidden_states
            return jnp.stack(xin),jnp.stack(xmid),jnp.max(jnp.abs(jax.vmap(lm.model.ln_f)(x)-ref)),jnp.max(jnp.abs(ref))
        finally:S3SUF.clear()
    fspec=reuse.get_filter_spec(rt32.base,rt32.cfg.training.spec_inner,'inner parameters')   # differentiate only the inner parameters (a whole-model fp32 gradient asked for a 23 GiB temporary: OOM)
    @eqx.filter_jit
    def ref_grad(m,c,st,pref_,kmask):
        S3SUF['kmask']=kmask;inner,outer_=eqx.partition(m,fspec)
        try:
            g=eqx.filter_grad(lambda ip:MetaModel.lm_loss(eqx.combine(ip,outer_),c,st,prefix_outputs=pref_)[0])(inner)
        finally:S3SUF.clear()
        p=g.language_model.model.h.suffix_blocks.feed_forward_prime;return p.w1.weight,p.w3.weight,p.w2.weight,optax.global_norm(jax.tree_util.tree_leaves(g))
    @eqx.filter_jit
    def outer(xin,xmid,d1,d3,d2):
        one=lambda a:(jnp.einsum('td,tf->df',a[0],a[1]),jnp.einsum('td,tf->df',a[0],a[2]),jnp.einsum('tf,td->fd',a[3],a[4]))   # one layer at a time (bounded temporaries)
        return jax.lax.map(one,(xin,d1,d3,xmid,d2))
    @eqx.filter_jit
    def add(acc,g,w):return tuple(a+w*b for a,b in zip(acc,g))
    @eqx.filter_jit
    def prof(G,xin,xmid,d1,d3,d2):
        """per in-chunk position bin (8 x 128) contribution to <G, sum_t d_t x_t^T>, summed over the 8 layers and the three matrices."""
        c=jnp.sum(jnp.einsum('ltd,ldf->ltf',xin,G[0])*d1,-1)+jnp.sum(jnp.einsum('ltd,ldf->ltf',xin,G[1])*d3,-1)+jnp.sum(jnp.einsum('ltf,lfd->ltd',xmid,G[2])*d2,-1)
        return c.sum(0).reshape(8,CH//8).sum(1)
    return dict(deltas=deltas,acts=acts,ref_grad=ref_grad,outer=outer,add=add,prof=prof,NL=NL,dff=dff,dm=dm)
def build_margin(L,rt32):
    """binding-margin gradients at W0 (fp32), as grad_components.main (R0 readout = empty context; a stated approximation): returns grad(L3 pair p) / grad(XP pair p) of
    sum(lp*gold) - sum(lp*foil) over the pair's items, inner-parameter trees. Copied from grad_components.py (margin_fn, margin_grad, chunk_inputs, item_rows, masks)."""
    W=int(rt32.cfg.model.sliding_window_size);T=L.T;Batch=L.Batch
    @eqx.filter_jit
    def prefix32(m,ids_,pos_,mask_):
        packing.PACK_PRE['mask']=mask_
        try:
            seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
            return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),clone_state(rt32.state_prefix),seq).last_hidden_state
        finally:packing.PACK_PRE.clear()
    def item_rows(items):
        rows=[]
        for it in items:
            for c in (it['gold'],it['foil']):
                ids,span=stage2.item_ids(T.tok,stage2.make_slot(T.tok,None),it['prefix'],c);rows.append((ids,(span[0]-1,span[1]-1)))
        return rows
    def margin_fn(m,ids_,tg_,pref_,seg_,loc_,wg_,wf_):
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:
            r=m.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt32.state_suffix),seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
            lg=r.logits.astype(jnp.float32);lp=jnp.take_along_axis(lg,tg_[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
        finally:packing.PACK_SUF.clear()
        return jnp.sum(lp*wg_)-jnp.sum(lp*wf_)
    @eqx.filter_jit
    def margin_grad(m,*args):return eqx.filter_grad(lambda m:margin_fn(m,*args))(m).inner_parameters()
    inp={}
    pairs=sorted({it['pair'] for it in L.fam['L3']},key=lambda s_:int(s_[1:]));xpairs=sorted({it['pair'] for it in L.fam['XP']},key=lambda s_:int(s_[1:]))
    with rt32.mesh,jax.default_matmul_precision('highest'):
        for fam,pl in (('L3',pairs),('XP',xpairs)):
            for p in pl:
                its=[it for it in L.fam[fam] if it['pair']==p];ch,where=packing.pack_rows(item_rows(its));assert len(ch)==1;ids,seg,loc=ch[0]
                pref=prefix32(rt32.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)));tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32)
                wg=np.zeros(CH,np.float32);wf=np.zeros(CH,np.float32)
                for r_,(c_,a_,b_) in enumerate(where):(wg if r_%2==0 else wf)[a_:b_]=1.0
                inp[(fam,p)]=(jnp.asarray(ids),jnp.asarray(tg),pref,jnp.asarray(seg),jnp.asarray(loc),jnp.asarray(wg),jnp.asarray(wf))
    return dict(grad=lambda fam,p:margin_grad(rt32.base,*inp[(fam,p)]),pairs=pairs,xpairs=xpairs)
def factors(O,c,st,pref_,kmask):
    d1,d3,d2=O['deltas'](c,st,pref_,kmask);xin,xmid,hd,hmax=O['acts'](c,st,pref_,kmask);return xin,xmid,d1,d3,d2,float(hd),float(hmax)
def recon_err(O,m,c,st,pref_,kmask,xin,xmid,d1,d3,d2):
    r1,r3,r2,gn=O['ref_grad'](m,c,st,pref_,kmask);rec=O['outer'](xin,xmid,d1,d3,d2)
    err=[float(jnp.linalg.norm(x-y)/jnp.maximum(jnp.linalg.norm(y),1e-30)) for x,y in zip(rec,(r1,r3,r2))]
    gn3=float(jnp.sqrt(sum(jnp.sum(x*x) for x in rec)));return err,float(gn),gn3
def pref_native32(rt,rt32,seq32,ids):
    try:return rt32.prefix(rt32.base,seq32),'fp32'
    except Exception as e:   # e.g. OOM of the 64K fp32 prefix: fall back to the bf16 natural prefix cast to fp32 (stated in the manifest)
        print('fp32 natural prefix failed:',repr(e)[:200],'-> bf16 prefix cast to fp32',flush=True)
        seq=rt.seq(ids,mask=np.asarray(seq32.loss_masks));return rt.prefix(rt.base,seq).astype(jnp.float32),'bf16->fp32'
def stream_ids(L,w):return np.asarray(L.T.P[w=='B'].ids)[:NCH*CH]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['compare','swap']);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--worlds',default='AB');ap.add_argument('--world',default='A')
    ap.add_argument('--chunks',default='all');ap.add_argument('--scales',default='1,2,4,8');ap.add_argument('--save-dir',default=None);ap.add_argument('--verify-step',action='store_true');ap.add_argument('--smoke',type=int,default=None);ap.add_argument('--no-first-order',action='store_true')
    ap.add_argument('--dump',default=str(DUMP));ap.add_argument('--nboot',type=int,default=4000);a=ap.parse_args()
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime();L=Ladder3(rt,a.seed,readout=(a.mode=='swap'));L.build3()
    if a.mode=='compare':L.rt32=runtime_fp32() if not hasattr(L,'rt32') else L.rt32
    rt32=L.rt32;m32=rt32.base;O=build_ops(L,rt32);W=WIN
    h0=reuse._hash_pytree(rt.base.inner_parameters())[0];man={'seed':a.seed,'mode':a.mode,'W0_inner_hash':h0,'kv_shape':KV_SHAPE,'code_sha':hashlib.sha256(open(__file__,'rb').read()).hexdigest()}
    Kc0=jnp.zeros(KV_SHAPE,jnp.float32);Vc0=jnp.zeros(KV_SHAPE,jnp.float32);keepM=jnp.asarray(np.r_[np.zeros(W,bool),np.ones(CH,bool)]);keepA=jnp.asarray(np.ones(W+CH,bool))
    if a.mode=='compare':
        w=a.world;ks=[int(x) for x in a.chunks.split(',')] if a.chunks!='all' else [0,1,2];rows=SP.chunk_rows(L.T.w,w=='B',L.T.tok,rt.bos,rt.eos,'ntp')[0];byk={x[3]['k']:x for x in rows};res={}
        with rt32.mesh,jax.default_matmul_precision('highest'):
            for k in ks:
                ids,mk,_,info=byk[k][0],byk[k][1],byk[k][2],byk[k][3];ids=np.asarray(ids,np.int32);tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32)
                c=L.Batch(input_ids=jnp.asarray(ids),target_tokens=jnp.asarray(tg),loss_masks=jnp.asarray(mk))
                pref=L.prefix_masked(m32,jnp.asarray(ids),Kc0,Vc0,jnp.int32(k*CH),keepM);st=clone_state(rt32.state_suffix)
                xin,xmid,d1,d3,d2,hd,hmax=factors(O,c,st,pref,keepM);z=np.load(Path(a.dump)/f's{a.seed}_{w}'/f'ntp_k{k:02d}.npz');r={}
                for nm,x in (('x_in',xin),('x_mid',xmid),('d1',d1),('d3',d3),('d2',d2)):
                    y=jnp.asarray(z[nm]);r[nm]={'rel_total':float(jnp.linalg.norm(x-y)/jnp.linalg.norm(y)),'rel_layer_max':float(jnp.max(jnp.linalg.norm((x-y).reshape(x.shape[0],-1),axis=1)/jnp.linalg.norm(y.reshape(y.shape[0],-1),axis=1)))}
                err,gn,gn3=recon_err(O,m32,c,st,pref,keepM,xin,xmid,d1,d3,d2);r['recon_err_w1_w3_w2']=err;r['gnorm_ref_vs_3matrix']=[gn,gn3];r['hidden_maxdiff']=hd;res[k]=r
                print(k,json.dumps(r),flush=True)
        worst=max(max(v[n]['rel_total'] for n in ('x_in','x_mid','d1','d3','d2')) for v in res.values());man['worst_rel_total']=worst;man['gate_1e-3']=worst<=1e-3
        json.dump({'manifest':man,'chunks':res},open(OUT/f'e29_compare_s{a.seed}_{a.world}.json','w'),indent=1);print('worst rel',worst,'gate 1e-3:',worst<=1e-3);return
    # ---- swap ----
    scales=[float(x) for x in a.scales.split(',')];st_all=clone_state(rt.state_all);eta0=None
    inner0=rt.base.inner_parameters();get3=lambda t:(t.language_model.model.h.suffix_blocks.feed_forward_prime.w1.weight,t.language_model.model.h.suffix_blocks.feed_forward_prime.w3.weight,t.language_model.model.h.suffix_blocks.feed_forward_prime.w2.weight)
    w0p=get3(inner0);eta=L.lr*float(rt.base.get_ilr_multiplier(st_all.get(rt.base.step_index)));clip=L.clip;man.update({'eta':eta,'clip':clip,'scales':scales})
    cells=[('A','A'),('M','M'),('M','A'),('A','M')]   # (x from, delta from)
    G={};PRO={};Gmean=None;MG=None
    if not a.no_first_order:   # G_D = mean_p grad(L3 margin of pair p) - mean_p grad(XP margin of pair p) at W0 (R0 readout approximation), three matrices
        MG=build_margin(L,rt32);gl=None;gx=None
        with rt32.mesh,jax.default_matmul_precision('highest'):
            for p in MG['pairs']:
                u=get3(MG['grad']('L3',p));gl=u if gl is None else tuple(x+y for x,y in zip(gl,u));del u
            for p in MG['xpairs']:
                u=get3(MG['grad']('XP',p));gx=u if gx is None else tuple(x+y for x,y in zip(gx,u));del u
            Gmean=tuple(x/len(MG['pairs'])-y/len(MG['xpairs']) for x,y in zip(gl,gx));del gl,gx
    for w in a.worlds:
        ids=stream_ids(L,w);mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True
        with rt32.mesh,jax.default_matmul_precision('highest'):
            seq32=rt32.seq(ids,mask=mask);pref_nat,pd=pref_native32(rt,rt32,seq32,ids);man[f'pref_dtype_{w}']=pd
            ks=list(range(NCH)) if a.chunks=='all' else (sorted({x[3]['k'] for x in SP.chunk_rows(L.T.w,w=='B',L.T.tok,rt.bos,rt.eos,'X')[0]}) if a.chunks=='c2' else [int(x) for x in a.chunks.split(',')])   # c2 = the 26 fact-intact chunks of the screen (NTP loss kept)
            if a.smoke:ks=ks[:a.smoke]
            nk={k:int(np.asarray(rt32.chunk(seq32,k).loss_masks).sum()) for k in ks};N=sum(nk.values());acc={cc:tuple(jnp.zeros(s,jnp.float32) for s in ((O['NL'],O['dm'],O['dff']),(O['NL'],O['dm'],O['dff']),(O['NL'],O['dff'],O['dm']))) for cc in cells}
            pro={cc:jnp.zeros(8,jnp.float32) for cc in cells};st=clone_state(rt32.state_suffix);t0=time.monotonic();chk={}
            for k in range(max(ks)+1):
                c=rt32.chunk(seq32,k);pref_c=pref_nat[k*CH:(k+1)*CH]
                if k in ks:
                    FA=factors(O,c,st,pref_c,keepA);pref_w=L.prefix_masked(m32,jnp.asarray(ids[k*CH:(k+1)*CH],dtype=jnp.int32),Kc0,Vc0,jnp.int32(k*CH),keepM);FM=factors(O,c,st,pref_w,keepM)
                    F={'A':FA,'M':FM};wk=nk[k]/N
                    for (P,Q) in cells:
                        acc[(P,Q)]=O['add'](acc[(P,Q)],O['outer'](F[P][0],F[P][1],F[Q][2],F[Q][3],F[Q][4]),wk)
                        if Gmean is not None:pro[(P,Q)]=pro[(P,Q)]+wk*O['prof'](Gmean,F[P][0],F[P][1],F[Q][2],F[Q][3],F[Q][4])
                    if a.verify_step or k==ks[0]:
                        eA=recon_err(O,m32,c,st,pref_c,keepA,*FA[:5]);eM=recon_err(O,m32,c,st,pref_w,keepM,*FM[:5])
                        chk[k]={'A_err':eA[0],'A_gn_ref_vs_3matrix':[eA[1],eA[2]],'M_err':eM[0],'M_gn_ref_vs_3matrix':[eM[1],eM[2]],'hidden_maxdiff_A_M':[FA[5],FM[5]]}
                        if max(eA[0]+eM[0])>1e-3:json.dump({'manifest':man,'check':chk},open(OUT/f'e29_swap_s{a.seed}_FAILED.json','w'),indent=1);print('RECONSTRUCTION CHECK FAILED',chk[k],flush=True);sys.exit(4)
                    if a.save_dir:
                        D_=Path(a.save_dir)/f's{a.seed}_{w}';D_.mkdir(parents=True,exist_ok=True)
                        for cd in ('A','M'):np.savez(D_/f'{cd}_k{k:02d}.npz',**dict(zip(('x_in','x_mid','d1','d3','d2'),[np.asarray(x) for x in F[cd][:5]])))
                    del F,FA,FM
                    if k%8==0:print(f'  {a.seed} {w} chunk {k} {time.monotonic()-t0:.0f}s',flush=True)
                st=L.read_state(m32,st,c,pref_c)
        G[w]=acc;PRO[w]={f'x{P}_d{Q}':np.asarray(v).tolist() for (P,Q),v in pro.items()};man[f'check_{w}']=chk
    # ---- realisation + readout ----
    def apply(Gs,s,coef):
        new=tuple((p-eta*coef*s*g.astype(p.dtype)).astype(p.dtype) for p,g in zip(w0p,Gs));return eqx.tree_at(get3,inner0,new)
    dots={}
    if Gmean is not None:
        with rt32.mesh,jax.default_matmul_precision('highest'):
            for fam,pl in (('L3',MG['pairs']),('XP',MG['xpairs'])):
                for p in pl:
                    u=get3(MG['grad'](fam,p))
                    for w in a.worlds:
                        for cc in cells:dots[(fam,w,cc,p)]=float(sum(jnp.vdot(x,y) for x,y in zip(u,G[w][cc])))
                    del u
    ev0=L.evaluate(rt.base);res={'cells':{},'profile_raw_bins_of_128':PRO,'profile_note':'sum over layers/matrices of d_t^T G_D x_t (G_D = grad of L3 margin mean minus XP margin mean, R0 readout), token-weighted over chunks, per in-chunk position bin (8 x 128); binding profile = (world A - world B)/2 with the sign of D_first_order = -(eta coef s)*that'};pairs_of=lambda f:[it['pair'] for it in L.fam[f]]
    def Mf(evA,evB,f):
        d={}
        for p,x in zip(pairs_of(f),(np.asarray(evA['_alt'][f])-np.asarray(evB['_alt'][f]))/2):d.setdefault(p,[]).append(x)
        ks=sorted(d,key=lambda s_:int(s_[1:]));return np.asarray([np.mean(d[k]) for k in ks])
    for (P,Q) in cells:
        for s in scales:
            ev={};info={}
            for w in a.worlds:
                Gs=G[w][(P,Q)];gn=float(jnp.sqrt(sum(jnp.sum(x*x) for x in Gs)));coef=min(1.0,clip/max(gn,1e-30));newp=apply(Gs,s,coef);mm=L.set_inner(rt.base,newp);ev[w]=L.evaluate(mm)
                info[w]={'gnorm':gn,'coef':coef,'canary_rise':max(ev[w]['_gen_nll']-ev0['_gen_nll'],ev[w]['_tail_nll']-ev0['_tail_nll'])}
            if len(a.worlds)==2:
                D=Mf(ev['A'],ev['B'],'L3')-Mf(ev['A'],ev['B'],'XP');res['cells'][f'x{P}_d{Q}_s{s:g}']={'D_alt_pairs':D.tolist(),'D_alt_mean':float(D.mean()),'info':info}
                if dots:
                    am=lambda fam,pl,p:{w:-eta*info[w]['coef']*s*dots[(fam,w,(P,Q),p)] for w in a.worlds}
                    b=np.asarray([(am('L3',None,p)['A']-am('L3',None,p)['B'])/2 for p in MG['pairs']]);bn=np.asarray([(am('XP',None,p)['A']-am('XP',None,p)['B'])/2 for p in MG['xpairs']])
                    res['cells'][f'x{P}_d{Q}_s{s:g}']['D_first_order_pairs']=(b-bn).tolist();res['cells'][f'x{P}_d{Q}_s{s:g}']['D_first_order_mean']=float((b-bn).mean())
                print(f'x<-{P} d<-{Q} s={s:g}: D_alt {D.mean():+.4f} canary rise {max(i["canary_rise"] for i in info.values()):+.3f} gn {[round(i["gnorm"],3) for i in info.values()]}',flush=True)
    rng=np.random.default_rng(29);lines=[f'# E29 factor exchange, seed {a.seed}, chunks {a.chunks}, pref dtype {[man.get(f"pref_dtype_{w}") for w in a.worlds]}\n','| scale | cell | D_alt [CI] | canary rise | first-order D (R0 readout, sign/size check only) |','|---|---|---|---|---|']
    def ci(v):
        v=np.asarray(v);b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(a.nboot)];return float(v.mean()),*np.percentile(b,[2.5,97.5]).tolist()
    if len(a.worlds)==2:
        for s in scales:
            for P,Q in cells:
                r=res['cells'][f'x{P}_d{Q}_s{s:g}'];t=ci(r['D_alt_pairs']);lines.append(f"| {s:g} | x<-{P} d<-{Q} | {t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}] | {max(i['canary_rise'] for i in r['info'].values()):+.3f} | {('%+.3f'%r['D_first_order_mean']) if 'D_first_order_mean' in r else '-'} |")
            D=lambda P,Q:np.asarray(res['cells'][f'x{P}_d{Q}_s{s:g}']['D_alt_pairs']);tot=D('M','M')-D('A','A');n=len(tot);bs=[]
            for _ in range(a.nboot):
                i=rng.integers(0,n,n);T=tot[i].mean();bs.append([(D('M','A')[i].mean()-D('A','A')[i].mean())/T,(D('A','M')[i].mean()-D('A','A')[i].mean())/T] if abs(T)>1e-9 else [np.nan,np.nan])
            T=tot.mean();sx=(D('M','A').mean()-D('A','A').mean())/T;sd=(D('A','M').mean()-D('A','A').mean())/T;bs=np.asarray(bs)
            lines.append(f"\nscale {s:g}: Delta_tot = D(M,M)-D(A,A) = {ci(tot)[0]:+.3f} [{ci(tot)[1]:+.3f},{ci(tot)[2]:+.3f}]; s_x {sx:.2f} [{np.nanpercentile(bs[:,0],2.5):.2f},{np.nanpercentile(bs[:,0],97.5):.2f}], s_delta {sd:.2f} [{np.nanpercentile(bs[:,1],2.5):.2f},{np.nanpercentile(bs[:,1],97.5):.2f}], s_int {1-sx-sd:.2f} (pair bootstrap, one seed; pool seeds in analysis)\n")
    if len(a.worlds)==2 and PRO:
        lines+=['\n## Position profile: binding profile (world A - world B)/2 of sum_t d_t^T G_D x_t per 128-position bin (raw, unscaled; first-order D = -(eta coef s) x this)\n','| cell | '+' | '.join(f'{i*128}-{i*128+127}' for i in range(8))+' | head(0-255)/tail(768-1023) |','|---|'+'---|'*9]
        for P,Q in cells:
            pa=np.asarray(PRO['A'][f'x{P}_d{Q}']);pb=np.asarray(PRO['B'][f'x{P}_d{Q}']);bp=(pa-pb)/2;lines.append(f'| x<-{P} d<-{Q} | '+' | '.join(f'{v:+.2e}' for v in bp)+f' | {bp[:2].sum()/bp[6:].sum() if abs(bp[6:].sum())>1e-30 else float("nan"):.2f} |')
    json.dump({'manifest':man,'result':res},open(OUT/f'e29_swap_s{a.seed}.json','w'),indent=1);(OUT/f'e29_swap_s{a.seed}.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
