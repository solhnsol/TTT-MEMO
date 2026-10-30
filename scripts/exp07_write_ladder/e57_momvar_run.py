"""E57 momentum-only write direction variants (job 520; GPU, one news cycle, no mask; post-hoc, exploratory; new file, wraps e57_adamdecomp_run like e57_adamb1_run / e57_slotmom_run). Base = the R2 arm of job 470 (R2B09): clipped gradient g, mu = .9 mu + .1 g, direction d0 = mu / (1 - .9^t), no sqrt(v), every chunk update rescaled to 0.5 x the NAT update norm of the chunk (same step size for every arm, applied by e55_core to the whole direction), all inner parameters. The variants change the DIRECTION d only; the transforms act on the w2 leaf (feed_forward_prime.w2.weight, shape [8 layers, 5632 (in = the key side), 2560], y = x @ W, convention of keystat.py) except CAUTIOUS and MULTI (all inner parameters):
  W_WHITEN_A / W_WHITEN_B  d_w2[l] = (C_l + lambda_l I)^-1 d0_w2[l] on the 5632-dim key side, lambda_l = alpha tr(C_l) / 5632, alpha = 0.1 (A) / 1.0 (B); C_l = bias-corrected EMA (decay .99, over the write chunks 0..k INCLUDING the current chunk) of the per-chunk second moment of the input of the layer's w2 (silu(w1 x) * w3 x averaged over the 1024 tokens as x^T x / 1024, from an extra forward at the weights before the write); the whitened matrix is rescaled to the Frobenius norm of d0_w2[l] (only the direction changes)
  W_CAUTIOUS               d = d0 where sign(d0) == sign(g_t) (g_t = the clipped gradient of the chunk), 0 elsewhere (then the common rescale to the step size)
  W_MULTI                  d = .8 mu_fast / (1 - .9^t) + .2 mu_slow / (1 - .99^t)  (mu_fast: beta .9 = R2B09, mu_slow: beta .99)
  W_MUON                   d_w2[l] = Newton-Schulz orthogonalisation (5 iterations, quintic coefficients 3.4445 / -4.7750 / 2.0315, fp32) of d0_w2[l], rescaled to the Frobenius norm of d0_w2[l]; no tuning
Self-test limits: alpha -> inf (whiten), mask off (cautious), weights 1 / 0 (multi) reproduce the R2B09 direction.
  python3 e57_momvar_run.py <the arguments of e57_adamdecomp_run.py with --arms W_WHITEN_A,W_WHITEN_B,W_CAUTIOUS,W_MULTI,W_MUON>   |   --selftest"""
import sys,json,time,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e55_swap as E
import e57_adamdecomp_run as AD
KINDS={'W_WHITEN_A':('whiten',dict(alpha=0.1)),'W_WHITEN_B':('whiten',dict(alpha=1.0)),'W_CAUTIOUS':('cautious',{}),'W_MULTI':('multi',dict(w_fast=0.8,w_slow=0.2)),'W_MUON':('muon',{})}
WHITEN_ARMS=tuple(a for a,(k,_) in KINDS.items() if k=='whiten')
for _k in KINDS:
    E.ADAM[_k]=('NAT',0.5,False)
    if _k not in E.ARMS:E.ARMS.append(_k)
    AD.NEW[_k]=('NAT',0.5,False)
BETA_C=0.99;DIN=5632
KS={'active':False,'C':None,'n':0}      # host EMA of the key second moments [8, 5632, 5632] (float32)
def ns_orth(M,steps=5):
    """Newton-Schulz orthogonalisation of one matrix (jnp, fp32): quintic iteration of Muon"""
    import jax.numpy as jnp
    a,b,c=3.4445,-4.7750,2.0315;X=M.astype(jnp.float32);X=X/(jnp.linalg.norm(X)+1e-7);tr=X.shape[0]>X.shape[1]
    if tr:X=X.T
    for _ in range(steps):
        A=X@X.T;B=b*A+c*(A@A);X=a*X+B@X
    return X.T if tr else X
def make_fn(jax,jnp,eqx,clip,kind,params):
    f32=jnp.float32;B1=.9;B2=.99
    def prep(g):
        g=jax.tree.map(lambda x:x.astype(f32),g);gn=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g)));coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));return jax.tree.map(lambda x:x*coef,g),gn,coef
    isw2=lambda p:'w2' in jax.tree_util.keystr(p)
    @eqx.filter_jit
    def base(g,mu,nu,t):
        g,gn,coef=prep(g);mu=jax.tree.map(lambda a,x:B1*a+(1-B1)*x,mu,g);t1=t+1.0;d0=jax.tree.map(lambda a:a/(1-B1**t1),mu);return d0,g,mu,gn,coef
    @eqx.filter_jit
    def multi(g,mu,nu,t):
        g,gn,coef=prep(g);mu=jax.tree.map(lambda a,x:B1*a+(1-B1)*x,mu,g);nu=jax.tree.map(lambda a,x:B2*a+(1-B2)*x,nu,g);t1=t+1.0
        d=jax.tree.map(lambda a,b:params['w_fast']*(a/(1-B1**t1))+params['w_slow']*(b/(1-B2**t1)),mu,nu);return d,mu,nu,gn,coef
    @eqx.filter_jit
    def cautious(d0,g):return jax.tree.map(lambda d,x:jnp.where(jnp.sign(d)==jnp.sign(x),d,0.0),d0,g)
    @eqx.filter_jit
    def muon(d0):
        def f(x):
            o=jax.vmap(ns_orth)(x);n0=jnp.linalg.norm(x.reshape(x.shape[0],-1),axis=1);n1=jnp.linalg.norm(o.reshape(o.shape[0],-1),axis=1);return o*(n0/jnp.maximum(n1,1e-30))[:,None,None]
        return jax.tree_util.tree_map_with_path(lambda p,x:f(x) if isw2(p) else x,d0)
    @eqx.filter_jit
    def whiten(d0,Cs,lam):
        def f(x):
            w=jax.vmap(lambda Cl,l,M:jnp.linalg.solve(Cl+l*jnp.eye(Cl.shape[0],dtype=Cl.dtype),M))(Cs,lam,x.astype(f32))
            n0=jnp.linalg.norm(x.reshape(x.shape[0],-1),axis=1);n1=jnp.linalg.norm(w.reshape(w.shape[0],-1),axis=1);return w*(n0/jnp.maximum(n1,1e-30))[:,None,None]
        return jax.tree_util.tree_map_with_path(lambda p,x:f(x) if isw2(p) else x,d0)
    if kind=='multi':return lambda g,mu,nu,t:multi(g,mu,nu,t)
    def fn(g,mu,nu,t):
        d0,gc,mu,gn,coef=base(g,mu,nu,t)
        if kind=='cautious':d=cautious(d0,gc)
        elif kind=='muon':d=muon(d0)
        elif kind=='whiten':
            Ch=KS['C']/(1-BETA_C**KS['n']);tr=np.trace(Ch,axis1=1,axis2=2);lam=params['alpha']*tr/DIN
            d=whiten(d0,jnp.asarray(Ch),jnp.asarray(lam.astype(np.float32)))
        else:raise ValueError(kind)
        return d,mu,nu,gn,coef
    return fn
def make_dirs(jax,jnp,eqx,clip):
    return {a:make_fn(jax,jnp,eqx,clip,k,p) for a,(k,p) in KINDS.items()}
def ema_update(row_list):
    """host EMA of the per-layer key second moments; row_list = the 8 arrays [5632, 5632] of the current chunk"""
    new=np.stack([np.asarray(r,np.float32) for r in row_list])
    KS['C']=new*(1-BETA_C) if KS['C'] is None else BETA_C*KS['C']+(1-BETA_C)*new;KS['n']+=1
def momvar_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    import jax,jax.numpy as jnp,equinox as eqx
    from ttt.model.transformer import SwiGLUMLP
    f32=jnp.float32;cap={'on':False,'rows':[]};orig_call=SwiGLUMLP.__call__
    def _cb(v):cap['rows'].append(np.asarray(v))
    def call(self,x):
        if not cap['on']:return orig_call(self,x)
        x2=jax.nn.silu(self.w1(x))*self.w3(x);x2f=x2.astype(f32).reshape(-1,x2.shape[-1]);jax.debug.callback(_cb,(x2f.T@x2f)/x2f.shape[0],ordered=True)
        return self.dropout(self.w2(x2))
    SwiGLUMLP.__call__=call
    S3SUF=s3.S3SUF
    @eqx.filter_jit
    def fwd_keys(m,st,c,pref,kmask):
        S3SUF['kmask']=kmask
        try:return m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c).logits.astype(f32).sum()
        finally:S3SUF.clear()
    L0=C.Ladder55;orig_init=L0.__init__;orig_run=L0.run_stream
    def init(self,*aa,**kk):
        orig_init(self,*aa,**kk);og=self.grad_mask
        def gm(m,st_suf,c,pref_,kmask):
            if KS['active']:
                cap['rows']=[];cap['on']=True
                try:fwd_keys(m,st_suf,c,pref_,kmask);jax.effects_barrier()
                finally:cap['on']=False
                assert len(cap['rows'])==16,len(cap['rows']);ema_update(cap['rows'][0::2]);cap['rows']=[]       # callbacks come in the order prime_0, ff_0, prime_1, ... (checked in jobs 490 / 500)
            return og(m,st_suf,c,pref_,kmask)
        self.grad_mask=gm
    def run(self,arm,*aa,**kk):
        KS['active']=arm in WHITEN_ARMS;KS['C']=None;KS['n']=0;return orig_run(self,arm,*aa,**kk)
    L0.__init__=init;L0.run_stream=run
    try:AD.decomp_arms_orig(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM)
    finally:L0.__init__=orig_init;L0.run_stream=orig_run;SwiGLUMLP.__call__=orig_call
def selftest():
    import jax,jax.numpy as jnp,equinox as eqx
    AD.selftest()
    import e57_adamb1_run as B1M
    rng=np.random.default_rng(5)
    def mk():return {'w1':{'weight':jnp.asarray(rng.standard_normal((2,6,8)).astype(np.float32))},'w2':{'weight':jnp.asarray(rng.standard_normal((2,8,6)).astype(np.float32))}}
    gs=[mk() for _ in range(8)];z=jax.tree.map(jnp.zeros_like,gs[0]);ref=B1M.make_dirs(jax,jnp,eqx,1e9)['R2B09']
    def run(fn):
        mu=jax.tree.map(jnp.zeros_like,gs[0]);nu=jax.tree.map(jnp.zeros_like,gs[0]);out=[]
        for t,g in enumerate(gs):
            d,mu,nu,_,_=fn(g,mu,nu,jnp.float32(t));out.append(d)
        return out
    R=run(ref)
    leaves=lambda d:[np.asarray(x) for x in jax.tree.leaves(d)]
    # multi with weights 1 / 0 == R2B09 exactly
    mf=make_fn(jax,jnp,eqx,1e9,'multi',dict(w_fast=1.0,w_slow=0.0));M=run(mf)
    for r,o in zip(R,M):
        for x,y in zip(leaves(r),leaves(o)):assert np.array_equal(x,y),'multi(1,0) != R2B09'
    # cautious: mask off == identity (the jitted function with all-true condition), mask on == numpy rule
    cf=make_fn(jax,jnp,eqx,1e9,'cautious',{});mu=jax.tree.map(jnp.zeros_like,gs[0]);d,_,_,_,_=cf(gs[0],mu,mu,jnp.float32(0.))
    for x,y,g in zip(leaves(R[0]),leaves(d),leaves(gs[0])):
        assert np.array_equal(y,np.where(np.sign(x)==np.sign(g),x,0.0))
    # first step: d0 = g so the cautious mask keeps everything
    for x,y in zip(leaves(R[0]),leaves(d)):assert np.array_equal(x,y)
    # whiten: alpha -> inf reproduces the direction (norm-preserved, identity metric), alpha finite changes it
    KS['C']=None;KS['n']=0
    X=rng.standard_normal((6,8,8)).astype(np.float32)   # not used
    Cs=np.stack([np.eye(8,dtype=np.float32)*(1+i) for i in range(2)]);KS['C']=Cs*(1-BETA_C);KS['n']=1
    import types
    DINsave=globals()['DIN'];globals()['DIN']=8
    try:
        wf=make_fn(jax,jnp,eqx,1e9,'whiten',dict(alpha=1e8));mu=jax.tree.map(jnp.zeros_like,gs[0]);d,_,_,_,_=wf(gs[0],mu,mu,jnp.float32(0.))
        for x,y in zip(leaves(R[0]),leaves(d)):assert np.allclose(x,y,rtol=1e-4,atol=1e-5),'whiten(alpha inf) != R2B09'
        KS['C']=np.stack([np.diag(np.arange(1,9,dtype=np.float32)) for _ in range(2)])*(1-BETA_C);wf2=make_fn(jax,jnp,eqx,1e9,'whiten',dict(alpha=0.01));d2,_,_,_,_=wf2(gs[0],mu,mu,jnp.float32(0.))
        w2=np.asarray(d2['w2']['weight']);w2r=np.asarray(R[0]['w2']['weight'])
        assert not np.allclose(w2,w2r),"whiten no change";assert np.allclose(np.linalg.norm(w2.reshape(2,-1),axis=1),np.linalg.norm(w2r.reshape(2,-1),axis=1),rtol=1e-4),("norm",np.linalg.norm(w2.reshape(2,-1),axis=1),np.linalg.norm(w2r.reshape(2,-1),axis=1));assert np.array_equal(np.asarray(d2["w1"]["weight"]),np.asarray(R[0]["w1"]["weight"])),"w1 changed"
    finally:globals()['DIN']=DINsave
    # muon: norm preserved, singular values of the result flattened towards 1 (relative spread smaller than the input's)
    mnf=make_fn(jax,jnp,eqx,1e9,'muon',{});mu=jax.tree.map(jnp.zeros_like,gs[0]);d,_,_,_,_=mnf(gs[0],mu,mu,jnp.float32(0.));o=np.asarray(d['w2']['weight']);r0=np.asarray(R[0]['w2']['weight'])
    assert np.allclose(np.linalg.norm(o.reshape(2,-1),axis=1),np.linalg.norm(r0.reshape(2,-1),axis=1),rtol=1e-4)
    sv=lambda M:np.linalg.svd(M[0],compute_uv=False);assert sv(o)[-1]/sv(o)[0]>sv(r0)[-1]/sv(r0)[0]
    # ema_update
    KS['C']=None;KS['n']=0;ema_update([np.ones((3,3),np.float32)]);ema_update([2*np.ones((3,3),np.float32)]);assert abs(KS['C'][0,0,0]-(BETA_C*(1-BETA_C)+(1-BETA_C)*2))<1e-6 and KS['n']==2
    print('e57_momvar_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    AD.make_dirs=make_dirs;AD.decomp_arms_orig=AD.decomp_arms;AD.decomp_arms=momvar_arms;AD.main()
if __name__=='__main__':main()
