"""Key-value probes for the fast-weight (prime SwiGLU FFN) layers of TTT-E2E (JAX; new file; thread-server 11/23; job 1344k spec v2, Backward-Lens style decomposition of the MLP gradient: dL/dW = sum_p x_p (x) delta_p). Nothing in e2e/ is edited: a subclass of ttt.model.transformer.SwiGLUMLP replaces the stacked prime FFN of the suffix blocks of a model copy (eqx.tree_at); the stacked leaves are scanned over the 8 suffix layers by the model, so the layer id `lid` (a stacked int leaf) arrives inside the scan body as a scalar.
  ProbeFwd : forward captures (layer input x [T,H] after ffn_prime_norm, x2 = silu(z1)*z3 [T,I]) through jax.debug.callback -> CAP['fwd'][lid] = (x, x2)
  ProbeBwd : identity taps (custom_vjp) on z1, z3, z2 whose backward rule sends the incoming cotangents (delta_z1, delta_z3 [T,I]; delta_z2 [T,H]) through jax.debug.callback -> CAP['bwd'][(name, lid)]
The two modes are two DIFFERENT classes (not a global flag) so that jax's trace caches (remat / scan are keyed by the callable) cannot mix them. Under jax.vmap with batch 1 the callbacks fire once per layer. With 1 chunk per call the callbacks are ordered by lid, not by time; call jax.effects_barrier() before reading CAP.
  python3 e59_kvprobe.py --selftest   (toy scan / remat / vmap / grad check, CPU)"""
import functools,dataclasses,sys
import numpy as np,jax,jax.numpy as jnp,equinox as eqx
CAP={'fwd':{},'bwd':{}}
def _cb_fwd(lid,x,x2):CAP['fwd'][int(lid)]=(np.asarray(x),np.asarray(x2))
def _cb_bwd(name,lid,d):CAP['bwd'][(name,int(lid))]=np.asarray(d)
@functools.partial(jax.custom_vjp,nondiff_argnums=(0,))
def tap(name,z,lid):return z
def _tap_f(name,z,lid):return z,lid
def _tap_b(name,lid,g):
    jax.debug.callback(functools.partial(_cb_bwd,name),lid,g);return g,None
tap.defvjp(_tap_f,_tap_b)
def _classes():
    from ttt.model.transformer import SwiGLUMLP
    class ProbeFwd(SwiGLUMLP):
        lid: jax.Array
        def __call__(self,x):
            z1=self.w1(x);z3=self.w3(x);x2=jax.nn.silu(z1)*z3
            jax.debug.callback(_cb_fwd,self.lid,x,x2)
            return self.dropout(self.w2(x2))
    class ProbeBwd(SwiGLUMLP):
        lid: jax.Array
        def __call__(self,x):
            z1=tap('z1',self.w1(x),self.lid);z3=tap('z3',self.w3(x),self.lid)
            z2=tap('z2',self.w2(jax.nn.silu(z1)*z3),self.lid)
            return self.dropout(z2)
    return ProbeFwd,ProbeBwd
_C={}
def classes():
    if not _C:_C['fwd'],_C['bwd']=_classes()
    return _C['fwd'],_C['bwd']
WHERE=lambda mm:mm.language_model.model.h.suffix_blocks.feed_forward_prime
def install(m,mode):
    """model copy whose stacked prime FFN is the probe subclass (mode 'fwd' | 'bwd'); same weights, same numerics"""
    old=WHERE(m);cls=classes()[0 if mode=='fwd' else 1];nl=old.w1.weight.shape[0];new=object.__new__(cls)
    for f in dataclasses.fields(type(old)):object.__setattr__(new,f.name,getattr(old,f.name))
    object.__setattr__(new,'lid',jnp.arange(nl,dtype=jnp.int32))
    return eqx.tree_at(WHERE,m,new)
def selftest():
    """toy check of the mechanism (scan over stacked layers + jax.checkpoint + custom_vjp taps + callbacks, both modes in ONE process without clearing caches): sum_p x (x) delta == autodiff gradient"""
    L,T,H,I=3,5,4,6;rng=np.random.default_rng(0)
    W1=jnp.asarray(rng.normal(size=(L,H,I)),jnp.float32);W3=jnp.asarray(rng.normal(size=(L,H,I)),jnp.float32);W2=jnp.asarray(rng.normal(size=(L,I,H)),jnp.float32);LID=jnp.arange(L)
    def mlp_fwd(h,w1,w2,w3,lid):
        x=h/jnp.sqrt((h**2).mean(-1,keepdims=True)+1e-6);z1=x@w1;z3=x@w3;x2=jax.nn.silu(z1)*z3;jax.debug.callback(_cb_fwd,lid,x,x2);return x2@w2
    def mlp_bwd(h,w1,w2,w3,lid):
        x=h/jnp.sqrt((h**2).mean(-1,keepdims=True)+1e-6);z1=tap('z1',x@w1,lid);z3=tap('z3',x@w3,lid);return tap('z2',(jax.nn.silu(z1)*z3)@w2,lid)
    def mk(mlp):
        def loss(W1,W2,W3,h0,wt):
            def layer(h,xs):
                w1,w2,w3,lid=xs;return h+jax.checkpoint(mlp)(h,w1,w2,w3,lid),None
            h,_=jax.lax.scan(layer,h0,(W1,W2,W3,LID));return jnp.sum(wt*(h**2).sum(-1))
        return loss
    def plain(h,w1,w2,w3,lid):
        x=h/jnp.sqrt((h**2).mean(-1,keepdims=True)+1e-6);return (jax.nn.silu(x@w1)*(x@w3))@w2
    h0=jnp.asarray(rng.normal(size=(T,H)),jnp.float32);wt=jnp.asarray(rng.random(T),jnp.float32)
    jax.jit(mk(mlp_fwd))(W1,W2,W3,h0,wt);jax.effects_barrier()
    jax.jit(jax.grad(mk(mlp_bwd),argnums=(0,1,2)))(W1,W2,W3,h0,wt);jax.effects_barrier()
    gr=jax.jit(jax.grad(mk(plain),argnums=(0,1,2)))(W1,W2,W3,h0,wt);worst=0.0
    for l in range(L):
        x,x2=CAP['fwd'][l];d1,d2,d3=CAP['bwd'][('z1',l)],CAP['bwd'][('z2',l)],CAP['bwd'][('z3',l)]
        for rec,ref in ((x2.T@d2,gr[1][l]),(x.T@d1,gr[0][l]),(x.T@d3,gr[2][l])):worst=max(worst,float(np.abs(rec-ref).max()/np.abs(ref).max()))
    assert worst<1e-5,worst;print(f'e59_kvprobe selftest ok: worst relative error of sum_p x (x) delta vs autodiff over {L} layers x (w1, w2, w3) = {worst:.2e}')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
