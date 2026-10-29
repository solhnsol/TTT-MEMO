"""E57 momentum-only write direction with other beta1 (job 470; GPU, one news cycle, no mask; post-hoc, exploratory; new file, wraps e57_adamdecomp_run unchanged). Same stream, same code path, same step size as job 430 (every chunk update has the norm 0.5 x the NAT update norm of the chunk, norm_targets of E57_main_NAT.arms.json) and the direction of 430 R2MOM, d = m / c1 with m = EMA(g, beta1), c1 = 1 - beta1^t, no division by sqrt(v), g = the clipped gradient, only beta1 differs:
  R2B05   beta1 = 0.5
  R2B09   beta1 = 0.9  (= 430 R2MOM, rerun as the in-job control)
  R2B099  beta1 = 0.99
  NATADAMH  Adam (m / c1) / (sqrt(v / c2) + 1e-8), beta1 .9, beta2 .999 (control, as in 430, e55_core adam_dir)
States at the chunks given by --read-at, canary there, OUT reads of the 42 questions of bank v1 at W0 and every state (e57_qcurve_run reader).
  python3 e57_adamb1_run.py <the arguments of e57_adamdecomp_run.py with --arms NATADAMH,R2B05,R2B09,R2B099>   |   --selftest"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e55_swap as E
import e57_adamdecomp_run as AD
B1S={'R2B05':0.5,'R2B09':0.9,'R2B099':0.99}
for _k in B1S:
    E.ADAM[_k]=('NAT',0.5,False)
    if _k not in E.ARMS:E.ARMS.append(_k)
    AD.NEW[_k]=('NAT',0.5,False)
def make_dirs(jax,jnp,eqx,clip):
    f32=jnp.float32
    def prep(g):
        g=jax.tree.map(lambda x:x.astype(f32),g);gn=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g)));coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));return jax.tree.map(lambda x:x*coef,g),gn,coef
    def mk(B1):
        @eqx.filter_jit
        def mom(g,mu,nu,t):
            g,gn,coef=prep(g);mu=jax.tree.map(lambda a,x:B1*a+(1-B1)*x,mu,g);t1=t+1.0;c1=1-B1**t1;d=jax.tree.map(lambda a:a/c1,mu);return d,mu,nu,gn,coef
        return mom
    return {k:mk(b) for k,b in B1S.items()}
def selftest():
    import jax,jax.numpy as jnp,equinox as eqx
    AD.selftest()
    D=make_dirs(jax,jnp,eqx,1.0);rng=np.random.default_rng(1);g={'a':jnp.asarray(rng.standard_normal((4,5)).astype(np.float32)*3),'b':jnp.asarray(rng.standard_normal(7).astype(np.float32))}
    g2={k:v*0.5+1 for k,v in g.items()};z=jax.tree.map(jnp.zeros_like,g)
    def clipped(x):
        n=np.sqrt(sum((np.asarray(v)**2).sum() for v in x.values()));c=min(1.0,1.0/n);return {k:np.asarray(v)*c for k,v in x.items()}
    c1_,c2_=clipped(g),clipped(g2)
    for arm,b in B1S.items():
        d,mu1,_,_,_=D[arm](g,z,z,jnp.float32(0.))
        for k in c1_:assert np.allclose(np.asarray(d[k]),c1_[k],rtol=1e-4,atol=1e-6)
        d2,_,_,_,_=D[arm](g2,mu1,z,jnp.float32(1.))
        for k in c1_:assert np.allclose(np.asarray(d2[k]),(b*(1-b)*c1_[k]+(1-b)*c2_[k])/(1-b**2),rtol=1e-4,atol=1e-6),(arm,k)
    assert all(a in E.ADAM and a in E.ARMS for a in B1S)
    print('e57_adamb1_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    AD.make_dirs=make_dirs;AD.main()
if __name__=='__main__':main()
