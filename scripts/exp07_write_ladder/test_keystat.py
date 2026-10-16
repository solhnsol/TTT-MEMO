"""CPU tests for keystat.py (registered checks of the KEYSTAT/PoE screen v1).
1 the three mean-removal forms (key-only, delta-only, both) equal G - N xbar deltabar^T (= t_a), with some delta rows zero.
2 t_b removes exactly the top-r key directions: U_r^T t_b(G) = 0 and t_b is idempotent; r = din gives 0.
3 t_c solves (C + alpha I) X = G.
4 poe_bias causality: the bias at position t uses only stream tokens at positions <= lo + t (changing any later token leaves it
  unchanged; changing an earlier one changes it), and equals the explicit count formula."""
import os,sys
os.environ.setdefault('JAX_PLATFORMS','cpu');sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
import keystat as K
res=[]
def check(n,ok,d=''):print(('PASS' if ok else 'FAIL')+': '+n+(f' ({d})' if d else ''),flush=True);res.append(bool(ok))
r=np.random.default_rng(0);T,din,dout=50,12,7
x=jnp.asarray(r.normal(size=(T,din)));d=r.normal(size=(T,dout));d[:9]=0;d=jnp.asarray(d)
G,sd,xbar,N,M=K.key_stats(x,d);nz=np.any(np.asarray(d)!=0,1);xn=np.asarray(x)[nz];dn=np.asarray(d)[nz]
xb=xn.mean(0);db=dn.mean(0)
f1=(xn-xb).T@dn;f2=xn.T@(dn-db);f3=(xn-xb).T@(dn-db);fa=np.asarray(K.t_a(G,sd,xbar));fN=np.asarray(G)-len(xn)*np.outer(xb,db)
check('1 key-only = delta-only = both = G - N xbar deltabar^T = t_a',max(np.abs(f1-f2).max(),np.abs(f1-f3).max(),np.abs(f1-fa).max(),np.abs(f1-fN).max())<1e-10,f'N={int(N)} of {T}')
for rr in (1,4):
    tb=K.t_b(G,M,rr);w,U=np.linalg.eigh(np.asarray(M));Ur=U[:,-rr:]
    check(f'2 t_b r={rr}: top-{rr} key directions removed, idempotent',np.abs(Ur.T@np.asarray(tb)).max()<1e-9 and np.abs(np.asarray(K.t_b(tb,M,rr))-np.asarray(tb)).max()<1e-9)
check('2b t_b r = din gives 0',np.abs(np.asarray(K.t_b(G,M,din))).max()<1e-8)
C=jnp.asarray(np.cov(r.normal(size=(200,din)).T));al=float(jnp.trace(C)/din)
X=K.t_c(G,C,al);check('3 t_c solves (C + alpha I) X = G',np.abs(np.asarray((C+al*jnp.eye(din))@X-G)).max()<1e-9)
V=20;CH=8;lo=16;ids=r.integers(0,V,size=40);b,cnt=K.poe_bias(ids,lo,CH,V,1.0,0.1)
ok=True
for t in range(CH):
    c=np.bincount(ids[:lo+t+1],minlength=V);exp=np.log((c+0.1)/(lo+t+1+0.1*V))
    ok&=np.abs(np.asarray(b[t])-exp).max()<1e-6
ids2=ids.copy();ids2[lo+5]=(ids2[lo+5]+1)%V;b2,_=K.poe_bias(ids2,lo,CH,V,1.0,0.1)
same_before=np.abs(np.asarray(b2[:5])-np.asarray(b[:5])).max()<1e-12;diff_at=np.abs(np.asarray(b2[5])-np.asarray(b[5])).max()>1e-6
check('4 poe_bias = explicit causal counts; a token change at lo+5 leaves positions < 5 unchanged and changes position 5',ok and same_before and diff_at)
print(f'{sum(res)}/{len(res)} passed');sys.exit(0 if all(res) else 1)
