"""E57 slot-momentum write direction (job 510; GPU, one news cycle, no mask; post-hoc, exploratory; new file, wraps e57_adamdecomp_run like e57_adamb1_run). Same stream, code path, step size (every chunk update has the norm 0.5 x the NAT update norm of the chunk), momentum-only direction (no sqrt(v)), beta1 = .9 and clipped gradient as arm R2B09 of job 470, but the momentum buffer is split into SLOTS: chunk k is routed to the slot r[k] of a routing table, only that slot is updated, mu_s = B1 * mu_s + (1 - B1) * g, and the direction is d = mu_s / (1 - B1^n_s) with n_s = the number of updates of THAT slot (own bias correction); the weights are updated with that direction scaled to 0.5 x the NAT norm of the chunk. Slots are allocated on first use (fp32, all inner parameters, 1.4 GB each).
  S_ONE     one slot (all chunks -> slot 0): must reproduce R2B09 of job 470 exactly (end-to-end check)
  S_ORACLE  routing table 'oracle_event' of e57_slot_routing_tables.json (dominant event by token share, 10 slots)
  S_KMEANS  routing table 'kmeans_C' (offline, non-causal k-means on the C key sketch of job 490; exploratory)
  S_RANDOM  routing table 'random'
  python3 e57_slotmom_run.py --routing TABLES.json <the arguments of e57_adamdecomp_run.py with --arms S_ONE,S_ORACLE,S_KMEANS,S_RANDOM>   |   --selftest"""
import sys,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e55_swap as E
import e57_adamdecomp_run as AD
SM={'routing':None}
ARMS={'S_ONE':None,'S_ORACLE':'oracle_event','S_KMEANS':'kmeans_C','S_RANDOM':'random'}
for _k in ARMS:
    E.ADAM[_k]=('NAT',0.5,False)
    if _k not in E.ARMS:E.ARMS.append(_k)
    AD.NEW[_k]=('NAT',0.5,False)
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def table_of(arm,tables,nch=280):
    key=ARMS[arm]
    if key is None:t=[0]*nch
    else:t=[int(x) for x in tables[key]]
    assert len(t)==nch and min(t)>=0 and max(t)<10,(arm,len(t));return t
def slot_fn(jax,jnp,eqx,clip,table):
    f32=jnp.float32;B1=.9
    def prep(g):
        g=jax.tree.map(lambda x:x.astype(f32),g);gn=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g)));coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));return jax.tree.map(lambda x:x*coef,g),gn,coef
    @eqx.filter_jit
    def upd(g,mu_s,cnt):
        g,gn,coef=prep(g);mu_s=jax.tree.map(lambda a,x:B1*a+(1-B1)*x,mu_s,g);c1=1-B1**cnt;d=jax.tree.map(lambda a:a/c1,mu_s);return d,mu_s,gn,coef
    def fn(g,mu,nu,t):
        k=int(t)
        if k==0 or not isinstance(mu,dict):mu={'free':mu,'slots':{},'cnt':{},'zref':None}
        s=table[k]
        if s not in mu['slots']:
            if mu['free'] is not None:mu['slots'][s]=mu['free'];mu['free']=None
            else:mu['slots'][s]=jax.tree.map(jnp.zeros_like,next(iter(mu['slots'].values())))
            mu['cnt'][s]=0
        mu['cnt'][s]+=1
        d,mu_s,gn,coef=upd(g,mu['slots'][s],jnp.asarray(float(mu['cnt'][s]),f32));mu['slots'][s]=mu_s
        return d,mu,nu,gn,coef
    return fn
def make_dirs(jax,jnp,eqx,clip):
    tables=json.load(open(SM['routing']));return {a:slot_fn(jax,jnp,eqx,clip,table_of(a,tables)) for a in ARMS}
def selftest():
    import jax,jax.numpy as jnp,equinox as eqx
    AD.selftest()
    import e57_adamb1_run as B1M
    rng=np.random.default_rng(3);mk=lambda:{'a':jnp.asarray(rng.standard_normal((4,5)).astype(np.float32)*3),'b':jnp.asarray(rng.standard_normal(7).astype(np.float32))}
    gs=[mk() for _ in range(12)];z=jax.tree.map(jnp.zeros_like,gs[0])
    ref=B1M.make_dirs(jax,jnp,eqx,1.0)['R2B09'];one=slot_fn(jax,jnp,eqx,1.0,[0]*12)
    mu_r=z;mu_o=jax.tree.map(jnp.zeros_like,gs[0])
    for t,g in enumerate(gs):
        dr,mu_r,_,_,_=ref(g,mu_r,z,jnp.float32(t));do,mu_o,_,_,_=one(g,mu_o,z,jnp.float32(t))
        for k in dr:assert np.array_equal(np.asarray(dr[k]),np.asarray(do[k])),('single slot != R2B09',t,k)
    tab=[0,1,0,1,1,0,2,2,0,1,2,0];f=slot_fn(jax,jnp,eqx,1.0,tab);mu=jax.tree.map(jnp.zeros_like,gs[0]);ref_mu={};cnt={}
    for t,g in enumerate(gs):
        d,mu,_,_,_=f(g,mu,z,jnp.float32(t));n=sum(float(np.sum(np.asarray(v)**2)) for v in g.values())**.5;c=min(1.0,1.0/n);s=tab[t];cnt[s]=cnt.get(s,0)+1
        gc={k:np.asarray(v)*c for k,v in g.items()};ref_mu[s]={k:(.9*ref_mu[s][k]+.1*gc[k]) if s in ref_mu else .1*gc[k] for k in gc}
        for k in gc:assert np.allclose(np.asarray(d[k]),ref_mu[s][k]/(1-.9**cnt[s]),rtol=1e-4,atol=1e-6),(t,k)
    print('e57_slotmom_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    SM['routing']=_pop('--routing');assert SM['routing']
    AD.make_dirs=make_dirs;AD.main()
if __name__=='__main__':main()
