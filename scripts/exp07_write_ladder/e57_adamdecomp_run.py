"""E57 decomposition of the Adam write step (job 430; GPU, one news cycle, no mask; post-hoc, exploratory; new file). Arms written for one cycle (280 chunks) with the same stream, the same code path and the same step size as NATADAMH (every chunk update has the norm 0.5 x the NAT (SGD) update norm of that chunk, norm_targets of E57_main_NAT.arms.json; gradient clip and loss as in e55_core), differing only in the DIRECTION d of the step (e55_core: newp = p - adam_scale * d, adam_scale = 0.5 x norm_target / ||d||):
  NATADAMH  Adam: m = EMA(g, .9), v = EMA(g^2, .999), d = (m / c1) / (sqrt(v / c2) + 1e-8)  (control, rerun)
  R1RMS     RMSprop: beta1 = 0:  d = g / (sqrt(v / c2) + 1e-8)
  R2MOM     momentum only:  d = m / c1  (m = EMA(g, .9), no division by sqrt(v))
  R3SGN     stateless sign:  d = sign(g)  (every chunk)
  R4VFIX    frozen second moment: v is updated only during the first 8 chunks and then fixed, m kept (e55_core adam_dir_vfix, VFIX_T = 8)
g = the clipped gradient of the chunk. States are saved at the chunks given by --read-at (default 99,179,279), the canary is read there, the 42 questions of bank v1 are read OUT (neutral SQuAD tail) at W0 and from every saved state (the reader of e57_qcurve_run: output E57_qcurve_scores.json, E57_main_<ARM>.arms.json).
  python3 e57_adamdecomp_run.py --bundle B --out O --cycles 1 --read-at 99,179,279 --canary-npos 1 --arms NATADAMH,R1RMS,R2MOM,R3SGN,R4VFIX --ntg-from E57_main_NAT.arms.json --stage arms|reads   |   --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
CH=1024
AD={'ntg':None}
NEW={'R1RMS':('NAT',0.5,False),'R2MOM':('NAT',0.5,False),'R3SGN':('NAT',0.5,False),'R4VFIX':('NAT',0.5,True)}
import e55_swap as E
for _k,_v in NEW.items():
    E.ADAM[_k]=_v
    if _k not in E.ARMS:E.ARMS.append(_k)
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def make_dirs(jax,jnp,eqx,clip):
    """-> {arm: fn(g, mu, nu, t) -> (d, mu, nu, gn, coef)} for R1RMS / R2MOM / R3SGN (the jitted functions are called like e55_core.adam_dir)"""
    f32=jnp.float32;B1=.9;B2=.999;EPS=1e-8
    def prep(g):
        g=jax.tree.map(lambda x:x.astype(f32),g);gn=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g)));coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));return jax.tree.map(lambda x:x*coef,g),gn,coef
    @eqx.filter_jit
    def rms(g,mu,nu,t):
        g,gn,coef=prep(g);nu=jax.tree.map(lambda a,x:B2*a+(1-B2)*x*x,nu,g);t1=t+1.0;c2=1-B2**t1;d=jax.tree.map(lambda x,b:x/(jnp.sqrt(b/c2)+EPS),g,nu);return d,mu,nu,gn,coef
    @eqx.filter_jit
    def mom(g,mu,nu,t):
        g,gn,coef=prep(g);mu=jax.tree.map(lambda a,x:B1*a+(1-B1)*x,mu,g);t1=t+1.0;c1=1-B1**t1;d=jax.tree.map(lambda a:a/c1,mu);return d,mu,nu,gn,coef
    @eqx.filter_jit
    def sgn(g,mu,nu,t):
        g,gn,coef=prep(g);d=jax.tree.map(jnp.sign,g);return d,mu,nu,gn,coef
    return {'R1RMS':rms,'R2MOM':mom,'R3SGN':sgn}
def selftest():
    import jax,jax.numpy as jnp,equinox as eqx
    D=make_dirs(jax,jnp,eqx,1.0);rng=np.random.default_rng(0);g={'a':jnp.asarray(rng.standard_normal((4,5)).astype(np.float32)*3),'b':jnp.asarray(rng.standard_normal(7).astype(np.float32))}
    z=jax.tree.map(jnp.zeros_like,g);gnp={k:np.asarray(v) for k,v in g.items()};gn=np.sqrt(sum((v**2).sum() for v in gnp.values()));coef=min(1.0,1.0/gn);gc_={k:v*coef for k,v in gnp.items()}
    d,mu,nu,gn_,c_=D['R1RMS'](g,z,z,jnp.float32(0.));assert abs(float(gn_)-gn)<1e-4 and abs(float(c_)-coef)<1e-5
    for k in gc_:assert np.allclose(np.asarray(d[k]),gc_[k]/(np.sqrt((1-.999)*gc_[k]**2/(1-.999))+1e-8),rtol=1e-4)    # first step: v_hat = g^2 -> d = g/(|g|+eps) = sign(g)
    d,mu,nu,_,_=D['R2MOM'](g,z,z,jnp.float32(0.))
    for k in gc_:assert np.allclose(np.asarray(d[k]),gc_[k],rtol=1e-4,atol=1e-6)        # first step: m_hat = g
    d,_,_,_,_=D['R3SGN'](g,z,z,jnp.float32(0.))
    for k in gc_:assert np.array_equal(np.asarray(d[k]),np.sign(gc_[k]))
    # second step R2: m = .9*.1*g1 + .1*g2 bias corrected
    g2={k:v*0.5+1 for k,v in g.items()};_,mu1,_,_,_=D['R2MOM'](g,z,z,jnp.float32(0.));d2,_,_,_,_=D['R2MOM'](g2,mu1,z,jnp.float32(1.))
    gn2=np.sqrt(sum((np.asarray(v)**2).sum() for v in g2.values()));c2=min(1.0,1.0/gn2)
    for k in gc_:
        m=.9*(.1*gc_[k])+.1*np.asarray(g2[k])*c2;assert np.allclose(np.asarray(d2[k]),m/(1-.9**2),rtol=1e-4,atol=1e-6)
    assert E.ADAM['R4VFIX'][2] is True and E.ADAM['R1RMS'][2] is False and 'R3SGN' in E.ARMS
    print('e57_adamdecomp_run selftest ok')
def decomp_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    assert N==1,'one cycle'
    ntg=list(json.load(open(AD['ntg']))['norm_targets'][:nch]);assert len(ntg)==nch
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'E57_adamdecomp_rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids_all[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids_all[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V of the stream recorded ({Kh0.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    DIRS=make_dirs(LD.jax,LD.jnp,LD.eqx,float(rt.cfg.training.optimizer_inner.clip_gradient))
    for arm in arms:
        fnA=out/f'E57_main_{arm}.arms.json'
        if fnA.exists() and not a.rerun:print('exists, reused:',fnA);continue
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_main_{arm}',canary=None);L.rec_cache=rc;traj=[]
        if arm in DIRS:L.adam_dir=DIRS[arm]
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),cycle=k//n+1,canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(arm,1,ids_all,nch=nch,tag=f'ad{arm}',log=lg,read_hook=hook,hook_chunks=set(c for c in cch if c<nch),read_chunks=set(c for c in rch if c<nch),chunks_meta=dummy_meta,norm_targets=ntg);log=res['log']
        assert set(res['reads'])>=set(c for c in rch if c<nch),('weights at some positions were not saved',sorted(rch),sorted(res['reads']))
        fnA.write_text(json.dumps(dict(arm=arm,engine=arm,bank=bsrc,cycles=N,n_chunks=nch,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','gnorm','scale') if k_ in e} for e in log],read_positions=sorted(int(c) for c in rch)),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def main():
    if '--selftest' in sys.argv:selftest();return
    AD['ntg']=_pop('--ntg-from')
    import e57_main_run as M,e57_qcurve_run as QC
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NEW if n not in M.ARMS_ALL);M.stage_arms=decomp_arms;M.stage_reads=QC.qcurve_reads
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='arms':assert AD['ntg']
    M.main()
if __name__=='__main__':main()
