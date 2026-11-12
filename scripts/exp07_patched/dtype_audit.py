"""E34 step 0: does the stored 3B state lose small updates to rounding? (CPU for 0a/0b, GPU only with --live)
0a  dtype of every leaf of saved state npz files (np.savez of the inner-parameter leaves, see Ladder.save_state) and, with --live, of rt.base.inner_parameters().
0b  per leaf, dW = state - W0 statistics against the bf16 ulp of the W0 value (ulp = 2^(floor(log2|w|)-7), hypothetical if the leaf is stored in a wider type):
    fraction dW == 0, fraction |dW| < ulp/2 (would be rounded away in bf16 storage), median |dW|/ulp, and whether dW lies on the ulp grid (quantised -> bf16 storage).
0c  (--live) one native chunk (chunk 0, stream A) at W0: the update stored in p.dtype (ladder.py step rule) vs the same update accumulated in fp32:
    fraction of elements with a zero stored update and ||stored||/||fp32|| over the inner leaves.
Usage: dtype_audit.py --w0 W0_state.npz --state STATE.npz [--state ...] [--live --seed 101] [--grep-log run2.log]   -> OUT/e34_dtype_audit.json
W0_state.npz: the state-0 file of a save-mode run (L0 'save' keeps state 0), or any W0 dump of the same leaf order."""
import os,sys,json,argparse,re
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--w0');ap.add_argument('--state',action='append',default=[]);ap.add_argument('--live',action='store_true');ap.add_argument('--seed',type=int,default=101)
ap.add_argument('--grep-log',action='append',default=[]);ap.add_argument('--out',default=None);a=ap.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'))
def f32(x):
    if x.dtype.kind=='V' and x.dtype.itemsize==2:return (x.view(np.uint16).astype(np.uint32)<<16).view(np.float32)   # bfloat16 saved by np.savez
    return x.astype(np.float32)
def bf16_ulp(w):
    w=np.abs(w);e=np.floor(np.log2(np.maximum(w,np.float32(1e-38))));return np.where(w>0,2.0**(e-7),0.0).astype(np.float32)
res={'dtype_logs':{},'npz':{},'delta':{}}
for lg in a.grep_log:
    res['dtype_logs'][lg]=[l.strip()[:300] for l in open(lg,errors='ignore') if 'param dtype' in l.lower()][:5]
if a.w0:
    z0=np.load(a.w0);res['npz']['w0']={k:str(z0[k].dtype) for k in z0.files}
    for sp in a.state:
        z=np.load(sp);res['npz'][sp]={k:str(z[k].dtype) for k in z.files};per={}
        assert z.files==z0.files,'leaf lists differ'
        tot=dict(n=0,zero=0,below_half_ulp=0);med=[]
        for k in z.files:
            w0=f32(z0[k]).ravel();w1=f32(z[k]).ravel();d=w1-w0;u=bf16_ulp(w0);nz=u>0
            n=d.size;zero=int((d==0).sum());bh=int(((np.abs(d)<u/2)&nz).sum());r=np.abs(d[nz&(d!=0)])/u[nz&(d!=0)]
            on_grid=float(np.mean(np.abs(r-np.round(r))<1e-3)) if r.size else float('nan')
            per[k]={'dtype':str(z[k].dtype),'n':n,'frac_zero':zero/n,'frac_below_half_bf16_ulp':bh/n,'median_abs_dW_over_ulp':float(np.median(r)) if r.size else None,'frac_on_ulp_grid':on_grid}
            tot['n']+=n;tot['zero']+=zero;tot['below_half_ulp']+=bh;med.append(r[:100000])
        m=np.concatenate(med) if med else np.zeros(0);res['delta'][sp]={'per_leaf':per,'total_frac_zero':tot['zero']/tot['n'],'total_frac_below_half_bf16_ulp':tot['below_half_ulp']/tot['n'],'median_abs_dW_over_ulp':float(np.median(m)) if m.size else None}
        print(sp,'zero',round(tot['zero']/tot['n'],4),'below half bf16 ulp',round(tot['below_half_ulp']/tot['n'],4),'median |dW|/ulp',res['delta'][sp]['median_abs_dW_over_ulp'],flush=True)
if a.live:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import s3 as S3M
    from s3 import *
    rt=Runtime();L=Ladder3(rt,a.seed,readout=False);L.build3();m=rt.base;inner=m.inner_parameters()
    res['live']={'inner_leaf_dtypes':sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(inner) if hasattr(x,'dtype')}),'n_leaves':len(jax.tree_util.tree_leaves(inner)),'param_dtype_cfg':str(getattr(rt.cfg.model,'param_dtype',None)),'compute_dtype_cfg':str(getattr(rt.cfg.model,'compute_dtype',None))}
    ids=np.asarray(L.T.P[False].ids)[:NCH*CH];mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);st_all=clone_state(rt.state_all)
    with rt.mesh:
        pref=rt.prefix(rt.base,seq)[:CH];g,ce=L.grad_nat(m,clone_state(rt.state_suffix),rt.chunk(seq,0),pref)
        gn=optax.global_norm(g);coef=jnp.where(gn<L.clip,1.0,L.clip/jnp.maximum(gn,1e-30));s=L.lr*m.get_ilr_multiplier(st_all.get(m.step_index))
        up=jax.tree.map(lambda p,x:(-s*coef*x),inner,g);stored=jax.tree.map(lambda p,x:(p+x.astype(p.dtype)).astype(p.dtype)-p,inner,up);full=jax.tree.map(lambda p,x:x.astype(jnp.float32),inner,up)
        nz=sum(int(jnp.sum(x!=0)) for x in jax.tree_util.tree_leaves(stored));n=sum(x.size for x in jax.tree_util.tree_leaves(stored))
        ns=float(jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32)**2) for x in jax.tree_util.tree_leaves(stored))));nf=float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(full))))
        res['live'].update({'chunk0_ce':float(ce),'gnorm':float(gn),'coef':float(coef),'frac_zero_stored_update':1-nz/n,'norm_stored_over_norm_fp32':ns/nf})
    print(json.dumps(res['live']),flush=True)
json.dump(res,open(Path(a.out) if a.out else OUT/'e34_dtype_audit.json','w'),indent=1);print('saved')
