"""E55 candidate readers (design 2.4, 2.5): 16-candidate span scores for every item at the saved read-point states and at W0, with the out-of-window neutral tail ('out') and the in-window real-stream tails ('in_e<lag>'). Modelled on e49_candidates.py / e54b.gpu_main: fp32 runtime, packing.Packed(ctx, raw_rows, GP=2).r1(m32) scores the packed rows (summed span log-prob), one Packed per tail (the frozen prefix layers do not depend on the weights, so the same Packed scores W0 and every arm's state).
The lab modules are INJECTED (Reader55.__init__), so that the orchestration can be tested on CPU with the tiny model of e55_cpu_harness.py (e55_units.py U11); on the server e55_run.py passes ladder / packing.
Record (one JSON per arm cell and seed; format of e55_read.py, W0 stored separately per context): {'cell','arm','p','M','s','bank','seed','meta':item_meta,'read':{'<a>':{'<ctx>':{'A':S,'B':S}}},'W0':{'out':S,'<ctx>@<a>':{'A':S,'B':S}},'d':{'<a>':{'<ctx>':[d per item]}},'hash':{...}} with S a list (n_items,16).
STATUS: CPU-tested with random rows and the tiny model; the 3B / real-tokenizer / SQuAD-tail path is NOT run."""
import json,time
import numpy as np
class Reader55:
    def __init__(self,rt,rt32,packing,TailCtx,get_filter_spec,hash_pytree,jax,eqx,jnp):
        self.rt,self.rt32,self.packing,self.TailCtx,self.hash_pytree,self.jax,self.eqx,self.jnp=rt,rt32,packing,TailCtx,hash_pytree,jax,eqx,jnp
        self.f=get_filter_spec(rt.base,rt.cfg.training.spec_inner,'inner parameters');self.f32=get_filter_spec(rt32.base,rt32.cfg.training.spec_inner,'inner parameters')
    def to_fp32(self,m):
        src=self.jax.tree_util.tree_leaves(self.eqx.partition(m,self.f)[0]);inner32,outer32=self.eqx.partition(self.rt32.base,self.f32);td=self.jax.tree_util.tree_structure(inner32);assert td.num_leaves==len(src)
        return self.eqx.combine(self.jax.tree_util.tree_unflatten(td,src),outer32)
    def load_state(self,fn,expect_hash=None):
        z=np.load(fn);leaves=[self.jnp.asarray(z[f'arr_{i}']) for i in range(len(z.files))];inner,outer=self.eqx.partition(self.rt.base,self.f);td=self.jax.tree_util.tree_structure(inner);assert td.num_leaves==len(leaves)
        m=self.eqx.combine(self.jax.tree_util.tree_unflatten(td,leaves),outer)
        if expect_hash is not None:assert self.hash_pytree(m.inner_parameters())[0]==expect_hash,('loaded weights differ from the streamed weights',fn)
        return m
    def packed(self,raw,tail_ids):
        with self.rt32.mesh,self.jax.default_matmul_precision('highest'):return self.packing.Packed(self.TailCtx(self.rt32,list(tail_ids)),raw,GP=2)
    def score(self,P,m,n_items,ncand):
        m32=self.to_fp32(m)
        with self.jax.default_matmul_precision('highest'):return np.asarray(P.r1(m32)).reshape(n_items,ncand)
def check_rows(raw,n_items,ncand):
    """every candidate of an item has the same span length (place_len) and the rows are item-major"""
    assert len(raw)==n_items*ncand,(len(raw),n_items,ncand);L=raw[0][1][1]-raw[0][1][0]
    for ids,(a,b) in raw:assert b-a==L and 0<a<b<=len(ids),(a,b,len(ids))
    return L
def read_world(rd,raw,n_items,ncand,plan_read,out_tail,state_files,w0_models=None,log=print,only_apps=None,which_ctx=('out','in'),w0_model=None,in_apps=None):
    """-> read[a][ctx][w] per arm, W0 dict.
    plan_read: e55_read.plan(world,enc)['read'] ({a: {'read_chunk','ctx':{name:{'tail_A','tail_B','d',...}}}}); out_tail: ids of the neutral tail;
    state_files: {arm: {a: {w: (file, hash)}}}. Order: per read point a, per context, per world w: ONE Packed (tail fixed), W0 scored first, then every arm's state (loaded one at a time)."""
    arms=sorted(state_files);read={ar:{} for ar in arms};W0={};m0=w0_model
    for a in sorted(plan_read):
        if only_apps is not None and a not in only_apps:continue
        ctxs=[('out',None)]+([(n,c) for n,c in plan_read[a]['ctx'].items()] if 'in' in which_ctx and (in_apps is None or a in in_apps) else [])
        for ctx,c in ctxs:
            for w in 'AB':
                t0=time.monotonic();tail=out_tail if ctx=='out' else c['tail_'+w];P=rd.packed(raw,tail)
                if ctx=='out':
                    if 'out' not in W0:W0['out']=rd.score(P,m0,n_items,ncand)
                else:W0.setdefault(f'{ctx}@{a}',{})[w]=rd.score(P,m0,n_items,ncand)
                for ar in arms:
                    ent=state_files[ar].get(a,{}).get(w)
                    if ent is None:continue
                    m=rd.load_state(*ent);read[ar].setdefault(a,{}).setdefault(ctx,{})[w]=rd.score(P,m,n_items,ncand);del m
                del P
                log(f'  read a={a} ctx={ctx} world {w}: {time.monotonic()-t0:.0f}s')
    return read,W0
def make_records(read,W0,meta,plan_read,info):
    """one record per arm (JSON-able). info: {'cell','p','M','s','bank','seed',...}"""
    out={}
    for ar,rd_ in read.items():
        rec={'arm':ar,**info,'meta':meta,'read':{str(a):{ctx:{w:np.asarray(S).tolist() for w,S in d.items()} for ctx,d in cs.items()} for a,cs in rd_.items()},
             'W0':{k:(np.asarray(v).tolist() if not isinstance(v,dict) else {w:np.asarray(S).tolist() for w,S in v.items()}) for k,v in W0.items()},
             }
        rec['meta']={**meta,'d':{str(a):{ctx:c['d'] for ctx,c in plan_read[a]['ctx'].items()} for a in rd_}}
        out[ar]=rec
    return out
