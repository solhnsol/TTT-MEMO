"""E55 candidate readers (pairless fact world, design 10.2): 16-candidate span scores for every item at the saved read-point states and at W0, with the out-of-window neutral tail ('out', exactly 8192 tokens) and the per-group in-window real-stream tails ('in_g<g>'). Modelled on e49_candidates.py / e54b.gpu_main: fp32 runtime, packing.Packed(ctx, raw_rows, GP=2).r1(m32) scores the packed rows (summed span log-prob), one Packed per tail (the frozen prefix layers do not depend on the weights, so the same Packed scores W0 and every arm's state); weights are never updated while reading.
The lab modules are INJECTED (Reader55.__init__), so that the orchestration can be tested on CPU with the tiny model of e55_cpu_harness.py (e55_units.py U11); on the server e55_run.py passes ladder / packing.
Record (one JSON per arm cell and seed): see e55_read.py. STATUS: CPU-tested with random rows and the tiny model; the 3B / real-tokenizer / SQuAD-tail path is NOT run."""
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
def read_world(rd,raw,n_items,ncand,plan_read,out_tail,state_files,log=print,only_apps=None,which_ctx=('out','in'),w0_model=None,in_apps=None):
    """pairless fact world (design 10.2) -> (read, W0, idx).
    read[arm][a]['out'] = (n_items,16) scores of ALL items in the neutral tail (out_tail: exactly 8192 tokens, built once into one Packed that scores W0 and every state); read[arm][a]['in_g<g>'] = (len(idx),16) scores of group g's items (plan_read[a]['ctx'][name]['items'] = idx, control items ride with group 0) in ITS OWN group's real-stream context (the 8 chunks that end just before the group's next appearance, 'tail'). W0['out'] / W0['in_g<g>@<a>'] have the same shapes. idx[a][name] = item indices of the rows.
    plan_read: e55_read.plan(world)['read']; state_files: {arm: {a: (file, hash)}}. Weights are FIXED during reading (no inner update); a state is loaded, scored on the shared Packed and dropped."""
    assert len(out_tail)==8192,('out-of-window context must be exactly 8192 tokens',len(out_tail))
    arms=sorted(state_files);read={ar:{} for ar in arms};W0={};idx={};m0=w0_model
    for a in sorted(plan_read):
        if only_apps is not None and a not in only_apps:continue
        if a not in sorted(plan_read):continue
        have=[ar for ar in arms if state_files[ar].get(a) is not None]
        if 'out' in which_ctx and have:
            t0=time.monotonic();P=rd.packed(raw,out_tail);S0=rd.score(P,m0,n_items,ncand)
            if 'out' not in W0:W0['out']=S0
            else:assert np.array_equal(W0['out'],S0),'W0 out scores changed between read points'
            for ar in have:m=rd.load_state(*state_files[ar][a]);read[ar].setdefault(a,{})['out']=rd.score(P,m,n_items,ncand);del m
            del P;log(f'  read a={a} out: {time.monotonic()-t0:.0f}s')
        if 'in' not in which_ctx or (in_apps is not None and a not in in_apps):continue
        for name,c in sorted(plan_read[a]['ctx'].items()):
            ii=np.asarray(c['items']);rows=[raw[i*ncand+k] for i in ii for k in range(ncand)];idx.setdefault(a,{})[name]=ii.tolist()
            t0=time.monotonic();P=rd.packed(rows,c['tail']);W0[f'{name}@{a}']=rd.score(P,m0,len(ii),ncand)
            for ar in have:m=rd.load_state(*state_files[ar][a]);read[ar].setdefault(a,{})[name]=rd.score(P,m,len(ii),ncand);del m
            del P;log(f'  read a={a} {name} (group {c["group"]}, {len(ii)} items): {time.monotonic()-t0:.0f}s')
    return read,W0,idx

def make_records(read,W0,meta,plan_read,idx,info):
    """one record per arm (JSON-able; format of e55_read / e55_metrics). info: {'k','n','p','bank','seed',...}"""
    out={}
    for ar,rd_ in read.items():
        a_s=[str(a) for a in rd_]
        m={**meta,**{k:v for k,v in info.items() if k in ('k','n','p','bank','seed')},'arm':ar,'cell':info.get('cell')}
        m['d']={str(a):{name:plan_read[a]['ctx'][name]['d'] for name in cs if name!='out'} for a,cs in rd_.items()}
        m['item_idx']={str(a):{name:idx[a][name] for name in cs if name!='out'} for a,cs in rd_.items()}
        rec={'arm':ar,**info,'meta':m,'read':{str(a):{ctx:np.asarray(S).tolist() for ctx,S in cs.items()} for a,cs in rd_.items()},'W0':{k:np.asarray(v).tolist() for k,v in W0.items()}}
        out[ar]=rec
    return out

class Reads:
    """frozen-weight reads ALONG a stream (inline mode, design 11c): called from e55_core.Ladder55.run_stream after the update of a chunk. Nothing here touches the fast weights, the optimizer, the RNG or the carried KV window: the weights are only passed to pure jitted scoring functions (an fp32 COPY is made by Reader55.to_fp32), the contexts are computed separately from the real stream tokens. W0 (before any update) is scored once and shared by all arms of a seed.
    plan_read: e55_read.plan(world)['read']; read_apps: appearances with an 'out' read (all items, neutral 8192-token tail); in_apps: appearances with the in-window reads (per group, tail = the real chunks that end just before the group's next appearance; shorter than 8 chunks early in the stream, see plan_read[a]['ctx'][name]['window_chunks'])."""
    def __init__(self,rd,raw,n_items,ncand,plan_read,out_tail,m0,read_apps,in_apps,log=print):
        assert len(out_tail)==8192;self.rd,self.raw,self.n,self.nc,self.plan,self.m0,self.log=rd,raw,n_items,ncand,plan_read,m0,log;self.apps=set(read_apps);self.in_apps=set(in_apps)
        self.P_out=rd.packed(raw,out_tail);self.W0={'out':rd.score(self.P_out,m0,n_items,ncand)};self.idx={}
    def at(self,a,m):
        """-> {'out': (n_items,16), 'in_g<g>': (n_g,16)} for the weights m after appearance a (only the requested kinds)"""
        out={}
        if a in self.apps:out['out']=self.rd.score(self.P_out,m,self.n,self.nc)
        if a in self.in_apps:
            for name,c in sorted(self.plan[a]['ctx'].items()):
                ii=np.asarray(c['items']);rows=[self.raw[i*self.nc+k] for i in ii for k in range(self.nc)];self.idx.setdefault(a,{})[name]=ii.tolist();P=self.rd.packed(rows,c['tail'])
                key=f'{name}@{a}'
                if key not in self.W0:self.W0[key]=self.rd.score(P,self.m0,len(ii),self.nc)
                out[name]=self.rd.score(P,m,len(ii),self.nc);del P
        return out
