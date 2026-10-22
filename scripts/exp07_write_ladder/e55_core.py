"""E55 one-pass runner core (design: E55-controlled-world-binding-design.md 2.1, 2.3). Built on s3.Ladder3 (record / prefix_masked / write_step / read_state / native_step are the lab's unchanged closures); this file adds only
  (1) compose_state: the carried suffix-layer SWA state (stacked over the suffix layers; K and V each [L, 8192, hidden], block of lag j = rows [(8-j)*1024, (9-j)*1024), merged heads, pre-RoPE, after qk_norm, RoPE applied inside the window at window-relative positions; all verified on the public e2e source a4fc4788, ttt/model/attention.py SWA) composed from the real / decoy / background-only chains per e55_swap.plan,
  (2) Ladder55.run_stream: the stream is read ONCE; chunk k is always WRITTEN with its real tokens; the attention window of the write forward is composed per arm (prefix layers: recorded post-RoPE K,V of the three streams via e55_swap.compose_window; suffix layers: compose_state), the three suffix chains are advanced with the SAME current weights, weights are saved at the read chunks.
Every arm goes through the same write path (prefix_masked + write_step); NAT = nothing swapped (U1 shows it equals native_step). Absolute chunk positions are unchanged: a decoy slot holds the K,V the decoy stream has at the same absolute chunk index (RoPE of the prefix layers is absolute, that of the suffix layers window-relative, so both stay aligned).
Arms: NAT, SWD, SWDS, SWDO, MSK implemented; SWB (background-only stream) needs a background-only realisation of the world that gen_world_ctl does not provide yet (stage 2): refused.
STATUS: tested on CPU with a tiny random e2e model (e55_units.py); the 3B GPU run is NOT done."""
import time,hashlib,json
from pathlib import Path
import numpy as np
import e55_swap as E
import s3
jax,jnp,eqx=s3.jax,s3.jnp,s3.eqx
CH=1024;WIN=8192;NW=E.NW
IMPLEMENTED=('NAT','SWD','SWDS','SWDO','MSK')
def compose_state(chains,pl,counter=None):
    """suffix-layer SWA state with the slot blocks taken from the chain that plan `pl` names ('real' | 'decoy' | 'bg'; 'mask' and absent slots keep the real chain's block, the mask is applied through the key mask). Every other leaf (chunk index) is the real chain's. counter (list) collects the number of KV leaves composed (must be 2: K and V)."""
    def f(*ls):
        l0=ls[0]
        if getattr(l0,'ndim',0)==3 and l0.shape[1]==WIN:
            if counter is not None:counter.append(1)
            blocks=[]
            for j in range(NW,0,-1):
                s=pl.get(j,'real');s='real' if s=='mask' else s
                blocks.append(ls[('real','decoy','bg').index(s)][:,(NW-j)*CH:(NW-j+1)*CH])
            return jnp.concatenate(blocks,axis=1)
        return l0
    return jax.tree.map(f,chains['real'],chains['decoy'],chains['bg'])
def keep_vector(pl):
    """bool [WIN+CH]: True = key kept; 'mask' slots False; same indexing as s3_sets.Sets.keep_vector (index j <-> absolute position k*CH-WIN+j; absent slots stay True, the kj>=0 test of the masks removes them)"""
    keep=np.ones(WIN+CH,bool)
    for j,s in pl.items():
        if s=='mask':keep[(NW-j)*CH:(NW-j+1)*CH]=False
    return keep
class Ladder55(s3.Ladder3):
    def __init__(self,rt,save_dir=None,canary=None):
        """no Readouts / old world: only what build3 and the step functions need"""
        from ttt.model.data import Batch
        self.rt=rt;cfg=rt.cfg.training.optimizer_inner;self.lr=float(cfg.lr);self.clip=float(cfg.clip_gradient);self.lr_scale=1.0;self.Batch=Batch
        self.f=s3.reuse.get_filter_spec(rt.base,rt.cfg.training.spec_inner,'inner parameters');self.save_dir=save_dir;self._nsave=0;self.canary=canary;self.build3()
    def set_inner(self,m,newp):
        inner,outer=eqx.partition(m,self.f);assert jax.tree_util.tree_structure(inner)==jax.tree_util.tree_structure(newp);return eqx.combine(newp,outer)
    def save_state(self,m,name=None):
        leaves=[np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.partition(m,self.f)[0])];Path(self.save_dir).mkdir(parents=True,exist_ok=True)
        fn=Path(self.save_dir)/(name or f'state_{self._nsave:03d}.npz');self._nsave+=1;np.savez(fn,*leaves);return {'_file':str(fn),'_hash':s3.reuse._hash_pytree(m.inner_parameters())[0]}
    def load_state(self,fn):
        z=np.load(fn);leaves=[jnp.asarray(z[f'arr_{i}']) for i in range(len(z.files))]
        inner,outer=eqx.partition(self.rt.base,self.f);td=jax.tree_util.tree_structure(inner);assert td.num_leaves==len(leaves);return eqx.combine(jax.tree_util.tree_unflatten(td,leaves),outer)
    def _stream_seq(self,ids,nch):
        rt=self.rt;ids=np.asarray(ids)[:nch*CH];mask=np.zeros(nch*CH,dtype=bool);mask[:len(ids)-1]=True;return rt.seq(ids,mask=mask)
    def window_inputs(self,arm,k,p,recs,sts,counter=None):
        """what the write forward of chunk k gets: (Kc,Vc,keep,st_suffix,plan). recs[name] = (Kh,Vh) numpy [layers,T,nh,d]; sts[name] = suffix chain state."""
        pl=E.plan(arm,k,p);keep=keep_vector(pl)
        if arm in ('NAT','MSK'):
            Kc,Vc=self.ctx_kv(*recs['real'],k);st=sts['real']
        else:
            src_k={n:recs[n][0] for n in recs};src_v={n:recs[n][1] for n in recs}
            Kw,_=E.compose_window(src_k,k,pl,CH,axis=1);Vw,_=E.compose_window(src_v,k,pl,CH,axis=1);Kc,Vc=jnp.asarray(Kw),jnp.asarray(Vw)
            st=compose_state({'real':sts['real'],'decoy':sts.get('decoy',sts['real']),'bg':sts.get('bg',sts['real'])},pl,counter)
        return Kc,Vc,keep,st,pl
    def run_stream(self,arm,p,real_ids,decoy_ids=None,bg_ids=None,read_chunks=(),nch=None,tag='',smoke=False,log=print,weights0=None):
        """-> {'log': per-chunk dicts, 'reads': {chunk: {...state file / hash / canary}}, 'stream_hash', 'clip_bind_frac'}"""
        assert arm in E.ARMS,arm
        if arm not in IMPLEMENTED:raise NotImplementedError(f'arm {arm} is not implemented (SWB needs a background-only stream)')
        rt=self.rt;n=len(real_ids)//CH;nch=nch or n;assert nch<=n and len(real_ids)%CH==0
        names=['real']+(['decoy'] if arm in ('SWD','SWDS','SWDO') else [])+(['bg'] if arm=='SWB' else [])
        ids_={'real':np.asarray(real_ids),'decoy':None if decoy_ids is None else np.asarray(decoy_ids),'bg':None if bg_ids is None else np.asarray(bg_ids)}
        for nm in names:assert ids_[nm] is not None and len(ids_[nm])>=nch*CH,(nm,'stream missing or too short')
        out={'log':[],'reads':{},'arm':arm,'p':p,'tag':tag,'n_chunks':nch,'stream_hash':hashlib.sha256(np.asarray(real_ids[:nch*CH],np.int64).tobytes()).hexdigest()}
        m=weights0 if weights0 is not None else rt.base;st_all=s3.clone_state(rt.state_all);cum=0.0
        with rt.mesh:
            seqs={nm:self._stream_seq(ids_[nm],nch) for nm in names};prefs={nm:rt.prefix(rt.base,seqs[nm]) for nm in names}
            recs={}
            for nm in names:
                t0=time.monotonic();Kh,Vh,_=self.record(ids_[nm][:nch*CH]);recs[nm]=(Kh,Vh);log(f'  recorded natural K,V of {nm} ({Kh.nbytes*2/2**30:.1f} GiB host) in {time.monotonic()-t0:.0f}s')
            sts={nm:s3.clone_state(rt.state_suffix) for nm in names}
            if self.canary:out['canary_w0']=self.canary(m)
            for k in range(nch):
                t0=time.monotonic();old=m;c=rt.chunk(seqs['real'],k);cnt=[]
                Kc,Vc,keep,st_w,pl=self.window_inputs(arm,k,p,recs,sts,cnt)
                if cnt:assert len(cnt)==2,('expected exactly the K and V leaves of the suffix SWA state, got',len(cnt))
                pref_w=self.prefix_masked(m,jnp.asarray(ids_['real'][k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep))
                newp,gn,coef,ce=self.write_step(m,st_all,st_w,c,pref_w,jnp.asarray(keep))
                for nm in names:sts[nm]=self.read_state(old,sts[nm],rt.chunk(seqs[nm],k),prefs[nm][k*CH:(k+1)*CH])   # same weights (start of the chunk) for every chain
                m=self.set_inner(m,newp)
                un=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=un
                ent={'k':k,'gnorm':float(gn),'coef':float(coef),'update_norm':un,'ce':float(ce),'sec':time.monotonic()-t0,'n_swapped_slots':sum(1 for v in pl.values() if v not in ('real',)),'n_masked_slots':sum(1 for v in pl.values() if v=='mask')}
                out['log'].append(ent)
                if k in read_chunks:
                    r={**self.save_state(m,f'state_{tag}_{k:03d}.npz')}
                    if self.canary:r['canary']=self.canary(m)
                    out['reads'][k]=r;log(f'  {arm} chunk {k+1}/{nch}: state saved, hash {r["_hash"]}')
                elif k%8==0:log(f'  {arm} chunk {k+1}/{nch} {ent["sec"]:.1f}s ce {ent["ce"]:.3f}')
            out['final_hash']=s3.reuse._hash_pytree(m.inner_parameters())[0];out['model']=m
        cs=[e['coef'] for e in out['log']];out['clip_bind_frac']=sum(c_<1 for c_ in cs)/len(cs);out['cum_update_norm']=cum
        return out
