"""Exp05-A: chunk-wise relational memory trajectory (World A/B, Exp04 clean-v2 seed). e2e/ and Exp04 artifacts are not modified."""
import os,sys,json,time,argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
from core import *            # Runtime, CH, jax, jnp, eqx, np, MetaModel, Batch, clone_state, reuse
import optax,hashlib
import stage2,gen_world as gw
OUT=Path(os.environ['EXP05_OUT']) if os.environ.get('EXP05_OUT') else ROOT/'experiments/exp05_write_dynamics'   # run-1 default; run2 sets EXP05_OUT
EXP4=ROOT/'experiments/exp04_what_ttt_retains/stage2'
NCH=gw.NCH
CKPT_CARRY=[10,14,20,26,33,39,45,51,57,62]   # C5 checkpoints (state after chunk k)
V3_K=[14,33,62]

def vram_mb():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']).decode()
        for l in out.splitlines():
            pid,mem=[x.strip() for x in l.split(',')]
            if int(pid)!=os.getpid():continue
            return int(mem)
    except Exception:pass
    return None
def gpu_total_used():
    try:return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().split()[0])
    except Exception:return None

class Traj:
    def __init__(self,rt,seed):
        self.rt=rt;self.seed=seed;self.w=gw.build_world(seed,v=2)
        cfg=rt.cfg.training.optimizer_inner;self.lr=float(cfg.lr);self.clip=float(cfg.clip_gradient)
        self.tok=rt.tok
        # --- jitted own-implementation of the native inner step (V2) ---
        lr=self.lr;clip=self.clip
        @eqx.filter_jit
        def own_update(m,state_tuple,seq,pref):
            gfn=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)
            (_l,(ce,tok_nll,_ns)),grads=gfn(m,seq,state_tuple[1],prefix_outputs=pref)
            g=grads.inner_parameters()
            gn=optax.global_norm(g)
            coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30))     # optax.clip_by_global_norm semantics
            mult=m.get_ilr_multiplier(state_tuple[0].get(m.step_index))
            step=lr*mult
            delta=jax.tree.map(lambda x:(-step*coef*x) if x is not None else None,g)
            newp=jax.tree.map(lambda p,d:(p+d.astype(p.dtype)).astype(p.dtype),m.inner_parameters(),delta)
            return newp,gn,coef,ce
        self.own_update=own_update
        @eqx.filter_jit
        def short_logp(m,ids_pad,spans_unused):
            seq=rt.seq(ids_pad);pref=rt.prefix(m,seq)
            _,lp=rt.chunk_logp(m,clone_state(rt.state_suffix),seq,pref);return lp
        self.short_logp=short_logp
        self.P={False:stage2.Probe(rt,self.w,False),True:stage2.Probe(rt,self.w,True)}
        self.l3=[it for it in self.P[False].items if it['level']=='L3']
        self.ctrl=self._control_items()
        self._tok_cache={}
    def _control_items(self):
        w=self.w;items=[]
        cp,cl=w['ctrl_persons'],w['ctrl_places']
        for p in range(4):
            a,b=2*p,2*p+1
            for x,y in ((a,b),(b,a)):
                items.append({'level':'C4','pair':f'c{p}','tid':100+x,'prefix':gw.PR_HOME.format(X=cp[x]),'gold':' '+cl[x],'foil':' '+cl[y]})
        return items
    # ---- short weights-only probe: zero KV, 1024-token chunk, RoPE from 0, no tail ----
    def _item_arrays(self,it):
        key=(it['prefix'],it['gold'],it['foil'])
        if key in self._tok_cache:return self._tok_cache[key]
        slot=stage2.make_slot(self.tok,None);res=[]
        for cand in (it['gold'],it['foil']):
            ids,span=stage2.item_ids(self.tok,slot,it['prefix'],cand);assert len(ids)<=CH
            res.append((np.asarray(ids+[self.rt.eos]*(CH-len(ids)),dtype=np.int32),span))
        self._tok_cache[key]=res;return res
    def probe_short(self,m,items):
        out=[]
        with self.rt.mesh:
            for it in items:
                sc=[]
                for ids,span in self._item_arrays(it):
                    lp=np.asarray(self.short_logp(m,jnp.asarray(ids),0));sc.append(float(lp[span[0]-1:span[1]-1].sum()))
                out.append(sc[0]-sc[1])
        return np.asarray(out)
    # ---- Exp04-protocol probes (stream tail 7168 tokens): weights-only (empty KV) or native carry KV ----
    def probe_exp04(self,world_swap,m,st,k,items,mode):
        P=self.P[world_swap];sc=[]
        stt=clone_state(self.rt.state_suffix) if mode=='empty' else st
        for it in items:
            (g,fo),(ng,nf)=P.score(m,stt,k,it,None);sc.append((g[0],fo[0]))
        return sc

    def pair_margin(self,items,margins):
        by={}
        for it,mg in zip(items,margins):by.setdefault(it['pair'],[]).append(mg)
        return {p:float(np.mean(v)) for p,v in by.items()}

    def chunk_classes(self,swap):
        """per (pair,chunk): own/other/filler + shadow status from actual token spans."""
        P=self.P[swap];spans=sorted(P.spans,key=lambda s:s['tok_start']);tri={t['id']:t['pair'] for t in self.w['triples']}
        prev={};ev=[]
        for s in spans:
            t=s['t'];pr=prev.get(t)
            shadow=(pr is not None and pr['tok_end']>=s['tok_start']-8192)
            ev.append({'t':t,'pair':tri[t],'round':s['round'],'tok_start':s['tok_start'],'tok_end':s['tok_end'],'chunk':s['tok_start']//CH,'chunk_end':s['tok_end']//CH,'first':pr is None,'shadowed':bool(shadow),'prev_gap':(s['tok_start']-pr['tok_end']) if pr else None})
            prev[t]=s
        return ev

    def run_world(self,swap,max_chunks=NCH,v1_repeat=False,do_probes=True,do_carry=True,do_v3=True):
        rt=self.rt;P=self.P[swap];tag='B' if swap else 'A'
        n=max_chunks;ids=P.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True
        seq=rt.seq(ids,mask=mask)
        with rt.mesh:
            pref=rt.prefix(rt.base,seq);jax.block_until_ready(pref)
            m=rt.base;opt=m.inner_optimizer(clone_state(rt.state_all)).init(m.inner_parameters());state_tuple=(clone_state(rt.state_all),clone_state(rt.state_suffix))
            rows=[];margins_state={};ctrl_state={};hashes=[];tok_nll=[];carry={};v3={};t_start=time.monotonic();timing=[]
            def do_state_probes(j,m_):
                if not do_probes:return
                mg=self.probe_short(m_,self.l3+self.ctrl)
                margins_state[j]=self.pair_margin(self.l3,mg[:len(self.l3)]);ctrl_state[j]=self.pair_margin(self.ctrl,mg[len(self.l3):])
            def hook_at(j,m_,st_):
                """j = number of chunks consumed. V3 / C5 probes on this state."""
                k=j-1
                if do_v3 and k in V3_K:
                    v3[k]={'empty':self.probe_exp04(swap,m_,st_,k,self.l3,'empty'),'carry':self.probe_exp04(swap,m_,st_,k,self.l3,'carry')}
                if do_carry and k in CKPT_CARRY and k not in V3_K:
                    carry[k]=self.probe_exp04(swap,m_,st_,k,self.l3,'carry')
            t0=time.monotonic();do_state_probes(0,m);timing.append(('probe_state',time.monotonic()-t0))
            h0=reuse._hash_pytree(m.inner_parameters())[0];hashes.append(h0)
            for k in range(n):
                c=rt.chunk(seq,k);p=pref[k*CH:(k+1)*CH];t0=time.monotonic()
                # own implementation (V2) on the pre-update model
                newp,gn,coef,ce=self.own_update(m,state_tuple,c,p);jax.block_until_ready(gn)
                # token NLL + native update
                _,lp=rt.chunk_logp(m,state_tuple[1],c,p);tok_nll.append(np.asarray(-lp,dtype=np.float32))
                old=m.inner_parameters();r=rt.update(m,opt,state_tuple,c,p);jax.block_until_ready(r.metrics)
                m,opt,state_tuple=r.new_model,r.new_optimizer_state,r.new_state;del r
                new=m.inner_parameters()
                dn=jax.tree.map(lambda a,b:(a.astype(jnp.float32)-b.astype(jnp.float32)) if eqx.is_inexact_array(a) else None,new,old)
                def sqn(t):return sum(float(jnp.sum(x*x)) for x in jax.tree.leaves(t) if x is not None)
                nat2=sqn(dn);err2=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(new),jax.tree.leaves(newp)))
                # per-leaf / per-layer update norms
                leafn={};layern=None
                for (pth,x) in jax.tree_util.tree_flatten_with_path(dn)[0]:
                    leafn[jax.tree_util.keystr(pth)]=float(jnp.sqrt(jnp.sum(x*x)))
                    if x.ndim>=3 and x.shape[0]==int(rt.cfg.model.suffix_len):
                        ln=np.asarray(jnp.sqrt(jnp.sum(x.reshape(x.shape[0],-1)**2,axis=1)));layern=ln if layern is None else np.sqrt(layern**2+ln**2)
                hashes.append(reuse._hash_pytree(m.inner_parameters())[0])
                rows.append({'chunk':k,'grad_norm_preclip':float(gn),'clip_coef':float(coef),'clipped':bool(float(gn)>self.clip),'ce_loss':float(ce),'update_norm':float(np.sqrt(nat2)),'v2_rel_err':float(np.sqrt(err2)/max(np.sqrt(nat2),1e-30)),
                             'layer_update_norm':None if layern is None else [float(x) for x in layern],'leaf_update_norm':leafn,'hash':hashes[-1],'pre_nll_mean':float(tok_nll[-1][:CH-1].mean()),'sec_update':time.monotonic()-t0})
                t1=time.monotonic();do_state_probes(k+1,m);rows[-1]['sec_probe']=time.monotonic()-t1
                if (do_v3 or do_carry):
                    t2=time.monotonic();hook_at(k+1,m,state_tuple[1]);rows[-1]['sec_hook']=time.monotonic()-t2
                rows[-1]['vram_mb']=vram_mb();rows[-1]['gpu_used_mb']=gpu_total_used()
                print(f'  [{tag}] chunk {k} gn {float(gn):.3f} coef {float(coef):.3f} v2err {rows[-1]["v2_rel_err"]:.2e} upd {rows[-1]["update_norm"]:.4f} sec {time.monotonic()-t0:.1f}',flush=True)
        return {'world':tag,'rows':rows,'margins_state':margins_state,'ctrl_state':ctrl_state,'hashes':hashes,'tok_nll':np.stack(tok_nll),'carry':carry,'v3':v3,
                'events':self.chunk_classes(swap),'total_sec':time.monotonic()-t_start,'lr':self.lr,'clip':self.clip,'xla_flags':os.environ.get('XLA_FLAGS',''),'jax':jax.__version__}

def save(res,path):
    d={k:v for k,v in res.items() if k!='tok_nll'}
    json.dump(d,open(str(path)+'.json','w'),default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o))
    np.savez_compressed(str(path)+'_tok_nll.npz',tok_nll=res['tok_nll'].astype(np.float16))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--max-chunks',type=int,default=NCH);ap.add_argument('--tag',default='full')
    ap.add_argument('--worlds',default='AB');ap.add_argument('--no-probes',action='store_true');ap.add_argument('--no-carry',action='store_true');ap.add_argument('--no-v3',action='store_true')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    rt=Runtime();T=Traj(rt,a.seed);print('loaded; lr',T.lr,'clip',T.clip,'gpu used',gpu_total_used(),'param dtypes',sorted({str(x.dtype) for x in jax.tree.leaves(rt.base.inner_parameters()) if hasattr(x,'dtype')}),'n inner leaves',len(jax.tree.leaves(rt.base.inner_parameters())),'suffix_len',int(rt.cfg.model.suffix_len),flush=True)
    for w in a.worlds:
        f=OUT/f'traj_seed{a.seed}_{a.tag}_{w}'
        if Path(str(f)+'.json').exists():print('exists',f);continue
        t0=time.monotonic();res=T.run_world(w=='B',a.max_chunks,do_probes=not a.no_probes,do_carry=not a.no_carry and a.max_chunks==NCH,do_v3=not a.no_v3 and a.max_chunks==NCH)
        save(res,f);print('saved',f,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
