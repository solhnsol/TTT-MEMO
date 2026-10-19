"""3B 2x2 stream, 1 pass (PROTOCOL_stage01.md "3B 2x2 stream (spec v2)" 22:50 KST + amendment 23:00 KST).
Write rule per chunk k (eta = lr x ilr multiplier, lr x1; c_k = min(1, clip/||g_k||) from the NTP gradient alone):
  nat      native NTP (g_k from the native carried-KV read)                     Delta = -eta c_k g_k
  natc2    native NTP + chunk-alone c2                                          Delta = -eta c_k (g_k + kappa lam v_k)
  msk      masked NTP (g_k chunk-alone, positions from 0)                       Delta = -eta c_k g_k
  mskc2    masked NTP + chunk-alone c2                                          Delta = -eta c_k (g_k + kappa lam v_k)
  mskgp    masked NTP + chunk-alone place-token NTP (control, no swap)         Delta = -eta c_k (g_k + kappa lam gP_k)
  mskrep   masked REPLACE (descriptive): chunk-alone c2 only                    Delta = -eta kappa lam v_k
v_k = gP(chunk) - gP(c2-swapped chunk), place-token loss (suppress.chunk_rows 'X' / 'Y'), at the current weights; v_k = 0 on
chunks without an exposure sentence fully inside. lam = the screen's per-seed c2ch shared scale (onestep2_screen_checks).
Both gradients at the same weights, one combined update. Per chunk the log has |g|, c_k, |v|, the update norm.
Modes as ladder.py: --mode save (stream only, states saved) then --mode eval (fp32 readout of the saved states).
Usage: stream2x2.py --cell mskc2 --kappa 4 --world-seed 101 --mode save|eval [--worlds AB] [--tag T]"""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
sys.path.insert(0,str(LD.ROOT/'scripts/exp04_what_ttt_retains'))
import suppress as SP
OUT=LD.OUT
class Ladder2(LD.Ladder):
    def build2(self):
        rt=self.rt;lr,clip=self.lr,self.clip;Batch=self.Batch
        def _g(m,ids_,tg_,mk_,pref_,seg_,loc_):
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_)
                (_l,(ce,_n,_s)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,seq,clone_state(rt.state_suffix),prefix_outputs=pref_)
            finally:packing.PACK_SUF.clear()
            return grads.inner_parameters(),ce
        @eqx.filter_jit
        def gP(m,ids_,tg_,mk_,pref_,seg_,loc_):return _g(m,ids_,tg_,mk_,pref_,seg_,loc_)[0]
        def apply(m,st_all,g,v,kl,use_c):
            gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));coef=jnp.where(use_c,coef,1.0)
            s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
            newp=jax.tree.map(lambda p,x,y:(p+(-s*coef*(x+kl*y)).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g,v)
            return newp,gn,coef
        @eqx.filter_jit
        def msk_step(m,st_all,ids_,tg_,mk_,pref_,seg_,loc_,v,kl,wg):
            g,ce=_g(m,ids_,tg_,mk_,pref_,seg_,loc_);g=jax.tree.map(lambda x:x*wg,g)   # wg = 0 for REPLACE (no NTP term)
            newp,gn,coef=apply(m,st_all,g,v,kl,wg>0);return newp,gn,coef,ce
        @eqx.filter_jit
        def nat_step(m,st_all,st_suf,c,p,v,kl):
            (_l,(ce,_n,new_st)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,c,st_suf,prefix_outputs=p)
            newp,gn,coef=apply(m,st_all,grads.inner_parameters(),v,kl,True);return newp,new_st,gn,coef,ce
        self.gP=gP;self.msk_step=msk_step;self.nat_step=nat_step
    def ev(self,m):
        if getattr(self,'canary_only',False):return {'_gen_nll':self.gen_nll(m),'_tail_nll':self.tail_nll(m),'_canary_only':True}
        return self.evaluate(m)
    def run_cell(self,swap,cell,kappa,lam,passes=1):
        """passes = 1: the registered 1-pass stream (unchanged semantics). passes > 1 (multi-pass c2 2x2): fresh filler every pass (ladder.fs_streams: pass 1 = the original stream,
        pass p >= 2 = the same exposure sentences with new background); weights carry over, the suffix KV restarts each pass; the c2 term uses each pass's own world for chunk_rows."""
        rt=self.rt;W=int(rt.cfg.model.sliding_window_size);log=[];m=rt.base;st_all=clone_state(rt.state_all)
        streams=self.fs_streams(swap,passes) if passes>1 else [self.T.P[swap].ids];worlds=self._fs_worlds[swap] if passes>1 else [self.T.w]
        seg=np.zeros(CH,np.int32);loc=np.arange(CH,dtype=np.int32);pm=jnp.asarray(packing.prefix_mask(seg,W));samenext=np.concatenate([np.ones(CH-1,bool),[False]])
        need_v=cell in ('natc2','mskc2','mskgp','mskrep')
        kl=jnp.float32(kappa*lam);zero=None;cum=0.0
        with rt.mesh:
            log.append({'epoch':0,'margins':self.ev(m)})
            for ps in range(1,passes+1):
                ids=np.asarray(streams[ps-1]);stats=[];rows={}
                if need_v:
                    rX,_=SP.chunk_rows(worlds[ps-1],swap,self.T.tok,rt.bos,rt.eos,'X');rY,_=SP.chunk_rows(worlds[ps-1],swap,self.T.tok,rt.bos,rt.eos,'Y')
                    for (iX,mX,nX,inX),(iY,mY,nY,inY) in zip(rX,rY):
                        k=inX['k'];assert inY['k']==k and list(iX)==list(ids[k*CH:(k+1)*CH]),('X rows differ from the stream chunk',k,ps);rows[k]=(iX,mX,iY,mY)
                if cell.startswith('nat'):
                    mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref=rt.prefix(rt.base,seq);st_suf=clone_state(rt.state_suffix)
                for k in range(getattr(self,'nch_run',NCH)):
                    v=None;vn=0.0
                    if need_v and k in rows:
                        iX,mX,iY,mY=rows[k]
                        def gp(ii,mm):
                            tg=np.concatenate([ii[1:],[rt.eos]]).astype(np.int32);pf=self.prefix_iso(rt.base,jnp.asarray(ii),jnp.asarray(loc),pm)
                            return self.gP(m,jnp.asarray(ii),jnp.asarray(tg),jnp.asarray(mm),pf,jnp.asarray(seg),jnp.asarray(loc))
                        gx=gp(iX,mX)
                        if cell=='mskgp':v=gx
                        else:gy=gp(iY,mY);v=jax.tree.map(lambda a,b:a-b,gx,gy);del gy
                        del gx;vn=float(optax.global_norm(v))
                    if v is None:
                        if zero is None:zero=jax.tree.map(jnp.zeros_like,m.inner_parameters())
                        v=zero;klk=jnp.float32(0.0)
                    else:klk=kl
                    old=m
                    if cell.startswith('nat'):
                        newp,st_suf,gn,coef,ce=self.nat_step(m,st_all,st_suf,rt.chunk(seq,k),pref[k*CH:(k+1)*CH],v,klk)
                    else:
                        c=ids[k*CH:(k+1)*CH];tg=np.concatenate([ids[k*CH+1:(k+1)*CH+1],[rt.eos]])[:CH].astype(np.int32)
                        pf=self.prefix_iso(rt.base,jnp.asarray(c),jnp.asarray(loc),pm)
                        newp,gn,coef,ce=self.msk_step(m,st_all,jnp.asarray(c),jnp.asarray(tg),jnp.asarray(samenext),pf,jnp.asarray(seg),jnp.asarray(loc),v,klk,jnp.float32(0.0 if cell=='mskrep' else 1.0))
                    m=self.set_inner(m,newp)
                    un=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=un
                    stats.append({'k':k,'gnorm':float(gn),'coef':float(coef),'vnorm':vn,'has_v':bool(need_v and k in rows),'update_norm':un,'ce':float(ce)})
                cs=[s_['coef'] for s_ in stats if cell!='mskrep']
                log.append({'epoch':ps,'cum_update_norm':cum,'chunks':stats,'clip_bind_frac':(sum(c_<1 for c_ in cs)/len(cs)) if cs else None,'mean_coef':(sum(cs)/len(cs)) if cs else None,
                            'stream_hash':hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),'inner_hash':reuse._hash_pytree(m.inner_parameters())[0],'margins':self.ev(m)})
                print(f'  {cell} kappa {kappa} {"B" if swap else "A"} pass {ps}: cum upd {cum:.4f}, clip binds {log[-1]["clip_bind_frac"]}, chunks with v {sum(s_["has_v"] for s_ in stats)}',flush=True)
        return log
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=['nat','natc2','msk','mskc2','mskgp','mskrep'],required=True);ap.add_argument('--kappa',type=float,default=0.0)
    ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--worlds',default='AB');ap.add_argument('--mode',choices=['save','eval'],required=True);ap.add_argument('--tag',default='');ap.add_argument('--keep',action='store_true');ap.add_argument('--canary-only',action='store_true',help='kappa search: pass-1 canaries only (no state save, no readout)');ap.add_argument('--passes',type=int,default=1,help='multi-pass (fresh filler) c2 2x2: passes > 1');ap.add_argument('--smoke-chunks',type=int,default=None,help='smoke test: only the first N chunks per pass (no files kept, must be used with --canary-only and a smoke tag)')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    lam=float(json.load(open(OUT/f'onestep2_screen_checks_s{a.world_seed}.json'))['lam']['c2ch'])
    tag=a.tag or (f'{a.cell}_k{a.kappa:g}'+('_cq' if a.canary_only else '')+(f'_mp{a.passes}' if a.passes>1 else ''));lev='S2X2';rt=Runtime();SD=OUT/'states'/f'{lev}_{tag}_s{a.world_seed}'
    L=Ladder2(rt,a.world_seed,readout=(a.mode!='save'),save_dir=SD);L.build2();L.canary_only=a.canary_only
    if a.smoke_chunks:L.nch_run=a.smoke_chunks;assert a.canary_only and 'smoke' in tag
    json.dump({'items':L.items,'fam_sizes':{k:len(v) for k,v in L.fam.items()},'lr':L.lr,'clip':L.clip,'cell':a.cell,'kappa':a.kappa,'lam':lam,'xla_flags':os.environ.get('XLA_FLAGS')},open(OUT/f'meta_{lev}_{tag}_s{a.world_seed}.json','w'),indent=1)
    for w in a.worlds:
        f=OUT/f'{lev}_{tag}_s{a.world_seed}_{w}.json';fs=OUT/f'{lev}_{tag}_s{a.world_seed}_{w}.saved.json'
        if f.exists():print('exists',f);continue
        if a.mode=='eval':
            d=json.load(open(fs));t0=time.monotonic()
            for e in d['log']:
                if 'margins' in e and '_file' in e['margins']:
                    fn=e['margins']['_file'];h=e['margins']['_hash'];mm=L.load_state(fn)
                    assert reuse._hash_pytree(mm.inner_parameters())[0]==h;e['margins']=L.evaluate(mm);e['hash']=h
                    if not a.keep:os.remove(fn)
            d['sec_eval']=time.monotonic()-t0;json.dump(d,open(f,'w'));print('scored',f,round(d['sec_eval']),'s',flush=True);continue
        L.save_dir=SD/w;L._nsave=0;(SD/w).mkdir(parents=True,exist_ok=True)
        if fs.exists():print('exists',fs);continue
        t0=time.monotonic();log=L.run_cell(w=='B',a.cell,a.kappa,lam,a.passes)
        json.dump({'log':log,'sec':time.monotonic()-t0,'lam':lam,'kappa':a.kappa,'cell':a.cell},open(fs,'w'));print('saved',fs,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
