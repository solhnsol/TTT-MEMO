"""Context-gain selector mcg (PROTOCOL_stage01.md "Context-gain selector mcg", registered 2026-10-17 10:23 KST). 3B, masked write, 1 pass.
Per chunk k at the current fast weights (stop-gradient): native read forward (stream with carried suffix KV, as native TTT reads)
and chunk-alone write forward (the Q3 x1 masked setting). g_t = NLL_masked(t) - NLL_native(t), r_t = ||p_native - e_y|| / ||p_masked - e_y||.
Eligible E: 'all' = every masked loss position (in-chunk targets), '64' = those with in-chunk index >= 64.
Selection: top ceil(q |E|) by g_t, ties -> earlier position.
Cells (flag --cell; default-off, nothing else changes):
  masked   all eligible positions (= ladder.py L0mw x1; q = 1 bit identity vs L0mw_x1 pass 1 is checked in --diag)
  mcg      top-q by g_t                       msurp  top-q by NLL_masked (same count)
  mrand    same count, random positions seeded by (world seed, chunk index)
  masknm   -eta c (||g_mcg|| / ||g_masked||) g_masked, c = min(1, clip / ||g_mcg||)
  mcgw     (optional, descriptive) position weights proportional to max(g_t, 0) over E
--diag: the masked trajectory, forward diagnostics only (no D, no canary): per chunk the 6 (q, E) selections vs the gold place
tokens (precision / recall / F1), per-token g_t and r_t, the q = 1 state hash vs L0mw_x1 pass 1.
Modes as stream2x2.py: --mode save then --mode eval.  --max-chunks N: smoke test (no outputs kept).
Usage: mcg.py --diag --world-seed 101 | mcg.py --cell mcg --q 0.1 --elig all --world-seed 101 --mode save|eval"""
import os,sys,json,time,math,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
import suppress as SP
OUT=LD.OUT
QS=(0.05,0.10,0.20);ES=('all','64')
def gold_tokens(w,swap,tok,bos):
    """per stream target position (index i predicts token i+1): place / name tokens of every exposure sentence, with the
    sentence's round, triple and whether an earlier-round sentence of the same triple ends earlier in the same chunk."""
    text,_,_=gw.realize(w,swap);e=tok.encode(text,add_special_tokens=False);offs=[(0,0)]+list(e.offsets);info=SP._sent_info(w,swap)
    def ctok(c):return next(j for j,(a,b) in enumerate(offs) if j>0 and b>c)
    for s in info:s['t0']=ctok(s['start']);s['t1']=ctok(s['end']-1)+1
    out={}
    for s in info:
        k_=(s['t0'])//CH;earlier=any(u['t']==s['t'] and u['round']<s['round'] and u['t0']>=k_*CH and u['t1']<=s['t0'] for u in info)
        spans=[('place',s['start']+s['p0'],s['start']+s['p0']+len(s['P']))]+[('name',s['start']+x0,s['start']+x0+len(s['X'])) for x0 in s['xs']]
        for kind,c0,c1 in spans:
            for j in SP._ptoks(offs,c0,c1):
                if j>0:out[j-1]={'kind':kind,'round':s['round'],'t':s['t'],'earlier_in_chunk':earlier}
    return out
def eligible(E):
    pos=np.arange(CH-1)   # masked loss positions: every in-chunk target (last position predicts the next chunk: excluded)
    return pos if E=='all' else pos[pos>=64]
def select(score,E,q):
    """top ceil(q |E|) eligible positions by score, ties -> earlier position."""
    el=eligible(E);n=math.ceil(q*len(el));sc=np.asarray(score)[el]
    order=np.lexsort((el,-sc));return np.sort(el[order[:n]])
def rand_select(E,q,wseed,k):
    el=eligible(E);n=math.ceil(q*len(el));r=np.random.default_rng([int(wseed),int(k)]);return np.sort(r.choice(el,n,replace=False))
def prf(sel,gold):
    s=set(int(x) for x in sel);g=set(int(x) for x in gold);tp=len(s&g);return tp,len(s)-tp,len(g)-tp
class MCG(LD.Ladder):
    def build3(self):
        rt=self.rt;lr,clip=self.lr,self.clip;Batch=self.Batch
        def _resid(lp,tg_):p=jnp.exp(lp);py=jnp.take_along_axis(p,tg_[:,None],1)[:,0];return jnp.sqrt(jnp.maximum(jnp.sum(p*p,-1)-2*py+1.0,0.0))
        @eqx.filter_jit
        def masked_fwd(m,ids_,tg_,pref_,seg_,loc_):
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                r_=m.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt.state_suffix),seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
            finally:packing.PACK_SUF.clear()
            lp=jax.nn.log_softmax(r_.logits.astype(jnp.float32),-1);return -jnp.take_along_axis(lp,tg_[:,None],1)[:,0],_resid(lp,tg_)
        @eqx.filter_jit
        def native_fwd(m,st,c,p):
            r_=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=c);lp=jax.nn.log_softmax(r_.logits.astype(jnp.float32),-1)
            return r_.new_state,-jnp.take_along_axis(lp,c.target_tokens[:,None],1)[:,0],_resid(lp,c.target_tokens)
        @eqx.filter_jit
        def grad(m,ids_,tg_,mk_,pref_,seg_,loc_):
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                (_l,(ce,_n,_s)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_),clone_state(rt.state_suffix),prefix_outputs=pref_)
            finally:packing.PACK_SUF.clear()
            return grads.inner_parameters(),ce
        @eqx.filter_jit
        def nm_apply(m,st_all,g_m,g_s):
            gm=optax.global_norm(g_m);gs=optax.global_norm(g_s);coef=jnp.where(gs<clip,1.0,clip/jnp.maximum(gs,1e-30))
            s=lr*m.get_ilr_multiplier(st_all.get(m.step_index));f=coef*gs/jnp.maximum(gm,1e-30)
            return jax.tree.map(lambda p,x:(p+(-s*f*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g_m),gs,coef,gm
        @eqx.filter_jit
        def wstep(m,st_all,ids_,tg_,wt_,pref_,seg_,loc_):
            """mcgw: weighted mean NLL (weights wt_, sum 1) over positions, clipped step as the inner rule."""
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            def lossf(mm):
                r_=mm.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt.state_suffix),seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
                lp=jax.nn.log_softmax(r_.logits.astype(jnp.float32),-1);return jnp.sum(-jnp.take_along_axis(lp,tg_[:,None],1)[:,0]*wt_)
            try:g=eqx.filter_grad(lambda mm:lossf(mm))(m).inner_parameters()
            finally:packing.PACK_SUF.clear()
            gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
            return jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g),gn,coef
        self.masked_fwd=masked_fwd;self.native_fwd=native_fwd;self.grad=grad;self.nm_apply=nm_apply;self.wstep=wstep
    def run(self,swap,cell,q,E,diag=False,max_chunks=None):
        rt=self.rt;W=int(rt.cfg.model.sliding_window_size);Pw=self.T.P[swap];ids=np.asarray(Pw.ids);log=[];m=rt.base;st_all=clone_state(rt.state_all)
        seg=np.zeros(CH,np.int32);loc=np.arange(CH,dtype=np.int32);pm=jnp.asarray(packing.prefix_mask(seg,W));samenext=np.concatenate([np.ones(CH-1,bool),[False]])
        need_fwd=diag or cell in ('mcg','msurp','masknm','mcgw');need_nat=diag or cell in ('mcg','masknm','mcgw')
        gold=gold_tokens(self.T.w,swap,self.T.tok,rt.bos) if diag else None;stats=[];cum=0.0;D={'chunks':[]}
        nch=NCH if max_chunks is None else max_chunks
        with rt.mesh:
            log.append({'epoch':0,'margins':(None if diag else self.evaluate(m))})
            if need_nat:
                mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref_n=rt.prefix(rt.base,seq);st_suf=clone_state(rt.state_suffix)
            for k in range(nch):
                t0=time.monotonic()
                c=ids[k*CH:(k+1)*CH];tg=np.concatenate([ids[k*CH+1:(k+1)*CH+1],[rt.eos]])[:CH].astype(np.int32)
                pref=self.prefix_iso(rt.base,jnp.asarray(c),jnp.asarray(loc),pm);a_=(jnp.asarray(c),jnp.asarray(tg))
                nm_=rm_=nn_=rn_=None
                if need_fwd:nm_,rm_=(np.asarray(x) for x in self.masked_fwd(m,*a_,pref,jnp.asarray(seg),jnp.asarray(loc)))
                if need_nat:st_suf,nn_,rn_=self.native_fwd(m,st_suf,rt.chunk(seq,k),pref_n[k*CH:(k+1)*CH]);nn_=np.asarray(nn_);rn_=np.asarray(rn_)
                gt=None if nn_ is None else nm_-nn_
                if diag:
                    el=eligible('all');gl=[i for i in el if (k*CH+i) in gold];gp=[i for i in gl if gold[k*CH+i]['kind']=='place']
                    ch={'k':k,'g':np.round(gt[el],5).tolist(),'r':np.round((rn_/np.maximum(rm_,1e-30))[el],5).tolist(),'nll_m':np.round(nm_[el],5).tolist(),
                        'gold':[{'i':int(i),**gold[k*CH+i]} for i in gl],'sel':{}}
                    for q_ in QS:
                        for E_ in ES:
                            s_=select(gt,E_,q_);ch['sel'][f'{q_:g}|{E_}']={'prf':prf(s_,gp),'pos_hist':np.histogram(s_,bins=16,range=(0,CH))[0].tolist()}
                    D['chunks'].append(ch)
                mk=samenext.copy();un_extra={}
                if cell in ('mcg','msurp','mrand'):
                    s_=select(gt,E,q) if cell=='mcg' else (select(nm_,E,q) if cell=='msurp' else rand_select(E,q,self.T.w['seed'],k))
                    mk=np.zeros(CH,bool);mk[s_]=True;un_extra['n_sel']=int(len(s_))
                if E=='64' and cell=='masked':mk[:64]=False
                old=m
                if cell=='masknm':
                    s_=select(gt,E,q);ms=np.zeros(CH,bool);ms[s_]=True;mE=np.zeros(CH,bool);mE[eligible(E)]=True
                    gM,ce=self.grad(m,*a_,jnp.asarray(mE),pref,jnp.asarray(seg),jnp.asarray(loc));gS,_=self.grad(m,*a_,jnp.asarray(ms),pref,jnp.asarray(seg),jnp.asarray(loc))
                    newp,gn,coef,gmn=self.nm_apply(m,st_all,gM,gS);del gM,gS;un_extra['g_masked_norm']=float(gmn);un_extra['n_sel']=int(len(s_))
                elif cell=='mcgw':
                    wt=np.zeros(CH,np.float32);el=eligible(E);wv=np.maximum(gt[el],0);wt[el]=wv/max(wv.sum(),1e-30);newp,gn,coef=self.wstep(m,st_all,*a_,jnp.asarray(wt),pref,jnp.asarray(seg),jnp.asarray(loc));ce=jnp.float32(np.nan)
                else:
                    newp,gn,coef,ce=self.step(m,st_all,jnp.asarray(c),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc),jnp.float32(1.0))
                m=self.set_inner(m,newp)
                un=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=un
                stats.append({'k':k,'gnorm':float(gn),'coef':float(coef),'update_norm':un,'ce':float(ce),'sec':time.monotonic()-t0,**un_extra})
                print(f'progress {cell} {"B" if swap else "A"} chunk {k+1}/{nch} {stats[-1]["sec"]:.1f}s',flush=True)   # timing only
            cs=[s_['coef'] for s_ in stats]
            log.append({'epoch':1,'cum_update_norm':cum,'chunks':stats,'clip_bind_frac':sum(c_<1 for c_ in cs)/len(cs),'mean_coef':sum(cs)/len(cs),
                        'stream_hash':hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),'inner_hash':reuse._hash_pytree(m.inner_parameters())[0],
                        'margins':(None if diag or max_chunks else self.evaluate(m))})
        return log,D
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=['masked','mcg','msurp','mrand','masknm','mcgw'],default='masked');ap.add_argument('--q',type=float,default=1.0)
    ap.add_argument('--elig',choices=ES,default='all');ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--worlds',default='AB')
    ap.add_argument('--mode',choices=['save','eval'],default='save');ap.add_argument('--diag',action='store_true');ap.add_argument('--max-chunks',type=int);ap.add_argument('--keep',action='store_true')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    if a.diag:assert a.cell=='masked' and a.q==1.0 and a.elig=='all'
    tag=('diag' if a.diag else f'{a.cell}_q{a.q:g}_{a.elig}')+(f'_smoke{a.max_chunks}' if a.max_chunks else '');lev='MCG';rt=Runtime();SD=OUT/'states'/f'{lev}_{tag}_s{a.world_seed}'
    L=MCG(rt,a.world_seed,readout=(a.mode!='save'),save_dir=SD);L.build3()
    if not a.max_chunks:json.dump({'items':L.items,'lr':L.lr,'clip':L.clip,'cell':a.cell,'q':a.q,'elig':a.elig,'xla_flags':os.environ.get('XLA_FLAGS')},open(OUT/f'meta_{lev}_{tag}_s{a.world_seed}.json','w'),indent=1)
    for w in a.worlds:
        f=OUT/f'{lev}_{tag}_s{a.world_seed}_{w}.json';fs=OUT/f'{lev}_{tag}_s{a.world_seed}_{w}.saved.json'
        if f.exists():print('exists',f);continue
        if a.mode=='eval':
            d=json.load(open(fs));t0=time.monotonic()
            for e in d['log']:
                if isinstance(e.get('margins'),dict) and '_file' in e['margins']:
                    fn=e['margins']['_file'];h=e['margins']['_hash'];mm=L.load_state(fn)
                    assert reuse._hash_pytree(mm.inner_parameters())[0]==h;e['margins']=L.evaluate(mm);e['hash']=h
                    if not a.keep:os.remove(fn)
            d['sec_eval']=time.monotonic()-t0;json.dump(d,open(f,'w'));print('scored',f,round(d['sec_eval']),'s',flush=True);continue
        L.save_dir=SD/w;L._nsave=0;(SD/w).mkdir(parents=True,exist_ok=True)
        if fs.exists():print('exists',fs);continue
        t0=time.monotonic();log,D=L.run(w=='B',a.cell,a.q,a.elig,diag=a.diag,max_chunks=a.max_chunks)
        if a.max_chunks:print('smoke ok',w,'chunks',len(log[-1]['chunks']),'inner_hash',log[-1]['inner_hash'][:16],round(time.monotonic()-t0),'s',flush=True);continue
        if a.diag:
            rf=OUT/f'L0mw_x1_s{a.world_seed}_{w}.saved.json';ref=json.load(open(rf))['log'][1]['margins']['_hash'] if rf.exists() else None;D['q1_hash']=log[-1]['inner_hash'];D['L0mw_x1_pass1_hash']=ref;D['q1_bit_identical']=None if ref is None else (ref==log[-1]['inner_hash'])
            print('q=1 bit identity vs L0mw_x1 pass 1:',D['q1_bit_identical'],flush=True)
            json.dump({'log':log,'diag':D,'sec':time.monotonic()-t0},open(OUT/f'{lev}_{tag}_s{a.world_seed}_{w}.diag.json','w'));continue
        json.dump({'log':log,'sec':time.monotonic()-t0,'cell':a.cell,'q':a.q,'elig':a.elig},open(fs,'w'));print('saved',fs,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
