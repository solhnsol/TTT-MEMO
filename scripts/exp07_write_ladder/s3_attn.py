"""S3 attention-based cells (criterion (2) of spec v1.1; definitions of 10/17 17:09 and 00:31 KST): attnsel, randattn, recencyattn. Separate file so the running S3 jobs (s3.py) stay untouched.
attnsel      per chunk, ONLINE at the current weights: the natural read forward's attention (suffix layers, targets = all loss positions of the chunk, head-max of the layer-mean probability per target),
             the per-target top-64 keys among the candidates (window before the chunk minus the first 4 stream positions and the last 128 tokens before the chunk), UNION over the targets = the masked set S
             (A head-max, k = 64, the policy selected on s101 in stage 1). The write forward is the S3 masked write (all layers), the state advance the natural read.
randattn     |S_attnsel| random positions (not O tokens) in the window before the chunk with the same distance-bin histogram as the attnsel set of the same (seed, world, pass, chunk), read from the attnsel cell's file.
recencyattn  the |S_attnsel| positions immediately before the chunk (same count read from the attnsel file).
Usage: s3_attn.py --cell attnsel|randattn|recencyattn --world-seed 101 --passes 4 --mode save|eval [--keep] [--worlds AB] ; --unit: sizes at W0 on a few chunks (no D)."""
import os,sys,json,time,argparse,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import s3_stage1 as ST
from s3_stage1 import *
import s3 as S3M
class Ladder3A(ST.Stage1):
    K_ATTN=64
    def build_attn(self):
        rt=self.rt;nh=32;KA=self.K_ATTN
        @eqx.filter_jit
        def capture_union(m,st_suf,c,pref_c,cand):
            blocks=m.language_model.model.h.suffix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0]
            ST.S3REC['on']=True;ST.S3REC['P']=jnp.zeros((nh,CH,WIN+CH),jnp.float32);ST.S3REC['LS']=jnp.zeros((WIN+CH,),jnp.float32);ST.S3REC['CNT']=jnp.zeros((WIN+CH,),jnp.float32)
            try:
                h=pref_c
                for l in range(nl):
                    blk=jax.tree.map(lambda a:a[l] if eqx.is_array(a) else a,blocks);stl=jax.tree.map(lambda s:s[l],st_suf);h,_=blk(h,stl,c,is_prefix=False)
                Pm=ST.S3REC['P']/nl
            finally:ST.S3REC.clear()
            M=jnp.max(Pm,axis=0)[:,:WIN];M=jnp.where(cand[None,:],M,-1.0);_,idx=jax.lax.top_k(M,KA)   # [CH, KA], ties -> lower index
            valid=jnp.take_along_axis(jnp.broadcast_to(cand[None,:],M.shape),idx,axis=1)
            return jnp.zeros((WIN,),bool).at[idx.reshape(-1)].max(valid.reshape(-1))
        def make_union(KA_):
            @eqx.filter_jit
            def f(m,st_suf,c,pref_c,cand):
                blocks=m.language_model.model.h.suffix_blocks;nl=jax.tree_util.tree_leaves(blocks)[0].shape[0]
                ST.S3REC['on']=True;ST.S3REC['P']=jnp.zeros((nh,CH,WIN+CH),jnp.float32);ST.S3REC['LS']=jnp.zeros((WIN+CH,),jnp.float32);ST.S3REC['CNT']=jnp.zeros((WIN+CH,),jnp.float32)
                try:
                    h=pref_c
                    for l in range(nl):
                        blk=jax.tree.map(lambda a:a[l] if eqx.is_array(a) else a,blocks);stl=jax.tree.map(lambda s:s[l],st_suf);h,_=blk(h,stl,c,is_prefix=False)
                    Pm=ST.S3REC['P']/nl
                finally:ST.S3REC.clear()
                Mx=jnp.max(Pm,axis=0)[:,:WIN];Mx=jnp.where(cand[None,:],Mx,-1.0);_,idx=jax.lax.top_k(Mx,KA_)
                valid=jnp.take_along_axis(jnp.broadcast_to(cand[None,:],Mx.shape),idx,axis=1)
                union=jnp.zeros((WIN,),bool).at[idx.reshape(-1)].max(valid.reshape(-1))
                A=jnp.where(cand,jnp.max(jnp.max(Pm,axis=0)[:,:WIN],axis=0),0.0)   # per candidate: head-max attention (max over heads of the layer-mean probability) maxed over the chunk's targets = the score whose top-k union the deterministic cell uses; the stochastic cell samples proportional to it (design 00:47)
                return union,A
            return f
        self.capture_union=capture_union;self.capture_union16=make_union(16);self.capture_union64=make_union(64)
    def cand_vec(self,k):
        W=WIN;pos=k*CH-W+np.arange(W);return jnp.asarray((pos>=4)&(pos<k*CH-128)&(pos>=0))
    def attn_set(self,cell,m,st_suf,c,pref_c,k,sets,ref,seed_=0):
        k0=k*CH;W=WIN
        if cell=='attnk16':   # deterministic k = 16 union (descriptive, s101)
            u,_=self.capture_union16(m,st_suf,c,pref_c,self.cand_vec(k));S=k0-W+np.nonzero(np.asarray(u))[0]
            bins=np.bincount((k0-S)//SETS.BIN,minlength=W//SETS.BIN).tolist() if len(S) else [0]*(W//SETS.BIN);return S,1.0,bins
        if cell=='attnstoch':   # stochastic (tau = 1, proportional to the head-max score), resampled every pass (descriptive, s101): n = the size of the k = 64 union at this chunk
            u,A=self.capture_union64(m,st_suf,c,pref_c,self.cand_vec(k));n=int(np.asarray(u).sum());A=np.asarray(A,np.float64);cand=np.asarray(self.cand_vec(k))
            if n==0 or A.sum()<=0:return np.zeros(0,int),1.0,[0]*(W//SETS.BIN)
            n=min(n,int((A>0).sum()));rng=np.random.default_rng([int(seed_),int(sets.passno),int(k)]);S=k0-W+np.sort(rng.choice(W,size=n,replace=False,p=A/A.sum()))
            bins=np.bincount((k0-S)//SETS.BIN,minlength=W//SETS.BIN).tolist();return S,1.0,bins
        if cell=='attnsel':
            u=np.asarray(self.capture_union(m,st_suf,c,pref_c,self.cand_vec(k)));S=k0-W+np.nonzero(u)[0]
            bins=np.bincount((k0-S)//SETS.BIN,minlength=W//SETS.BIN).tolist() if len(S) else [0]*(W//SETS.BIN);return S,1.0,bins
        bins=ref['bins'];n=int(ref['n_masked'])
        if n==0:return np.zeros(0,int),1.0,bins
        if cell=='recencyattn':return np.arange(max(0,k0-n),k0),1.0,bins
        if cell=='randattn':
            Oset=set(sets.oracle(k));lo=max(0,k0-W);pool={}
            for p in range(lo,k0):
                if p not in Oset:pool.setdefault((k0-p)//SETS.BIN,[]).append(p)
            need=[b for b,cnt in enumerate(bins) for _ in range(cnt)]
            ch,fr=SETS._fill(need,pool,random.Random(f'{sets.seed}|{sets.passno}|{k}|randattn'));return np.asarray(ch,int),fr,bins
        raise ValueError(cell)
    def run_cell(self,swap,cell,passes=1,nch=None,seed=0):
        rt=self.rt;log=[];m=rt.base;st_all=clone_state(rt.state_all);streams,worlds=self.streams_worlds(swap,passes);nch=nch or NCH;cum=0.0
        refd=None
        if cell in ('randattn','recencyattn'):
            refd=json.load(open(OUT/f'S3_attnsel_p{passes}_s{seed}_{"B" if swap else "A"}.saved.json'))['log']
        with rt.mesh:
            log.append({'epoch':0,'margins':self.ev(m)})
            for ps in range(1,passes+1):
                ids=np.asarray(streams[ps-1])[:NCH*CH];stats=[];t_pass=time.monotonic()
                mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);pref_nat=rt.prefix(rt.base,seq);st_suf=clone_state(rt.state_suffix)
                t0=time.monotonic();Kh,Vh,_=self.record(ids);print(f'  recorded natural K,V in {time.monotonic()-t0:.0f}s',flush=True)
                sets=SETS.Sets(SETS.sentence_table(worlds[ps-1],swap,self.T.tok,gw,SP),seed,ps)
                for k in range(nch):
                    t0=time.monotonic();old=m;c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH]
                    ref=refd[ps]['chunks'][k] if refd is not None else None
                    S,fr,bins=self.attn_set(cell,m,st_suf,c,pref_c,k,sets,ref,seed);keep=sets.keep_vector(k,S);Kc,Vc=self.ctx_kv(Kh,Vh,k)
                    pref_w=self.prefix_masked(m,jnp.asarray(ids[k*CH:(k+1)*CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep))
                    newp,gn,coef,ce=self.write_step(m,st_all,st_suf,c,pref_w,jnp.asarray(keep));st_suf=self.read_state(m,st_suf,c,pref_c)
                    m=self.set_inner(m,newp)
                    un=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=un
                    O_=sets.oracle(k);candn=np.asarray(self.cand_vec(k));Oc=set(p_ for p_ in O_ if k*CH-WIN<=p_<k*CH and candn[p_-(k*CH-WIN)]);Ss=set(int(x) for x in S);tp_=len(Ss&Oc)
                    stats.append({'k':k,'gnorm':float(gn),'coef':float(coef),'update_norm':un,'ce':float(ce),'sec':time.monotonic()-t0,'n_masked':int(len(S)),'match_frac':float(fr),'bins':bins,
                                  'S':[int(x) for x in S],'n_O':len(O_),'n_O_cand':len(Oc),'tp':tp_,'fp':len(Ss)-tp_,'fn':len(Oc)-tp_})   # masked set stored; TP / FP / FN against the selectable O tokens (descriptive F1)
                    if k%8==0:print(f'  {cell} {"B" if swap else "A"} pass {ps} chunk {k+1}/{nch} {stats[-1]["sec"]:.1f}s masked {len(S)}',flush=True)
                cs=[s_['coef'] for s_ in stats]
                log.append({'epoch':ps,'cum_update_norm':cum,'chunks':stats,'clip_bind_frac':sum(c_<1 for c_ in cs)/len(cs),'mean_coef':sum(cs)/len(cs),'n_masked_total':sum(s_['n_masked'] for s_ in stats),
                            'stream_hash':hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),'inner_hash':reuse._hash_pytree(m.inner_parameters())[0],'sec_pass':time.monotonic()-t_pass,'margins':self.ev(m)})
                print(f'  {cell} {"B" if swap else "A"} pass {ps}: cum upd {cum:.3f}, clip binds {log[-1]["clip_bind_frac"]:.3f}, masked tokens {log[-1]["n_masked_total"]}, {log[-1]["sec_pass"]:.0f}s',flush=True)
                del Kh,Vh
        return log
    def unit_attn(self,seed):
        """at W0 (natural chain, no update) on pass 1 / world A: the attnsel set size per eligible chunk and the match with the stage-1 numbers (mean masked tokens ~4031 on s101 for k = 64), the matched cells' sizes."""
        rt=self.rt;m=rt.base;ids=np.asarray(self.T.P[False].ids)[:NCH*CH];mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);rows=[]
        with rt.mesh:
            pref_nat=rt.prefix(rt.base,seq);sets=SETS.Sets(SETS.sentence_table(self.T.w,False,self.T.tok,gw,SP),seed,1);st=clone_state(rt.state_suffix)
            for k in range(NCH):
                c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH]
                if self.answers(sets,k):
                    S,fr,bins=self.attn_set('attnsel',m,st,c,pref_c,k,sets,None);ref={'n_masked':len(S),'bins':bins};R,fr2,_=self.attn_set('randattn',m,st,c,pref_c,k,sets,ref);Rc,_,_=self.attn_set('recencyattn',m,st,c,pref_c,k,sets,ref)
                    rows.append(dict(k=k,n_attn=len(S),n_rand=len(R),rand_bin_match=fr2,n_recency=len(Rc),O_overlap_attn=len(set(S)&set(sets.oracle(k))),n_O=len(sets.oracle(k)),rand_overlap_O=len(set(R)&set(sets.oracle(k)))));print(rows[-1],flush=True)
                st=self.read_state(m,st,c,pref_c)
        return rows
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=['attnsel','randattn','recencyattn','attnk16','attnstoch'],default='attnsel');ap.add_argument('--world-seed',type=int,default=101)
    ap.add_argument('--worlds',default='AB');ap.add_argument('--mode',choices=['save','eval'],default='save');ap.add_argument('--tag',default='');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--keep',action='store_true')
    ap.add_argument('--canary-only',action='store_true');ap.add_argument('--smoke-chunks',type=int,default=None);ap.add_argument('--unit',action='store_true')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime()
    if a.unit:
        L=Ladder3A(rt,a.world_seed,readout=False);L.build3();L.build1();L.build_attn();rows=L.unit_attn(a.world_seed);json.dump(rows,open(OUT/f's3_attn_unit_s{a.world_seed}.json','w'),indent=1)
        import numpy as _np;print('mean attn-sel size',_np.mean([r['n_attn'] for r in rows]),'chunks',len(rows));return
    tag=a.tag or f'{a.cell}_p{a.passes}';lev='S3';SD=OUT/'states'/f'{lev}_{tag}_s{a.world_seed}'
    L=Ladder3A(rt,a.world_seed,readout=(a.mode!='save'),save_dir=SD);L.build3();L.build1();L.build_attn();L.canary_only=a.canary_only
    if a.smoke_chunks:assert a.canary_only and 'smoke' in tag
    if not a.smoke_chunks:json.dump({'items':L.items,'fam_sizes':{k:len(v) for k,v in L.fam.items()},'lr':L.lr,'clip':L.clip,'cell':a.cell,'passes':a.passes,'xla_flags':os.environ.get('XLA_FLAGS')},open(OUT/f'meta_{lev}_{tag}_s{a.world_seed}.json','w'),indent=1)
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
        t0=time.monotonic();log=L.run_cell(w=='B',a.cell,a.passes,a.smoke_chunks,seed=a.world_seed)
        json.dump({'log':log,'sec':time.monotonic()-t0,'cell':a.cell,'passes':a.passes},open(fs,'w'));print('saved',fs,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
