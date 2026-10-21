"""Exp07 write ladder, L0 vs L1 (seed 101 clean-v2 worlds A/B, same tokens, same W0, same inner parameters).
L0  native TTT-E2E: online stream, chunk 1024, SWA 8K context, one clipped SGD step per chunk (inner rule), suffix MLP.
L1  offline NTP fine-tuning on the SAME exposure sentences, each isolated (no other sentence in context: no shadowing,
    no filler), shuffled every epoch, many epochs; same inner parameters and the same step rule (SGD with the config lr
    x ilr multiplier, global-norm clip 1.0, mean token loss over a 1024-token chunk of packed isolated sentences).
Readout: R1 (neutral tail) through the packed scorer (scripts/exp05_write_dynamics/packing.py), used only after
verify_packing.py has passed. Families: L3 (binding), XP (cross-pair null), PS (pseudo null), FN/FP (familiarity of
exposed names/places vs never-exposed foils), FN0/FP0 (familiarity nulls), L3r, L2 (secondary relation directions).
Outputs per evaluation point: per-world item margins; analysis in analyze_ladder.py. e2e/ is not modified."""
import os,sys,json,time,argparse,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp05_write_dynamics'))
import readouts as RD
RD.FAMILIARITY=True
from readouts import *
import packing,optax
OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'))

# ---- fp32 readout side (bf16 readout carries ~0.05 nat/candidate rounding noise; see verify_packing*.json) ----
_dpa=jax.nn.dot_product_attention
def _dpa_fp32_xla(*a,**k):
    """cuDNN cannot run fp32: redirect only fp32 calls to XLA; the bf16 stream keeps cuDNN."""
    if k.get('implementation')=='cudnn' and a[0].dtype==jnp.float32:k=dict(k,implementation='xla')
    return _dpa(*a,**k)
jax.nn.dot_product_attention=_dpa_fp32_xla
def runtime_fp32():
    orig=reuse.config3b.compose_3b_128k_config
    def comp():
        cfg=orig();cfg.model.compute_dtype='fp32';return cfg
    reuse.config3b.compose_3b_128k_config=comp
    try:return Runtime()
    finally:reuse.config3b.compose_3b_128k_config=orig
class _TailCtx:
    """what packing.Packed needs from a Readouts object, built on the fp32 runtime."""
    def __init__(self,rt32,tail_ids):
        self.rt=rt32;self.tail_ids=tail_ids
        with rt32.mesh,jax.default_matmul_precision('highest'):
            seq=rt32.seq(tail_ids+[rt32.eos]*CH);self._tail_seq=seq;self._tail_pref=rt32.prefix(rt32.base,seq)[:len(tail_ids)]

def squad_tail(tok,bos,forbid):
    """alternative readout context: SQuAD dev Wikipedia paragraphs (expository, unlike the fiction stream and the
    pg1342 tail), BOS + 7167 tokens; checked to contain none of the world's names."""
    d=json.load(open(ROOT/'data/squad/dev-v1.1.json'));paras=[p['context'] for a in d['data'] for p in a['paragraphs']]
    paras=[' '.join(x.split()) for x in paras[:400]];paras=[x for x in paras if not any(nm in x for nm in forbid)]
    text='\n\n'.join(paras);ids=[bos]+stage2.enc(tok,text).ids
    for nm in forbid:assert nm not in text,nm
    return ids[:8192-CH]
class Ladder:
    def __init__(self,rt,seed=101,readout=True,save_dir=None):
        """readout=False: streaming-only process (no fp32 model, no packed scorer); evaluate() then saves the inner
        weights to save_dir and a separate process scores them (the 64K stream + update + fp32 readout OOM together)."""
        self.rt=rt;self.readout=readout;self.save_dir=save_dir;self._nsave=0;self.T=T=RD.Readouts(rt,seed);w=T.w;name=T.P[False].name
        extra=[it for it in gw.probes(w,name) if it['level'] in ('L3r','L2')]
        self.fam=dict(T.fam);self.fam['L3r']=[it for it in extra if it['level']=='L3r'];self.fam['L2']=[it for it in extra if it['level']=='L2']
        self.items=[it for v in self.fam.values() for it in v];raw=[]
        for it in self.items:
            for c in (it['gold'],it['foil']):
                ids,span=stage2.item_ids(T.tok,stage2.make_slot(T.tok,None),it['prefix'],c);raw.append((ids,(span[0]-1,span[1]-1)))
        if readout:
            self.rt32=rt32=runtime_fp32();self.ctx=_TailCtx(rt32,T.tail_ids)
            with jax.default_matmul_precision('highest'):self.P=packing.Packed(self.ctx,raw,GP=2)
            self.f32=reuse.get_filter_spec(rt32.base,rt32.cfg.training.spec_inner,'inner parameters')
            forbid=[t['X'] for t in w['triples']]+[t['P'] for t in w['triples']]+list(w['ctrl_persons'])+list(w['ctrl_places'])
            self.ctx_alt=_TailCtx(rt32,squad_tail(T.tok,rt.bos,forbid))
            with jax.default_matmul_precision('highest'):self.P_alt=packing.Packed(self.ctx_alt,raw,GP=2)
        cfg=rt.cfg.training.optimizer_inner;self.lr=float(cfg.lr);self.clip=float(cfg.clip_gradient)
        self.f=reuse.get_filter_spec(rt.base,rt.cfg.training.spec_inner,'inner parameters')
        from ttt.model.data import Batch
        lr,clip=self.lr,self.clip
        @eqx.filter_jit
        def prefix_iso(m,ids_,pos_,mask_):
            packing.PACK_PRE['mask']=mask_
            try:
                seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
                return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),clone_state(rt.state_prefix),seq).last_hidden_state
            finally:packing.PACK_PRE.clear()
        @eqx.filter_jit
        def step(m,st_all,ids_,tg_,mk_,pref_,seg_,loc_):
            """one inner-rule step on a chunk of packed isolated sentences (empty suffix KV, no carry)."""
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_)
                (_l,(ce,_n,_s)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,seq,clone_state(rt.state_suffix),prefix_outputs=pref_)
            finally:packing.PACK_SUF.clear()
            g=grads.inner_parameters();gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30))
            s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
            newp=jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g)
            return newp,gn,coef,ce
        @eqx.filter_jit
        def native_scaled(m,st_all,st_suf,c,p,scale):
            """the native inner step re-implemented (as Exp05 V2) with the step size multiplied by `scale` after clipping."""
            (_l,(ce,_n,new_st)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,c,st_suf,prefix_outputs=p)
            g=grads.inner_parameters();gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30))
            s_=lr*m.get_ilr_multiplier(st_all.get(m.step_index))*scale
            newp=jax.tree.map(lambda q,x:(q+(-s_*coef*x).astype(q.dtype)).astype(q.dtype),m.inner_parameters(),g)
            return newp,new_st
        self.prefix_iso=prefix_iso;self.step=step;self.native_scaled=native_scaled
    def set_inner(self,m,newp):
        inner,outer=eqx.partition(m,self.f)
        assert jax.tree_util.tree_structure(inner)==jax.tree_util.tree_structure(newp)
        return eqx.combine(newp,outer)
    def to_fp32(self,m):
        """same weight values in the fp32-compute model (params are stored fp32 in both)."""
        src=jax.tree_util.tree_leaves(eqx.partition(m,self.f)[0]);inner32,outer32=eqx.partition(self.rt32.base,self.f32)
        td=jax.tree_util.tree_structure(inner32);assert td.num_leaves==len(src)
        return eqx.combine(jax.tree_util.tree_unflatten(td,src),outer32)
    def _gen_setup(self):
        """held-out general text (pg11, not a stream or tail book), 8 chunks, for the overshoot check."""
        rt=self.rt
        t=(ROOT/'data/narrative_gutenberg/pg11.txt').read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t
        t=' '.join(t.replace('\r','').split());ids=([rt.bos]+stage2.enc(self.T.tok,t).ids)[:8*CH];assert len(ids)==8*CH
        with rt.mesh:self._gseq=rt.seq(ids);self._gpref=rt.prefix(rt.base,self._gseq)
    def gen_nll(self,m):
        rt=self.rt
        if not hasattr(self,'_gseq'):self._gen_setup()
        with rt.mesh:
            st=clone_state(rt.state_suffix);tot=[]
            for j in range(8):
                st,lp=rt.chunk_logp(m,st,rt.chunk(self._gseq,j),self._gpref[j*CH:(j+1)*CH]);tot.append(-np.asarray(lp,np.float64))
        return float(np.concatenate(tot)[:-1].mean())
    def tail_nll(self,m):
        """second canary: the name-scrubbed neutral tail (pg1342, 7 chunks); pg11 turned out to be memorized (NLL ~0.19)."""
        rt=self.rt
        if not hasattr(self,'_tseq'):
            with rt.mesh:self._tseq=rt.seq(self.T.tail_ids);self._tpref=rt.prefix(rt.base,self._tseq)
        with rt.mesh:
            st=clone_state(rt.state_suffix);tot=[]
            for j in range(len(self.T.tail_ids)//CH):
                st,lp=rt.chunk_logp(m,st,rt.chunk(self._tseq,j),self._tpref[j*CH:(j+1)*CH]);tot.append(-np.asarray(lp,np.float64))
        return float(np.concatenate(tot)[:-1].mean())
    def save_state(self,m):
        leaves=[np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.partition(m,self.f)[0])]
        fn=Path(self.save_dir)/f'state_{self._nsave:03d}.npz';self._nsave+=1;np.savez(fn,*leaves)
        return {'_file':str(fn),'_hash':reuse._hash_pytree(m.inner_parameters())[0]}
    def load_state(self,fn):
        z=np.load(fn);leaves=[jnp.asarray(z[f'arr_{i}']) for i in range(len(z.files))]
        inner,outer=eqx.partition(self.rt.base,self.f);td=jax.tree_util.tree_structure(inner);assert td.num_leaves==len(leaves)
        return eqx.combine(jax.tree_util.tree_unflatten(td,leaves),outer)
    def evaluate(self,m):
        if not self.readout:return self.save_state(m)
        m32=self.to_fp32(m)
        with jax.default_matmul_precision('highest'):mg=self.P.r1(m32);ma=self.P_alt.r1(m32)
        gn_=self.gen_nll(m);mag,maf=ma[0::2],ma[1::2];mgg,mgf=mg[0::2],mg[1::2];ma=mag-maf;alt={};altg={};altf={};j=0
        for f,its in self.fam.items():alt[f]=ma[j:j+len(its)].tolist();altg[f]=mag[j:j+len(its)].tolist();altf[f]=maf[j:j+len(its)].tolist();j+=len(its)
        mg=mgg-mgf;out={};outg={};outf={};i=0
        for f,its in self.fam.items():out[f]=mg[i:i+len(its)].tolist();outg[f]=mgg[i:i+len(its)].tolist();outf[f]=mgf[i:i+len(its)].tolist();i+=len(its)
        out['_gen_nll']=gn_;out['_tail_nll']=self.tail_nll(m);out['_alt']=alt
        out['_gold']=outg;out['_foil']=outf;out['_alt_gold']=altg;out['_alt_foil']=altf   # E47: the summed log-probs of the gold / foil span themselves (margin = gold - foil); additive fields, the margins above are unchanged
        return out
    def sentences(self,swap):
        """the exposure sentences of world A/B exactly as realized in the stream, tokenized alone ('\\n\\n' + sentence)."""
        _,_,spans=gw.realize(self.T.w,swap);return [stage2.enc(self.T.tok,'\n\n'+s['text']).ids for s in spans]
    def l1_rows(self,swap,variant):
        """training rows (ids, (a,b)) for L1 variants; loss positions a..b-1 (position i predicts token i+1).
        plain: each exposure sentence alone. ctx: the previous round's two pair sentences + the current sentence, loss on
        the current sentence only (offline write WITH in-context shadowing). fill: plain rows plus the world's own background
        paragraphs (the filler of the native stream) as further isolated rows with loss on all their tokens."""
        _,_,spans=gw.realize(self.T.w,swap);enc=lambda t:stage2.enc(self.T.tok,t).ids;tri={t['id']:t for t in self.T.w['triples']}
        rows=[]
        if variant in ('plain','fill'):rows=[(x,(0,len(x)-1)) for x in (enc('\n\n'+s_['text']) for s_ in spans)]
        if variant=='ctx':
            by={}
            for s_ in spans:by.setdefault((tri[s_['t']]['pair'],s_['round']),[]).append(s_['text'])
            for s_ in spans:
                cur=enc('\n\n'+s_['text']);r=s_['round'];pr=tri[s_['t']]['pair']
                ctx=enc('\n\n'+'\n\n'.join(by[(pr,r-1)])) if r>=2 else []
                rows.append((ctx+cur,(max(len(ctx)-1,0),len(ctx)+len(cur)-1)))
        if variant=='fill':
            for t in self.T.w['texts']:
                if t['kind']=='bg':x=enc('\n\n'+t['text']);rows.append((x[:1024],(0,min(len(x),1024)-1)))
        return rows
    def run_L1(self,swap,epochs,seed=0,eval_every=1,save_epochs=(),save_dir=None,variant='plain'):
        rt=self.rt;W=int(rt.cfg.model.sliding_window_size);rows=self.l1_rows(swap,variant);rng=random.Random(seed);log=[]
        m=rt.base;st_all=clone_state(rt.state_all)
        with rt.mesh:
            log.append({'epoch':0,'step':0,'margins':self.evaluate(m)})
            nstep=0;cum=0.0
            for ep in range(1,epochs+1):
                order=list(range(len(rows)));rng.shuffle(order)
                chunks,where=packing.pack_rows([rows[i] for i in order]);mks=[np.zeros(1024,bool) for _ in chunks]
                for c_,a_,b_ in where:mks[c_][a_:b_]=True   # loss positions inside each row (plain rows: all in-row targets)
                for (ids,seg,loc),mk in zip(chunks,mks):
                    tg=np.concatenate([ids[1:],[rt.eos]]).astype(np.int32)
                    pref=self.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
                    newp,gn,coef,ce=self.step(m,st_all,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
                    old=m;m=self.set_inner(m,newp);nstep+=1
                    d=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())));cum+=d**0.5
                    log.append({'epoch':ep,'step':nstep,'grad_norm':float(gn),'clip_coef':float(coef),'ce':float(ce),'update_norm':d**0.5,'cum_update_norm':cum})
                if ep in save_epochs:
                    Path(save_dir).mkdir(parents=True,exist_ok=True);leaves=[np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.partition(m,self.f)[0])]
                    np.savez(Path(save_dir)/f'epoch_{ep:03d}.npz',*leaves)
                if ep%eval_every==0:log.append({'epoch':ep,'step':nstep,'margins':self.evaluate(m)});print(f'  L1 {"B" if swap else "A"} epoch {ep} steps {nstep} ce {log[-2]["ce"]:.3f}',flush=True)
        return log
    def run_L0_multipass(self,swap,passes):
        """native TTT re-reading the same stream `passes` times: weights carry over, suffix KV restarts at each pass
        (a fresh read of the document); evaluation at the end of every pass (epoch = pass, comparable to L1 epochs)."""
        rt=self.rt;Pw=self.T.P[swap];ids=Pw.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);log=[]
        with rt.mesh:
            pref=rt.prefix(rt.base,seq);m=rt.base;opt=m.inner_optimizer(clone_state(rt.state_all)).init(m.inner_parameters());st_all=clone_state(rt.state_all);cum=0.0
            log.append({'epoch':0,'margins':self.evaluate(m)})
            for ps in range(1,passes+1):
                st=(st_all,clone_state(rt.state_suffix))
                for k in range(NCH):
                    old=m;r=rt.update(m,opt,st,rt.chunk(seq,k),pref[k*CH:(k+1)*CH]);m,opt,st=r.new_model,r.new_optimizer_state,r.new_state;del r
                    cum+=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5
                st_all=st[0]
                log.append({'epoch':ps,'cum_update_norm':cum,'margins':self.evaluate(m)});print(f'  L0mp {"B" if swap else "A"} pass {ps}',flush=True)
        return log
    def _native_passes(self,streams,tag):
        """native TTT over a list of token streams (one per pass); weights carry over, KV restarts each pass."""
        rt=self.rt;log=[];m=rt.base;opt=m.inner_optimizer(clone_state(rt.state_all)).init(m.inner_parameters());st_all=clone_state(rt.state_all);cum=0.0
        with rt.mesh:
            log.append({'epoch':0,'margins':self.evaluate(m)})
            for ps,ids in enumerate(streams,1):
                n=len(ids);pad=(-n)%CH;full=list(ids)+[rt.eos]*pad;mask=np.zeros(len(full),dtype=bool);mask[:n-1]=True
                seq=rt.seq(full,mask=mask);pref=rt.prefix(rt.base,seq);st=(st_all,clone_state(rt.state_suffix))
                for k in range(len(full)//CH):
                    old=m;r=rt.update(m,opt,st,rt.chunk(seq,k),pref[k*CH:(k+1)*CH]);m,opt,st=r.new_model,r.new_optimizer_state,r.new_state;del r
                    cum+=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5
                st_all=st[0];del pref
                log.append({'epoch':ps,'chunks':len(full)//CH,'cum_update_norm':cum,'margins':self.evaluate(m)});print(f'  {tag} pass {ps}',flush=True)
        return log
    def run_L0_oracle(self,swap,passes):
        """(a) oracle: pass 1 = the full native stream; passes 2.. re-read ONLY the exposure sentences, in stream order,
        as one continuous stream (window context kept, filler removed)."""
        Pw=self.T.P[swap];_,_,spans=gw.realize(self.T.w,swap)
        expo=[self.rt.bos]+stage2.enc(self.T.tok,'\n\n'.join(s['text'] for s in spans)).ids
        return self._native_passes([Pw.ids]+[expo]*(passes-1),'L0or')
    def run_L0_fillerswap(self,swap,passes):
        """(b) every pass reads the same exposure sentences at the same schedule positions in the text list, with the
        background paragraphs replaced by a different draw (pass 1 = the original stream)."""
        import copy
        Pw=self.T.P[swap];streams=[Pw.ids];w0=self.T.w
        for ps in range(2,passes+1):
            w=copy.deepcopy(w0);pool=gw.background(random.Random(7000+ps));bi=0;new=[]
            for t in w['texts']:
                if t['kind']=='bg':new.append({'kind':'bg','text':pool[bi%len(pool)]});bi+=1
                else:new.append(t)
            w['texts']=new;ids=[self.rt.bos]+stage2.enc(self.T.tok,gw.realize(w,swap)[0]).ids
            while len(ids)<NCH*CH:
                w['texts'].append({'kind':'bg','text':pool[bi%len(pool)]});bi+=1;ids=[self.rt.bos]+stage2.enc(self.T.tok,gw.realize(w,swap)[0]).ids
            streams.append(ids[:NCH*CH])
        return self._native_passes(streams,'L0fs')
    def run_L0_ksteps(self,swap,K,eval_every=16):
        """one pass, K native inner steps on every chunk (same chunk, same incoming KV); KV for the next chunk from the
        last step's forward. K sweep (reviewer stage 3); canary rule applies."""
        rt=self.rt;Pw=self.T.P[swap];ids=Pw.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);log=[]
        with rt.mesh:
            pref=rt.prefix(rt.base,seq);m=rt.base;st_all=clone_state(rt.state_all);st=clone_state(rt.state_suffix);cum=0.0
            log.append({'state':0,'margins':self.evaluate(m)})
            for k in range(NCH):
                c=rt.chunk(seq,k);p_=pref[k*CH:(k+1)*CH]
                for _ in range(K):
                    newp,st_new=self.native_scaled(m,st_all,st,c,p_,jnp.float32(1.0));old=m;m=self.set_inner(m,newp)
                    cum+=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5
                st=st_new
                if (k+1)%eval_every==0 or k+1==NCH:log.append({'state':k+1,'cum_update_norm':cum,'margins':self.evaluate(m)});print(f'  L0k{K} {"B" if swap else "A"} state {k+1}',flush=True)
        return log
    def run_L0_scaled(self,swap,scale,eval_every=16):
        """one pass, every native step multiplied by `scale` (intensity control for multi-pass repetition)."""
        rt=self.rt;Pw=self.T.P[swap];ids=Pw.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);log=[]
        with rt.mesh:
            pref=rt.prefix(rt.base,seq);m=rt.base;st_all=clone_state(rt.state_all);st=clone_state(rt.state_suffix);cum=0.0
            log.append({'state':0,'margins':self.evaluate(m)})
            for k in range(NCH):
                newp,st=self.native_scaled(m,st_all,st,rt.chunk(seq,k),pref[k*CH:(k+1)*CH],jnp.float32(scale));old=m;m=self.set_inner(m,newp)
                cum+=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5
                if (k+1)%eval_every==0 or k+1==NCH:log.append({'state':k+1,'cum_update_norm':cum,'margins':self.evaluate(m)});print(f'  L0x{scale} {"B" if swap else "A"} state {k+1}',flush=True)
        return log
    def run_L0(self,swap,eval_states):
        rt=self.rt;Pw=self.T.P[swap];ids=Pw.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask);log=[]
        with rt.mesh:
            pref=rt.prefix(rt.base,seq);m=rt.base;opt=m.inner_optimizer(clone_state(rt.state_all)).init(m.inner_parameters());st=(clone_state(rt.state_all),clone_state(rt.state_suffix));cum=0.0
            if 0 in eval_states:log.append({'state':0,'margins':self.evaluate(m)})
            for k in range(NCH):
                old=m;r=rt.update(m,opt,st,rt.chunk(seq,k),pref[k*CH:(k+1)*CH]);m,opt,st=r.new_model,r.new_optimizer_state,r.new_state;del r
                cum+=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5
                if k+1 in eval_states:log.append({'state':k+1,'cum_update_norm':cum,'margins':self.evaluate(m)});print(f'  L0 {"B" if swap else "A"} state {k+1}',flush=True)
        return log

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--level',choices=['L0','L1','L0mp','L0x','L0or','L0fs','L0k'],required=True);ap.add_argument('--passes',type=int,default=8);ap.add_argument('--scale',type=float,default=8.0);ap.add_argument('--worlds',default='AB')
    ap.add_argument('--epochs',type=int,default=32);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--tag',default='main');ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--mode',choices=['full','save','eval'],default='full');ap.add_argument('--save-epochs',default='');ap.add_argument('--keep',action='store_true');ap.add_argument('--k',type=int,default=1);ap.add_argument('--l1-variant',default='plain',choices=['plain','ctx','fill'])
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    vp=ROOT/'experiments/exp05_write_dynamics/run3_readouts/verify_packing_fp32.log'
    base=[l for l in open(vp) if l.startswith('base ')]
    import ast;b=ast.literal_eval(base[0][5:]) if base else {}
    assert b and max(b['S_maxdiff_R1'],b['S_maxdiff_R0'],b['L_maxdiff_R1'])<=1e-3,'packed fp32 readout not verified - refusing to run'
    rt=Runtime();SD=OUT/'states'/f'{a.level}_{a.tag}_s{a.world_seed}'
    L=Ladder(rt,a.world_seed,readout=(a.mode!='save'),save_dir=SD)
    meta={'items':L.items,'fam_sizes':{k:len(v) for k,v in L.fam.items()},'lr':L.lr,'clip':L.clip,'level':a.level,'epochs':a.epochs,'seed':a.seed,'xla_flags':os.environ.get('XLA_FLAGS')}
    json.dump(meta,open(OUT/f'meta_{a.level}_{a.tag}_s{a.world_seed}.json','w'),indent=1)
    for w in a.worlds:
        f=OUT/f'{a.level}_{a.tag}_s{a.world_seed}_{w}.json';fs=OUT/f'{a.level}_{a.tag}_s{a.world_seed}_{w}.saved.json'
        if f.exists():print('exists',f);continue
        if a.mode=='eval':
            d=json.load(open(fs));t0=time.monotonic()
            for e in d['log']:
                if 'margins' in e and '_file' in e['margins']:
                    fn=e['margins']['_file'];h=e['margins']['_hash'];mm=L.load_state(fn)
                    assert reuse._hash_pytree(mm.inner_parameters())[0]==h,'loaded weights differ from the streamed weights'
                    e['margins']=L.evaluate(mm);e['hash']=h
                    if not a.keep:os.remove(fn)
            d['sec_eval']=time.monotonic()-t0;json.dump(d,open(f,'w'));print('scored',f,round(d['sec_eval']),'s',flush=True);continue
        if a.mode=='save':
            L.save_dir=SD/w;L._nsave=0;(SD/w).mkdir(parents=True,exist_ok=True)
            if fs.exists():print('exists',fs);continue
        t0=time.monotonic()
        se=tuple(int(x) for x in a.save_epochs.split(',') if x)
        if a.level=='L1':log=L.run_L1(w=='B',a.epochs,a.seed,save_epochs=se,save_dir=SD/w,variant=a.l1_variant)
        elif a.level=='L0mp':log=L.run_L0_multipass(w=='B',a.passes)
        elif a.level=='L0or':log=L.run_L0_oracle(w=='B',a.passes)
        elif a.level=='L0fs':log=L.run_L0_fillerswap(w=='B',a.passes)
        elif a.level=='L0x':log=L.run_L0_scaled(w=='B',a.scale)
        elif a.level=='L0k':log=L.run_L0_ksteps(w=='B',a.k)
        else:log=L.run_L0(w=='B',set(range(NCH+1)))
        json.dump({'log':log,'sec':time.monotonic()-t0},open(fs if a.mode=='save' else f,'w'));print('saved',fs if a.mode=='save' else f,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
