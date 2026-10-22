"""E55 design 22.3 stage 0: offline isolated-row training (ladder.py run_L1 'plain' variant) on the E55 v3 world. Every exposure sentence of the real stream is a separate ROW ('\\n\\n' + sentence, loss on all its tokens, no other sentence, no background, empty suffix KV); rows of round r = the 16 sentences of chunk r (exposure r of each fact), shuffled within the round; rounds follow in order, so after a rounds every fact has exactly a exposures. Rows are packed first-fit into 1024-token chunks (exp05 packing: segment mask in the prefix layers, per-row suffix attention), ONE inner-rule step per packed chunk (SGD, lr x ilr multiplier, global-norm clip, same code as ladder.Ladder.step). A read point is a segment boundary: the segment (a0, a1] is packed on its own, so the weights at the read point have seen exactly a1 exposures per fact.
Cumulative update norm matched to NAT (design 22.3): within a segment every step is rescaled AFTER clipping (as NATM, direction unchanged) so that the segment's total update norm equals the NAT cumulative norm increment T(a1) - T(a0) (T from the per-chunk update norms of the NAT run of the same seed), spread equally over the J steps of the segment; the raw (unscaled) step norms are logged.
Reads: frozen weights through the Reads object (out-of-window neutral tail, unexposed read frames), W0 from it. CPU tested with a tiny model (e55_units.py U26); the 3B GPU path is NOT run."""
import time,random
import numpy as np
import e55_core as C
import s3
jax,jnp,eqx=s3.jax,s3.jnp,s3.eqx
LD=s3.LD
class Iso55(C.Ladder55):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.build_iso()
    def build_iso(self):
        """closures as in ladder.Ladder.__init__ (prefix_iso, step), rebuilt here because Ladder55 does not run the old-world constructor"""
        rt=self.rt;lr,clip=self.lr,self.clip;Batch=self.Batch;packing=LD.packing;MetaModel=LD.MetaModel;optax=LD.optax;clone_state=LD.clone_state
        @eqx.filter_jit
        def prefix_iso(m,ids_,pos_,mask_):
            packing.PACK_PRE['mask']=mask_
            try:
                seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
                return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),clone_state(rt.state_prefix),seq).last_hidden_state
            finally:packing.PACK_PRE.clear()
        @eqx.filter_jit
        def iso_grad(m,ids_,tg_,mk_,pref_,seg_,loc_):
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_)
                (_l,(ce,_n,_s)),grads=eqx.filter_value_and_grad(MetaModel.lm_loss,has_aux=True)(m,seq,clone_state(rt.state_suffix),prefix_outputs=pref_)
            finally:packing.PACK_SUF.clear()
            return grads.inner_parameters(),ce
        @eqx.filter_jit
        def sgd_apply(m,st_all,g):
            """the inner rule: clip by global norm, step lr x ilr multiplier (same formula as ladder.Ladder.step / s3 write_step)"""
            gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
            return jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g),gn,coef
        self.prefix_iso=prefix_iso;self.iso_grad=iso_grad;self.sgd_apply=sgd_apply
    @staticmethod
    def norm_between(newp,old):
        f32=jnp.float32;return sum(float(jnp.sum((a.astype(f32)-b.astype(f32))**2)) for a,b in zip(jax.tree.leaves(newp),jax.tree.leaves(old.inner_parameters())))**0.5
    @staticmethod
    def rescale(newp,old,s):
        f32=jnp.float32;return jax.tree.map(lambda a,b:(b.astype(f32)+(a.astype(f32)-b.astype(f32))*s).astype(a.dtype),newp,old.inner_parameters())
def iso_rows(world,enc,seed,apps):
    """-> per segment (a0, a1] the list of rows (ids, (0, len-1)) in training order. enc: text -> ids (the model tokenizer; '\\n\\n' + sentence)."""
    A_=world['chunks'];assert world['p']==1 and max(apps)<=len(A_) and sorted(apps)==list(apps);segs=[];a0=0
    for a1 in apps:
        rows=[]
        for r in range(a0,a1):
            sents=[x['text'] for x in A_[r]['sents']];rng=random.Random(f'E55iso|{world["seed"]}|{seed}|{r}');rng.shuffle(sents);rows+=[(enc('\n\n'+t),t) for t in sents]
        segs.append((a0,a1,[(ids,(0,len(ids)-1)) for ids,_ in rows],[t for _,t in rows]));a0=a1
    return segs
def cum_target(norms):
    """T[a] = sum of the first a per-chunk update norms of the NAT run (T[0] = 0)"""
    return np.concatenate([[0.0],np.cumsum(np.asarray(norms,float))])
def iso_run(L,rt,segs,T=None,read=None,eos=None,log=print,smoke_steps=None,weights0=None,read_at=None):
    """-> {'log': per step dicts, 'reads': {a1: read(a1, m) result}, 'cum_update_norm': {a1: ...}, 'model': final weights}. T = cum_target of the NAT run (None: no matching, raw SGD steps)."""
    packing=LD.packing;clone_state=LD.clone_state;W=int(rt.cfg.model.sliding_window_size);m=weights0 if weights0 is not None else rt.base;st_all=clone_state(rt.state_all);out={'log':[],'reads':{},'cum_update_norm':{}};cum=0.0;eos=rt.eos if eos is None else eos
    with rt.mesh:
        for a0,a1,rows,texts in segs:
            chunks,where=packing.pack_rows(rows);mks=[np.zeros(1024,bool) for _ in chunks]
            for c_,a_,b_ in where:mks[c_][a_:b_]=True
            J=len(chunks);inc=None if T is None else (T[a1]-T[a0])/J;seg_cum=0.0
            for j,((ids,seg,loc),mk) in enumerate(zip(chunks,mks)):
                if smoke_steps is not None and j>=smoke_steps:break
                tg=np.concatenate([ids[1:],[eos]]).astype(np.int32);t0=time.monotonic()
                pref=L.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
                g,ce=L.iso_grad(m,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc));newp,gn,coef=L.sgd_apply(m,st_all,g)
                raw=L.norm_between(newp,m);s=1.0 if inc is None else (inc/raw if raw>0 else 1.0)
                if inc is not None:newp=L.rescale(newp,m,s)
                un=L.norm_between(newp,m);m=L.set_inner(m,newp);cum+=un;seg_cum+=un
                out['log'].append({'a0':a0,'a1':a1,'step':j,'rows_in_chunk':int(seg.max())+1,'grad_norm':float(gn),'clip_coef':float(coef),'ce':float(ce),'raw_update_norm':raw,'scale':s,'update_norm':un,'cum_update_norm':cum,'sec':time.monotonic()-t0})
            out['cum_update_norm'][a1]=cum;log(f'  ISO segment ({a0},{a1}]: {J} packed chunks, {len(rows)} rows, update norm {seg_cum:.4g}'+('' if T is None else f' (NAT target {T[a1]-T[a0]:.4g})'))
            if read is not None and (read_at is None or a1 in read_at):out['reads'][a1]=read(a1,m)
    out['model']=m;return out

def rows_bg(world):
    """E55 design 24.3 (a) ISO+background (ISOB): per exposure round r ONE segment holding the tokens of stream chunk r itself (fact sentences AND the non-repeating background of that chunk), cut into isolated rows: every fact sentence is a row, every run of background tokens between sentences is a row (no row sees another; positions restart per row). The rows concatenated are exactly the 1024 tokens of the chunk, so they pack into exactly one chunk per round (128 steps, the same text, the same fact share 31-37% as the online stream, the same loss tokens up to one per row). -> segments (a0, a1, rows, texts)"""
    segs=[];A_=world['chunks'];sep=len(world['sep'])
    for r,c in enumerate(A_):
        ids=list(c['ids']);cuts=[0];
        for x in sorted(c['sents'],key=lambda x:x['start']):cuts+=[x['start'],x['end']]
        cuts=sorted(set(cuts+[len(ids)]));rows=[(ids[a:b],(0,b-a-1)) for a,b in zip(cuts,cuts[1:]) if b>a];assert sum(len(x[0]) for x in rows)==len(ids)==1024
        segs.append((r,r+1,rows,[]))
    return segs
def rows_density(world,enc,seed):
    """(c) ISO-density (ISOD): per exposure round r ONE segment with the 16 sentences of round r as isolated rows ('\\n\\n' + sentence, loss on their tokens only, shuffled within the round), one chunk (about 400 tokens, rest padding) per round = 128 steps."""
    segs=[]
    for r,c in enumerate(world['chunks']):
        sents=[x['text'] for x in c['sents']];random.Random(f'E55isod|{world["seed"]}|{seed}|{r}').shuffle(sents);rows=[]
        for t in sents:ids=enc('\n\n'+t);rows.append((ids,(0,len(ids)-1)))
        segs.append((r,r+1,rows,sents))
    return segs
def bigstep_targets(norms,segs_iso,apps):
    """(b) NAT-bigstep (NATB): per-chunk update-norm targets of the NAT stream = NAT norm x (chunks of the segment / packed ISO steps of the segment) for the segment (a0, a1] that contains the chunk, i.e. every chunk step gets the size of one stage-0 ISO step of its segment (the ISO steps are the NAT segment norm spread over fewer steps). segs_iso = iso_rows segments of stage 0 (same apps)."""
    packing=LD.packing;t=np.asarray(norms,float).copy()
    for a0,a1,rows,_ in segs_iso:
        J=len(packing.pack_rows(rows)[0]);t[a0:a1]*= (a1-a0)/J
    return t

def fos_stream(segs,pad_ids):
    """(d) fact-only stream (FOS): the isolated rows of stage-0 segments concatenated as ORDINARY continuous text (no isolation, the SWA window of the previous 8 chunks is present, the write path of the online stream), each segment padded with `pad_ids` ('\\n\\n' tokens) to a multiple of 1024 tokens so that a read point falls on a chunk boundary (at most 1023 pad tokens per segment). -> (ids, chunk index of the last chunk of each segment, tokens of pad)"""
    ids=[];ends=[];pad=0
    for a0,a1,rows,_ in segs:
        for r,_ab in rows:ids+=list(r)
        while len(ids)%1024:ids.append(pad_ids[pad%len(pad_ids)]);pad+=1
        ends.append(len(ids)//1024-1)
    return np.asarray(ids,np.int64),ends,pad
def fos_targets(T,segs,ends):
    """per-chunk norm targets of FOS: the NAT cumulative increment of the segment spread equally over the segment's chunks (cumulative norm at the read points == NAT's, as stage 0)"""
    t=[];s0=0
    for (a0,a1,_r,_t),e in zip(segs,ends):
        n=e-s0+1;t+=[(T[a1]-T[a0])/n]*n;s0=e+1
    return np.asarray(t)
