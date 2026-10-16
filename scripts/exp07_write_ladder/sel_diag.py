"""Label-free v2 selector diagnostic (PROTOCOL "Label-free v2", 2026-10-17 00:33 KST, section 2): W0 forwards only, s101-103,
worlds A and B. For the NER-v2 PERSON-swapped chunks, candidates = target tokens after the PERSON span inside the swapped
sentence (first target after the name, PER spans, cut tail and pad excluded); per swapped sentence the top-3 by
KL-sel = W0 next-token KL(p_orig || p_swap) (fp32, full vocabulary, pmap-aligned) and by surprisal-sel = W0 NLL on the
original input (ties -> earlier). Output: gold-place precision of each selector pooled over the 3 seeds; the higher one is
"sel" (tie to 3 decimals -> surprisal-sel). No gradient, no D, no canary. Usage: NERV2=<tag> python sel_diag.py"""
import os,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
sys.path.insert(0,str(LD.ROOT/'scripts/exp04_what_ttt_retains'))
import suppress_v2 as V
OUT=LD.OUT
def main():
    seeds=(101,102,103);tagv2=os.environ.get('NERV2','ner_v2');res={'seeds':{},'sentence_splitter_sha':hashlib.sha256(V.SENT_RE.pattern.encode()).hexdigest()}
    rt=Runtime();tot={'kl':[0,0],'surprisal':[0,0]}
    for sd in seeds:
        L=LD.Ladder(rt,sd,readout=True);rt32=L.rt32;m0=rt32.base;T=L.T;W=int(rt.cfg.model.sliding_window_size);Batch=L.Batch
        seg=jnp.asarray(np.zeros(CH,np.int32));loc=jnp.asarray(np.arange(CH,dtype=np.int32));pm=jnp.asarray(packing.prefix_mask(np.zeros(CH,np.int32),W))
        @eqx.filter_jit
        def logp(m,ids_):
            packing.PACK_PRE['mask']=pm
            try:
                sq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=loc)
                pf=m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),clone_state(rt32.state_prefix),sq).last_hidden_state
            finally:packing.PACK_PRE.clear()
            packing.PACK_SUF['seg']=seg;packing.PACK_SUF['loc']=loc
            try:r=m.language_model.suffix_call(prefix_outputs=pf,state=clone_state(rt32.state_suffix),seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
            finally:packing.PACK_SUF.clear()
            return jax.nn.log_softmax(r.logits.astype(jnp.float32),-1)
        ext=V.external_names(T.w,V.gw.ntok,T.w['name_len']);res['seeds'][sd]={}
        for w in 'AB':
            sw=w=='B';sp=[tuple(x) for x in json.load(open(OUT/f'{tagv2}_s{sd}_{w}.json'))['spans']]
            pr,cnt=V.lf_chunk_pairs_v2(T.w,sw,T.tok,rt.bos,rt.eos,'lfswap',sp,ext,sd,'after_span_sentence')
            text,_,_=V.gw.realize(T.w,sw);e=T.tok.encode(text,add_special_tokens=False);offs=[(0,0)]+list(e.offsets);gp=V.place_spans(T.w,sw)
            isplace=lambda j:any(offs[j][1]>a and offs[j][0]<b for a,b in gp)
            sel={'kl':[],'surprisal':[]};ncand=0
            with rt32.mesh,jax.default_matmul_precision('highest'):
                for o,ed in pr:
                    inf=o[3];lo=inf['lo'];cand=sorted({j for g in inf['groups'] for j in g});ncand+=len(cand)
                    if not cand:continue
                    lo_=logp(m0,jnp.asarray(o[0]));le_=logp(m0,jnp.asarray(ed[0]))
                    po=jnp.asarray([j-lo-1 for j in cand]);pe=jnp.asarray([inf['pmap'][j]-1 for j in cand]);tgt=jnp.asarray([int(o[0][j-lo]) for j in cand])
                    a_=lo_[po];kl=np.asarray(jnp.sum(jnp.exp(a_)*(a_-le_[pe]),-1));nll=np.asarray(-jnp.take_along_axis(a_,tgt[:,None],1)[:,0])
                    for nm,v in (('kl',kl),('surprisal',nll)):sel[nm]+=V.kl_select(inf['groups'],dict(zip(cand,v.tolist())))
                    del lo_,le_
            r={'n_candidates':ncand,**cnt}
            for nm in sel:
                hit=sum(1 for j in sel[nm] if isplace(j));r[nm]={'n_selected':len(sel[nm]),'gold_place_hits':hit,'precision':hit/max(len(sel[nm]),1)};tot[nm][0]+=hit;tot[nm][1]+=len(sel[nm])
            res['seeds'][sd][w]=r;print(sd,w,json.dumps(r),flush=True)
        del L
    for nm in tot:res[nm+'_pooled_precision']=tot[nm][0]/max(tot[nm][1],1)
    pk,ps=round(res['kl_pooled_precision'],3),round(res['surprisal_pooled_precision'],3)
    res['sel']='kl' if pk>ps else 'surprisal';res['rule']='higher pooled gold-place precision; tie to 3 decimals -> surprisal'
    json.dump(res,open(OUT/'v2_selector_diagnostic.json','w'),indent=1);print('SELECTOR',res['sel'],'KL',pk,'surprisal',ps,flush=True)
if __name__=='__main__':main()
