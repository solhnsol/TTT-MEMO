"""Exp07 stage 2: does the next-token gradient on the exposure sentences point along the binding direction?
At weights W (W0, and the L1 s101 'ck' weights after epochs 1, 2, 4, 8, separately for worlds A and B), in fp32:
  g_w      = mean over packed chunks of the gradient of the isolated-sentence NTP loss of world w (the L1 update signal)
  u_w(p)   = gradient of pair p's L3 margin (canonical items: gold = world-A place) under an empty-context readout (R0)
  f_w(p)   = gradient of pair p's name-familiarity margin (FN items)
  a_w(p)   = -<u_w(p), g_w>   first-order change of the margin per unit step along the update direction
Binding component  b(p) = (a_A(p) - a_B(p))/2   (world B's update should LOWER the canonical margin)
Common component   c(p) = (a_A(p) + a_B(p))/2
Familiarity rate   r(p) = (-<f_A(p),g_A> - <f_B(p),g_B>)/2
Reported per state: mean b, c, r with pair-bootstrap CIs, |b|/|r|, and the first-order prediction checked against the
measured epoch-to-epoch change of D where available. H-seq predicts b ~ 0 at W0 and b > 0 after familiarity is learned.
The R0 readout (no context) is used for u to avoid differentiating through the 7K-token tail; this is a stated
approximation. e2e/ is not modified."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
import ladder as LD
from ladder import *
OUT=LD.OUT;STATES=[0,1,2,4,8]
def run_checks(ns):
    """Unit checks of the exact functions used by the measurement (GRAD_CHECK=1). Results: grad_checks_s101.json."""
    import json as _j
    rt=ns['rt'];rt32=ns['rt32'];L=ns['L'];T=ns['T'];W=ns['W'];Batch=ns['Batch']
    prefix32=ns['prefix32'];loss_grad=ns['loss_grad'];margin_grad=ns['margin_grad'];dot=ns['dot'];norm=ns['norm']
    dloss=ns['dloss'];distill_value=ns['distill_value'];distill_grad=ns['distill_grad'];build_targets=ns['build_targets'];logits_fn=ns['logits_fn']
    dist=ns['dist'];dist_in=ns['dist_in'];inp=ns['inp'];pairs=ns['pairs'];xpairs=ns['xpairs']
    R={};m0=rt32.base;ok=True
    with rt32.mesh,jax.default_matmul_precision('highest'):
        # C1: packed NTP gradient on one isolated sentence == gradient of the native lm_loss (the quantity TTT-E2E's inner
        #     step uses; Exp05 V2 showed own value_and_grad step == native update bitwise) on the same sentence, unpatched.
        sent0=L.sentences(False)[0];ch,_=packing.pack_rows([(sent0,(0,0))]);ids,seg,loc=ch[0]
        tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32);mk=(seg>=0)&(np.concatenate([seg[1:],[-1]])==seg)
        pref=prefix32(m0,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
        g_pack=loss_grad(m0,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
        seqn=Batch(input_ids=jnp.asarray(ids),target_tokens=jnp.asarray(tg),loss_masks=jnp.asarray(mk));prefn=rt32.prefix(rt32.base,seqn)
        g_nat=eqx.filter_jit(lambda m,sq,pf:eqx.filter_grad(lambda m:MetaModel.lm_loss(m,sq,clone_state(rt32.state_suffix),prefix_outputs=pf)[0])(m).inner_parameters())(m0,seqn,prefn)
        diff=jax.tree.map(lambda a,b:a-b,g_pack,g_nat);R['C1_rel_diff']=norm(diff)/norm(g_nat);R['C1_pass']=R['C1_rel_diff']<=1e-4;del g_pack,g_nat,diff
        # C2c: teacher contexts exist and contain the previous round's pair sentences
        tri={t['id']:t for t in T.w['triples']};_,name,spans=gw.realize(T.w,False)
        nrows=len(dist['A'][0]);R['C2c_rows']={'A':nrows,'B':len(dist['B'][0])}
        full,(ta,tb)=dist['A'][0][0];txt=T.tok.decode(full[:ta]);s1=[s_ for s_ in spans if s_['round']>=2][0]
        pr=tri[s1['t']]['pair'];prev=[s_ for s_ in spans if s_['round']==s1['round']-1 and tri[s_['t']]['pair']==pr]
        R['C2c_example_context']=txt[:400];R['C2c_context_tokens']=ta
        R['C2c_pass']=nrows>=400 and all(p_['text'] in txt for p_ in prev) and ta>0
        # C2a: finite difference vs gradient along a random inner direction (both signals, one student chunk)
        si=dist_in['A'][2];tgts=build_targets('A',m0);inp0=si[0];tgt,mask=tgts[0];tgt=jnp.asarray(tgt);mask=jnp.asarray(mask)
        inner,outer=eqx.partition(m0,L.f32);lv,td=jax.tree_util.tree_flatten(inner);key=jax.random.PRNGKey(7)
        vs=[jax.random.normal(k_,x.shape,jnp.float32) for k_,x in zip(jax.random.split(key,len(lv)),lv)];vn=float(np.sqrt(sum(float(jnp.sum(v*v)) for v in vs)));vs=[v/vn for v in vs]
        vtree=jax.tree_util.tree_unflatten(td,vs)
        for kind,nm in ((0,'ctxKL'),(1,'logitMSE')):
            g=distill_grad(m0,*inp0,tgt,mask,jnp.int32(kind));gv=dot(g,vtree);del g
            eps=1e-3;mp=eqx.combine(jax.tree.map(lambda x,v:x+eps*v,inner,vtree),outer);mm=eqx.combine(jax.tree.map(lambda x,v:x-eps*v,inner,vtree),outer)
            fd=(float(distill_value(mp,*inp0,tgt,mask,jnp.int32(kind)))-float(distill_value(mm,*inp0,tgt,mask,jnp.int32(kind))))/(2*eps);del mp,mm
            R[f'C2a_{nm}']={'grad_dot_v':gv,'finite_diff':fd,'rel':abs(gv-fd)/max(abs(fd),1e-12)};R[f'C2a_{nm}_pass']=R[f'C2a_{nm}']['rel']<=2e-2
        # C2b: teacher == student (teacher logits = the student's own logits on the same isolated input) -> zero gradient
        own=logits_fn(m0,*inp0)
        for kind,nm in ((0,'ctxKL'),(1,'logitMSE')):
            g_self=distill_grad(m0,*inp0,own,mask,jnp.int32(kind));g_real=distill_grad(m0,*inp0,tgt,mask,jnp.int32(kind))
            R[f'C2b_{nm}']={'grad_norm_self':norm(g_self),'grad_norm_real':norm(g_real),'ratio':norm(g_self)/max(norm(g_real),1e-30),'loss_self':float(distill_value(m0,*inp0,own,mask,jnp.int32(kind)))}
            R[f'C2b_{nm}_pass']=R[f'C2b_{nm}']['ratio']<=1e-3 and (kind==0 or R[f'C2b_{nm}']['loss_self']<=1e-8);del g_self,g_real
        del own,tgts
        # C3: positive control. Delta_w = W_w(L1 epoch 4) - W0 carries binding (L1 D = +1.18 at epoch 4). The measure
        #     <u_w(p), Delta_w> must give a clearly positive binding part; the cross-pair null must be small.
        SD=ns['OUT']/'states'/'L1_ck_s101';pc={}
        for w in 'AB':
            mw=L.to_fp32(L.load_state(SD/w/'epoch_004.npz'));dl=jax.tree.map(lambda a,b:a-b,eqx.partition(mw,L.f32)[0],eqx.partition(m0,L.f32)[0]);del mw
            for fam,pl in (('L3',pairs),('XP',xpairs)):
                for p in pl:u=margin_grad(m0,*inp[(fam,p)]);pc[(w,fam,p)]=dot(u,dl);del u
            del dl
        b=np.asarray([(pc[('A','L3',p)]-pc[('B','L3',p)])/2 for p in pairs]);b0=np.asarray([(pc[('A','XP',p)]-pc[('B','XP',p)])/2 for p in xpairs])
        rng=np.random.default_rng(3);bs=[b[rng.integers(0,len(b),len(b))].mean() for _ in range(4000)]
        R['C3']={'b_mean':float(b.mean()),'b_ci':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'null_mean':float(b0.mean()),'null_over_b':float(abs(b0.mean())/max(abs(b.mean()),1e-12))}
        R['C3_pass']=R['C3']['b_ci'][0]>0 and R['C3']['null_over_b']<=0.2
    R['ALL_PASS']=all(v for k,v in R.items() if k.endswith('_pass'))
    _j.dump(R,open(ns['OUT']/'grad_checks_s101.json','w'),indent=1,default=str);print(_j.dumps(R,indent=1,default=str),flush=True)
    if not R['ALL_PASS']:sys.exit(3)
def main():
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime();L=LD.Ladder(rt,101);rt32=L.rt32;W=int(rt32.cfg.model.sliding_window_size);T=L.T
    from ttt.model.data import Batch
    @eqx.filter_jit
    def prefix32(m,ids_,pos_,mask_):
        packing.PACK_PRE['mask']=mask_
        try:
            seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
            return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),clone_state(rt32.state_prefix),seq).last_hidden_state
        finally:packing.PACK_PRE.clear()
    def chunk_inputs(rows):
        ch,where=packing.pack_rows(rows);assert len(ch)==1,len(ch);ids,seg,loc=ch[0]
        with rt32.mesh,jax.default_matmul_precision('highest'):pref=prefix32(rt32.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
        tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32);return ids,tg,seg,loc,pref,where
    def item_rows(items):
        rows=[]
        for it in items:
            for c in (it['gold'],it['foil']):
                ids,span=stage2.item_ids(T.tok,stage2.make_slot(T.tok,None),it['prefix'],c);rows.append((ids,(span[0]-1,span[1]-1)))
        return rows
    @eqx.filter_jit
    def margin_grad(m,ids_,tg_,pref_,seg_,loc_,wg_,wf_):
        """gradient (inner params) of sum(lp * wg) - sum(lp * wf): weights mark gold / foil span positions."""
        def f(m):
            packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
            try:
                r=m.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt32.state_suffix),seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
                lg=r.logits.astype(jnp.float32);lp=jnp.take_along_axis(lg,tg_[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
            finally:packing.PACK_SUF.clear()
            return jnp.sum(lp*wg_)-jnp.sum(lp*wf_)
        return eqx.filter_grad(f)(m).inner_parameters()
    @eqx.filter_jit
    def loss_grad(m,ids_,tg_,mk_,pref_,seg_,loc_):
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:
            seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_)
            g=eqx.filter_grad(lambda m:MetaModel.lm_loss(m,seq,clone_state(rt32.state_suffix),prefix_outputs=pref_)[0])(m)
        finally:packing.PACK_SUF.clear()
        return g.inner_parameters()
    def dot(a,b):return float(sum(jnp.vdot(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))))
    def norm(a):return dot(a,a)**0.5
    # pair inputs (built once; prefixes are weight-independent)
    pairs=sorted({it['pair'] for it in L.fam['L3']},key=lambda s:int(s[1:]))
    def masks(where,n_items):
        wg=np.zeros(1024,np.float32);wf=np.zeros(1024,np.float32)
        for r,(c,a,b) in enumerate(where):(wg if r%2==0 else wf)[a:b]=1.0
        return jnp.asarray(wg),jnp.asarray(wf)
    inp={}
    xpairs=sorted({it['pair'] for it in L.fam['XP']},key=lambda s:int(s[1:]))
    for fam,pl in (('L3',pairs),('FN',pairs),('XP',xpairs)):
        for p in pl:
            its=[it for it in L.fam[fam] if it['pair']==p];ids,tg,seg,loc,pref,where=chunk_inputs(item_rows(its));wg,wf=masks(where,len(its))
            inp[(fam,p)]=(jnp.asarray(ids),jnp.asarray(tg),pref,jnp.asarray(seg),jnp.asarray(loc),wg,wf)
    sent={}
    for w in 'AB':
        ch,_=packing.pack_rows([(s,(0,0)) for s in L.sentences(w=='B')]);lst=[]
        for ids,seg,loc in ch:
            tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32);mk=(seg>=0)&(np.concatenate([seg[1:],[-1]])==seg)
            with rt32.mesh,jax.default_matmul_precision('highest'):pref=prefix32(rt32.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
            lst.append((jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc)))
        sent[w]=lst
    # ---- write-signal screening (reviewer 2026-10-15): teacher = same weights with the previous round's pair sentences in
    # context; student = the current exposure sentence alone. Signals: ctxKL = soft-target CE to the teacher's next-token
    # distribution on the current sentence; logitMSE = squared difference of centred logits (approximation of TTCD's
    # window-difference distillation, which uses hidden states). Rounds 2-16 only (round 1 has no previous exposure).
    def distill_rows(w):
        _,_,spans=gw.realize(T.w,w=='B');tri={t['id']:t for t in T.w['triples']};by={}
        for s_ in spans:by.setdefault((tri[s_['t']]['pair'],s_['round']),[]).append(s_['text'])
        trows=[];srows=[]
        for s_ in spans:
            pr=tri[s_['t']]['pair'];r=s_['round']
            if r<2:continue
            ctx_ids=stage2.enc(T.tok,'\n\n'+'\n\n'.join(by[(pr,r-1)])).ids;cur=stage2.enc(T.tok,'\n\n'+s_['text']).ids
            full=ctx_ids+cur;off=len(ctx_ids);trows.append((full,(off,len(full)-1)));srows.append((cur,(0,len(cur)-1)))
        return trows,srows
    def pack_inputs(rows):
        ch,where=packing.pack_rows(rows);out=[]
        for ids,seg,loc in ch:
            tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32)
            with rt32.mesh,jax.default_matmul_precision('highest'):pref=prefix32(rt32.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
            out.append((jnp.asarray(ids),jnp.asarray(tg),pref,jnp.asarray(seg),jnp.asarray(loc)))
        return out,where
    dist={w:distill_rows(w) for w in 'AB'};dist_in={}
    for w in 'AB':
        ti,tw=pack_inputs(dist[w][0]);si,sw=pack_inputs(dist[w][1]);dist_in[w]=(ti,tw,si,sw)
        print('distill rows',w,len(dist[w][0]),'teacher chunks',len(ti),'student chunks',len(si),flush=True)
    @eqx.filter_jit
    def logits_fn(m,ids_,tg_,pref_,seg_,loc_):
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:r=m.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt32.state_suffix),seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
        finally:packing.PACK_SUF.clear()
        return r.logits.astype(jnp.float32)
    def dloss(m,ids_,tg_,pref_,seg_,loc_,tlogits,mask,kind):
        """distillation write loss on one packed student chunk (kind 0: soft-target CE = ctxKL up to the teacher entropy;
        kind 1: logitMSE on centred logits). Positions with mask = 0 do not contribute."""
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:r=m.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt32.state_suffix),seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
        finally:packing.PACK_SUF.clear()
        sl=r.logits.astype(jnp.float32);n=jnp.maximum(mask.sum(),1.0)
        ce=-(jax.nn.softmax(tlogits,-1)*jax.nn.log_softmax(sl,-1)).sum(-1)
        d=(sl-sl.mean(-1,keepdims=True))-(tlogits-tlogits.mean(-1,keepdims=True));mse=(d*d).mean(-1)
        return jnp.where(kind==0,(ce*mask).sum()/n,(mse*mask).sum()/n)
    distill_value=eqx.filter_jit(dloss)
    @eqx.filter_jit
    def distill_grad(m,*args):return eqx.filter_grad(dloss)(m,*args).inner_parameters()
    def build_targets(w,m_teacher):
        """per student chunk: (teacher logits placed at the student positions, mask). Teacher = m_teacher reading the
        previous round's two pair sentences + the current sentence."""
        ti,tw,si,sw=dist_in[w];rowlog={}
        for ci,inp_ in enumerate(ti):
            lg_=np.asarray(logits_fn(m_teacher,*inp_))
            for ri,(tc,ta,tb) in enumerate(tw):
                if tc==ci:rowlog[ri]=lg_[ta:tb].copy()
            del lg_
        V=next(iter(rowlog.values())).shape[-1];out=[]
        for cj in range(len(si)):
            tgt=np.zeros((1024,V),np.float32);mask=np.zeros(1024,np.float32)
            for ri,(sc,sa,sb) in enumerate(sw):
                if sc!=cj:continue
                tgt[sa:sb]=rowlog[ri];mask[sa:sb]=1.0
            out.append((tgt,mask))
        return out
    def signal_grad(m32,w,kind):
        ti,tw,si,sw=dist_in[w];g=None;nch=len(si)
        for inp_,(tgt,mask) in zip(si,build_targets(w,rt32.base)):   # FIXED teacher: W0 with the previous round in context
            gi=distill_grad(m32,*inp_,jnp.asarray(tgt),jnp.asarray(mask),jnp.int32(kind));g=gi if g is None else jax.tree.map(lambda x,y:x+y,g,gi);del gi
        return jax.tree.map(lambda x:x/nch,g)
    if os.environ.get('GRAD_CHECK')=='1':
        ns=dict(locals());ns['OUT']=OUT;run_checks(ns);return
    SD=OUT/'states'/'L1_ck_s101';res={}
    for k in STATES:
        t0=time.monotonic();a={};gn={}
        for w in 'AB':
            m=rt.base if k==0 else L.load_state(SD/w/f'epoch_{k:03d}.npz');m32=L.to_fp32(m)
            with rt32.mesh,jax.default_matmul_precision('highest'):
                g=None
                for c_ in sent[w]:
                    gi=loss_grad(m32,*c_);g=gi if g is None else jax.tree.map(lambda x,y:x+y,g,gi);del gi
                g=jax.tree.map(lambda x:x/len(sent[w]),g);gn[w]=norm(g)
                gsig={'NTP':g}
                if k in (0,1):
                    for kind,nm in ((0,'ctxKL'),(1,'logitMSE')):gsig[nm]=signal_grad(m32,w,kind);gn[(w,nm)]=norm(gsig[nm])
                for fam,pl in (('L3',pairs),('FN',pairs),('XP',xpairs)):
                    for p in pl:
                        u=margin_grad(m32,*inp[(fam,p)])
                        for nm,gg in gsig.items():a[(w,fam,p) if nm=='NTP' else (w,fam,p,nm)]=-dot(u,gg)
                        del u
                del gsig
            del m32
        b=np.asarray([(a[('A','L3',p)]-a[('B','L3',p)])/2 for p in pairs]);c=np.asarray([(a[('A','L3',p)]+a[('B','L3',p)])/2 for p in pairs])
        r=np.asarray([(a[('A','FN',p)]+a[('B','FN',p)])/2 for p in pairs])
        bn=np.asarray([(a[('A','XP',p)]-a[('B','XP',p)])/2 for p in xpairs])
        res[k]={'b':b.tolist(),'b_null':bn.tolist(),'c':c.tolist(),'r':r.tolist(),'gnorm':{w:gn[w] for w in 'AB'},'sec':time.monotonic()-t0}
        if k in (0,1):
            for nm in ('ctxKL','logitMSE'):
                res[k][nm]={'b':[(a[('A','L3',p,nm)]-a[('B','L3',p,nm)])/2 for p in pairs],'b_null':[(a[('A','XP',p,nm)]-a[('B','XP',p,nm)])/2 for p in xpairs],'c':[(a[('A','L3',p,nm)]+a[('B','L3',p,nm)])/2 for p in pairs],
                            'r':[(a[('A','FN',p,nm)]+a[('B','FN',p,nm)])/2 for p in pairs],'gnorm':{w:gn[(w,nm)] for w in 'AB'}}
        print(f'state {k}: b {b.mean():+.4f} c {c.mean():+.4f} r {r.mean():+.4f} |g| {gn}',flush=True)
        json.dump(res,open(OUT/'grad_components_s101.json','w'),indent=1)
    rng=np.random.default_rng(19)
    def ci(v):
        v=np.asarray(v);bs=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(4000)];return f'{v.mean():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]'
    md=['# Exp07 stage 2: gradient components on the exposure sentences (seed 101; L1 ck weights)\n','| state (L1 epoch) | binding b | common c | familiarity rate r | b / r | grad norm A, B |','|---|---|---|---|---|---|']
    for k,v in res.items():md.append(f"| {k} | {ci(v['b'])} | {ci(v['c'])} | {ci(v['r'])} | {np.mean(v['b'])/np.mean(v['r']):+.4f} | {v['gnorm']['A']:.3f}, {v['gnorm']['B']:.3f} |")
    md+=['','## Write-signal screening (per unit gradient norm; b = binding, b_null = cross-pair null, r = name familiarity)','| state | signal | b / |g| | (b - b_null) / |g| | r / |g| | b / r |','|---|---|---|---|---|---|']
    for k,v in res.items():
        for nm in ('NTP','ctxKL','logitMSE'):
            x=v if nm=='NTP' else v.get(nm)
            if not x:continue
            gnm=(x['gnorm']['A']+x['gnorm']['B'])/2;bb=np.asarray(x['b']);rr=np.asarray(x['r']);bn_=np.asarray(x['b_null'])
            md.append(f"| {k} | {nm} | {ci(bb/gnm)} | {ci((bb-bn_)/gnm)} | {ci(rr/gnm)} | {bb.mean()/rr.mean():+.4f} |")
    (OUT/'grad_components_s101.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
