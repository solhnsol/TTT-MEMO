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
        # fp32 loss values have ~5e-7 resolution at loss ~4, so the step must move the loss by many ulps: direction 1 =
        # the normalised gradient itself (eps 1e-3), direction 2 = a random unit direction with a larger step (eps 1.0).
        def fd_check(g,dirn,eps,kind):
            gv=dot(g,dirn);mp=eqx.combine(jax.tree.map(lambda x,v:x+eps*v,inner,dirn),outer);mm=eqx.combine(jax.tree.map(lambda x,v:x-eps*v,inner,dirn),outer)
            fd=(float(distill_value(mp,*inp0,tgt,mask,jnp.int32(kind)))-float(distill_value(mm,*inp0,tgt,mask,jnp.int32(kind))))/(2*eps);del mp,mm
            return {'grad_dot_v':gv,'finite_diff':fd,'eps':eps,'rel':abs(gv-fd)/max(abs(fd),1e-12)}
        for kind,nm in ((0,'ctxKL'),(1,'logitMSE')):
            g=distill_grad(m0,*inp0,tgt,mask,jnp.int32(kind));gn_=norm(g);gdir=jax.tree.map(lambda x:x/gn_,g)
            r1=fd_check(g,gdir,1e-3,kind);r2=fd_check(g,vtree,1.0,kind);del g,gdir
            R[f'C2a_{nm}']={'gradient_direction':r1,'random_direction':r2};R[f'C2a_{nm}_pass']=r1['rel']<=2e-2 and r2['rel']<=5e-2
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
def run_onestep(ns):
    """W0 one-step realization test (PROTOCOL_stage01.md, registered 2026-10-16 00:35 KST). One step from W0 along the NTP
    mean gradient g_w (stage-2 g) or along its binding-only part g_b,w = (<g_w,U>/<U,U>) U, U = sum_p grad M_p (L3, R0, W0),
    with the native step size eta (lr x ilr multiplier) and the clip coefficient of g_w, times s in {1,2,4,8}. Full readout
    (R1, R1-alt, canaries) per point, plus the R0 margins and the first-order prediction. Unit checks O1-O4 run first."""
    import json as _j
    rt=ns['rt'];rt32=ns['rt32'];L=ns['L'];W=ns['W'];loss_grad=ns['loss_grad'];margin_grad=ns['margin_grad'];margin_val=ns['margin_val']
    dot=ns['dot'];norm=ns['norm'];inp=ns['inp'];pairs=ns['pairs'];xpairs=ns['xpairs'];sent=ns['sent'];OUT=ns['OUT'];m0=rt32.base
    SC=(1,2,4,8);R={}
    inner0,_=eqx.partition(rt.base,L.f);td=jax.tree_util.tree_structure(inner0);p0=jax.tree_util.tree_leaves(inner0)
    def make(v,step):
        """W0 - step * v as a model of the bf16-compute runtime (same cast rule as the inner step in ladder.py)."""
        vl=jax.tree_util.tree_leaves(v);assert len(vl)==len(p0)
        return L.set_inner(rt.base,jax.tree_util.tree_unflatten(td,[(p+(-step*x).astype(p.dtype)).astype(p.dtype) for p,x in zip(p0,vl)]))
    def r0(m):
        m32=L.to_fp32(m);out={}
        with rt32.mesh,jax.default_matmul_precision('highest'):
            for fam,pl in (('L3',pairs),('FN',pairs),('XP',xpairs)):out[fam]=[float(margin_val(m32,*inp[(fam,p)])) for p in pl]
        return out
    with rt.mesh:eta=float(L.lr*rt.base.get_ilr_multiplier(clone_state(rt.state_all).get(rt.base.step_index)))
    R['eta']=eta;g={};gb={};coef={};gn={}
    with rt32.mesh,jax.default_matmul_precision('highest'):
        U=None
        for p in pairs:u=margin_grad(m0,*inp[('L3',p)]);U=u if U is None else jax.tree.map(lambda x,y:x+y,U,u);del u
        UU=dot(U,U)
        for w in 'AB':
            gg=None
            for c_ in sent[w]:
                gi=loss_grad(m0,*c_);gg=gi if gg is None else jax.tree.map(lambda x,y:x+y,gg,gi);del gi
            g[w]=jax.tree.map(lambda x:x/len(sent[w]),gg);del gg;gn[w]=norm(g[w]);coef[w]=min(1.0,L.clip/max(gn[w],1e-30))
            k=dot(g[w],U)/UU;gb[w]=jax.tree.map(lambda x:k*x,U);R[f'gU_{w}']=dot(g[w],U);R[f'gb_norm_{w}']=norm(gb[w])
            # O3: projection identity <g_b,U> == <g,U>
            R[f'O3_{w}_rel']=abs(dot(gb[w],U)-R[f'gU_{w}'])/max(abs(R[f'gU_{w}']),1e-30)
        # first-order per-pair predictions per unit step (a = -<u_p, v>) for both directions
        pred={}
        for fam,pl in (('L3',pairs),('FN',pairs),('XP',xpairs)):
            for p in pl:
                u=margin_grad(m0,*inp[(fam,p)])
                for w in 'AB':pred[('g',w,fam,p)]=-dot(u,g[w]);pred[('gb',w,fam,p)]=-dot(u,gb[w])
                del u
    R['gnorm']=gn;R['coef']=coef;R['O3_pass']=all(R[f'O3_{w}_rel']<=1e-4 for w in 'AB')
    # O1: make(v, 0) reproduces W0 exactly
    R['O1_pass']=reuse._hash_pytree(make(g['A'],0.0).inner_parameters())[0]==reuse._hash_pytree(rt.base.inner_parameters())[0]
    # O2: eta equals the step size the native-rule inner step (ladder.py step) actually applies on one exposure chunk
    ch,_=packing.pack_rows([(x,(0,0)) for x in L.sentences(False)]);ids,seg,loc=ch[0]
    tg=np.concatenate([ids[1:],[rt.eos]]).astype(np.int32);mk=(seg>=0)&(np.concatenate([seg[1:],[-1]])==seg)
    with rt.mesh:
        pref=L.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
        newp,gn1,cf1,_=L.step(rt.base,clone_state(rt.state_all),jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
        dl=float(np.sqrt(sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(newp),jax.tree.leaves(rt.base.inner_parameters())))))
    R['O2_native_step']=dl/(float(gn1)*float(cf1));R['O2_rel']=abs(R['O2_native_step']-eta)/eta;R['O2_pass']=R['O2_rel']<=1e-2;del newp
    # O4: the measured R0 change along g_b matches the first-order prediction for a small step (central difference)
    base_r0=r0(rt.base);tot=sum(pred[('gb','A','L3',p)] for p in pairs);h=min(1.0,0.2/max(abs(tot),1e-30))
    mp_=r0(make(gb['A'],h));mm_=r0(make(gb['A'],-h));fd=sum((a-b)/(2*h) for a,b in zip(mp_['L3'],mm_['L3']))
    R['O4']={'h':h,'pred':tot,'finite_diff':fd,'rel':abs(fd-tot)/max(abs(tot),1e-30)};R['O4_pass']=R['O4']['rel']<=5e-2
    R['ALL_PASS']=all(v for k,v in R.items() if k.endswith('_pass'))
    _j.dump(R,open(OUT/'onestep_checks_s101.json','w'),indent=1,default=str);print(_j.dumps(R,indent=1,default=str),flush=True)
    if not R['ALL_PASS']:sys.exit(3)
    meta={'eta':eta,'coef':coef,'gnorm':gn,'scales':SC,'pred':{'|'.join(k):v for k,v in pred.items()},'r0_base':base_r0,'pairs':pairs,'xpairs':xpairs}
    _j.dump(meta,open(OUT/'onestep_meta_s101.json','w'),indent=1)
    e0=L.evaluate(rt.base)                                   # W0 is the same model for both worlds and both directions
    for vn,V in (('gb',gb),('g',g)):
        logs={}
        for w in 'AB':
            logs[w]=[{'state':0,'margins':e0,'r0':base_r0}]
            for sc in SC:
                m=make(V[w],eta*coef[w]*sc);logs[w].append({'state':sc,'step':eta*coef[w]*sc,'margins':L.evaluate(m),'r0':r0(m)});del m
                print('onestep',vn,w,sc,'done',flush=True)
                _j.dump({'log':logs[w],'meta':{'v':vn,'eta':eta,'coef':coef[w]}},open(OUT/f'ONESTEP_{vn}_s101_{w}.json','w'),indent=1)
def run_onestep2(ns):
    """One-step replication (STAGE=rep) and suppression-signal screen (STAGE=screen), PROTOCOL_stage01.md 2026-10-16.
    s bases: 'lr' = the ONESTEP definition, step = eta * c_w * s * v (c_w = clip coefficient of the isolated NTP g_w);
    'nm' (norm-matched, primary for the screen) = eta * c_w * s * |g_w| * v / |v|. For v = g both are identical.
    Signals: g (isolated NTP), gb (projection of g on U), and for the screen c1/c2 contrasts (suppress.py) in the iso and
    chunk conditions plus gntp (the masked write's own NTP direction over all chunks). Per signal the scales run upward
    and stop after the first scale with a canary rise > 0.10 (either canary, either world). Checks run first."""
    import json as _j,suppress as SP
    rt=ns['rt'];rt32=ns['rt32'];L=ns['L'];W=ns['W'];loss_grad=ns['loss_grad'];margin_grad=ns['margin_grad'];margin_val=ns['margin_val']
    prefix32=ns['prefix32'];dot=ns['dot'];norm=ns['norm'];inp=ns['inp'];pairs=ns['pairs'];sent=ns['sent'];OUT=ns['OUT'];m0=rt32.base
    SEED=ns['SEED'];STAGE=os.environ.get('STAGE','rep');T=L.T;R={'seed':SEED,'stage':STAGE}
    inner0,_=eqx.partition(rt.base,L.f);td=jax.tree_util.tree_structure(inner0);p0=jax.tree_util.tree_leaves(inner0)
    def make(v,step):
        vl=jax.tree_util.tree_leaves(v);assert len(vl)==len(p0)
        return L.set_inner(rt.base,jax.tree_util.tree_unflatten(td,[(p+(-step*x).astype(p.dtype)).astype(p.dtype) for p,x in zip(p0,vl)]))
    def r0(m):
        m32=L.to_fp32(m)
        with rt32.mesh,jax.default_matmul_precision('highest'):return [float(margin_val(m32,*inp[('L3',p)])) for p in pairs]
    def add(a,b):return b if a is None else jax.tree.map(lambda x,y:x+y,a,b)
    def scal(a,k):return jax.tree.map(lambda x:x*k,a)
    def wgrad(chunks):
        """token-weighted mean gradient over packed chunks: sum_c n_c * grad(mean loss of c) / sum_c n_c."""
        g=None;n=0
        for args,nc in chunks:
            if nc==0:continue
            g=add(g,scal(loss_grad(m0,*args),float(nc)));n+=nc
        return scal(g,1.0/n),n
    def iso_chunks(rows):
        ch,where=packing.pack_rows(rows);mks=[np.zeros(CH,bool) for _ in ch]
        for c_,a_,b_ in where:mks[c_][a_:b_]=True
        out=[]
        for (ids,seg,loc),mk in zip(ch,mks):
            tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32)
            pref=prefix32(m0,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
            out.append(((jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc)),int(mk.sum())))
        return out
    seg0=np.zeros(CH,np.int32);loc0=np.arange(CH,dtype=np.int32);pm0=jnp.asarray(packing.prefix_mask(seg0,W))
    def chunk_chunks(rows):
        out=[]
        for ids,mk,n,_ in rows:
            tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32);pref=prefix32(m0,jnp.asarray(ids),jnp.asarray(loc0),pm0)
            out.append(((jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg0),jnp.asarray(loc0)),n))
        return out
    with rt.mesh:eta=float(L.lr*rt.base.get_ilr_multiplier(clone_state(rt.state_all).get(rt.base.step_index)))
    g={};gb={};coef={};gn={}
    with rt32.mesh,jax.default_matmul_precision('highest'):
        U=None
        for p in pairs:U=add(U,margin_grad(m0,*inp[('L3',p)]))
        UU=dot(U,U)
        for w in 'AB':
            gg=None
            for c_ in sent[w]:gg=add(gg,loss_grad(m0,*c_))
            g[w]=scal(gg,1.0/len(sent[w]));del gg;gn[w]=norm(g[w]);coef[w]=min(1.0,L.clip/max(gn[w],1e-30))
            gb[w]=scal(U,dot(g[w],U)/UU)
            R[f'O3_{w}_rel']=abs(dot(gb[w],U)-dot(g[w],U))/max(abs(dot(g[w],U)),1e-30)
    R.update(eta=eta,gnorm=gn,coef=coef,gb_norm={w:norm(gb[w]) for w in 'AB'});R['O3_pass']=all(R[f'O3_{w}_rel']<=1e-4 for w in 'AB')
    R['O1_pass']=reuse._hash_pytree(make(g['A'],0.0).inner_parameters())[0]==reuse._hash_pytree(rt.base.inner_parameters())[0]
    ch,_=packing.pack_rows([(x,(0,0)) for x in L.sentences(False)]);ids,seg,loc=ch[0]
    tg=np.concatenate([ids[1:],[rt.eos]]).astype(np.int32);mk=(seg>=0)&(np.concatenate([seg[1:],[-1]])==seg)
    with rt.mesh:
        pref=L.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
        newp,gn1,cf1,_=L.step(rt.base,clone_state(rt.state_all),jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
        dl=float(np.sqrt(sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(newp),jax.tree.leaves(rt.base.inner_parameters())))))
    R['O2_rel']=abs(dl/(float(gn1)*float(cf1))-eta)/eta;R['O2_pass']=R['O2_rel']<=1e-2;del newp
    def step_for(v,w,basis,s):return eta*coef[w]*s*(1.0 if basis=='lr' else gn[w]/norm(v))
    # O5: the realised update norm of a norm-matched step equals eta*c*s*|g| (checked on g_b, s = 1)
    st=step_for(gb['A'],'A','nm',1);m_=make(gb['A'],st)
    dn=float(np.sqrt(sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m_.inner_parameters()),jax.tree.leaves(rt.base.inner_parameters())))));del m_
    R['O5']={'realised':dn,'target':eta*coef['A']*gn['A']};R['O5_pass']=abs(dn-R['O5']['target'])/R['O5']['target']<=1e-3
    sig={}
    if STAGE=='rep':plan=[('g_lr','g','lr',(1,2,4,8)),('gb_lr','gb','lr',(1,2,4,8)),('gb_nm','gb','nm',(1,2,4,8))];sig['g']=g;sig['gb']=gb
    else:
        plan=[]
        for nm_ in ('c1iso','c2iso','c1ch','c2ch'):plan+=[(nm_+'_nm',nm_,'nm',(1,2,4,8)),(nm_+'_lr',nm_,'lr',(1,8))]
        plan.append(('gntpch_nm','gntpch','nm',(1,2,4,8)))
        info={}
        with rt32.mesh,jax.default_matmul_precision('highest'):
            # K3: causality check of the contrast on single sentences (P before every X -> exactly zero; X before P -> nonzero)
            si=SP._sent_info(T.w,False);before=[i for i,s in enumerate(si) if all(x>s['p0'] for x in s['xs'])];after=[i for i,s in enumerate(si) if any(x<s['p0'] for x in s['xs'])]
            rx=SP.iso_rows(T.w,False,T.tok,'X');r0_=SP.iso_rows(T.w,False,T.tok,'null');k3={}
            for nm_,i in (('P_first',before[0]),('X_first',after[0])):
                a,_=wgrad(iso_chunks([rx[i]]));b,_=wgrad(iso_chunks([r0_[i]]));k3[nm_]=norm(jax.tree.map(lambda x,y:x-y,a,b))/max(norm(a),1e-30);del a,b
            R['K3']=k3;R['K3_pass']=k3['P_first']<=1e-6 and k3['X_first']>=1e-3
            for w in 'AB':
                sw=w=='B';gx,nx=wgrad(iso_chunks(SP.iso_rows(T.w,sw,T.tok,'X')))
                for c_,kind in (('c1iso','null'),('c2iso','Y')):
                    gk,nk=wgrad(iso_chunks(SP.iso_rows(T.w,sw,T.tok,kind)));sig.setdefault(c_,{})[w]=jax.tree.map(lambda x,y:x-y,gx,gk);info[(c_,w)]=(nx,nk);del gk
                del gx;rows,skx=SP.chunk_rows(T.w,sw,T.tok,rt.bos,rt.eos,'X');gx,nx=wgrad(chunk_chunks(rows))
                for c_,kind in (('c1ch','null'),('c2ch','Y')):
                    rows,sk=SP.chunk_rows(T.w,sw,T.tok,rt.bos,rt.eos,kind);gk,nk=wgrad(chunk_chunks(rows));sig.setdefault(c_,{})[w]=jax.tree.map(lambda x,y:x-y,gx,gk);info[(c_,w)]=(nx,nk,sk);del gk
                del gx;rows,_=SP.chunk_rows(T.w,sw,T.tok,rt.bos,rt.eos,'ntp');sig.setdefault('gntpch',{})[w],_=wgrad(chunk_chunks(rows))
        R['signal_norms']={c_:{w:norm(v[w]) for w in 'AB'} for c_,v in sig.items()};R['signal_dot_U']={c_:{w:dot(v[w],U) for w in 'AB'} for c_,v in sig.items()}
        R['cos_with_g']={c_:{w:dot(v[w],g[w])/(norm(v[w])*gn[w]) for w in 'AB'} for c_,v in sig.items()};R['loss_tokens']={f'{k[0]}|{k[1]}':v for k,v in info.items()}
        del g,gb
    R['ALL_PASS']=all(v for k,v in R.items() if k.endswith('_pass'))
    _j.dump(R,open(OUT/f'onestep2_{STAGE}_checks_s{SEED}.json','w'),indent=1,default=str);print(_j.dumps(R,indent=1,default=str),flush=True)
    if not R['ALL_PASS']:sys.exit(3)
    e0=L.evaluate(rt.base);base0=r0(rt.base)
    for tag,sn,basis,scales in plan:
        logs={w:[{'state':0,'margins':e0,'r0':base0}] for w in 'AB'}
        for s in scales:
            for w in 'AB':
                v=sig[sn][w];st=step_for(v,w,basis,s);m=make(v,st);logs[w].append({'state':s,'step':st,'margins':L.evaluate(m),'r0':r0(m)});del m
                _j.dump({'log':logs[w],'meta':{'signal':sn,'basis':basis,'eta':eta,'coef':coef[w],'gnorm':gn[w],'vnorm':norm(v)}},open(OUT/f'OS2_{tag}_s{SEED}_{w}.json','w'),indent=1)
            rise=max(logs[w][-1]['margins'][k]-e0[k] for w in 'AB' for k in ('_gen_nll','_tail_nll'))
            print('os2',tag,s,'canary rise',round(rise,4),flush=True)
            if rise>0.10:print('os2',tag,'damaged at s',s,'-> larger s skipped',flush=True);break
def run_nsweep(ns):
    """N-sweep (PROTOCOL_stage01.md): NTP g on the exposure sentences of N pairs only; b, r for those pairs, at W0."""
    import json as _j,random as _r
    rt32=ns['rt32'];L=ns['L'];T=ns['T'];W=ns['W'];prefix32=ns['prefix32'];loss_grad=ns['loss_grad'];margin_grad=ns['margin_grad']
    dot=ns['dot'];inp=ns['inp'];pairs=ns['pairs'];m0=rt32.base
    tri={t['id']:t for t in T.w['triples']};res=[]
    def chunks_for(w,S):
        _,_,spans=gw.realize(T.w,w=='B');sents=L.sentences(w=='B');keep=[x for x,s_ in zip(sents,spans) if tri[s_['t']]['pair'] in S]
        ch,_=packing.pack_rows([(x,(0,0)) for x in keep]);out=[]
        for ids,seg,loc in ch:
            tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32);mk=(seg>=0)&(np.concatenate([seg[1:],[-1]])==seg)
            pref=prefix32(m0,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
            out.append((jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc)))
        return out
    rng=_r.Random(11);plan=[]
    for N in (1,2,4,8):
        for i in range(6):plan.append((N,sorted(rng.sample(pairs,N),key=lambda s:int(s[1:]))))
    plan.append((15,list(pairs)))
    with rt32.mesh,jax.default_matmul_precision('highest'):
        for N,S in plan:
            a={}
            for w in 'AB':
                g=None;ch=chunks_for(w,set(S))
                for c_ in ch:
                    gi=loss_grad(m0,*c_);g=gi if g is None else jax.tree.map(lambda x,y:x+y,g,gi);del gi
                g=jax.tree.map(lambda x:x/len(ch),g)
                for fam in ('L3','FN'):
                    for p in S:u=margin_grad(m0,*inp[(fam,p)]);a[(w,fam,p)]=-dot(u,g);del u
                del g
            b=[(a[('A','L3',p)]-a[('B','L3',p)])/2 for p in S];r=[(a[('A','FN',p)]+a[('B','FN',p)])/2 for p in S]
            res.append({'N':N,'pairs':S,'b':b,'r':r});print('N',N,S,'b/r',np.mean(b)/np.mean(r),flush=True)
            _j.dump(res,open(ns['OUT']/'nsweep_s101.json','w'),indent=1)
    # fit log q = a + s log N on subset-level ratios (subsets with mean r <= 0 are excluded and counted)
    pts=[(np.log(x['N']),np.log(np.mean(x['b'])/np.mean(x['r']))) for x in res if np.mean(x['r'])>0 and np.mean(x['b'])>0]
    nexcl=len(res)-len(pts);X=np.asarray([p[0] for p in pts]);Y=np.asarray([p[1] for p in pts])
    s=float(np.polyfit(X,Y,1)[0]);bs=[]
    rr=np.random.default_rng(5)
    for _ in range(4000):
        q=[]
        for x in res:
            i=rr.integers(0,len(x['b']),len(x['b']));mb=np.mean(np.asarray(x['b'])[i]);mr=np.mean(np.asarray(x['r'])[i])
            if mb>0 and mr>0:q.append((np.log(x['N']),np.log(mb/mr)))
        if len(q)>=3:q=np.asarray(q);bs.append(np.polyfit(q[:,0],q[:,1],1)[0])
    lo,hi=np.percentile(bs,[2.5,97.5]);v='supports statistical dominance (slope ~ -1)' if (lo<=-1<=hi and hi<0) else ('no N dependence' if lo<=0<=hi else 'other')
    out={'slope':s,'ci':[float(lo),float(hi)],'excluded_subsets':nexcl,'verdict':v}
    _j.dump({'subsets':res,'fit':out},open(ns['OUT']/'nsweep_s101.json','w'),indent=1)
    md=['# N-sweep (seed 101, W0, NTP)\n','| N | mean b/r over subsets | subsets |','|---|---|---|']
    for N in (1,2,4,8,15):
        q=[np.mean(x['b'])/np.mean(x['r']) for x in res if x['N']==N];md.append(f'| {N} | {np.mean(q):+.4f} (range {min(q):+.4f} .. {max(q):+.4f}) | {len(q)} |')
    md.append(f"\n**slope s = {s:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {v}** (excluded subsets with b or r <= 0: {nexcl})")
    (ns['OUT']/'nsweep_s101.md').write_text('\n'.join(md)+'\n');print('\n'.join(md),flush=True)
def run_ctx3(ns):
    """Training-free shadowing check at W0 (PROTOCOL_stage01.md): NTP write direction on the exposure sentences when the
    sentence is written alone (plain), with the previous round's same-pair sentences in context (ctx = same fact), or with
    a same-length context of other pairs' sentences (irr). Loss on the current sentence only. b, r per pair, |g|, and
    the mean current-sentence loss."""
    import json as _j
    rt32=ns['rt32'];L=ns['L'];W=ns['W'];prefix32=ns['prefix32'];loss_grad=ns['loss_grad'];margin_grad=ns['margin_grad']
    dot=ns['dot'];norm=ns['norm'];inp=ns['inp'];pairs=ns['pairs'];m0=rt32.base;Batch=ns['Batch'];res={}
    @eqx.filter_jit
    def loss_val(m,ids_,tg_,mk_,pref_,seg_,loc_):
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:return MetaModel.lm_loss(m,Batch(input_ids=ids_,target_tokens=tg_,loss_masks=mk_),clone_state(rt32.state_suffix),prefix_outputs=pref_)[0]
        finally:packing.PACK_SUF.clear()
    with rt32.mesh,jax.default_matmul_precision('highest'):
        for v in ('plain','ctx','irr'):
            a={};gn={};lv={}
            for w in 'AB':
                ch,where=packing.pack_rows(L.l1_rows(w=='B',v));mks=[np.zeros(1024,bool) for _ in ch]
                for c_,a_,b_ in where:mks[c_][a_:b_]=True
                g=None;ls=[]
                for (ids,seg,loc),mk in zip(ch,mks):
                    tg=np.concatenate([ids[1:],[rt32.eos]]).astype(np.int32)
                    pref=prefix32(m0,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
                    args=(jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
                    gi=loss_grad(m0,*args);g=gi if g is None else jax.tree.map(lambda x,y:x+y,g,gi);del gi;ls.append(float(loss_val(m0,*args)))
                g=jax.tree.map(lambda x:x/len(ch),g);gn[w]=norm(g);lv[w]=float(np.mean(ls))
                for fam in ('L3','FN'):
                    for p in pairs:u=margin_grad(m0,*inp[(fam,p)]);a[(w,fam,p)]=-dot(u,g);del u
                del g
            b=[(a[('A','L3',p)]-a[('B','L3',p)])/2 for p in pairs];r=[(a[('A','FN',p)]+a[('B','FN',p)])/2 for p in pairs]
            res[v]={'b':b,'r':r,'gnorm':gn,'loss':lv};print(v,'b',np.mean(b),'r',np.mean(r),'|g|',gn,'loss',lv,flush=True)
            _j.dump(res,open(ns['OUT']/'ctx3_w0_s101.json','w'),indent=1)
    rng=np.random.default_rng(23)
    def ci(x):
        x=np.asarray(x);bs=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(4000)];return f'{x.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]'
    md=['# W0 write direction under three write contexts (seed 101)\n','| context | binding b | familiarity r | b / r | mean |g| | mean current-sentence loss |','|---|---|---|---|---|---|']
    for v,x in res.items():md.append(f"| {v} | {ci(x['b'])} | {ci(x['r'])} | {np.mean(x['b'])/np.mean(x['r']):+.4f} | {(x['gnorm']['A']+x['gnorm']['B'])/2:.3f} | {(x['loss']['A']+x['loss']['B'])/2:.3f} |")
    (ns['OUT']/'ctx3_w0_s101.md').write_text('\n'.join(md)+'\n');print('\n'.join(md),flush=True)
def main():
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    SEED=int(os.environ.get('SEED','101'));rt=Runtime();L=LD.Ladder(rt,SEED);rt32=L.rt32;W=int(rt32.cfg.model.sliding_window_size);T=L.T
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
    def margin_fn(m,ids_,tg_,pref_,seg_,loc_,wg_,wf_):
        """sum(lp * wg) - sum(lp * wf) under the empty-context readout (R0): weights mark gold / foil span positions."""
        packing.PACK_SUF['seg']=seg_;packing.PACK_SUF['loc']=loc_
        try:
            r=m.language_model.suffix_call(prefix_outputs=pref_,state=clone_state(rt32.state_suffix),seq=Batch(input_ids=ids_,target_tokens=tg_,loss_masks=jnp.ones_like(ids_,dtype=bool)))
            lg=r.logits.astype(jnp.float32);lp=jnp.take_along_axis(lg,tg_[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
        finally:packing.PACK_SUF.clear()
        return jnp.sum(lp*wg_)-jnp.sum(lp*wf_)
    margin_val=eqx.filter_jit(margin_fn)
    @eqx.filter_jit
    def margin_grad(m,*args):
        """gradient (inner params) of margin_fn."""
        return eqx.filter_grad(lambda m:margin_fn(m,*args))(m).inner_parameters()
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
    if os.environ.get('CTX3')=='1':
        ns=dict(locals());ns['OUT']=OUT;run_ctx3(ns);return
    if os.environ.get('ONESTEP2')=='1':
        sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'));ns=dict(locals());ns['OUT']=OUT;run_onestep2(ns);return
    if os.environ.get('ONESTEP')=='1':
        ns=dict(locals());ns['OUT']=OUT;run_onestep(ns);return
    if os.environ.get('NSWEEP')=='1':
        ns=dict(locals());ns['OUT']=OUT;run_nsweep(ns);return
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
