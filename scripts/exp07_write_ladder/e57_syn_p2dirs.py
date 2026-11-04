"""E57 synthetic s64, P2 pre-check of the update direction (new file; GPU forward / backward at W0 weights, NO stream training; investigation thread 'binding learned fast at few exposures', Hans 11/04 12:58 KST; design E57 122-AB addendum 16): at the chunks 16 / 32 / 64 / 96 / 127 of the stream of seed 64 (cell v3_16_1, ALLQ256 mask on every row, as in the write arms) the gradient of the inner MLP (feed_forward_prime w1 / w2 / w3 of the 8 suffix layers) of the weights W0 for two losses: ALLQ256 plain mean CE, and M3 w=8 (weight 8 on the deciding rows = top 30% by window gain). Five directions: raw gradient; sign (the first Adam step); input-side whitening G -> C^-1 G with C = X^T X / N + rho tr(C) / d I for rho = 0.1 and 0.01 (X = the input of the matrix: the ffn_prime_norm output for w1 / w3, silu(w1 x) * w3 x for w2, per layer, from an explicit forward of the suffix blocks that is checked against suffix_call); Muon (Newton-Schulz, 5 iterations, per layer matrix). Each direction is scaled to the update norm 0.5 x the NAT norm of that chunk (E55_v3a_NAT_s64) and applied ONCE to W0; the 16 facts are read per token (e57_syn_tokread.r1_tokens, the read of jobs 676 / 710) and the canary is measured. Stage analyze: per loss and direction first-token dA, d(A-B), binding share d(A-B) / dA, branch-token and first-token rank ratio, 95% bootstrap CIs over the 16 facts (the five chunks merged by the per-fact mean), paired differences to the raw gradient and to sign, cosine with the raw gradient, canary change after the step.
  python3 e57_syn_p2dirs.py --stage run --out O | --stage analyze --out O | --selftest"""
import sys,json,time,gc,hashlib,argparse,pathlib,types
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics',ROOT/'e2e'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;WIN_=8*CH;CHUNKS=(16,32,64,96,127);DIRS=('raw','sign','white01','white001','muon');LOSSES=('plain','m3w8')
def ns5(G,steps=5,xp=np):
    """Newton-Schulz orthogonalisation of one matrix (quintic coefficients of Muon)"""
    a,b,c=3.4445,-4.7750,2.0315;X=G/(xp.sqrt((G*G).sum())+1e-7);tr=X.shape[0]>X.shape[1]
    if tr:X=X.T
    for _ in range(steps):
        A=X@X.T;B=b*A+c*(A@A);X=a*X+B@X
    return X.T if tr else X
def whiten(G,X,rho,xp=np):
    """G [d_in, d_out], X [N, d_in] -> C^-1 G with C = X^T X / N + rho tr(C)/d I"""
    N,d=X.shape;C=(X.T@X)/N;C=C+rho*xp.trace(C)/d*xp.eye(d,dtype=C.dtype);return xp.linalg.solve(C,G)
def selftest():
    rng=np.random.default_rng(0);G=rng.standard_normal((40,100)).astype(np.float64);U=ns5(G);s=np.linalg.svd(U,compute_uv=False);assert s.min()>0.5 and s.max()<1.5,(s.min(),s.max())
    X=rng.standard_normal((2000,30));Gm=rng.standard_normal((30,7));W=whiten(Gm,X,1e-9);C=X.T@X/2000;assert np.allclose(C@W,Gm,atol=1e-6) and np.allclose(whiten(Gm,np.eye(30)*np.sqrt(30),0.0),Gm)
    Xa=X*np.linspace(0.1,3,30);Wa=whiten(Gm,Xa,0.1);assert np.isfinite(Wa).all()
    print('e57_syn_p2dirs selftest ok')
def run(a):
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);import jax,jax.numpy as jnp,equinox as eqx,e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G,e57_syn_pass as SP,e57_syn_tokread as TR
    from ttt.model.transformer import Block
    lg=lambda *x:print(*x,flush=True);seed=64;ro=[16,32,64,128]
    ntg=[e['update_norm'] for e in json.load(open(EXP/'E55_v3a_NAT_s64.saved.json'))['log']];M30=np.load(EXP/'e57_synq/syn_s64_m30_masks.npz',allow_pickle=True)
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed),seed,read_apps=set(ro),arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];ids=np.asarray(ids);nch=128;P_=R.plan(w,enc);items=P_['items'];tok=rt.tok
    raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w));Pk=rd.packed(raw,out_tail);tokids=[[int(x) for x in ids_[a_+1:b_+1]] for ids_,(a_,b_) in raw]
    def tokens(m):
        with LD.jax.default_matmul_precision('highest'):return [x.tolist() for x in TR.r1_tokens(Pk,rt32,rd.to_fp32(m))]
    ip=rt.base.inner_parameters();leaves=jax.tree_util.tree_flatten_with_path(ip)[0];kinds=[]
    for pth,lf in leaves:
        ks=jax.tree_util.keystr(pth);kinds.append('in' if ('w1' in ks or 'w3' in ks) else 'mid' if 'w2' in ks else 'other');lg('inner leaf',ks,lf.shape,lf.dtype)
    assert 'other' not in kinds,kinds
    def tree_norm(t):return float(jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32)**2) for x in jax.tree.leaves(t))))
    def tree_dot(a_,b_):return float(sum(jnp.sum(x.astype(jnp.float32)*y.astype(jnp.float32)) for x,y in zip(jax.tree.leaves(a_),jax.tree.leaves(b_))))
    S3SUF=s3.S3SUF
    @eqx.filter_jit
    def gradf(m,st,c,pref,kmask,wts):
        def total(mm):
            S3SUF['kmask']=kmask
            try:out=mm.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
            finally:S3SUF.clear()
            lp=jax.nn.log_softmax(out.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wts*nll),nll
        (_l,nll),g=eqx.filter_value_and_grad(total,has_aux=True)(m);valid=c.loss_masks.astype(jnp.float32);return g.inner_parameters(),jnp.sum(valid*nll)/jnp.maximum(jnp.sum(valid),1e-10)
    @eqx.filter_jit
    def acts(m,st,c,pref,kmask):
        lm=m.language_model;H=lm.model.h;NL=jax.tree_util.tree_leaves(H.suffix_blocks)[0].shape[0];S3SUF['kmask']=kmask
        try:
            layer=lambda i:jax.tree.map(lambda a_:a_[i] if eqx.is_array(a_) else a_,H.suffix_blocks);stl=lambda i:jax.tree.map(lambda a_:a_[i],st);x=pref;xin=[];xmid=[]
            for i in range(NL):
                b=layer(i);cfg_b=b.config;rmsf=lambda n,u:n(u)
                so,_=b.seq_modeling_forward(lambda blk,inp,sq,s_:type(blk).__call__(blk,inp,sq,s_,is_prefix=False),rmsf,b.seq_norm,b.seq_modeling_block,b.seq_post_norm,x,stl(i),c)
                x=x+so;f=b.feed_forward_prime;inp=jax.vmap(lambda u:b.ffn_prime_norm(u))(x) if cfg_b.pre_norm else x
                xin.append(inp);z1=f.w1(inp);z3=f.w3(inp);mid=jax.nn.silu(z1)*z3;xmid.append(mid);o=f.w2(mid)
                if cfg_b.post_norm:o=jax.vmap(lambda u:b.ffn_prime_post_norm(u))(o)
                x=x+o;fo=b.ffn_forward(lambda ff,u:ff(u),rmsf,b.ffn_norm,b.feed_forward,b.ffn_post_norm,x);x=x+fo
            ref=lm.suffix_call(prefix_outputs=pref,state=st,seq=c).last_hidden_states
            return jnp.stack(xin).astype(jnp.float32),jnp.stack(xmid).astype(jnp.float32),jnp.max(jnp.abs(jax.vmap(lm.model.ln_f)(x)-ref)),jnp.max(jnp.abs(ref))
        finally:S3SUF.clear()
    def map_leaves(fn,tree):
        lv,td=jax.tree_util.tree_flatten(tree);return jax.tree_util.tree_unflatten(td,[fn(k_,x) for k_,x in zip(kinds,lv)])
    def d_sign(g):return jax.tree.map(jnp.sign,g)
    def d_muon(g):return jax.tree.map(lambda x:jax.vmap(lambda M_:ns5(M_.astype(jnp.float32),5,jnp))(x),g)
    def d_white(g,xin,xmid,rho):
        def f(kind,x):
            X=xin if kind=='in' else xmid;return jnp.stack([whiten(x[l].astype(jnp.float32),X[l],rho,jnp) for l in range(x.shape[0])])
        return map_leaves(f,g)
    store={};info={};t00=time.monotonic()
    def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
        keep=np.asarray(keep,bool);kp=CM.band_keep(keep,256);pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(kp));c=ctx['c'];st=ctx['st_w']
        valid=np.asarray(c.loss_masks,np.float32);dec=CM.up(M30[f'qb_TOP30_{k}'],CH).astype(np.float32);nv=max(valid.sum(),1e-10)
        W_={'plain':valid/nv,'m3w8':valid*(1.0+7.0*dec)/nv}
        with rt.mesh,jax.default_matmul_precision('highest'):
            xin,xmid,hd,hmax=acts(m,st,c,pm,jnp.asarray(kp));info[k]=dict(hidden_maxdiff=float(hd),hidden_max=float(hmax));lg(f'  chunk {k}: explicit forward vs suffix_call max|diff| {float(hd):.2e} (max {float(hmax):.2f})')
            for ln in LOSSES:
                g,ce=gradf(m,st,c,pm,jnp.asarray(kp),jnp.asarray(W_[ln]));gn=tree_norm(g);dirs={'raw':g,'sign':d_sign(g),'white01':d_white(g,xin,xmid,0.1),'white001':d_white(g,xin,xmid,0.01),'muon':d_muon(g)}
                for dn,d in dirs.items():
                    nd=tree_norm(d);scale=0.5*ntg[k]/nd;f32=jnp.float32
                    newp=jax.tree.map(lambda p_,x:(p_.astype(f32)-scale*x.astype(f32)).astype(p_.dtype),m.inner_parameters(),d)
                    store[(k,ln,dn)]=[np.asarray(x) for x in jax.tree.leaves(newp)];info[(k,ln,dn)]=dict(cos_with_raw=tree_dot(d,g)/(nd*gn),scale=scale,dir_norm=nd,grad_norm=gn,ce=float(ce))
                lg(f'  chunk {k} loss {ln}: grad norm {gn:.3f}, directions built ({time.monotonic()-t00:.0f}s)')
                del dirs,g;gc.collect()
        return dict(done=True)
    fn.wants_ctx=True
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
    del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();lg(f'natural K,V recorded in {time.monotonic()-t00:.0f}s')
    L=C.Ladder55(rt,save_dir=out/'states'/'p2',canary=None);L.rec_cache=rc;_ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]
    L.run_stream('NAT',1,ids,nch=nch,tag='p2dirs',log=lambda *x:None,chunks_meta=[{'sents':[]} for _ in range(nch)],diag_fn=fn,diag_chunks=set(CHUNKS))
    # reads of the stored models
    treedef=jax.tree_util.tree_structure(rt.base.inner_parameters());res={'W0':tokens(rt.base),'tokids':tokids,'chunks':list(CHUNKS),'info':{},'tokens':{},'canary_w0':can(rt.base)}
    for k,v in info.items():
        if isinstance(k,int):res['info'][str(k)]=v
    for key in sorted(store):
        k,ln,dn=key;newp=jax.tree_util.tree_unflatten(treedef,[jnp.asarray(x) for x in store[key]]);m2=L.set_inner(rt.base,newp);res['tokens']['%d|%s|%s'%key]=tokens(m2);cn=can(m2);i_=dict(info[key]);i_['canary_gen_change']=cn['_gen_nll']-res['canary_w0']['_gen_nll'];i_['canary_tail_change']=cn['_tail_nll']-res['canary_w0']['_tail_nll'];res['info']['%d|%s|%s'%key]=i_
        lg(f'  read {key}: canary gen {i_["canary_gen_change"]:+.4f} tail {i_["canary_tail_change"]:+.4f} cos {i_["cos_with_raw"]:+.3f} ({time.monotonic()-t00:.0f}s)')
    (out/'e57_syn_p2dirs.json').write_text(json.dumps(res,default=float));lg('wrote',out/'e57_syn_p2dirs.json')
def analyze(a):
    import e57_syn_tokread as TR
    out=pathlib.Path(a.out);d=json.load(open(out/'e57_syn_p2dirs.json'));meta=json.load(open(EXP/'e57_synq/E55_synq_ALLQ256_s64.read.json'))['meta'];tokids=d['tokids'];W0=TR.metrics(d['W0'],tokids,meta);facts=sorted(W0);M={k:TR.metrics(t,tokids,meta) for k,t in d['tokens'].items()}
    def pf(ln,dn,fn_):                            # per-fact mean over the five chunks
        v=[]
        for f in facts:
            xs=[fn_(M['%d|%s|%s'%(k,ln,dn)][f],W0[f]) for k in d['chunks']];xs=[x for x in xs if x is not None];v.append(np.mean(xs) if xs else np.nan)
        return np.asarray(v)
    FN={'first_dA':lambda m,w:m['A_first']-w['A_first'],'first_dAB':lambda m,w:(m['A_first']-w['A_first'])-(m['B_first']-w['B_first']) if m['B_first'] is not None and w['B_first'] is not None else None,'rr_branch_improvement':lambda m,w:(w['rank_branch']-m['rank_branch'])/16.0,'rr_first_improvement':lambda m,w:(w['rank_first']-m['rank_first'])/16.0,'full_dA':lambda m,w:m['A_full']-w['A_full']}
    res={'chunks':d['chunks'],'hidden_check':{k:v for k,v in d['info'].items() if k in [str(c) for c in d['chunks']]},'by_loss':{}}
    for ln in LOSSES:
        r={}
        for dn in DIRS:
            e={k:TR.boot(pf(ln,dn,f)) for k,f in FN.items()};dA=pf(ln,dn,FN['first_dA']);dAB=pf(ln,dn,FN['first_dAB']);ok=~np.isnan(dAB)
            e['binding_share_dAB_over_dA']=TR.ratio(dAB[ok],dA[ok]) if ok.sum()>2 and abs(dA[ok].mean())>1e-9 else None
            inf=[d['info']['%d|%s|%s'%(k,ln,dn)] for k in d['chunks']];e['cos_with_raw_mean']=float(np.mean([x['cos_with_raw'] for x in inf]));e['canary_gen_change_mean']=float(np.mean([x['canary_gen_change'] for x in inf]));e['canary_tail_change_mean']=float(np.mean([x['canary_tail_change'] for x in inf]));e['canary_gen_change_per_chunk']=[x['canary_gen_change'] for x in inf];e['scale_mean']=float(np.mean([x['scale'] for x in inf]))
            for ref in ('raw','sign'):
                if ref!=dn:e[f'minus_{ref}']={k:TR.boot(pf(ln,dn,f)-pf(ln,ref,f)) for k,f in FN.items() if k!='full_dA'}
            r[dn]=e
        res['by_loss'][ln]=r
    (out/'e57_syn_p2_report.json').write_text(json.dumps(res,indent=1,default=float));print('wrote',out/'e57_syn_p2_report.json')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    {'run':run,'analyze':analyze}[a.stage](a)
if __name__=='__main__':main()
