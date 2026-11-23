"""Key / error swap with ONE-STEP application and matched reads (job 1344m; spec v2 items G(ii) and H; E2E 3B JAX, frozen W0; new file; thread-server 11/23; EXPLORATORY; NOT queued: Hans decides after the final 1344k result). Same setting as e59_kv1344k.py (U12 world seeds 231..238, 8 chunks 16..127, conditions C0 = SAME_U1U2, SAMEVIS = OTHERONLY, NATVIS = NAT; update-path masks of e59_shadow.mk_uv; probes of e59_kvprobe.py). The three conditions are streamed one after the other; at every measured chunk the raw layer inputs x, x2 and error signals delta_z1, delta_z3, delta_z2 of all 8 prime layers are kept in host memory for the three conditions (+ the autodiff gradient), and when the LAST condition (NATVIS) reaches that chunk (the weights are W0 in all streams, only the attention masks differ) the combinations below are built, applied as ONE step with the ORIGINAL rule (L.sgd_apply: lr, global clip, ilr multiplier) to W0 and read.
G(ii): for the combination (A key, B error): G = sum_p x^A_p (x) delta^B_p for w1, w3 (x = ffn_prime_norm input) and w2 (x2) of all 8 layers, every position; weights W0 - step(G); metric = mean over the 16 read facts of the out-of-window margin (gold - best wrong, summed log p, neutral 8192 tail, fp32 reads of e55_gpu.Reader55) minus the W0 margin. Combinations: ACT[c] (autodiff gradient of condition c, c in C0, SAMEVIS, NATVIS = the real step), G[A,B] for A, B in {C0, SAMEVIS} (A = B must reproduce ACT), the NATVIS analogues, SIZE-RESTORED versions (error direction of B, per-position and per-layer size |delta_z2| of C0: delta^B * |delta^C0|/|delta^B|), COMMON-KEY versions (w2 key x2 = c^A_p u_l + x2^{perp}_p with the component c_p along the unit mean read key u_l of the 16 facts replaced by that of another condition: separates the fact-specific address from the shared (mean) component of the key), and the POSITION SPLIT of G[A,A]: the step built from the rows of the answer positions of the 16 read facts / of the other facts' answer positions / of the remaining positions, each applied with the clip coefficient of the FULL step (linear response; the three effects should add up to the full one).
H: weights W0, W_C0, W_SAMEVIS, W_NATVIS (one step each, from every measured chunk, with the autodiff gradient) x read contexts {neutral 8192 tail, SAME-FACT prefix (the sentence of that fact in chunk k-1 in front of the question, its answer tokens hidden from every later token by attention), OTHER-FACT prefix (the sentence of the fact (f+1) mod 16 of chunk k, its own answer tokens hidden)}: 12 cells of out-of-window margins per fact. The hidden answer tokens are asserted (the gold answer tokens occur in the visible prefix nowhere).
Saved: kvstep_s<seed>.json (per chunk, per combination / cell: mean margin change and the 16 per-fact values; validation of G[C0,C0] against the autodiff gradient). Analysis: e59_kvstep_analyze.py. NOT available: end-of-stream weights of NAT / band 1024 / ISO at seeds 71-78 (states are not stored), so H is done on the one-step weights only.
  python3 e59_kvstep.py --seed 231 --out O [--chunks 16,...] [--nch N] | --selftest"""
import sys,os,json,time,pathlib,argparse,itertools
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_shadow as SH
CH=1024;NL=8;NREAD=16
CONDS={'C0':'SAME_U1U2','SAMEVIS':('ha','OTHERONLY'),'NATVIS':('nat','NAT')}
def combos():
    """list of dict(name, A, B, size, common, pos) ; 'act' entries use the autodiff gradient"""
    L=[dict(name=f'ACT[{c}]',act=c) for c in CONDS]
    for A,B in itertools.product(('C0','SAMEVIS'),('C0','SAMEVIS')):L.append(dict(name=f'G[{A},{B}]',A=A,B=B))
    for A,B in (('NATVIS','C0'),('C0','NATVIS'),('NATVIS','NATVIS')):L.append(dict(name=f'G[{A},{B}]',A=A,B=B))
    for A,B in (('C0','SAMEVIS'),('SAMEVIS','SAMEVIS'),('C0','NATVIS'),('NATVIS','NATVIS')):L.append(dict(name=f'G[{A},{B}|size=C0]',A=A,B=B,size='C0'))
    for B in ('C0','SAMEVIS'):L.append(dict(name=f'G[SAMEVIS,{B}|common=C0]',A='SAMEVIS',B=B,common='C0'))
    L.append(dict(name='G[SAMEVIS,SAMEVIS|size=C0,common=C0]',A='SAMEVIS',B='SAMEVIS',size='C0',common='C0'));L.append(dict(name='G[C0,C0|common=SAMEVIS]',A='C0',B='C0',common='SAMEVIS'))
    L.append(dict(name='G[NATVIS,NATVIS|common=C0]',A='NATVIS',B='NATVIS',common='C0'))
    for c in ('C0','SAMEVIS','NATVIS'):
        for pos in ('own','other','rest'):L.append(dict(name=f'G[{c},{c}]@{pos}',A=c,B=c,pos=pos))
    return L
def build_G(store,keys_mean_unit,cmb,posmask,tau=1e-30):
    """-> dict name -> [NL, in, out] float32 arrays (w1, w2, w3) for the combination. store[cond] = dict(x [NL,T,2560], x2 [NL,T,I], dz1, dz3 [NL,T,I], dz2 [NL,T,2560]); posmask[class] = bool [T]; keys_mean_unit [NL,I]"""
    A,B=cmb['A'],cmb['B'];a,b=store[A],store[B];out={'w1':[],'w2':[],'w3':[]}
    for l in range(NL):
        x=a['x'][l];x2=a['x2'][l]
        if cmb.get('common'):
            u=keys_mean_unit[l];c_other=store[cmb['common']]['x2'][l]@u;c_a=x2@u;x2=x2+(c_other-c_a)[:,None]*u[None,:]
        s=np.ones(x.shape[0],np.float32)
        if cmb.get('size'):s=(np.linalg.norm(store[cmb['size']]['dz2'][l],axis=1)/(np.linalg.norm(b['dz2'][l],axis=1)+tau)).astype(np.float32)
        if cmb.get('pos'):s=s*posmask[cmb['pos']].astype(np.float32)
        d1=b['dz1'][l]*s[:,None];d3=b['dz3'][l]*s[:,None];d2=b['dz2'][l]*s[:,None]
        out['w1'].append(x.T@d1);out['w2'].append(x2.T@d2);out['w3'].append(x.T@d3)
    return {k:np.stack(v).astype(np.float32) for k,v in out.items()}
def pack_rows_masked(rows,CH=1024,one_per_chunk=False,rowmask=None):
    """copy of packing.pack_rows; positions listed in rowmask[id(ids)] get their OWN unique segment id (no other token of the chunk attends to them; they keep their positions)"""
    chunks=[];where=[];cur=[];seg=[];loc=[];s=0;uniq=[10**6]
    def flush():
        n=len(cur);chunks.append((np.asarray(cur+[cur[-1]]*(CH-n),np.int32),np.asarray(seg+[-1]*(CH-n),np.int32),np.asarray(loc+list(range(CH-n)),np.int32)))
    for ids,(a,b) in rows:
        if cur and (one_per_chunk or len(cur)+len(ids)>CH):flush();cur,seg,loc,s=[],[],[],0
        off=len(cur);where.append((len(chunks),off+a,off+b));cur+=list(ids);seg+=[s]*len(ids);loc+=list(range(len(ids)))
        for p in (rowmask or {}).get(id(ids),[]):seg[off+p]=uniq[0];uniq[0]+=1
        s+=1
    if cur:flush()
    return chunks,where
def find_sub(ids,sub,end):
    """all start positions of the token subsequence `sub` in ids[:end]"""
    n=len(sub);return [i for i in range(end-n+1) if list(ids[i:i+n])==list(sub)]
def selftest():
    # (1) masked packing: unique segment ids on the hidden positions, rest identical to packing.pack_rows
    rows=[([5,6,7,8,9,10],(3,5)),([1,2,3,4],(2,3))];rm={id(rows[0][0]):[1,2]}
    ch,wh=pack_rows_masked(rows,16,False,rm);seg=ch[0][1];assert list(seg[:6])==[0,10**6,10**6+1,0,0,0] and list(seg[6:10])==[1]*4 and wh==[(0,3,5),(0,8,9)],(list(seg[:10]),wh)
    ch0,wh0=pack_rows_masked(rows,16,False,None);assert list(ch0[0][1][:6])==[0]*6
    # (2) G assembly: A=B reproduces the plain sum; size restore keeps the direction; common restore changes only the key along u
    rng=np.random.default_rng(0);T=12;H=6;I=9;mk=lambda:dict(x=rng.normal(size=(NL,T,H)).astype(np.float32),x2=rng.normal(size=(NL,T,I)).astype(np.float32),dz1=rng.normal(size=(NL,T,I)).astype(np.float32),dz3=rng.normal(size=(NL,T,I)).astype(np.float32),dz2=rng.normal(size=(NL,T,H)).astype(np.float32))
    st={c:mk() for c in CONDS};u=rng.normal(size=(NL,I)).astype(np.float32);u/=np.linalg.norm(u,axis=1,keepdims=True);pm={'own':np.arange(T)<3,'other':(np.arange(T)>=3)&(np.arange(T)<6),'rest':np.arange(T)>=6}
    by={c['name']:c for c in combos() if 'act' not in c}
    G=build_G(st,u,by['G[C0,C0]'],pm);assert np.allclose(G['w2'][2],st['C0']['x2'][2].T@st['C0']['dz2'][2],rtol=1e-4,atol=1e-4)
    Gs=build_G(st,u,by['G[SAMEVIS,SAMEVIS|size=C0]'],pm);dn=np.linalg.norm(st['C0']['dz2'][1],axis=1);sc=dn/np.linalg.norm(st['SAMEVIS']['dz2'][1],axis=1);assert np.allclose(Gs['w2'][1],st['SAMEVIS']['x2'][1].T@(st['SAMEVIS']['dz2'][1]*sc[:,None]),rtol=1e-4,atol=1e-4)
    Gc=build_G(st,u,by['G[SAMEVIS,SAMEVIS|common=C0]'],pm);x2c=st['SAMEVIS']['x2'][3]+((st['C0']['x2'][3]@u[3])-(st['SAMEVIS']['x2'][3]@u[3]))[:,None]*u[3][None,:];assert np.allclose(Gc['w2'][3],x2c.T@st['SAMEVIS']['dz2'][3],rtol=1e-4,atol=1e-4) and np.allclose((x2c@u[3]),st['C0']['x2'][3]@u[3],atol=1e-4)
    P=sum(build_G(st,u,by[f'G[C0,C0]@{c}'],pm)['w2'] for c in ('own','other','rest'));assert np.allclose(P,G['w2'],rtol=1e-3,atol=1e-3),'the position classes must add up to the full G'
    print(f'e59_kvstep selftest ok: {len(combos())} combinations ({sum(1 for c in combos() if "act" in c)} autodiff), masked packing, G assembly, size / common restoration, position split adds up')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_read as R,e55_gpu as RG,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_kvprobe as KP
    from tokenizers import Tokenizer
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);chunks=[int(x) for x in a.chunks.split(',')];t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    if os.environ.get('E59_CPU_TEST'):
        _cur=jax.nn.dot_product_attention;jax.nn.dot_product_attention=lambda *a_,**k_:_cur(*a_,**dict(k_,implementation='xla'))
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,R55.forbid_list(w));tkz=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    keys=np.load(a.keys) if a.keys else np.load(out/f'kv_s{seed}_keys.npy');assert keys.shape[0]==NREAD
    ku=keys[:,0].mean(0);ku=(ku/np.linalg.norm(ku,axis=-1,keepdims=True)).astype(np.float32)        # [NL, I] unit mean read key per layer (first answer token)
    with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));Pout=LD.packing.Packed(T_,raw,GP=2)
    def margins(S_):                                                                     # per fact (16): gold - best wrong
        S_=np.asarray(S_)[:NREAD];g=np.asarray([items[i]['gold_idx'] for i in range(NREAD)]);k=np.arange(NREAD);gold=S_[k,g];w_=S_.copy();w_[k,g]=-np.inf;return gold-w_.max(1)
    def read_out(m):
        with jax.default_matmul_precision('highest'):return margins(Pout.r1(rd.to_fp32(m)) .reshape(len(items),G.NCAND))
    W0m=read_out(rt.base);lg(f'seed {seed}: W0 margin of the 16 read facts mean {W0m.mean():.3f}')
    base=rt.base;inner0,outer0=eqx.partition(base,rd.f)
    def model_from(newp):return eqx.combine(newp,outer0)
    # H contexts: prefix sentences with hidden answer tokens (built per measured chunk)
    gold_tok=[np.asarray(raw[i*G.NCAND+items[i]['gold_idx']][0][raw[i*G.NCAND+items[i]['gold_idx']][1][0]+1:raw[i*G.NCAND+items[i]['gold_idx']][1][1]+1]) for i in range(NREAD)]
    def prefix_rows(k,mode):
        """rows of the 16 facts (16 candidates each) with the planted prefix sentence; -> raw rows, rowmask"""
        its=[];hid=[]
        for f in range(NREAD):
            if mode=='same':ck=k-1;fs=f
            else:ck=k;fs=(f+1)%NREAD
            sn=[x for x in w['chunks'][ck]['sents'] if int(x['fact'])==fs][0];toks=ids0[ck*CH+int(sn['start']):ck*CH+int(sn['end'])];txt=tkz.decode(toks.tolist())
            its.append(dict(prefix=txt+'\n'+items[f]['prefix'],cands=items[f]['cands']))
        rr=R55.rows_for(LD,rt,its);rm={}
        for f in range(NREAD):
            fs=f if mode=='same' else (f+1)%NREAD
            for c in range(G.NCAND):
                ids,(a_,b_)=rr[f*G.NCAND+c];pos=find_sub(ids,gold_tok[fs],a_);assert len(pos)==1,('the hidden answer must occur exactly once in the prefix',f,mode,pos)
                rm[id(ids)]=list(range(pos[0],pos[0]+len(gold_tok[fs])))
                if mode=='same':assert not find_sub(ids,gold_tok[f],a_)[1:],'answer visible'
        return rr,rm
    def read_prefix(k,mode,models):
        rr,rm=prefix_rows(k,mode);orig=LD.packing.pack_rows
        LD.packing.pack_rows=lambda rows,CH_=1024,opc=False:pack_rows_masked(rows,CH_,opc,rm)
        try:
            with rt32.mesh,jax.default_matmul_precision('highest'):PP=LD.packing.Packed(LD._TailCtx(rt32,list(out_tail)),rr,GP=2)
        finally:LD.packing.pack_rows=orig
        res={}
        with jax.default_matmul_precision('highest'):
            for nm,m in models.items():
                S_=np.asarray(PP.r1(rd.to_fp32(m))).reshape(NREAD,G.NCAND);g=np.asarray([items[i]['gold_idx'] for i in range(NREAD)]);kk=np.arange(NREAD);gold=S_[kk,g];w_=S_.copy();w_[kk,g]=-np.inf;res[nm]=(gold-w_.max(1)).tolist()
        return res
    @eqx.filter_jit
    def fwd_probe(mp,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=mp.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        return jnp.sum(o.logits[:8,:8].astype(jnp.float32))
    def loss_fn(mm,st_suf,c,pref_,kmask,wt):
        s3.S3SUF['kmask']=kmask
        try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
    @eqx.filter_jit
    def ref_grad(m,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(m,st_suf,c,pref_,kmask,wt);return g.inner_parameters(),nll
    @eqx.filter_jit
    def bwd_probe(mp,st_suf,c,pref_,kmask,wt):
        (_l,nll),g=eqx.filter_value_and_grad(loss_fn,has_aux=True)(mp,st_suf,c,pref_,kmask,wt);return nll,jnp.sum(KP.WHERE(g).w2.weight.astype(jnp.float32))
    def named(g):
        d={}
        for pth,leaf in jax.tree_util.tree_flatten_with_path(g)[0]:
            s=jax.tree_util.keystr(pth)
            for nm in ('w1','w2','w3'):
                if f'.{nm}.' in s+'.' or f'{nm}.weight' in s:d[nm]=(pth,leaf)
        assert sorted(d)==['w1','w2','w3'];return d
    def tree_with(gtemplate,arrs):
        """gradient pytree like gtemplate with the w1 / w2 / w3 leaves replaced (cast to the leaf dtype)"""
        paths,treedef=jax.tree_util.tree_flatten_with_path(gtemplate);new=[]
        for pth,leaf in paths:
            s=jax.tree_util.keystr(pth);rep=None
            for nm in ('w1','w2','w3'):
                if f'.{nm}.' in s+'.' or f'{nm}.weight' in s:rep=jnp.asarray(arrs[nm],leaf.dtype)
            new.append(rep if rep is not None else leaf)
        return jax.tree_util.tree_unflatten(treedef,new)
    RES=dict(seed=seed,chunks=chunks,W0_margin=W0m.tolist(),rows=[],H=[],validation=[],args=vars(a));CMB=[c for c in combos()];STORE={}                  # STORE[(cond,k)] = dict arrays; GRAD[(cond,k)] = dict(w1,w2,w3 np arrays, tree)
    GRAD={};cond_list=list(CONDS)
    for cn in cond_list:
        arm=CONDS[cn];cnt=[0];L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{cn}',canary=can);L.rec_cache={}
        def ws(m,st_all,st_suf,c,pref,kmask,cnt=cnt,cn=cn,L=L):
            k=cnt[0];cnt[0]+=1
            if k in chunks:
                t1=time.monotonic();valid=np.asarray(c.loss_masks,np.float32);Wall=SH.token_weights(ap[k],valid>0)['all'];wt=jnp.asarray(Wall)
                KP.CAP['fwd'].clear();o=fwd_probe(KP.install(m,'fwd'),st_suf,c,pref,kmask);jax.block_until_ready(o);jax.effects_barrier();xs=dict(KP.CAP['fwd'])
                g,nll=ref_grad(m,st_suf,c,pref,kmask,wt);nl_=named(g);GRAD[(cn,k)]=dict(tree=g,**{nm:np.asarray(v[1],np.float32) for nm,v in nl_.items()})
                KP.CAP['bwd'].clear();nll2,_s=bwd_probe(KP.install(m,'bwd'),st_suf,c,pref,kmask,wt);jax.block_until_ready(nll2);jax.effects_barrier();bw=dict(KP.CAP['bwd'])
                STORE[(cn,k)]=dict(x=np.stack([xs[l][0] for l in range(NL)]).astype(np.float32),x2=np.stack([xs[l][1] for l in range(NL)]).astype(np.float32),dz1=np.stack([bw[('z1',l)] for l in range(NL)]).astype(np.float32),dz3=np.stack([bw[('z3',l)] for l in range(NL)]).astype(np.float32),dz2=np.stack([bw[('z2',l)] for l in range(NL)]).astype(np.float32))
                if cn==cond_list[-1]:
                    st={c_:STORE.pop((c_,k)) for c_ in cond_list};T=nll.shape[0];posm={'own':np.zeros(T,bool),'other':np.zeros(T,bool),'rest':np.zeros(T,bool)}
                    for d in ap[k]:
                        for t in d['second']:
                            if 0<=t-1<T:posm['own' if d['fact']<NREAD else 'other'][t-1]=True
                    posm['rest']=~(posm['own']|posm['other']);row=dict(k=k,combos={},posn={kk:int(v.sum()) for kk,v in posm.items()})
                    # validation: G[C0,C0] vs the autodiff gradient of C0
                    Gcc=build_G(st,ku,[c_ for c_ in CMB if c_.get('name')=='G[C0,C0]'][0],posm);ga=GRAD[('C0',k)];RES['validation'].append(dict(k=k,**{nm:float(np.linalg.norm(Gcc[nm]-ga[nm])/(np.linalg.norm(ga[nm])+1e-30)) for nm in ('w1','w2','w3')}))
                    models={'W0':m};step={}
                    for cmb in CMB:
                        if 'act' in cmb:gt=GRAD[(cmb['act'],k)]['tree']
                        else:
                            Ga=build_G(st,ku,cmb,posm)
                            if cmb.get('pos'):                                       # linear response: the clip coefficient of the FULL step G[c,c] is used for every position class
                                full=row['combos'][f"G[{cmb['A']},{cmb['A']}]"]['coef'];fac=full                         # step = coef_full x G_class (the clip of sgd_apply does not bind again because |coef_full x G_class| <= |coef_full x G| <= 1)
                                Ga={nm:v*fac for nm,v in Ga.items()}
                            gt=tree_with(GRAD[('C0',k)]['tree'],Ga)
                        newp,gn,coef=L.sgd_apply(m,st_all,gt);mn=model_from(newp);mg=read_out(mn);row['combos'][cmb['name']]=dict(gnorm=float(gn),coef=float(coef),dmargin=float((mg-W0m).mean()),dmargin_facts=(mg-W0m).tolist())
                        if cmb.get('act') is not None:models['W_'+cmb['act']]=mn
                        del gt,newp
                    # position-split effects use the clip coefficient of the FULL step of the same condition (linear response)
                    row['H']={}
                    for mode in ('same','other'):row['H'][mode]=read_prefix(k,mode,models)
                    row['H']['neutral']={nm:(margins(Pout.r1(rd.to_fp32(mm)).reshape(len(items),G.NCAND))).tolist() for nm,mm in models.items()}
                    RES['rows'].append(row);lg(f'seed {seed} k={k}: combos done ({len(CMB)}) + H in {time.monotonic()-t1:.0f}s; G[C0,C0] vs autodiff rel.err w2 {RES["validation"][-1]["w2"]:.2e}; dmargin ACT C0 {row["combos"]["ACT[C0]"]["dmargin"]:.3f} G[C0,C0] {row["combos"]["G[C0,C0]"]["dmargin"]:.3f}')
                    for kk in [x for x in list(GRAD) if x[1]==k]:GRAD.pop(kk)
            return m.inner_parameters(),jnp.asarray(0.0,jnp.float32),jnp.asarray(1.0,jnp.float32),jnp.asarray(0.0,jnp.float32)
        L.write_step=ws;uv=SH.mk_uv(arm,w,seed,lg);t0=time.monotonic();r=L.run_stream(NM.ENGINE,1,ids0,read_chunks=[],canary_chunks=[],nch=(a.nch or nch),tag=f'kvstep_{cn}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=w['chunks'],uv=uv);r.pop('model',None);lg(f'seed {seed} {cn} ({arm}): stream done in {time.monotonic()-t0:.0f}s')
        L.rec_cache=None;del L
    json.dump(RES,open(out/f'kvstep_s{seed}.json','w'),default=float);lg('wrote',out/f'kvstep_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--chunks',default=','.join(map(str,SH.CHUNKS)));ap_.add_argument('--nch',type=int,default=0);ap_.add_argument('--keys',default='');ap_.add_argument('--selftest',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
