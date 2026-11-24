"""Split of the REAL one-step effect by position class (job 1344p; GPU; new file derived from e59_kvstep.py; thread-server 11/24; Hans 'B' (seeds 8, chunks 16 and 80); EXPLORATORY; NOT queued): same world / conditions (C0 and SAMEVIS) / one-step rule / reads as job 1344m (L.sgd_apply, window 8, out-of-window margin of the 16 read facts, W0-relative), but the step is built from G = sum_p x_p (x) delta_p restricted to one class of positions and applied with the clip coefficient of the FULL step (linear response). Classes (a partition of all positions of the chunk): ans = answer tokens of the 16 read facts, name = first-entity (name) tokens of the read facts, frame = the other tokens of the sentences of the read facts, extra = all tokens of the 16 extra (U2) facts' sentences, filler = every other position. Fact level (facts 0..3): the step built from the answer tokens of fact f only ('f<f>_ans') or its name tokens only ('f<f>_name'); the margin change of ALL 16 read facts is stored, so the diagonal (own fact) and off-diagonal (the other read facts) can be separated. Also stored: the real full step ACT[c], the full G[c,c] (reproduction check), validation of G[C0,C0] against the autodiff gradient, token counts per class. Output kvpos_s<seed>.json. Analysis e59_kvpos_analyze.py.
  python3 e59_kvpos.py --seed 231 --out O --keys K [--chunks 16,80] [--nch N] | --selftest"""
import sys,os,json,time,pathlib,argparse,itertools
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_shadow as SH
CH=1024;NL=8;NREAD=16
CONDS={'C0':'SAME_U1U2','SAMEVIS':('ha','OTHERONLY')};CLASSES=('ans','name','frame','extra','filler');NFACT=4
def combos():
    """real steps, full G, the 5 position classes, the fact-level steps of facts 0..3 (answer tokens / name tokens), for every condition"""
    L=[dict(name=f'ACT[{c}]',act=c) for c in CONDS]+[dict(name=f'G[{c},{c}]',A=c,B=c) for c in CONDS]
    for c in CONDS:
        for pos in CLASSES:L.append(dict(name=f'G[{c},{c}]@{pos}',A=c,B=c,pos=pos))
    for c in CONDS:
        for f in range(NFACT):
            for kind in ('ans','name'):L.append(dict(name=f'G[{c},{c}]@f{f}_{kind}',A=c,B=c,pos=f'f{f}_{kind}'))
    return L
def class_masks(k,T,ap,w):
    """bool masks over the NLL index i = token position - 1 of chunk k: the 5 classes (a partition) and the fact-level masks f<f>_ans / f<f>_name for facts 0..NFACT-1"""
    m={c:np.zeros(T,bool) for c in CLASSES};byf={d['fact']:d for d in ap[k]};insent=np.zeros(T,bool)
    for x in w['chunks'][k]['sents']:
        f=int(x['fact']);idx=list(range(max(int(x['start'])-1,0),min(int(x['end'])-1,T)));insent[idx]=True
        if f>=NREAD:m['extra'][idx]=True;continue
        d=byf[f];second={t-1 for t in d['second']};first={t-1 for t in d['first']}
        for p in idx:m['ans' if p in second else ('name' if p in first else 'frame')][p]=True
    m['filler']=~insent
    for f in range(NFACT):
        for kind,key in (('ans','second'),('name','first')):
            v=np.zeros(T,bool)
            for t in byf[f][key]:
                if 0<=t-1<T:v[t-1]=True
            m[f'f{f}_{kind}']=v
    return m
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
    # the 5 classes partition the positions; the fact-level masks lie inside ans / name; the class steps add up to the full G
    class W:pass
    w={'chunks':[{'sents':[dict(fact=0,start=1,end=10),dict(fact=1,start=10,end=19),dict(fact=17,start=19,end=30)]}]};T=40
    ap=[[dict(fact=0,first=[2,3],second=[8,9]),dict(fact=1,first=[11],second=[17]),dict(fact=2,first=[],second=[]),dict(fact=3,first=[],second=[]),dict(fact=17,first=[20],second=[28])]]
    m=class_masks(0,T,ap,w);tot=sum(m[c].astype(int) for c in CLASSES);assert (tot==1).all(),tot
    assert m['ans'].sum()==3 and m['name'].sum()==3 and m['extra'].sum()==11 and m['f0_ans'].sum()==2 and not (m['f0_ans']&~m['ans']).any() and not (m['f1_name']&~m['name']).any()
    rng=np.random.default_rng(0);Tn=T;H=6;I=9;mk=lambda:dict(x=rng.normal(size=(NL,Tn,H)).astype(np.float32),x2=rng.normal(size=(NL,Tn,I)).astype(np.float32),dz1=rng.normal(size=(NL,Tn,I)).astype(np.float32),dz3=rng.normal(size=(NL,Tn,I)).astype(np.float32),dz2=rng.normal(size=(NL,Tn,H)).astype(np.float32))
    st={c:mk() for c in CONDS};u=rng.normal(size=(NL,I)).astype(np.float32);u/=np.linalg.norm(u,axis=1,keepdims=True);by={c['name']:c for c in combos() if 'act' not in c}
    full=build_G(st,u,by['G[C0,C0]'],m);P=sum(build_G(st,u,by[f'G[C0,C0]@{c}'],m)['w2'] for c in CLASSES);assert np.allclose(P,full['w2'],rtol=1e-3,atol=1e-3)
    fa=build_G(st,u,by['G[C0,C0]@f0_ans'],m)['w2'];assert np.allclose(fa,build_G(st,u,dict(A='C0',B='C0',pos='ans'),dict(ans=m['f0_ans']))['w2'])
    print(f'e59_kvpos selftest ok: {len(combos())} combinations (2 conditions: 2 real steps, 2 full G, 10 classes, 16 fact-level steps), classes partition the positions and add up to the full G')
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
        """rows of the 16 facts (16 candidates each) with the prefix sentence; -> raw rows, rowmask. The hidden answer tokens are found as the token subsequence of the gold answer in the row; if the answer is tokenised differently inside that sentence (punctuation attached) the sentence of the SAME fact from the next candidate chunk is used (same mode: chunks k-1, k-2, k-3, k+1; other mode: k, k-1, k+1), and the chosen chunk is recorded in PFXCHUNK"""
        its=[];hid=[];ch_used={}
        for f in range(NREAD):
            fs=f if mode=='same' else (f+1)%NREAD;cands_ck=[k-1,k-2,k-3,k+1] if mode=='same' else [k,k-1,k+1]
            for ck in cands_ck:
                sn=[x for x in w['chunks'][ck]['sents'] if int(x['fact'])==fs][0];toks=ids0[ck*CH+int(sn['start']):ck*CH+int(sn['end'])];txt=tkz.decode(toks.tolist())
                rr1=R55.rows_for(LD,rt,[dict(prefix=txt+'\n'+items[f]['prefix'],cands=items[f]['cands'])])
                if all(len(find_sub(ids,gold_tok[fs],a_))==1 for ids,(a_,b_) in rr1):its.append(dict(prefix=txt+'\n'+items[f]['prefix'],cands=items[f]['cands']));ch_used[f]=ck;break
            else:raise AssertionError(('no prefix sentence with a unique hidden answer',f,mode,k))
        rr=R55.rows_for(LD,rt,its);rm={};PFX.setdefault((k,mode),ch_used)
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
    PFX={};RES=dict(seed=seed,chunks=chunks,W0_margin=W0m.tolist(),rows=[],H=[],validation=[],args=vars(a));CMB=[c for c in combos()];STORE={}                  # STORE[(cond,k)] = dict arrays; GRAD[(cond,k)] = dict(w1,w2,w3 np arrays, tree)
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
                    st={c_:STORE.pop((c_,k)) for c_ in cond_list};T=nll.shape[0];posm=class_masks(k,T,ap,w);row=dict(k=k,combos={},posn={kk:int(v.sum()) for kk,v in posm.items() if not kk.startswith('f')})
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
                    RES['rows'].append(row);lg(f'seed {seed} k={k}: combos done ({len(CMB)}) in {time.monotonic()-t1:.0f}s; G[C0,C0] vs autodiff rel.err w2 {RES["validation"][-1]["w2"]:.2e}; dmargin ACT C0 {row["combos"]["ACT[C0]"]["dmargin"]:.3f} G[C0,C0] {row["combos"]["G[C0,C0]"]["dmargin"]:.3f}')
                    for kk in [x for x in list(GRAD) if x[1]==k]:GRAD.pop(kk)
            return m.inner_parameters(),jnp.asarray(0.0,jnp.float32),jnp.asarray(1.0,jnp.float32),jnp.asarray(0.0,jnp.float32)
        L.write_step=ws;uv=SH.mk_uv(arm,w,seed,lg);t0=time.monotonic();r=L.run_stream(NM.ENGINE,1,ids0,read_chunks=[],canary_chunks=[],nch=(a.nch or nch),tag=f'kvpos_{cn}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=w['chunks'],uv=uv);r.pop('model',None);lg(f'seed {seed} {cn} ({arm}): stream done in {time.monotonic()-t0:.0f}s')
        L.rec_cache=None;del L
    RES['prefix_chunks']={f'{k}|{m}':v for (k,m),v in PFX.items()};json.dump(RES,open(out/f'kvpos_s{seed}.json','w'),default=float);lg('wrote',out/f'kvpos_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--chunks',default='16,80');ap_.add_argument('--nch',type=int,default=0);ap_.add_argument('--keys',default='');ap_.add_argument('--selftest',action='store_true');a=ap_.parse_args()
    selftest() if a.selftest else run(a)
