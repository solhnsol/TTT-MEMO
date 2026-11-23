"""Shadowing check vs learning-path check (E2E 3B, GPU, frozen W0, NO weight update; new file; thread-server 11/23; Hans '섀도잉 점검 만들어' with the point-A idea merged; EXPLORATORY). Question: when sentences of OTHER facts that share the sentence structure of the read facts are visible, (a) is the loss on the target tokens LOWER (shadowing: attention copies / pattern-matches, the gradient shrinks) or (b) is the loss the same but the GRADIENT direction different (learning path distorted towards the structure)? Worlds: the U12 world of jobs 1328 / 1339 (32 facts: 16 read facts U1 + 16 extra facts U2, one sentence per fact per chunk, seeds 231..238, 128 chunks) in its original form ('intact') and with the tokens of every U2 sentence permuted ('scr', e59_scr.py). Conditions (update-path attention masks of the sentence rows, window 8 chunks, e59_u12.SetHideUV, all with the earlier sentences of the SAME fact hidden): C0 = all other facts hidden (SAME_U1U2, the reference), U1vis (U2 hidden, SAME_U2), U2vis (U1 hidden, SAME_U1), ALLVIS (SAME) in the intact world; C0 and U2vis(scr) in the scrambled world. At W0, in the 8 chunks 16, 32, 48, 64, 80, 96, 112, 127 (the engine path of e59_stage0_run.py: write step replaced, parameters unchanged), for every condition: the per-token NLL of the chunk (mean NLL of the ANSWER tokens = the second-mentioned entity of the U1 sentences, of the FIRST-mentioned entity, of the FRAME tokens = the other tokens of the U1 sentences, of all tokens) and the gradient w.r.t. the fast weights (262144-coordinate sketch + exact global norm) of three objectives: g_ans (mean NLL of the answer tokens of the U1 sentences), g_frame (mean NLL of their frame tokens), g_all (the update objective: mean NLL of all tokens of the chunk, before clipping).
  python3 e59_shadow.py --seed 231 --out O [--chunks 16,32,...] [--conds ...] | --selftest [--seed 231]"""
import sys,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NS=262144;NREAD=16
CHUNKS=(16,32,48,64,80,96,112,127)
COND_NEW={'intact':{'C0':'SAME_U1U2','SAMEVIS':('ha','OTHERONLY'),'NATVIS':('nat','NAT')}}      # job 1344: the same-fact sentences VISIBLE while every other fact is hidden (e59_hidearms OTHERONLY: hides the earlier sentences of all other facts, the same fact stays visible) and no mask at all (NAT: same fact and all other facts visible)
def mk_uv(spec,w,seed,lg):
    import e59_u12 as U
    if isinstance(spec,str):return U.uv_for(spec,w['chunks'],lg)
    kind,arm=spec
    if kind=='ha':import e59_hidearms as H;return H.uv_for(arm,w['chunks'],seed,lg)
    if kind=='nat':import e57_qband_run as QR;return QR.SelMaskQ('NATQ',{},lg)
    raise KeyError(spec)
COND={'intact':{'C0':'SAME_U1U2','U1vis':'SAME_U2','U2vis':'SAME_U1','ALLVIS':'SAME'},'scr':{'C0':'SAME_U1U2','U2vis':'SAME_U1'}}
def token_weights(ap_chunk,valid):
    """chunk-position -> NLL-index weights (index = target position = token position - 1): answer (second entity), first entity, frame tokens of the U1 sentences; each normalised to sum 1; all = valid mask / sum"""
    n=len(valid);wa=np.zeros(n,np.float32);wf=np.zeros(n,np.float32);wr=np.zeros(n,np.float32)
    for d in ap_chunk:
        if d['fact']>=NREAD:continue
        a,b=d['span'];ent=set(d['place'])|set(d['name'])
        for t in d['second']:
            if t>=1:wa[t-1]+=1
        for t in d['first']:
            if t>=1:wf[t-1]+=1
        for t in range(max(a,1),b):
            if t not in ent:wr[t-1]+=1
    wa=wa*valid;wf=wf*valid;wr=wr*valid;wall=valid.astype(np.float32);out={}
    for nm,w in (('ans',wa),('first',wf),('frame',wr),('all',wall)):
        assert w.sum()>0,(nm,'no tokens');out[nm]=(w/w.sum()).astype(np.float32)
    return out
def selftest(seed=231):
    import e55_run as R55,e57_syn_pass as SP,e59_u12 as U,e59_leak_probe as LP,e59_scr as SC,types
    U.install();a=types.SimpleNamespace(cell=U.CELL,seed=seed,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT');enc,w,tf=R55.build(a,seed,read_apps={16,32,64,128},arms=['NAT']);tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,128);assert miss==0,(miss,tot)
    valid=np.ones(CH,bool);valid[-1]=False
    for k in CHUNKS:
        W=token_weights(ap[k],valid);assert all(abs(W[n].sum()-1)<1e-5 for n in W)
        for n in ('ans','first','frame'):assert (W[n]>0).sum()>0
        assert not ((W['ans']>0)&(W['frame']>0)).any(),'answer tokens are not frame tokens'
        assert not ((W['first']>0)&(W['frame']>0)).any()
    n_u1=sum(1 for d in ap[16] if d['fact']<NREAD);assert n_u1==16,n_u1
    import e59_hidearms as H
    for world,cs in COND.items():
        for cn,arm in cs.items():
            if isinstance(arm,str) or arm[0]=='ha':mk_uv(arm,w,seed,lambda *x:None)
    for world,cs in COND_NEW.items():
        for cn,arm in cs.items():
            if isinstance(arm,str) or arm[0]=='ha':mk_uv(arm,w,seed,lambda *x:None)
    print('e59_shadow selftest ok: token weights for the answer / first-entity / frame / all objectives (U1 sentences only), conditions',COND)
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,e59_leak_probe as LP,e59_scr as SC
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);chunks=[int(x) for x in a.chunks.split(',')]
    U.install();cell=U.CELL;ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids0,sha=S['A'];assert sha==G.stream_sha(G.realize(w));ids0=np.asarray(ids0);nch=w['n_chunks'];tk=LP.tk_load();ap,miss,tot=LP.answer_positions(w,tk,nch);assert miss==0
    inner=rt.base.inner_parameters();leaves=jax.tree_util.tree_leaves(inner);sizes=[int(np.prod(x.shape)) for x in leaves];totp=sum(sizes);rng=np.random.default_rng(0);idx=[]
    for s_ in sizes:c=max(1,int(round(NS*s_/totp)));idx.append(np.sort(rng.choice(s_,size=min(c,s_),replace=False)).astype(np.int32))
    idx_dev=[jnp.asarray(i) for i in idx]
    @eqx.filter_jit
    def gsk(g):
        lv=jax.tree_util.tree_leaves(g);sq=jnp.sum(jnp.stack([jnp.sum(x.astype(jnp.float32)**2) for x in lv]));return jnp.sqrt(sq),jnp.concatenate([x.astype(jnp.float32).reshape(-1)[i] for x,i in zip(lv,idx_dev)])
    @eqx.filter_jit
    def grad_fn(m,st_suf,c,pref_,kmask,wt):
        def total(mm):
            s3.S3SUF['kmask']=kmask
            try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
            finally:s3.S3SUF.clear()
            lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
        (_l,nll),g=eqx.filter_value_and_grad(total,has_aux=True)(m);return g.inner_parameters(),nll
    res=dict(seed=seed,chunks=chunks,rows=[],args=vars(a));SK={};t_all=time.monotonic()
    CONDS=COND_NEW if a.set=='new' else COND
    for world,cs in CONDS.items():
        if a.worlds and world not in a.worlds.split(','):continue
        ids=ids0 if world=='intact' else SC.scramble_u2(ids0,w['chunks'],seed);L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}_{world}',canary=can);L.rec_cache={}
        for cn,arm in cs.items():
            cnt=[0]
            def ws(m,st_all,st_suf,c,pref,kmask,cnt=cnt,world=world,cn=cn):
                k=cnt[0];cnt[0]+=1
                if k in chunks:
                    valid=np.asarray(c.loss_masks,np.float32);W=token_weights(ap[k],valid>0);row=dict(world=world,cond=cn,k=k);gs={}
                    for obj in ('ans','frame','all'):
                        g,nll=grad_fn(m,st_suf,c,pref,kmask,jnp.asarray(W[obj]));nll=np.asarray(nll,np.float32);gn,sk=gsk(g);gs[obj]=np.asarray(sk);row['gnorm_'+obj]=float(gn);del g
                    for nm in ('ans','first','frame','all'):row['nll_'+nm]=float((nll*W[nm]).sum())
                    SK[(world,cn,k)]=np.stack([gs['ans'],gs['frame'],gs['all']]).astype(np.float32);res['rows'].append(row)
                return m.inner_parameters(),jnp.asarray(0.0,jnp.float32),jnp.asarray(1.0,jnp.float32),jnp.asarray(0.0,jnp.float32)
            L.write_step=ws;uv=mk_uv(arm,w,seed,lg);t0=time.monotonic()
            r=L.run_stream(NM.ENGINE,1,ids,read_chunks=[],canary_chunks=[],nch=(a.nch or nch),tag=f'shadow_{world}_{cn}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=w['chunks'],uv=uv);r.pop('model',None);lg(f'seed {seed} {world} {cn} ({arm}): {a.nch or nch} chunks in {time.monotonic()-t0:.0f}s')
        L.rec_cache=None;del L
    json.dump(res,open(out/f'shadow_s{seed}.json','w'));keys=sorted(SK);np.savez(out/f'shadow_s{seed}_sketch.npz',keys=np.array([f'{a_}|{b_}|{c_}' for a_,b_,c_ in keys]),sk=np.stack([SK[k] for k in keys]));lg('wrote',out/f'shadow_s{seed}.json',len(res['rows']),'rows',f'{time.monotonic()-t_all:.0f}s')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=231);ap.add_argument('--out');ap.add_argument('--chunks',default=','.join(map(str,CHUNKS)));ap.add_argument('--worlds',default='');ap.add_argument('--nch',type=int,default=0);ap.add_argument('--set',default='old');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest(a.seed)
    else:run(a)
