"""Leak probe 1 for the SHOWOTHER gain (E2E 3B, GPU forward only, NO weight update; new file; thread-server 11/21; Hans '점검 1 만들어'; EXPLORATORY). Question: do the visible other-fact sentences of SHOWOTHER (their natural K,V were recorded in a context that contained the same-fact sentences) carry the ANSWER of the sentence being predicted? At the frozen W0 weights, in the synthetic world v3_16_1 (the 1333 world, seeds 231..238, 128 chunks, 16 fact sentences per chunk), the NLL of the ANSWER tokens (the place tokens of every fact sentence; the 16-candidate tokens of the reads) and of all fact-sentence tokens is read in four attention conditions of the same stream: NAT (natural window), ALLHID (every row sees only its own chunk), SHOWSAME (sentence rows additionally see the earlier same-fact sentences), SHOWOTHER (sentence rows additionally see the earlier other-fact sentences); masks = e59_hidearms.uv_for (all 32 layers, the engine path of jobs 1333 / 1334); natural K,V recorded once per stream and shared (rec_cache); forward only (no gradient, parameters returned unchanged), as e59_stage0_run.py. Main metric: second-entity (answer) NLL per sentence (sum over its tokens), mean over chunks >= --first (16) and facts, SHOWOTHER - ALLHID (paired over seeds); SHOWSAME - ALLHID is the copy bound (answer visible), NAT - ALLHID the natural bound. No-leak reading: SHOWOTHER ~ ALLHID (a vocabulary effect can lower the NLL by at most about log 16 = 2.8 nat); leak reading: SHOWOTHER ~ SHOWSAME.
Answer tokens: the templates of the world put the place before OR after the person name; for every fact sentence the token ids of the place and of the person (variants with / without leading space; tokenizer data/tokenizer/llama3/tokenizer.json) are located inside the sentence span by id-subsequence match; the SECOND-mentioned entity is the binding-dependent answer (predicted given the other one), the FIRST-mentioned entity is the vocabulary / format control (its NLL cannot depend on a binding); token t is predicted at stream position t-1 (nll index (t-1) % 1024 of chunk (t-1)//1024, asserted against the engine's target tokens).
  python3 e59_leak_probe.py --seed 231 --out O [--chunks N] [--first 16] | --selftest [--seed 231]"""
import sys,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;CONDS=('NAT','ALLHID','SHOWSAME','SHOWOTHER')
def tk_load():
    from tokenizers import Tokenizer;return Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
def find_sub(ids,a,b,variants):
    """last occurrence of any token-id variant inside ids[a:b] -> list of positions or None"""
    best=None
    for p in variants:
        n=len(p)
        for i in range(b-n,a-1,-1):
            if ids[i:i+n]==p:
                if best is None or i>best[0]:best=(i,list(range(i,i+n)))
                break
    return None if best is None else best[1]
def answer_positions(w,tk,nch):
    """-> per chunk list of dict(fact, span, place=[idx], name=[idx], order 'place_first' | 'name_first', second=[idx of the later-mentioned entity = predicted given the other], first=[...]) (indices inside the chunk). The templates of the world put the place before OR after the person name; the SECOND-mentioned entity is the binding-dependent one."""
    var=lambda t:[tk.encode(' '+t,add_special_tokens=False).ids,tk.encode(t,add_special_tokens=False).ids]
    pv={int(f['id']):var(f['P']) for f in w['facts']};nv={int(f['id']):var(f['X']) for f in w['facts']};out=[];miss=0;tot=0
    for k in range(nch):
        ids=[int(x) for x in w['chunks'][k]['ids']];rows=[]
        for x in sorted(w['chunks'][k]['sents'],key=lambda s:s['start']):
            f=int(x['fact']);a,b=int(x['start']),int(x['end']);tot+=1;pp=find_sub(ids,a,b,pv[f]);nn=find_sub(ids,a,b,nv[f])
            if pp is None or nn is None:miss+=1;continue
            pf=pp[0]<nn[0];rows.append(dict(fact=f,span=(a,b),place=pp,name=nn,order='place_first' if pf else 'name_first',second=nn if pf else pp,first=pp if pf else nn))
        out.append(rows)
    return out,miss,tot
def selftest(seed=231):
    import e55_run as R55,e57_syn_pass as SP,e59_hidearms as H
    enc,w,tf=R55.build(SP.synth_args(seed,'v3_16_1'),seed,read_apps={16,32,64,128},arms=['NAT']);tk=tk_load();ap,miss,tot=answer_positions(w,tk,128)
    assert tot==128*16 and miss==0,(miss,tot)
    assert all(len(r)==16 for r in ap)
    import collections;print('orders',collections.Counter(d['order'] for r in ap for d in r),'second-entity token counts',collections.Counter(len(d['second']) for r in ap for d in r))
    for k in (3,50):
        base=(k*CH-8*CH+np.arange(9*CH))>=0
        for arm in ('ALLHID','SHOWSAME','SHOWOTHER'):kp,_=H.uv_for(arm,w['chunks'],seed,lambda *x:None).keep(k,None,None,None,None,base);assert kp.shape==(CH,9*CH)
    assert all(d['second'][0]>d['first'][-1] for r in ap for d in r)
    print('e59_leak_probe selftest ok: place and name tokens found for',tot,'sentences')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_hidearms as H
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;cell='v3_16_1';out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ci=OR.register_cell(cell);ro=list(ci['read_apps']);LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w))
    nch=a.chunks or w['n_chunks'];ids=np.asarray(ids)[:nch*CH];tk=tk_load();ap,miss,tot=answer_positions(w,tk,nch);assert miss==0,miss
    @eqx.filter_jit
    def fwd_fn(m,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);return -jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    L=ISO.Iso55(rt,save_dir=out/'states'/f's{seed}',canary=can);L.rec_cache={};recs={}
    for cond in CONDS:
        rec={};cnt=[0]
        def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt):
            k=cnt[0];cnt[0]+=1;nll=np.asarray(fwd_fn(m,st_suf,c,pref,kmask),np.float32);rec[k]=(nll,np.asarray(c.target_tokens),np.asarray(c.loss_masks,bool));ce=float((nll*rec[k][2]).sum()/max(rec[k][2].sum(),1))
            return m.inner_parameters(),jnp.asarray(0.0,jnp.float32),jnp.asarray(1.0,jnp.float32),jnp.asarray(ce,jnp.float32)
        L.write_step=ws;uv=QR.SelMaskQ('NATQ',{},lg) if cond=='NAT' else H.uv_for(cond,w['chunks'][:nch] if False else w['chunks'],seed,lg);t0=time.monotonic()
        r=L.run_stream(NM.ENGINE,1,ids,read_chunks=[],canary_chunks=[],nch=nch,tag=f'leak_{cond}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=w['chunks'],uv=uv);r.pop('model',None)
        assert len(rec)==nch
        for k,(nll,tg,vm) in rec.items():
            idx=np.flatnonzero(vm);assert len(idx)>=CH-1 and np.array_equal(tg[idx],ids[k*CH+idx+1]),('target tokens differ from the stream',k)
        recs[cond]=rec;lg(f'seed {seed} cond {cond}: {nch} chunks in {time.monotonic()-t0:.0f}s')
    def tn(rec,t):
        k=(t-1)//CH;i=(t-1)%CH;nll,tg,vm=rec[k];assert vm[i],(k,i);return float(nll[i])
    rows=[]
    for k in range(1,nch):
        for d in ap[k]:
            s0,s1=d['span'];row=dict(k=k,fact=d['fact'],order=d['order'],n_second=len(d['second']),n_first=len(d['first']))
            for cond in CONDS:
                for nm in ('second','first','place','name'):row[f'{nm}_{cond}']=sum(tn(recs[cond],k*CH+t) for t in d[nm])
                row['sent_'+cond]=float(np.mean([tn(recs[cond],k*CH+t) for t in range(max(s0,1),s1)]))
            rows.append(row)
    res=dict(seed=seed,chunks=nch,rows=rows,args=vars(a));json.dump(res,open(out/f'leak_probe_s{seed}.json','w'))
    first=[r_ for r_ in rows if r_['k']>=a.first];lg('seed',seed,'second-entity NLL per sentence (chunks >=',a.first,'):',{c:round(float(np.mean([r_['second_'+c] for r_ in first])),3) for c in CONDS},'first-entity:',{c:round(float(np.mean([r_['first_'+c] for r_ in first])),3) for c in CONDS});L.rec_cache=None
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=231);ap.add_argument('--out');ap.add_argument('--chunks',type=int,default=0);ap.add_argument('--first',type=int,default=16);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest(a.seed)
    else:run(a)
