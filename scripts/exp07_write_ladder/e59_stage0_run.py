"""Natural-document stage-0 check, E2E 3B (GPU forward only, NO weight update; new file; thread-server 11/21; Hans '점검부터 해봐야겠네'; EXPLORATORY; inputs = the branch claude/natural-recur-stage0 of TTT-reg, data/natural-text/recur/: sel24_books.jsonl (24 PG-19 test books) and sel24_events.json (18,850 events: doc, pair, k, prev, c, tok, sent / m1 / m2 char spans)). For every selected event (all k >= 1, plus --n-k0 random k = 0 events as the control, seeded by --sel-seed) the summed NLL of the m2 tokens is read in two conditions of the SAME frozen weights (W0 = the E2E checkpoint, original layers / fast weights, NOTHING written): (A) the natural window (the model's own SWA window of 8192 carried through the book stream in 1024-token chunks, no BOS, no EOS; the first tokens of a book have a shorter window) and (B) the band-1024 mask in ALL 32 layers (prefix 'ctx' keep + suffix kmask, the engine path of the MASK1024 arm of e59_srv_run.py): every token sees only the previous 1024 tokens. g = nllB - nllA (computed by the caller / analysis). Mechanics = the SGDSEL engine of e55_core.run_stream with the write step replaced by a forward-only NLL (no gradient, the parameters are returned unchanged), the natural K,V of the stream recorded once per book and shared by both conditions (rec_cache).
Alignment: stream = llama3 tokens of the book text (tokenizer data/tokenizer/llama3/tokenizer.json, add_special_tokens False), truncated to the first 400000 tokens (as recur_events.py) and to whole 1024-token chunks; token t is predicted at stream position t-1 (nll index (t-1) % 1024 of chunk (t-1)//1024, asserted against the engine's target tokens); m2 tokens = tokens whose char span overlaps the m2 char span; events whose m2 tokens fall beyond the stream or at token 0 are dropped (counted).
Output (one json per run): events [{doc, k, prev, c, tok, pos (first m2 token), ntok, nllA, nllB, pair}] + per-chunk mean NLL of both conditions + condition checks. python3 e59_stage0_run.py --data DIR --out O [--books N] [--max-chunks M] [--n-k0 2000] [--sel-seed 0] | --selftest [--data DIR]"""
import sys,json,time,hashlib,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;MAXTOK=400000
def load(data,books=None):
    ev=json.load(open(pathlib.Path(data)/'sel24_events.json'));docs=[json.loads(l) for l in open(pathlib.Path(data)/'sel24_books.jsonl')]
    if books:docs=docs[:books]
    return docs,ev
def tokenize(tk,text):
    e=tk.encode(text,add_special_tokens=False);ids=np.asarray(e.ids,np.int64);offs=e.offsets
    maxc=offs[min(len(offs),MAXTOK)-1][1];n=min(len(ids),MAXTOK);return ids[:n],offs[:n],maxc
def select(events,n_k0,sel_seed):
    rng=np.random.default_rng(sel_seed);k0=[i for i,e in enumerate(events) if e['k']==0];pick=set(rng.choice(k0,size=min(n_k0,len(k0)),replace=False).tolist()) if n_k0 else set()
    return [i for i,e in enumerate(events) if e['k']>=1 or i in pick]
def m2_tokens(offs,n,span):
    c0,c1=span;t=[i for i in range(n) if offs[i][1]>c0 and offs[i][0]<c1];return t
def event_positions(doc_events,offs,ntok_stream):
    """-> list of (event, [token indices of m2]) with every token index in 1..ntok_stream-1; dropped events counted"""
    import bisect
    starts=[o[0] for o in offs];ends=[o[1] for o in offs];out=[];dropped=0
    for e in doc_events:
        c0,c1=e['m2'];i0=max(bisect.bisect_right(ends,c0),0);i1=bisect.bisect_left(starts,c1);t=[i for i in range(i0,min(i1,len(starts))) if ends[i]>c0 and starts[i]<c1]
        if not t or t[0]<1 or t[-1]>=ntok_stream:dropped+=1;continue
        out.append((e,t))
    return out,dropped
def selftest(data):
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));docs,ev=load(data,2);sel=select(ev,2000,0);assert len(sel)==5301+2000,len(sel)
    sel2=select(ev,2000,0);assert sel==sel2
    for d in docs:
        ids,offs,maxc=tokenize(tk,d['text']);n=(len(ids)//CH)*CH;text=d['text'];de=[e for e in ev if e['doc']==d['id']];pos,dropped=event_positions(de,offs,n)
        bad=0
        for e,t in pos[:400]:
            s=text[e['m2'][0]:e['m2'][1]].strip();dec=tk.decode(ids[t].tolist()).strip();bad+= s not in dec and dec not in s
        assert bad<=0.01*min(400,len(pos)),('m2 tokens do not match the m2 text',bad)
        print(d['id'][:30],'tokens',len(ids),'chunks',n//CH,'events',len(de),'kept',len(pos),'dropped',dropped,'tok field vs first sentence token mismatches',sum(1 for e,t in pos[:200] if abs(t[0]-e['tok'])>400))
    print('e59_stage0_run selftest ok (selection, tokenisation, m2 token alignment)')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    from tokenizers import Tokenizer
    import e55_run as R55,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True)
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    docs,ev=load(a.data,a.books);sel=select(ev,a.n_k0,a.sel_seed);selset=set(sel);LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt)
    @eqx.filter_jit
    def fwd_fn(m,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);return -jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    res=dict(events=[],books=[],dropped=0,args=vars(a));t_all=time.monotonic()
    for di,d in enumerate(docs):
        ids,offs,maxc=tokenize(tk,d['text']);nch=len(ids)//CH
        if a.max_chunks:nch=min(nch,a.max_chunks)
        n=nch*CH;stream=ids[:n].astype(np.int64);de=[e for i,e in enumerate(ev) if e['doc']==d['id'] and i in selset];pos,dropped=event_positions(de,offs,n);res['dropped']+=dropped
        L=ISO.Iso55(rt,save_dir=out/'states'/f'b{di}',canary=can);L.rec_cache={};nllc={}
        for cond in ('A','B'):
            rec={};cnt=[0]
            def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt):
                k=cnt[0];cnt[0]+=1;nll=np.asarray(fwd_fn(m,st_suf,c,pref,kmask),np.float32);rec[k]=(nll,np.asarray(c.target_tokens),np.asarray(c.loss_masks,bool));ce=float((nll*rec[k][2]).sum()/max(rec[k][2].sum(),1))
                return m.inner_parameters(),jnp.asarray(0.0,jnp.float32),jnp.asarray(1.0,jnp.float32),jnp.asarray(ce,jnp.float32)
            L.write_step=ws;uv=QR.SelMaskQ('ALLQ1024',{'band_ALLQ1024':np.asarray(1024),'all_ALLQ1024':np.asarray(1)},lg) if cond=='B' else QR.SelMaskQ('NATQ',{},lg)
            meta=[dict(sents=[]) for _ in range(nch)];t0=time.monotonic();r=L.run_stream(NM.ENGINE,1,stream,read_chunks=[],canary_chunks=[],nch=nch,tag=f'b{di}{cond}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=meta,uv=uv);r.pop('model',None)
            assert len(rec)==nch,(len(rec),nch)
            for k,(nll,tg,vm) in rec.items():                                           # alignment: engine targets = the next stream tokens
                nv=int(vm.sum());assert nv>=CH-1,(k,nv);idx=np.flatnonzero(vm);assert np.array_equal(tg[idx],stream[k*CH+idx+1]),('target tokens differ from the stream',k)
            nllc[cond]=rec;lg(f'book {di} {d["id"][:40]} cond {cond}: {nch} chunks in {time.monotonic()-t0:.0f}s, mean NLL {np.mean([x[0][x[2]].mean() for x in nllc[cond].values()]):.3f}')
        def tok_nll(rec,t):
            k=(t-1)//CH;i=(t-1)%CH;nll,tg,vm=rec[k];assert vm[i];return float(nll[i])
        for e,t in pos:
            if t[-1]>=n:continue
            res['events'].append(dict(doc=e['doc'],k=e['k'],prev=e['prev'],c=e['c'],tok=e['tok'],pos=t[0],ntok=len(t),nllA=sum(tok_nll(nllc['A'],x) for x in t),nllB=sum(tok_nll(nllc['B'],x) for x in t),pair=e['pair']))
        c0=np.mean([nllc['A'][0][0][nllc['A'][0][2]].mean()-nllc['B'][0][0][nllc['B'][0][2]].mean()]);res['books'].append(dict(doc=d['id'],chunks=nch,kept=len(pos),dropped=dropped,chunk0_A_minus_B=float(c0),mean_A=float(np.mean([x[0][x[2]].mean() for x in nllc['A'].values()])),mean_B=float(np.mean([x[0][x[2]].mean() for x in nllc['B'].values()]))))
        assert abs(c0)<2e-2,('chunk 0: the band-1024 mask must reproduce the natural window (all tokens within 1024)',c0)
        json.dump(res,open(out/'stage0_e2e_events.partial.json','w'));L.rec_cache=None;del L,nllc
    res['sec']=time.monotonic()-t_all;json.dump(res,open(out/'stage0_e2e_events.json','w'));lg('wrote',out/'stage0_e2e_events.json',len(res['events']),'events')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT.parent/'TTT-reg-stage0/data/natural-text/recur'));ap.add_argument('--out');ap.add_argument('--books',type=int,default=0);ap.add_argument('--max-chunks',type=int,default=0);ap.add_argument('--n-k0',type=int,default=2000);ap.add_argument('--sel-seed',type=int,default=0);ap.add_argument('--selftest',action='store_true')
    a=ap.parse_args()
    if a.selftest:selftest(a.data)
    else:run(a)
