"""E61 InfiniteBench natural-novel pilot: same-fact hiding in the WRITE of a whole book, frozen-weight question reads (GPU; new file; thread-server 11/23; Hans 11/23 'E61 smoke on the shortest book + estimate for 16 books'; spec reports/server-request-1344k-v2-and-E61.md and experiments/E61-infbench-natural-pilot.md; EXPLORATORY; derived from e59_nat_write.py). Data: HF xinrongzhang2022/InfiniteBench longbook_qa_eng.jsonl (data/natural-text/infbench/), mask spans = data/natural-text/infbench/infbench_pilot_mask_compact.tsv (char spans of the evidence-mention sentences of each pilot question, in the book context string), distractors = infbench_pilot_distractors.tsv (3 per question). Book id = md5(context)[:10]. The book is streamed as 1024-token chunks (llama3 tokens, no BOS / EOS, whole chunks, weights from W0), original E2E write rule (clipped SGD lr 1 clip 1 on the mean NLL of all tokens of the chunk; engine of e59_nat_write.py / e55_core.run_stream, K0 = 0), window 8 chunks.
ARMS (update-path attention masks of the write forward, all 32 layers; nothing else differs): NAT (no mask); SAME (every row of a LATER evidence-mention sentence of question q loses the keys of the EARLIER evidence-mention sentences of the same q that lie in the 8 earlier chunks; rows are the tokens of the later sentence inside the chunk; current-chunk keys stay visible; mask from chunk 0 on); RAND (placebo: the same rows, the same number of hidden key sentences, each earlier evidence sentence replaced by a random sentence of the same chunk and the closest token length that contains no evidence span of any question of the book, not the answer string, not a proper name of the question). INTERPRETATION (the spec text 'hide the evidence-mention sentence in later writes while it is in the window, from chunk 0' admits two readings; implemented: rows = the later mention sentences of the same question, as the sentence rows of job 1333 / the pairs of e59_nat_write HIDE_PAIR).
READS (frozen weights, no book context, no KV): prompt 'Question: <q>\\nAnswer:', candidates = correct + 3 distractors, summed log-probability of the candidate tokens (margin = log p(correct) - mean log p(distractor)); read points: W0, after every chunk that contains an evidence-mention token of any question of the book, after the last chunk. Checks: masked pairs per chunk, equality of the update norms of SAME / RAND / NAT before the first masked chunk, evidence spans contain the answer string, weight hash before and after every read.
  python3 e61_book_run.py --book aa0310e147 --out O [--arms NAT,SAME,RAND] [--max-chunks N] | --selftest [--book ID]"""
import sys,json,time,bisect,hashlib,re,pathlib,argparse,random
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
DATA=ROOT/'data/natural-text/infbench';CH=1024;NW=8;WIN=NW*CH;ARMS=('NAT','SAME','RAND')
def load_book(bid,jsonl=DATA/'longbook_qa_eng.jsonl',mask=DATA/'infbench_pilot_mask_compact.tsv',dist=DATA/'infbench_pilot_distractors.tsv'):
    recs={}
    for line in open(jsonl):
        r=json.loads(line);recs[r['id']]=r
    ctx=None;qs=[]
    D={}
    for ln in open(dist):
        if ln.startswith('#') or ln.startswith('idx\t'):continue
        f=ln.rstrip('\n').split('\t');D[(f[1],int(f[2]))]=dict(question=f[3],correct=f[4],distractors=f[5:8])
    for ln in open(mask):
        if ln.startswith('#'):continue
        f=ln.rstrip('\n').split('\t')
        if f[1]!=bid:continue
        qid=int(f[2]);r=recs[qid];assert hashlib.md5(r['context'].encode()).hexdigest()[:10]==bid;ctx=r['context']
        spans=sorted({tuple(int(x) for x in s.split('-')) for s in f[4].split()});d=D[(bid,qid)];assert d['question'].strip()==r['input'].strip(),(d['question'],r['input'])
        qs.append(dict(idx=int(f[0]),qid=qid,answer=d['correct'],question=r['input'].strip(),distractors=d['distractors'],spans=spans))
    assert ctx is not None and qs,bid;return ctx,qs
def char_to_tok(offs):
    starts=[o[0] for o in offs];ends=[o[1] for o in offs]
    def f(c0,c1):return bisect.bisect_right(ends,c0),bisect.bisect_left(starts,c1)       # token interval [ts,te) covering the char span
    return f
def sentences(ctx):
    out=[];i=0
    for m in re.finditer(r'(?<=[.!?])\s+|\n+',ctx):
        if m.start()>i:out.append((i,m.start()))
        i=m.end()
    if i<len(ctx):out.append((i,len(ctx)))
    return out
def structure(ctx,qs,offs,n_tok,seed=0):
    """-> per question: evidence token spans (sorted, merged duplicates), chunk of the first token; list of all non-evidence candidate sentences (token spans) for RAND"""
    c2t=char_to_tok(offs);Q=[]
    for q in qs:
        ev=sorted({c2t(a,b) for a,b in q['spans']});ev=[(ts,te) for ts,te in ev if te>ts and te<=n_tok];txt=[ctx[a:b] for a,b in q['spans']];Q.append(dict(q,ev=ev,ev_text=txt))
    evall=sorted({s for q in Q for s in q['ev']});sent=[c2t(a,b) for a,b in sentences(ctx)];sent=[(ts,te) for ts,te in sent if te>ts and te<=n_tok]
    ovl=lambda s,t:s[0]<t[1] and t[0]<s[1]
    return Q,evall,sent
class E61UV:
    """UV object of the e55 engine: .keep(k,m,Kc,Vc,ids_k,keep_all) -> (bool [CH, WIN+CH], info)"""
    def __init__(self,Q,evall,sent,ctx,offs,mode,n_stream,seed=0,log=print):
        assert mode in ARMS;self.Q=Q;self.mode=mode;self.n=n_stream;self.log=log;self.last_info={};self.rep={};rng=random.Random(f'E61|{seed}')
        self.bad=[]
        for q in Q:
            ans=re.compile(r'(?<![\w])'+re.escape(q['answer'])+r'(?![\w])',re.I);names=set(w for w in re.findall(r"\b[A-Z][a-z]+",q['question']) if w not in ('What','Who','Where','When','Why','How','Which','Whom','Whose','Is','Does','Did','Do','The'))
            self.bad.append((ans,names))
        evs=sorted(evall);self.evs=evs
        if mode=='RAND':
            byc={}
            for s in sent:byc.setdefault(s[0]//CH,[]).append(s)
            def ok(s):
                if any(s[0]<t[1] and t[0]<s[1] for t in evs):return False
                tx=ctx[offs[s[0]][0]:offs[min(s[1],len(offs))-1][1]]
                return not any(a.search(tx) or any(nm in tx for nm in nms) for a,nms in self.bad)
            self.cand={c:[s for s in L if ok(s)] for c,L in byc.items()}
            for qi,q in enumerate(Q):
                for s in q['ev']:
                    c=s[0]//CH;L=self.cand.get(c,[]);ln=s[1]-s[0]
                    if not L:L=[x for cc in (c-1,c+1) for x in self.cand.get(cc,[])]
                    best=sorted(L,key=lambda x:abs((x[1]-x[0])-ln));top=best[:5];self.rep[(qi,s)]=rng.choice(top) if top else None
        self.hidden_real=0
    def rows_and_keys(self,k):
        lo=k*CH;hi=lo+CH;wlo=lo-WIN;out=[]
        for qi,q in enumerate(self.Q):
            ev=q['ev']
            for j,(ts,te) in enumerate(ev):
                if te<=lo or ts>=hi:continue
                keys=[]
                for e in ev[:j]:
                    if e[1]>wlo and e[0]<lo and e!=(ts,te) and e[1]<=ts:keys.append(e)         # earlier evidence sentence of the same question
                if not keys:continue
                if self.mode=='RAND':keys=[self.rep[(qi,e)] for e in keys if self.rep.get((qi,e)) is not None]
                a,b=max(ts,lo)-lo,min(te,hi)-lo;iv=[]
                for s0,s1 in set(keys):
                    s0=max(s0,wlo);s1=min(s1,lo)
                    if s1>s0:iv.append((s0,s1))
                out.append(((a,b),iv))
        return out
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if self.mode=='NAT' or k==0:self.last_info={'masked_pairs':0};return base,{'uv_arm':self.mode,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();wlo=k*CH-WIN;rows=0
        for (a,b),iv in self.rows_and_keys(k):
            rows+=b-a
            for s0,s1 in iv:kp[a:b,s0-wlo:s1-wlo]=False
        n=int((~kp&base[None,:]).sum());self.last_info={'masked_pairs':n,'sentence_rows':rows};return kp,{'uv_arm':self.mode,'q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def prep(bid,max_chunks=0):
    from tokenizers import Tokenizer
    import e59_stage0_run as S0
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));ctx,qs=load_book(bid);e=tk.encode(ctx,add_special_tokens=False);ids=np.asarray(e.ids,np.int64);offs=e.offsets;nch=len(ids)//CH
    if max_chunks:nch=min(nch,max_chunks)
    n=nch*CH;Q,evall,sent=structure(ctx,qs,offs,n);return tk,ctx,ids[:n],offs,nch,Q,evall,sent
def selftest(bid):
    tk,ctx,stream,offs,nch,Q,evall,sent=prep(bid);print(f'book {bid}: {len(ctx)} chars, {len(stream)} tokens in {nch} whole chunks, {len(Q)} questions, {len(evall)} evidence sentences, {len(sent)} sentences')
    for q in Q:
        ev=q['ev'];assert ev,q['qid'];ans=re.compile(r'(?<![\w])'+re.escape(q['answer'])+r'(?![\w])',re.I)
        txt=[tk.decode(stream[ts:te].tolist()) if te<=len(stream) else '' for ts,te in ev];hit=sum(1 for t in txt if ans.search(t));print(f'  q{q["qid"]} answer {q["answer"]!r}: {len(ev)} evidence token spans, chunk range {ev[0][0]//CH}..{ev[-1][0]//CH}, {hit} of them contain the answer string after the char->token conversion')
        assert hit>=max(1,len(ev)//2),'evidence spans do not contain the answer'
    for mode in ('SAME','RAND'):
        u=E61UV(Q,evall,sent,ctx,offs,mode,len(stream));tot=0;tot_rows=0
        for k in range(1,nch):
            base=np.ones((CH,WIN+CH),bool);kp,info=u.keep(k,None,None,None,None,base);tot+=info['masked_pairs'];tot_rows+=info['sentence_rows']
            assert (kp[:,WIN:]==base[:,WIN:]).all()
            for (a,b),iv in u.rows_and_keys(k):assert 0<=a<b<=CH and all(wlo_<s1<=k*CH for s0,s1 in iv for wlo_ in [k*CH-WIN] if s0>=wlo_)
        print(f'  {mode}: total masked (row,key) pairs {tot}, sentence rows {tot_rows}')
        if mode=='SAME':same_pairs=tot
        else:print(f'  RAND / SAME masked pairs ratio {tot/max(same_pairs,1):.3f}, replaced sentences: {sum(1 for v in u.rep.values() if v is not None)} of {len(u.rep)}')
    # evidence sentences with an earlier mention inside / outside the window
    inw=out_=0
    for q in Q:
        for j,(ts,te) in enumerate(q['ev']):
            if j==0:continue
            prev=q['ev'][j-1];d=(ts//CH)-(prev[1]-1)//CH;inw+=d<=NW;out_+=d>NW
    print(f'  repeated mentions: previous mention within the window {inw}, outside {out_}');print('e61 selftest ok')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    import e55_run as R55,e55_iso as ISO,e55_gpu as RG,e57_natmask_run as NM,e57_qband_run as QR
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    tk,ctx,stream,offs,nch,Q,evall,sent=prep(a.book,a.max_chunks);n=nch*CH;arms=a.arms.split(',')
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    items=[dict(prefix=f"Question: {q['question']}\nAnswer:",cands=[' '+q['answer']]+[' '+d for d in q['distractors']]) for q in Q];raw=R55.rows_for(LD,rt,items);NC=4
    with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,[]);P=LD.packing.Packed(T_,raw,GP=2)
    def read(m):
        m32=rd.to_fp32(m)
        with jax.default_matmul_precision('highest'):return np.asarray(P.r1(m32)).reshape(len(Q),NC)
    hash_ = lambda m:LD.reuse._hash_pytree(m.inner_parameters())[0]
    W0r=read(rt.base);lg('W0 read (summed log p of [correct, d1, d2, d3]):',np.round(W0r,2).tolist())
    hook_chunks=sorted({s[0]//CH for s in evall if s[0]//CH<nch}|{s[1]//CH for s in evall if (s[1]-1)//CH<nch and (s[1]-1)//CH>=0}|{nch-1});hook_chunks=sorted({k for k in hook_chunks if k<nch});lg(f'{nch} chunks, read points after chunks {hook_chunks}')
    @eqx.filter_jit
    def grad_fn(m,st_suf,c,pref_,kmask,wt):
        def total(mm):
            s3.S3SUF['kmask']=kmask
            try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
            finally:s3.S3SUF.clear()
            lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
        (_l,nll),g=eqx.filter_value_and_grad(total,has_aux=True)(m);return g.inner_parameters(),nll
    res=dict(book=a.book,nch=nch,questions=[{k:v for k,v in q.items() if k not in ('spans',)}|dict(ev_chunks=[s[0]//CH for s in q['ev']]) for q in Q],W0=W0r.tolist(),hook_chunks=hook_chunks,arms={},args=vars(a))
    L=ISO.Iso55(rt,save_dir=out/'states'/a.book,canary=can);L.rec_cache={}
    for arm in arms:
        uv=E61UV(Q,evall,sent,ctx,offs,arm,n,log=lg);rec=dict(reads={},hashes=[],steps=[]);cnt=[0]
        def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,uv=uv):
            k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=valid/valid.sum();g,nll=grad_fn(m,st_suf,c,pref,kmask,jnp.asarray(wt));nll=np.asarray(nll,np.float32)
            newp,gn,coef=L.sgd_apply(m,st_all,g);del g;ce=float((nll*valid).sum()/valid.sum());rec['steps'].append(dict(k=k,gnorm=float(gn),coef=float(coef),train_ce=ce,masked_pairs=int(uv.last_info.get('masked_pairs',0))))
            return newp,gn,coef,jnp.asarray(ce,jnp.float32)
        def hook(k,m,rec=rec):
            h0=hash_(m);r=read(m);h1=hash_(m);assert h0==h1,'weights changed by a read';rec['hashes'].append(h0);return r.tolist()
        L.write_step=ws;t0=time.monotonic();meta=[dict(sents=[]) for _ in range(nch)]
        r=L.run_stream(NM.ENGINE,1,stream,read_chunks=[],canary_chunks=[],nch=nch,tag=f'{a.book}{arm}',log=lg,read_hook=hook,hook_chunks=hook_chunks,chunks_meta=meta,uv=uv);r.pop('model',None)
        rec['reads']={str(k):v for k,v in r.get('inline',{}).items()};rec['sec']=time.monotonic()-t0;res['arms'][arm]=rec
        mm=[x['masked_pairs'] for x in rec['steps']];lg(f'arm {arm}: {nch} chunks in {rec["sec"]:.0f}s, masked pairs total {sum(mm)} over {sum(1 for x in mm if x)} chunks, mean |g| {np.mean([x["gnorm"] for x in rec["steps"]]):.3f}, reads at {len(rec["reads"])} points')
        json.dump(res,open(out/f'e61_{a.book}.partial.json','w'),default=float)
    if 'NAT' in res['arms']:
        first=min((i for ar in arms if ar!='NAT' for i,x in enumerate(res['arms'][ar]['steps']) if x['masked_pairs']),default=nch)
        for ar in arms:
            if ar=='NAT':continue
            d=max((abs(res['arms'][ar]['steps'][i]['gnorm']-res['arms']['NAT']['steps'][i]['gnorm']) for i in range(min(first,nch))),default=0.0);res['arms'][ar]['pre_first_mask_max_dgnorm_vs_NAT']=d;lg(f'check: {ar} vs NAT update norm before the first masked chunk ({first}): max |d gnorm| {d:.2e}')
    res['sec']=time.monotonic()-t_all;json.dump(res,open(out/f'e61_{a.book}.json','w'),default=float);lg('wrote',out/f'e61_{a.book}.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--book',default='aa0310e147');ap.add_argument('--out');ap.add_argument('--arms',default=','.join(ARMS));ap.add_argument('--max-chunks',type=int,default=0);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    selftest(a.book) if a.selftest else run(a)
