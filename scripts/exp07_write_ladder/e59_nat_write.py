"""Natural-document WRITE experiment, E2E 3B (GPU; new file; thread-server 11/22; Hans '자연 문서 쓰기 만들어'; EXPLORATORY; inputs = TTT-reg branch claude/natural-recur-stage0, data/natural-text/recur/: sel24_books.jsonl, sel24_events.json). Every PG-19 book is streamed as 1024-token chunks (llama3 tokens, no BOS / EOS, first 400000 tokens, whole chunks; weights restart from W0 for every book) and WRITTEN with the original rule (clipped SGD lr 1 clip 1 on the mean NLL of all tokens of the chunk, original E2E engine of e59_srv_run.py / e55_core.run_stream, SGDSEL, K0 = 0). Arms (update-path attention masks, all 32 layers, window 8 chunks; ONLY the write forward is masked): NAT (no mask); HIDE_PAIR (the rows of every event sentence that has an earlier occurrence (k >= 1) lose the keys of the sentences of the EARLIER occurrences of the SAME pair); HIDE_EV (the same rows lose the keys of ALL earlier event sentences of earlier chunks, any pair); MASK1024 (band 1024 on every row, e57 ALLQ1024). An event sentence = the spaCy sentence that contains a pair (sel24_events 'sent' char span -> token span by the llama3 offsets); rows = the tokens of that sentence inside the chunk; keys = tokens of the earlier occurrences' sentences inside the 8 earlier chunks (the sentence of the current event itself is never hidden from itself; current-chunk keys stay visible).
READ (prequential, every chunk, every arm): BEFORE the write step of chunk k the chunk is forwarded once more with the weights of that moment and the NATURAL window (no mask; for the masked arms an extra forward-only pass using the stashed window K,V; for NAT the write forward itself) and the per-token NLL is stored; the m2 tokens of the selected events (all k >= 1, plus --n-k0 random k = 0 events, seeded) are read out of it: nll = summed NLL over the m2 tokens. Main measurement: events whose previous occurrence is farther than 8192 tokens (the earlier mention is OUTSIDE the natural window, only weights can carry it), (arm - NAT) per event, book-clustered, minus the same difference of the k = 0 control events.
Output (one json per run): events [{doc, k, prev, c, pos, ntok, pair, nll: {arm: summed natural-window NLL of the m2 tokens}}], per book and arm summaries (mean natural-window NLL of all tokens, mean training loss, masked pairs, step norms). python3 e59_nat_write.py --out O [--books N] [--arms NAT,HIDE_PAIR,HIDE_EV,MASK1024] [--max-chunks M] [--n-k0 2000] [--sel-seed 0] [--data DIR] | --selftest"""
import sys,json,time,bisect,pathlib,argparse,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e59_stage0_run as S0
CH=1024;NW=8;WIN=NW*CH;ARMS=('NAT','HIDE_PAIR','HIDE_EV','MASK1024')
def book_struct(doc,events,offs,ntok):
    """-> list of event dicts (all events of the book in order): pair tuple, k, tok, token span of the sentence (ts, te), m2 token list; sentence table {(ts,te): [event idx]}"""
    starts=[o[0] for o in offs];ends=[o[1] for o in offs];ev=[]
    for e in events:
        c0,c1=e['sent'];ts=bisect.bisect_right(ends,c0);te=bisect.bisect_left(starts,c1);m0,m1=e['m2'];i0=bisect.bisect_right(ends,m0);i1=bisect.bisect_left(starts,m1);t=[i for i in range(i0,min(i1,ntok)) if ends[i]>m0 and starts[i]<m1]
        ev.append(dict(pair=tuple(e['pair']),k=e['k'],c=e['c'],prev=e['prev'],tok=e['tok'],ts=ts,te=te,m2=t,idx=e.get('_i')))
    return ev
class NatHideUV:
    """UV object of the e55 engine (.keep(k,m,Kc,Vc,ids_k,keep_all) -> (bool [CH, WIN+CH], info)); also stashes the window inputs of the chunk (Kc, Vc, natural 1-D keep) for the natural-window read of the write step"""
    def __init__(self,ev,mode,nstream,log=print):
        assert mode in ('NAT','HIDE_PAIR','HIDE_EV','MASK1024');self.ev=ev;self.mode=mode;self.n=nstream;self.log=log;self.stash=None;self.last_info={}
        bysent={}
        for i,e in enumerate(ev):bysent.setdefault((e['ts'],e['te']),[]).append(i)
        self.sents=sorted(bysent);self.bysent=bysent;self.bypair={}
        for i,e in enumerate(ev):self.bypair.setdefault(e['pair'],[]).append(i)
        self.cum=0
    def rows_and_keys(self,k):
        """-> list of (row slice (a,b) inside the chunk, [key token intervals (absolute) hidden for these rows]) for chunk k"""
        lo=k*CH;hi=lo+CH;wlo=lo-WIN;out=[]
        i0=bisect.bisect_left(self.sents,(lo-10**6,0))
        for (ts,te) in self.sents:
            if te<=lo or ts>=hi:continue
            idxs=self.bysent[(ts,te)];rec=[i for i in idxs if self.ev[i]['k']>=1]
            if not rec:continue
            a,b=max(ts,lo)-lo,min(te,hi)-lo;keys=[]
            if self.mode=='HIDE_PAIR':
                for i in rec:
                    for j in self.bypair[self.ev[i]['pair']]:
                        if self.ev[j]['k']<self.ev[i]['k'] and (self.ev[j]['ts'],self.ev[j]['te'])!=(ts,te):keys.append((self.ev[j]['ts'],self.ev[j]['te']))
            elif self.mode=='HIDE_EV':keys=[s for s in self.sents if s!=(ts,te) and s[0]<lo]
            iv=[]
            for s0,s1 in set(keys):
                s0=max(s0,wlo);s1=min(s1,lo)
                if s1>s0:iv.append((s0,s1))
            out.append(((a,b),iv))
        return out
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool);self.stash=(k,Kc,Vc,keep_all)
        if self.mode in ('NAT','MASK1024') or k==0:self.last_info={'masked_pairs':0};return base,{'uv_arm':self.mode,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();wlo=k*CH-WIN;rows=0
        for (a,b),iv in self.rows_and_keys(k):
            rows+=b-a
            for s0,s1 in iv:kp[a:b,s0-wlo:s1-wlo]=False
        n=int((~kp&base[None,:]).sum());self.last_info={'masked_pairs':n,'sentence_rows':rows};return kp,{'uv_arm':self.mode,'q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def selftest(data):
    from tokenizers import Tokenizer
    tk=S0.tk_load() if hasattr(S0,'tk_load') else Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    docs,evs=S0.load(data,1);d=docs[0];ids,offs,maxc=S0.tokenize(tk,d['text']);n=(len(ids)//CH)*CH;de=[e for e in evs if e['doc']==d['id']];ev=book_struct(d,de,offs,n)
    assert len(ev)==len(de) and all(e['ts']<e['te'] for e in ev)
    # first-mention / later-mention structure: k is the occurrence index; prev token distances agree with the sentence starts
    byp={}
    for e in ev:byp.setdefault(e['pair'],[]).append(e)
    bad=0
    for L in byp.values():
        L.sort(key=lambda x:x['k'])
        for a,b in zip(L,L[1:]):
            if b['prev'] is not None and abs((b['tok']-a['tok'])-b['prev'])>2:bad+=1
    assert bad==0,bad
    base_all=lambda k:(k*CH-WIN+np.arange(WIN+CH))>=0
    res={}
    for mode in ('HIDE_PAIR','HIDE_EV'):
        u=NatHideUV(ev,mode,n)
        for k in (3,9,20,60,150,300):
            if (k+1)*CH>n:continue
            base=base_all(k);kp,info=u.keep(k,None,None,None,None,base);H=~kp&base[None,:];exp=np.zeros((CH,WIN+CH),bool);lo=k*CH;wlo=lo-WIN
            for e in ev:
                if e['k']<1:continue
                ts,te=e['ts'],e['te']
                if te<=lo or ts>=lo+CH:continue
                a,b=max(ts,lo)-lo,min(te,lo+CH)-lo;sel=[]
                if mode=='HIDE_PAIR':sel=[(x['ts'],x['te']) for x in byp[e['pair']] if x['k']<e['k'] and (x['ts'],x['te'])!=(ts,te)]
                else:sel=[(x['ts'],x['te']) for x in ev if (x['ts'],x['te'])!=(ts,te) and x['ts']<lo]
                for s0,s1 in sel:
                    s0=max(s0,wlo);s1=min(s1,lo)
                    if s1>s0:exp[a:b,s0-wlo:s1-wlo]=True
            exp&=base[None,:];assert np.array_equal(H,exp),(mode,k,int(H.sum()),int(exp.sum()));assert (kp[:,WIN:]==base[None,WIN:]).all()
            res[(mode,k)]=int(H.sum())
    print('e59_nat_write selftest ok (brute-force masks of HIDE_PAIR / HIDE_EV on book 0 chunks',sorted({k for _,k in res}),'): hidden pairs',{f'{m}@{k}':v for (m,k),v in res.items()})
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    from tokenizers import Tokenizer
    import e55_run as R55,e55_iso as ISO,e57_natmask_run as NM,e57_qband_run as QR
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    docs,evs=S0.load(a.data,a.books);allsel=set(S0.select(evs,a.n_k0,a.sel_seed));evs_i=[dict(e,_i=i) for i,e in enumerate(evs)];LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);arms=a.arms.split(',')
    @eqx.filter_jit
    def grad_fn(m,st_suf,c,pref_,kmask,wt):
        def total(mm):
            s3.S3SUF['kmask']=kmask
            try:o=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
            finally:s3.S3SUF.clear()
            lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(wt*nll),nll
        (_l,nll),g=eqx.filter_value_and_grad(total,has_aux=True)(m);return g.inner_parameters(),nll
    @eqx.filter_jit
    def fwd_fn(m,st_suf,c,pref_,kmask):
        s3.S3SUF['kmask']=kmask
        try:o=m.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(o.logits.astype(jnp.float32),axis=-1);return -jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    res=dict(events=[],books=[],args=vars(a));t_all=time.monotonic();evmap={}
    for di,d in enumerate(docs):
        ids,offs,maxc=S0.tokenize(tk,d['text']);nch=len(ids)//CH
        if a.max_chunks:nch=min(nch,a.max_chunks)
        n=nch*CH;stream=ids[:n].astype(np.int64);de=[e for e in evs_i if e['doc']==d['id']];ev=book_struct(d,de,offs,len(ids))
        pos=[(i,e) for i,e in enumerate(ev) if e['idx'] in allsel and e['m2'] and e['m2'][0]>=1 and e['m2'][-1]<n]
        L=ISO.Iso55(rt,save_dir=out/'states'/f'b{di}',canary=can);L.rec_cache={};perarm={}
        for arm in arms:
            uv=NatHideUV(ev,arm,n,lg) if arm!='MASK1024' else None
            if arm=='MASK1024':
                base_uv=QR.SelMaskQ('ALLQ1024',{'band_ALLQ1024':np.asarray(1024),'all_ALLQ1024':np.asarray(1)},lg);holder=NatHideUV(ev,'MASK1024',n,lg)
                class Wrap:
                    def keep(s,k,m,Kc,Vc,ids_k,keep_all):holder.keep(k,m,Kc,Vc,ids_k,keep_all);return base_uv.keep(k,m,Kc,Vc,ids_k,keep_all)
                uv_obj=Wrap();stashsrc=holder
            else:uv_obj=uv;stashsrc=uv
            rec={};cnt=[0];stats=dict(steps=[])
            def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,stats=stats,arm=arm,stashsrc=stashsrc):
                k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=valid/valid.sum();g,nll_arm=grad_fn(m,st_suf,c,pref,kmask,jnp.asarray(wt));nll_arm=np.asarray(nll_arm,np.float32)
                if arm=='NAT':nll_nat=nll_arm
                else:
                    kk,Kc,Vc,keep_nat=stashsrc.stash;assert kk==k,(kk,k);ids_k=jnp.asarray(stream[k*CH:(k+1)*CH],dtype=jnp.int32)
                    pref_nat=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_nat));nll_nat=np.asarray(fwd_fn(m,st_suf,c,pref_nat,jnp.asarray(keep_nat)),np.float32)
                newp,gn,coef=L.sgd_apply(m,st_all,g);del g;rec[k]=(nll_nat,np.asarray(c.target_tokens),np.asarray(c.loss_masks,bool),nll_arm)
                stats['steps'].append(dict(k=k,gnorm=float(gn),coef=float(coef),train_ce=float((nll_arm*valid).sum()/valid.sum()),nat_ce=float((nll_nat*valid).sum()/valid.sum()),masked_pairs=int(stashsrc.last_info.get('masked_pairs',0))))
                return newp,gn,coef,jnp.asarray(stats['steps'][-1]['train_ce'],jnp.float32)
            L.write_step=ws;t0=time.monotonic();meta=[dict(sents=[]) for _ in range(nch)]
            r=L.run_stream(NM.ENGINE,1,stream,read_chunks=[],canary_chunks=[],nch=nch,tag=f'b{di}{arm}',log=lg,read_hook=None,hook_chunks=[],chunks_meta=meta,uv=uv_obj);r.pop('model',None);assert len(rec)==nch
            for k,(nl,tg,vm,_) in rec.items():
                idx=np.flatnonzero(vm);assert len(idx)>=CH-1 and np.array_equal(tg[idx],stream[k*CH+idx+1]),('targets differ from the stream',k)
            perarm[arm]=(rec,stats);lg(f'book {di} {d["id"][:40]} arm {arm}: {nch} chunks in {time.monotonic()-t0:.0f}s, mean natural-window NLL {np.mean([s_["nat_ce"] for s_ in stats["steps"]]):.3f}, mean training NLL {np.mean([s_["train_ce"] for s_ in stats["steps"]]):.3f}')
        def tn(rec,t):
            k=(t-1)//CH;i=(t-1)%CH;nl,tg,vm,_=rec[k];assert vm[i],(k,i);return float(nl[i])
        for i,e in pos:
            row=dict(doc=d['id'],k=e['k'],prev=e['prev'],c=e['c'],pos=e['m2'][0],ntok=len(e['m2']),pair=list(e['pair']),nll={arm:sum(tn(perarm[arm][0],t) for t in e['m2']) for arm in arms});res['events'].append(row)
        res['books'].append(dict(doc=d['id'],chunks=nch,events=len(pos),arms={arm:dict(mean_nat_ce=float(np.mean([s_['nat_ce'] for s_ in perarm[arm][1]['steps']])),mean_train_ce=float(np.mean([s_['train_ce'] for s_ in perarm[arm][1]['steps']])),masked_pairs=int(sum(s_['masked_pairs'] for s_ in perarm[arm][1]['steps'])),mean_gnorm=float(np.mean([s_['gnorm'] for s_ in perarm[arm][1]['steps']])),nat_ce_chunk=[s_['nat_ce'] for s_ in perarm[arm][1]['steps']]) for arm in arms}))
        assert all(abs(res['books'][-1]['arms'][ar]['nat_ce_chunk'][0]-res['books'][-1]['arms'][arms[0]]['nat_ce_chunk'][0])<2e-2 for ar in arms),'chunk 0 must be identical across arms'
        json.dump(res,open(out/'nat_write.partial.json','w'));L.rec_cache=None;del L,perarm
    res['sec']=time.monotonic()-t_all;json.dump(res,open(out/'nat_write.json','w'));lg('wrote',out/'nat_write.json',len(res['events']),'events')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT.parent/'TTT-reg-stage0/data/natural-text/recur'));ap.add_argument('--out');ap.add_argument('--books',type=int,default=0);ap.add_argument('--arms',default=','.join(ARMS));ap.add_argument('--max-chunks',type=int,default=0);ap.add_argument('--n-k0',type=int,default=2000);ap.add_argument('--sel-seed',type=int,default=0);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest(a.data)
    else:run(a)
