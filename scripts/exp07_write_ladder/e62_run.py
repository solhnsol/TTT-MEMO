"""E62 P0a run: invented facts planted into 3 PG-19 books, NAT / SAME / RAND writes, frozen-weight reads (GPU; new file; thread-server 11/23; E62 P0 request v1; EXPLORATORY; derived from e61_book_run.py / e59_nat_write.py; planting = e62_plant.py, pool = e62_templates.py). Per book (first NCH = 180 chunks of the planted text, llama3 tokens, whole chunks, weights from W0, original E2E write rule: clipped SGD lr 1 clip 1 on the mean NLL of all tokens, window 8 chunks): arms NAT (no mask), SAME (the rows of every later planted sentence of a fact lose the keys of ALL earlier planted sentences of the same fact that lie in the 8 earlier chunks OR earlier in the same chunk; all 32 layers, update path only; mask from chunk 0), RAND (placebo: the same rows, the same number of hidden sentences, each replaced by a book sentence of the same chunk and the closest token length that is not a planted sentence and contains no name / answer of any planted fact). READS (frozen weights, no KV carry, no book context; neutral 8192-token tail + question frame): 3 frames per fact (e62_templates.FRAMES), candidates = gold + 15 same-type answers (other books' planted answers first, then unplanted generated answers), score = summed log-probability of the answer span; main quantity: margin = gold - max(wrong) minus the W0 margin; read points: W0, after the chunk of the 1st / 2nd / 4th mention of every fact, after its last mention, after the last chunk; positive control (W0): the planted sentence directly in front of the question. Facts: W0 margin > 0 (mean over frames) -> the answer is redrawn (<= 6 rounds).
  python3 e62_run.py --out O [--books 3] [--arms NAT,SAME,RAND] [--max-chunks N] [--seed 0] | --selftest"""
import sys,json,time,bisect,re,pathlib,argparse,random,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e62_plant as PL,e62_templates as TP,e62_facts as FA
CH=1024;NW=8;WIN=NW*CH;ARMS=('NAT','SAME','RAND');NC=16
def sent_spans(text,offs):
    """token spans of the book sentences (regex boundaries) of the planted text"""
    starts=[o[0] for o in offs];b=[0]+PL.sentence_bounds(text)+[len(text)];out=[]
    for x,y in zip(b,b[1:]):
        ts=bisect.bisect_left(starts,x);te=bisect.bisect_left(starts,y)
        if te>ts:out.append((ts,te))
    return out
class E62UV:
    """UV object of the e55 engine: .keep(k,m,Kc,Vc,ids_k,keep_all) -> (bool [CH, WIN+CH], info). Rows = tokens of the later planted sentences of a fact inside the chunk; keys = tokens of the earlier planted sentences of the same fact located in [chunk start - WIN, row sentence start)"""
    def __init__(self,facts_ev,sents,txt_bad,text,offs,mode,seed=0):
        assert mode in ARMS;self.ev=facts_ev;self.mode=mode;self.last_info={};self.rep={};rng=random.Random(f'E62UV|{seed}')
        if mode=='RAND':
            planted=sorted(s for ev in facts_ev for s in ev);byc={}
            for s in sents:
                if any(s[0]<p[1] and p[0]<s[1] for p in planted):continue
                tx=text[offs[s[0]][0]:offs[min(s[1],len(offs))-1][1]].lower()
                if any(w in tx for w in txt_bad):continue
                byc.setdefault(s[0]//CH,[]).append(s)
            self.byc=byc
            for fi,ev in enumerate(facts_ev):
                for e in ev:
                    c=e[0]//CH;L=byc.get(c,[]) or byc.get(c-1,[]);ln=e[1]-e[0];self.rep[(fi,e)]=sorted(L,key=lambda x:abs((x[1]-x[0])-ln))[:6]
        self.rng=rng
    def rows_and_keys(self,k):
        lo=k*CH;hi=lo+CH;wlo=lo-WIN;out=[]
        for fi,ev in enumerate(self.ev):
            for j,(ts,te) in enumerate(ev):
                if te<=lo or ts>=hi:continue
                keys=[e for e in ev[:j] if e[1]>wlo and e[1]<=ts]
                if not keys:continue
                if self.mode=='RAND':
                    ks=[]
                    for e in keys:
                        c=[x for x in self.rep[(fi,e)] if x[1]<=ts and x[1]>wlo]
                        if c:ks.append(c[0])
                    keys=ks
                a,b=max(ts,lo)-lo,min(te,hi)-lo;iv=[]
                for s0,s1 in set(keys):
                    s0=max(s0,wlo);s1=min(s1,hi)
                    if s1>s0:iv.append((s0,s1))
                out.append(((a,b),iv))
        return out
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool);self.stash=(k,Kc,Vc,keep_all)
        if self.mode=='NAT':self.last_info={'masked_pairs':0,'hidden_key_tokens':0};return base,{'uv_arm':self.mode,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();wlo=k*CH-WIN;rows=0;hk=0
        for (a,b),iv in self.rows_and_keys(k):
            rows+=b-a
            for s0,s1 in iv:kp[a:b,s0-wlo:s1-wlo]=False;hk+=s1-s0
        n=int((~kp&base[None,:]).sum());self.last_info={'masked_pairs':n,'sentence_rows':rows,'hidden_key_tokens':hk};return kp,{'uv_arm':self.mode,'q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def make_cands(rng,facts_by_book,ans_pool):
    """facts_by_book[b] = list of dicts(type, answer); -> per fact list of 16 candidates (gold first): other books' planted answers of the same type first, then unplanted generated answers"""
    out={}
    for b,F in facts_by_book.items():
        for i,f in enumerate(F):
            oth=[g['answer'] for bb,G in facts_by_book.items() if bb!=b for g in G if g['type']==f['type']];rng.shuffle(oth)
            planted=set(g['answer'] for G in facts_by_book.values() for g in G);fill=[x for x in ans_pool[f['type']] if x not in planted and x!=f['answer']];rng.shuffle(fill)
            c=(oth+fill)[:NC-1];out[(b,i)]=[f['answer']]+c
    return out
def build_items(facts,cands,b):
    items=[]
    for i,f in enumerate(facts):
        for fr in TP.FRAMES[f['type']]:items.append(dict(prefix=fr.replace('{F}',f['name'][0]+' '+f['name'][1]),cands=[' '+x for x in cands[(b,i)]]))
    return items
def selftest(seed=0):
    PL.selftest(seed);print('e62_run selftest ok (planting); GPU parts are covered by the job smoke')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    from tokenizers import Tokenizer
    import e55_run as R55,e55_iso as ISO,e55_gpu as RG,e57_natmask_run as NM,e57_qband_run as QR
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic();seed=a.seed
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));allb=PL.load_books();ans,names=FA.gen_facts(tk,allb,seed) if hasattr(FA,'gen_facts') else PL.gen_facts(tk,allb,seed)
    bks=[k for k in allb if any(k.startswith(b) for b in PL.BOOKS)][:a.books];arms=a.arms.split(',')
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    rng=random.Random(f'E62|{seed}|run');nm=list(names);facts_by_book={};cells_by_book={};used=collections.Counter()
    for bk in bks:
        cells=PL.schedule(seed,bk);cells_by_book[bk]=cells;F=[]
        for c in cells:
            F.append(dict(type=c['type'],answer=ans[c['type']][used[c['type']]],name=nm.pop(0),n=c['n'],kind=c['kind']));used[c['type']]+=1
        facts_by_book[bk]=F
    forbid=[w for F in facts_by_book.values() for f in F for w in (f['answer'],f['name'][0],f['name'][1])]
    out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,forbid)
    # ---- W0 screen: margin (gold - max wrong, summed log-prob, mean over 3 frames) > 0 -> redraw the answer
    def margins(bk,F,cands):
        items=build_items(F,cands,bk);raw=R55.rows_for(LD,rt,items)
        with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));P=LD.packing.Packed(T_,raw,GP=2)
        with jax.default_matmul_precision('highest'):S=np.asarray(P.r1(rd.to_fp32(rt.base))).reshape(len(F),3,NC)
        return S,P
    for rnd in range(7):
        cands=make_cands(rng,facts_by_book,ans);bad=0
        for bk,F in facts_by_book.items():
            S,_=margins(bk,F,cands);mg=(S[:,:,0]-S[:,:,1:].max(2)).mean(1)
            for i,f in enumerate(F):
                f['w0_margin']=float(mg[i])
                if mg[i]>0:
                    bad+=1;pool=[x for x in ans[f['type']] if x not in [g['answer'] for G in facts_by_book.values() for g in G]];f['answer']=rng.choice(pool)
        lg(f'W0 screen round {rnd}: {bad} facts with margin > 0 redrawn')
        if bad==0:break
    else:raise RuntimeError('W0 screen did not settle')
    cands=make_cands(rng,facts_by_book,ans);res=dict(seed=seed,books=[],args=vars(a))
    for bk in bks:
        F=facts_by_book[bk];cells=cells_by_book[bk];text=allb[bk];e0=tk.encode(text,add_special_tokens=False);cut=e0.offsets[min(len(e0.ids),(PL.NCH+40)*CH)-1][1];text=text[:cut]
        newt,rows,stt=PL.plant_book(tk,text,cells,[(f['name'],f['answer']) for f in F],seed,bk);enc=tk.encode(newt,add_special_tokens=False);nch=min(len(enc.ids)//CH,PL.NCH,a.max_chunks or 10**9);n=nch*CH;stream=np.asarray(enc.ids[:n],np.int64)
        by=collections.defaultdict(list)
        for r in rows:by[r['cell']].append(r)
        evs=[sorted([r['tok'] for r in by[ci] if r['tok'][1]<=n]) for ci in range(len(F))];sents=sent_spans(newt,enc.offsets);bad=[w.lower() for f in F for w in (f['answer'],f['name'][0],f['name'][1])]
        items=build_items(F,cands,bk);raw=R55.rows_for(LD,rt,items)
        with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));P=LD.packing.Packed(T_,raw,GP=2)
        def read(m):
            with jax.default_matmul_precision('highest'):return np.asarray(P.r1(rd.to_fp32(m))).reshape(len(F),3,NC)
        # positive control (W0): the planted sentence directly in front of the question
        pcit=[];
        for i,f in enumerate(F):
            sent=sorted([r for r in by[i]],key=lambda r:r['mention'])[0]['text']
            for fr in TP.FRAMES[f['type']]:pcit.append(dict(prefix=sent+'\n'+fr.replace('{F}',f['name'][0]+' '+f['name'][1]),cands=[' '+x for x in cands[(bk,i)]]))
        rawpc=R55.rows_for(LD,rt,pcit)
        with rt32.mesh,jax.default_matmul_precision('highest'):Ppc=LD.packing.Packed(LD._TailCtx(rt32,list(out_tail)),rawpc,GP=2)
        with jax.default_matmul_precision('highest'):W0pc=np.asarray(Ppc.r1(rd.to_fp32(rt.base))).reshape(len(F),3,NC)
        W0r=read(rt.base);hash_=lambda m:LD.reuse._hash_pytree(m.inner_parameters())[0]
        hooks=sorted({r['chunk'] for ci in range(len(F)) for r in by[ci] if r['mention'] in (0,1,3,r['n']-1) and r['chunk']<nch}|{nch-1})
        lg(f'book {bk[:40]}: {nch} chunks, {sum(len(e) for e in evs)} mentions in the stream, planted share {100*stt["planted_share"]:.2f}%, {len(hooks)} read points')
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
        planted_mask=np.zeros(n+1,bool)
        for r in rows:
            if r['tok'][1]<=n:planted_mask[r['tok'][0]:r['tok'][1]]=True
        ment_by_chunk=collections.defaultdict(list)
        for r in rows:
            if r['tok'][1]<=n and r['chunk']<nch:ment_by_chunk[r['chunk']].append(r)
        brec=dict(book=bk,nch=nch,facts=F,cands={str(i):cands[(bk,i)] for i in range(len(F))},manifest=rows,plant_stats={k:v for k,v in stt.items() if k!='exact_repeat_share'},exact_repeat_share={str(k):v for k,v in stt['exact_repeat_share'].items()},W0=W0r.tolist(),W0_positive_control=W0pc.tolist(),hooks=hooks,arms={})
        L=ISO.Iso55(rt,save_dir=out/'states'/bk[:12].replace(' ','_'),canary=can);L.rec_cache={}
        for arm in arms:
            uv=E62UV(evs,sents,bad,newt,enc.offsets,arm,seed);rec=dict(steps=[],hashes=[],mention_nll=[]);cnt=[0]
            def ws(m,st_all,st_suf,c,pref,kmask,rec=rec,cnt=cnt,uv=uv):
                k=cnt[0];cnt[0]+=1;valid=np.asarray(c.loss_masks,np.float32);wt=valid/valid.sum();g,nll=grad_fn(m,st_suf,c,pref,kmask,jnp.asarray(wt));nll=np.asarray(nll,np.float32)
                if arm=='NAT':nll_nat=nll
                else:
                    kk,Kc,Vc,keep_nat=uv.stash;assert kk==k,(kk,k);ids_k=jnp.asarray(stream[k*CH:(k+1)*CH],dtype=jnp.int32);pref_nat=L.prefix_masked(m,ids_k,Kc,Vc,jnp.int32(k*CH),jnp.asarray(keep_nat));nll_nat=np.asarray(fwd_fn(m,st_suf,c,pref_nat,jnp.asarray(keep_nat)),np.float32)
                newp,gn,coef=L.sgd_apply(m,st_all,g);del g;ce=float((nll*valid).sum()/valid.sum());lo=k*CH;npm=~planted_mask[lo+1:lo+CH+1][:len(nll)]&(valid>0)
                for r in ment_by_chunk.get(k,[]):
                    ix=[t-1-lo for t in r['ans_tok'] if 0<=t-1-lo<len(nll)-1];rec['mention_nll'].append(dict(cell=r['cell'],mention=r['mention'],k=k,ntok=len(r['ans_tok']),nll_arm=[float(nll[i]) for i in ix],nll_nat=[float(nll_nat[i]) for i in ix]))
                rec['steps'].append(dict(k=k,gnorm=float(gn),coef=float(coef),train_ce=ce,nat_ce=float((nll_nat*valid).sum()/valid.sum()),nonplanted_ce_arm=float(nll[npm].mean()) if npm.any() else None,nonplanted_ce_nat=float(nll_nat[npm].mean()) if npm.any() else None,masked_pairs=int(uv.last_info.get('masked_pairs',0)),hidden_key_tokens=int(uv.last_info.get('hidden_key_tokens',0)),sentence_rows=int(uv.last_info.get('sentence_rows',0))))
                return newp,gn,coef,jnp.asarray(ce,jnp.float32)
            def hook(k,m,rec=rec):
                h0=hash_(m);r=read(m);h1=hash_(m);assert h0==h1,'weights changed by a read';rec['hashes'].append(h0);return r.tolist()
            L.write_step=ws;t0=time.monotonic();meta=[dict(sents=[]) for _ in range(nch)]
            r=L.run_stream(NM.ENGINE,1,stream,read_chunks=[],canary_chunks=[],nch=nch,tag=f'{bk[:8]}{arm}',log=lg,read_hook=hook,hook_chunks=hooks,chunks_meta=meta,uv=uv);r.pop('model',None)
            rec['reads']={str(k):v for k,v in r.get('inline',{}).items()};rec['sec']=time.monotonic()-t0;brec['arms'][arm]=rec;mm=[x['masked_pairs'] for x in rec['steps']]
            lg(f'  arm {arm}: {nch} chunks in {rec["sec"]:.0f}s, masked pairs {sum(mm)} in {sum(1 for x in mm if x)} chunks, hidden key tokens {sum(x["hidden_key_tokens"] for x in rec["steps"])}, sentence rows {sum(x["sentence_rows"] for x in rec["steps"])}, mean |g| {np.mean([x["gnorm"] for x in rec["steps"]]):.3f}')
        if 'NAT' in brec['arms']:
            first=min((i for ar in arms if ar!='NAT' for i,x in enumerate(brec['arms'][ar]['steps']) if x['masked_pairs']),default=nch)
            for ar in arms:
                if ar=='NAT':continue
                d=max((abs(brec['arms'][ar]['steps'][i]['gnorm']-brec['arms']['NAT']['steps'][i]['gnorm']) for i in range(min(first,nch))),default=0.0);brec['arms'][ar]['pre_first_mask_max_dgnorm_vs_NAT']=d;lg(f'  check {ar} vs NAT before the first masked chunk ({first}): max |d gnorm| {d:.2e}')
        res['books'].append(brec);json.dump(res,open(out/'e62_p0a.partial.json','w'),default=float);L.rec_cache=None;del L
    res['sec']=time.monotonic()-t_all;json.dump(res,open(out/'e62_p0a.json','w'),default=float);lg('wrote',out/'e62_p0a.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--books',type=int,default=3);ap.add_argument('--arms',default=','.join(ARMS));ap.add_argument('--max-chunks',type=int,default=0);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    selftest(a.seed) if a.selftest else run(a)
