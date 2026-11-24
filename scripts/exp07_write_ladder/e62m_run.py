"""E62 MAIN run (design 1; code e62m_run.py derived from e62_run.py; thread-server 11/24; Hans 'as recommended'; NOT queued): 12 book units x 14 cells (n = 2, 4, 8, 16, 32 mentions x gap kinds; 124 chunks per unit), arms NAT and SAME (RAND dropped: placebo = 0 in P0a), ONE question frame (frame 0 of every type; P0a: the three frames agree, per-fact correlation 0.93-0.97, SD +1.7% for one frame), reads at W0 / 1st / 2nd / 4th / last mention / unit end. FAMILIARITY vs BINDING: the 14 cells are grouped by n (A: n=2 close/far/edge, B: n=4 close/mixed, C: n=4 far/edge, D: n=8 close/mixed/edge, E: n=16 close/mixed, F: n=32 close/mixed); the 6 groups carry the 6 attribute types, rotated from unit to unit (unit u: group g gets type (g + u) mod 6, each group meets each type twice over the 12 units). The answers of the OTHER facts of the same group (same book, same type, same number of exposures) are EXPOSED, wrongly bound decoys of a fact (1-2 per fact), placed at candidate positions 1..m; the remaining 15 - m candidates are unexposed generated words (never planted anywhere in the run). binding = (gold - mean exposed decoys); familiarity = (mean exposed decoys - mean unexposed); gold - max over all candidates stays as the headline. Units: the first 124 chunks of 11 PG-19 test books (Carmen Ariza, Ragged Trousered Philanthropists, Old Mortality, Coningsby, Life of Thomas Paine II, Nature's Serial Story, Civil War Through the Camera, Jennie Gerhardt, Leavenworth Case, Life of Gordon II, Patrol of the Sun Dance Trail) plus a second, non-overlapping segment of Carmen Ariza (from chunk 200); several of them are non-fiction prose or biography. Everything else as e62_run.py (planting e62_plant.py, templates e62_templates.py, masks, pre-write NLL records).
  python3 e62m_run.py --out O [--books 12] [--arms NAT,SAME] [--max-chunks N] [--seed 0] | --selftest"""
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
UNITS=[('pg19test_21',0),('pg19test_37',0),('pg19test_91',0),('pg19test_92',0),('pg19test_44',0),('pg19test_90',0),('pg19test_56',0),('pg19test_18',0),('pg19test_49',0),('pg19test_14',0),('pg19test_29',0),('pg19test_21',1)]
NCH_M=124;SEG_STEP=200;GROUP_OF=[0,0,0,1,1,2,2,3,3,3,4,4,5,5];TYPES6=PL.TYPES
def full_key(allb,p):return [k for k in allb if k.startswith(p+'_')][0]
def unit_text(tk,allb,bk):
    k0,sg=bk.split('|seg');text=allb[full_key(allb,k0)];e=tk.encode(text,add_special_tokens=False);i0=int(sg)*SEG_STEP*CH;c0=0
    if i0>0:
        c=e.offsets[i0][0];c0=[x for x in PL.sentence_bounds(text) if x>=c][0]
    j=min(len(e.ids),i0+(NCH_M+40)*CH)-1;return text[c0:e.offsets[j][1]]
def make_cands_m(rng,facts_by_book,ans_pool):
    """gold first, then the EXPOSED same-type answers of the other facts of the same book (same group = same number of exposures), then unexposed generated words (never planted in the run)"""
    out={};planted=set(g['answer'] for G in facts_by_book.values() for g in G)
    for b,F in facts_by_book.items():
        for i,f in enumerate(F):
            exp=[g['answer'] for j,g in enumerate(F) if j!=i and g['type']==f['type']];f['exposed']=exp
            fill=[x for x in ans_pool[f['type']] if x not in planted and x!=f['answer']];rng.shuffle(fill);out[(b,i)]=[f['answer']]+(exp+fill)[:NC-1]
    return out
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
        for fr in TP.FRAMES[f['type']][:1]:items.append(dict(prefix=fr.replace('{F}',f['name'][0]+' '+f['name'][1]),cands=[' '+x for x in cands[(b,i)]]))
    return items
def selftest(seed=0):
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));allb=PL.load_books();ans,names=PL.gen_facts(tk,allb,seed,pool_per_type=110,n_names=len(UNITS)*14+14);nm=list(names);used=collections.Counter();fb={};tot=0;ntk=collections.Counter()
    for ui,(k0,sg) in enumerate(UNITS):
        bk=f'{k0}|seg{sg}';cells=PL.schedule(seed,bk,NCH_M);F=[]
        for ci,c in enumerate(cells):
            c['type']=TYPES6[(GROUP_OF[ci]+ui)%6];F.append(dict(type=c['type'],group=GROUP_OF[ci],answer=ans[c['type']][used[c['type']]],name=nm.pop(0),n=c['n'],kind=c['kind']));used[c['type']]+=1
        fb[bk]=F;text=unit_text(tk,allb,bk);newt,rows,stt=PL.plant_book(tk,text,cells,[(f['name'],f['answer']) for f in F],seed,bk,NCH_M)
        by=collections.defaultdict(list)
        for r in rows:by[r['cell']].append(r)
        for ci,c in enumerate(cells):
            ch=[r['chunk'] for r in sorted(by[ci],key=lambda r:r['mention'])];g=[b-a for a,b in zip(ch,ch[1:])];assert ch==c['chunks'];n,kind=c['n'],c['kind']
            assert all(x<=5 for x in g) if kind=='close' else True;assert all(x>=11 for x in g) if kind=='far' else True;assert all(6<=x<=10 for x in g) if kind=='edge' else True
            if kind=='mixed':assert any(x<=5 for x in g) and sum(x>=11 for x in g)>=(2 if n==32 else 1)
            ntk.update(len(r['ans_tok']) for r in by[ci])
        assert max(r['chunk'] for r in rows)<NCH_M and stt['planted_share']<=0.06;tot+=len(rows);print(f'  unit {ui:2d} {bk[:34]:34s} {len(rows)} mentions, planted {100*stt["planted_share"]:.2f}%, types per group {[TYPES6[(g+ui)%6] for g in range(6)]}')
    print('  answer tokens per mention (write-time tokenisation):',dict(ntk),f'share exactly 3: {ntk[3]/sum(ntk.values()):.3f}');assert ntk[3]/sum(ntk.values())>=0.9
    cands=make_cands_m(random.Random(0),fb,ans)
    for bk,F in fb.items():
        for i,f in enumerate(F):
            c=cands[(bk,i)];assert len(c)==NC and len(set(c))==NC and c[0]==f['answer'] and c[1:1+len(f['exposed'])]==f['exposed'] and 1<=len(f['exposed'])<=2,(bk,i,f['exposed']);assert all(g['n']==f['n'] for g in F if g['answer'] in f['exposed'])
    print(f'e62m_run selftest ok: 12 units, {tot} mentions, groups share a type and the same n, 1-2 exposed decoys per fact; GPU parts are covered by the job smoke')
def run(a):
    import jax,jax.numpy as jnp,equinox as eqx
    from tokenizers import Tokenizer
    import e55_run as R55,e55_iso as ISO,e55_gpu as RG,e57_natmask_run as NM,e57_qband_run as QR
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic();seed=a.seed
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));allb=PL.load_books();ans,names=PL.gen_facts(tk,allb,seed,pool_per_type=110,n_names=len(UNITS)*14+14)
    bks=[f'{k0}|seg{sg}' for k0,sg in UNITS][:a.books];arms=a.arms.split(',')
    LD,s3,C=R55.lab();rt=LD.Runtime();can=R55.Canary55(LD,rt);rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    rng=random.Random(f'E62|{seed}|run');nm=list(names);facts_by_book={};cells_by_book={};used=collections.Counter()
    for ui,bk in enumerate(bks):
        cells=PL.schedule(seed,bk,NCH_M)
        for ci,c in enumerate(cells):c['type']=TYPES6[(GROUP_OF[ci]+ui)%6];c['group']=GROUP_OF[ci]
        cells_by_book[bk]=cells;F=[]
        for c in cells:
            F.append(dict(type=c['type'],group=c['group'],answer=ans[c['type']][used[c['type']]],name=nm.pop(0),n=c['n'],kind=c['kind']));used[c['type']]+=1
        facts_by_book[bk]=F
    forbid=[w for F in facts_by_book.values() for f in F for w in (f['answer'],f['name'][0],f['name'][1])]
    out_tail=R55.out_tail_8192(LD,rt.tok,rt.bos,forbid)
    # ---- W0 screen: margin (gold - max wrong, summed log-prob, mean over 3 frames) > 0 -> redraw the answer
    def margins(bk,F,cands):
        items=build_items(F,cands,bk);raw=R55.rows_for(LD,rt,items)
        with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));P=LD.packing.Packed(T_,raw,GP=2)
        with jax.default_matmul_precision('highest'):S=np.asarray(P.r1(rd.to_fp32(rt.base))).reshape(len(F),1,NC)
        return S,P
    for rnd in range(7):
        cands=make_cands_m(rng,facts_by_book,ans);bad=0
        for bk,F in facts_by_book.items():
            S,_=margins(bk,F,cands);mg=(S[:,:,0]-S[:,:,1:].max(2)).mean(1)
            for i,f in enumerate(F):
                f['w0_margin']=float(mg[i])
                if mg[i]>0:
                    bad+=1;pool=[x for x in ans[f['type']] if x not in [g['answer'] for G in facts_by_book.values() for g in G]];f['answer']=rng.choice(pool)
        lg(f'W0 screen round {rnd}: {bad} facts with margin > 0 redrawn')
        if bad==0:break
    else:raise RuntimeError('W0 screen did not settle')
    cands=make_cands_m(rng,facts_by_book,ans);res=dict(seed=seed,books=[],args=vars(a))
    for bk in bks:
        F=facts_by_book[bk];cells=cells_by_book[bk];text=unit_text(tk,allb,bk)
        newt,rows,stt=PL.plant_book(tk,text,cells,[(f['name'],f['answer']) for f in F],seed,bk,NCH_M);enc=tk.encode(newt,add_special_tokens=False);nch=min(len(enc.ids)//CH,NCH_M,a.max_chunks or 10**9);n=nch*CH;stream=np.asarray(enc.ids[:n],np.int64)
        by=collections.defaultdict(list)
        for r in rows:by[r['cell']].append(r)
        evs=[sorted([r['tok'] for r in by[ci] if r['tok'][1]<=n]) for ci in range(len(F))];sents=sent_spans(newt,enc.offsets);bad=[w.lower() for f in F for w in (f['answer'],f['name'][0],f['name'][1])]
        items=build_items(F,cands,bk);raw=R55.rows_for(LD,rt,items)
        with rt32.mesh,jax.default_matmul_precision('highest'):T_=LD._TailCtx(rt32,list(out_tail));P=LD.packing.Packed(T_,raw,GP=2)
        def read(m):
            with jax.default_matmul_precision('highest'):return np.asarray(P.r1(rd.to_fp32(m))).reshape(len(F),1,NC)
        # positive control (W0): the planted sentence directly in front of the question
        pcit=[];
        for i,f in enumerate(F):
            sent=sorted([r for r in by[i]],key=lambda r:r['mention'])[0]['text']
            for fr in TP.FRAMES[f['type']][:1]:pcit.append(dict(prefix=sent+'\n'+fr.replace('{F}',f['name'][0]+' '+f['name'][1]),cands=[' '+x for x in cands[(bk,i)]]))
        rawpc=R55.rows_for(LD,rt,pcit)
        with rt32.mesh,jax.default_matmul_precision('highest'):Ppc=LD.packing.Packed(LD._TailCtx(rt32,list(out_tail)),rawpc,GP=2)
        with jax.default_matmul_precision('highest'):W0pc=np.asarray(Ppc.r1(rd.to_fp32(rt.base))).reshape(len(F),1,NC)
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
        L=ISO.Iso55(rt,save_dir=out/'states'/f'u{bks.index(bk)}',canary=can);L.rec_cache={}
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
            r=L.run_stream(NM.ENGINE,1,stream,read_chunks=[],canary_chunks=[],nch=nch,tag=f'u{bks.index(bk)}{arm}',log=lg,read_hook=hook,hook_chunks=hooks,chunks_meta=meta,uv=uv);r.pop('model',None)
            rec['reads']={str(k):v for k,v in r.get('inline',{}).items()};rec['sec']=time.monotonic()-t0;brec['arms'][arm]=rec;mm=[x['masked_pairs'] for x in rec['steps']]
            lg(f'  arm {arm}: {nch} chunks in {rec["sec"]:.0f}s, masked pairs {sum(mm)} in {sum(1 for x in mm if x)} chunks, hidden key tokens {sum(x["hidden_key_tokens"] for x in rec["steps"])}, sentence rows {sum(x["sentence_rows"] for x in rec["steps"])}, mean |g| {np.mean([x["gnorm"] for x in rec["steps"]]):.3f}')
        if 'NAT' in brec['arms']:
            first=min((i for ar in arms if ar!='NAT' for i,x in enumerate(brec['arms'][ar]['steps']) if x['masked_pairs']),default=nch)
            for ar in arms:
                if ar=='NAT':continue
                d=max((abs(brec['arms'][ar]['steps'][i]['gnorm']-brec['arms']['NAT']['steps'][i]['gnorm']) for i in range(min(first,nch))),default=0.0);brec['arms'][ar]['pre_first_mask_max_dgnorm_vs_NAT']=d;lg(f'  check {ar} vs NAT before the first masked chunk ({first}): max |d gnorm| {d:.2e}')
        res['books'].append(brec);json.dump(res,open(out/'e62m.partial.json','w'),default=float);L.rec_cache=None;del L
    res['sec']=time.monotonic()-t_all;json.dump(res,open(out/'e62m.json','w'),default=float);lg('wrote',out/'e62m.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--books',type=int,default=12);ap.add_argument('--arms',default='NAT,SAME');ap.add_argument('--max-chunks',type=int,default=0);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    selftest(a.seed) if a.selftest else run(a)
