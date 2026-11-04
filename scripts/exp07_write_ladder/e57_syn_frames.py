"""E57 synthetic long world (v3_16_1_L1024, seed 64): are the person-place bindings more than 'same-sentence' links? (new file; investigation thread, Hans 11/04 16:41 KST; read only on the saved states of job 881, NO weight update). Read frames: (1) the base read of job 881 (3 question frames, 16 place candidates); (2) INTERVENING CLAUSE, 3 phrasings: '{X}, who may be a friend of {Y}, now lives in the village of' ..., Y = the person of ANOTHER fact (fact (i+1) mod 16), candidates = the same 16 places; (3) REVERSE, 3 phrasings: 'The person who lives in {P} is' ..., candidates = the 16 person names. States: W0, ALLQ256 (M3W1N05), M3 w=8 (M3W8N05) and NAT (NATSGD) at a = 256 and 768 (chunk 255 / 767). Stage check (CPU): no word 6-gram of a new prefix (names / places / slots masked) in the 300 sentence frames or the 24 background books, key phrases not in the decoded stream, no sentence of the stream with both X and Y. Stage read (GPU): per-token reads of the 7 models, the base frames are packed like job 881 (reproduction check against its stored reads). Stage analyze (CPU): per frame kind first-token dA, d(A-B), branch / first-token rank ratio, top-1 (fact bootstrap, 3 phrasings averaged per fact), M3 minus ALLQ256 pairs.
  python3 e57_syn_frames.py --stage check|read|analyze --out O | --selftest"""
import sys,re,json,hashlib,time,gc,argparse,pathlib,random
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_longcell
EXP=ROOT/'experiments/exp07_write_ladder';CELL='v3_16_1_L1024'
FRIEND=['{X}, who may be a friend of {Y}, now lives in the village of','Some say {X}, a possible friend of {Y}, makes his home in the settlement of','{X}, perhaps a friend of {Y}, has his home in the town of']
REVERSE=['The person who lives in {P} is','Whoever makes a home in {P} goes by the name of','The name of the resident of {P} is']
POINTS=(255,767)
def mask_words(text):
    t=re.sub(r'\{\w+\}',' <s> ',text.lower());return re.findall(r"<s>|[a-z']+",t)
def ngrams(words,n):return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}
def shared(prefix_words,corpus_sets,n):return ngrams(prefix_words,n)&corpus_sets[n]
def build_world():
    import e55_run as R55,types,e57_syn_pass as SP
    args=SP.synth_args(64,CELL);enc,w,tf=R55.build(args,64,read_apps={16,32,64,128},arms=['NAT']);return R55,enc,w,tf
def make_items(w,base_items):
    """-> {'friend': [3 lists of 16 items], 'reverse': [3 lists of 16 items]}; every item has prefix, cands, cand_fact, gold_idx"""
    F=w['facts'];k=len(F);out={'friend':[],'reverse':[]}
    for j,fr in enumerate(FRIEND):
        its=[]
        for i in range(k):
            b=next(x for x in base_items if x['kind']=='fact' and x['fact']==i);its.append(dict(kind='fact',fact=i,prefix=fr.format(X=F[i]['X'],Y=F[(i+1)%k]['X']),cands=list(b['cands']),cand_fact=list(b['cand_fact']),gold_idx=b['gold_idx'],qframe=j))
        out['friend'].append(its)
    for j,fr in enumerate(REVERSE):
        its=[]
        for i in range(k):
            r=random.Random(f"E57frames|rev|{w['seed']}|{j}|{i}");order=list(range(k));r.shuffle(order)
            its.append(dict(kind='fact',fact=i,prefix=fr.format(P=F[i]['P']),cands=[' '+F[o]['X'] for o in order],cand_fact=order,gold_idx=order.index(i),qframe=j))
        out['reverse'].append(its)
    return out
def check_stage(a):
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);R55,enc,w,tf=build_world();import gen_facts as G,e55_read as R
    P_=R.plan(w,enc);base=P_['items'];its=make_items(w,base);F=w['facts']
    prefs=[(kind,j,it['prefix']) for kind in its for j in range(3) for it in its[kind][j][:2]]            # 2 examples per phrasing show the surface; the n-gram test uses the slot-masked frame
    frames=[('friend',j,fr) for j,fr in enumerate(FRIEND)]+[('reverse',j,fr) for j,fr in enumerate(REVERSE)]
    
    # corpus n-gram sets: bank sentences (slots masked) and the 24 background books
    sets={n:set() for n in (4,5,6)};
    for t in w['bank'] if isinstance(w['bank'],list) else []:
        ws=mask_words(t)
        for n in sets:sets[n]|=ngrams(ws,n)
    bsets={n:set() for n in (4,5,6)};books=G.CELL_BOOKS[CELL];txt=[]
    for b in books:
        f=ROOT/'data/narrative_gutenberg'/b
        if f.exists():
            ws=re.findall(r"[a-z']+",f.read_text(errors='ignore').lower())
            for n in bsets:bsets[n]|=ngrams(ws,n)
    res={'frames':[],'bank_size':len(w['bank']) if isinstance(w['bank'],list) else None,'books':len(books)}
    for kind,j,fr in frames:
        ws=mask_words(fr);e={'kind':kind,'phrasing':j,'frame':fr}
        for n in (4,5,6):
            sb=shared(ws,sets,n);sk=shared(ws,bsets,n);e[f'bank_shared_{n}grams']=[' '.join(x) for x in sorted(sb)];e[f'books_shared_{n}grams']=[' '.join(x) for x in sorted(sk)][:10];e[f'books_shared_{n}gram_count']=len(sk)
        res['frames'].append(e)
    # key phrases in the decoded stream
    import e55_run as R55b;S=R55b.streams(w,enc);ids,sha=S['A']
    import e57_pqa2_stream as S2;tk=S2.tokenizer();text=' '.join(tk.decode(list(map(int,ids[i:i+65536]))) for i in range(0,len(ids),65536)).lower();text=re.sub(r'\s+',' ',text)
    keys=['friend of','a friend','possible friend','perhaps a friend','may be a friend','person who lives','who lives in','lives in','makes a home','make a home','makes his home','goes by the name','name of the resident','resident of','has his home','his home in','now lives','is settled in']
    res['key_phrase_counts_in_stream']={k_:text.count(k_) for k_ in keys}
    # sentences with both X and Y (friend frames): the fact sentences of the world
    both=0;samec=0;n_sent=0
    for c in w['chunks']:
        names_in_chunk=set()
        for s in c['sents']:
            n_sent+=1;names=[f['X'] for f in F if f['X'] in s['text']];names_in_chunk.add(F[s['fact']]['X']);both+=int(len(names)>1)
        samec+=int(len(names_in_chunk)>1)
    res['stream_sentences']=n_sent;res['sentences_with_two_person_names']=both;res['chunks_with_more_than_one_person']=samec;res['n_chunks']=len(w['chunks'])
    res['examples']={f'{kind}{j}':its[kind][j][0]['prefix'] for kind in its for j in range(3)}
    bad=[(e['kind'],e['phrasing']) for e in res['frames'] if e['bank_shared_6grams'] or e['books_shared_6grams']]
    res['six_gram_overlap_frames']=bad;res['ok']=not bad and sum(res['key_phrase_counts_in_stream'][k_] for k_ in ('friend of','person who lives','goes by the name','name of the resident','has his home','makes a home','make a home','makes his home'))==0 and both==0
    (out/'e57_syn_frames_check.json').write_text(json.dumps(res,indent=1));print(json.dumps({k_:v for k_,v in res.items() if k_ not in ('frames',)},indent=1)[:2500]);
    for e in res['frames']:print(e['kind'],e['phrasing'],'bank 4/5/6-grams shared',[len(e[f'bank_shared_{n}grams']) for n in (4,5,6)],'books',[e[f'books_shared_{n}gram_count'] for n in (4,5,6)])
def read_stage(a):
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True)
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G,e57_syn_tokread as TR,types
    LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    import e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(64,CELL),64,read_apps={16,32,64,128},arms=['NAT']);P_=R.plan(w,enc);base=P_['items'];tok=rt.tok;its=make_items(w,base)
    out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    groups={'base':base};[groups.update({f'{k_}{j}':its[k_][j]}) for k_ in its for j in range(3)];packs={};tokids={}
    for gname,gi in groups.items():
        raw=R55.rows_for(LD,rt,gi);RG.check_rows(raw,len(gi),G.NCAND);packs[gname]=rd.packed(raw,out_tail);tokids[gname]=[[int(x) for x in ids_[a_+1:b_+1]] for ids_,(a_,b_) in raw]
        lg(f'  group {gname}: {len(gi)} items, {len(raw)} rows')
    def tokens(m):
        m32=rd.to_fp32(m);res={}
        with LD.jax.default_matmul_precision('highest'):
            for gname,P in packs.items():res[gname]=[x.tolist() for x in TR.r1_tokens(P,rt32,m32)]
        return res
    arms=TR.arms_curve(64,'synqL');want={l:s for l,s in arms.items() if int(l.split('@c')[1]) in POINTS}
    res={'groups':{g:[dict(fact=it['fact'],prefix=it['prefix'],cand_fact=it['cand_fact'],gold_idx=it['gold_idx'],kind=it['kind']) for it in gi] for g,gi in groups.items()},'tokids':tokids,'W0':tokens(rt.base),'states':{},'labels':sorted(want)}
    for l in sorted(want):
        t0=time.monotonic();m=rd.load_state(want[l]['file'],expect_hash=want[l]['hash']);res['states'][l]=tokens(m);del m;lg(f'  {l}: {time.monotonic()-t0:.0f}s')
    # reproduction of the base read against the stored per-token reads of job 881
    st=json.load(open(EXP/'e57_syn_long_curve/e57_syn_tokread_s64.json'));rep={}
    for l in res['states']:
        if l in st['arms']:rep[l]=float(max(abs(np.asarray(x)-np.asarray(y)).max() for x,y in zip(res['states'][l]['base'],st['arms'][l]) if len(x)==len(y)))
    res['reproduction_base_vs_job881_max_abs_token_logprob_diff']=rep;lg('reproduction',rep)
    (out/'e57_syn_frames_read.json').write_text(json.dumps(res))
def analyze_stage(a):
    import e57_syn_tokread as TR
    out=pathlib.Path(a.out);d=json.load(open(out/'e57_syn_frames_read.json'));meta_base=json.load(open(next(iter(TR.arms_curve(64,'synqL').values()))['read_json']))['meta'];labels=d['labels']
    def meta_of(g):
        if g=='base':return meta_base
        it=d['groups'][g];return dict(kind=[x['kind'] for x in it],fact=[x['fact'] for x in it],cand_fact=np.asarray([x['cand_fact'] for x in it]),gi=[x['gold_idx'] for x in it],gold_idx=[x['gold_idx'] for x in it])
    M={};W0={}
    for g in d['groups']:
        W0[g]=TR.metrics(d['W0'][g],d['tokids'][g],meta_of(g))
        for l in labels:M[(l,g)]=TR.metrics(d['states'][l][g],d['tokids'][g],meta_of(g))
    facts=sorted(W0['base']);kinds={'base':['base'],'friend':[f'friend{j}' for j in range(3)],'reverse':[f'reverse{j}' for j in range(3)]}
    def pf(l,kind,fn):
        v=[]
        for f in facts:
            xs=[fn(M[(l,g)][f],W0[g][f]) for g in kinds[kind]];xs=[x for x in xs if x is not None];v.append(np.mean(xs) if xs else np.nan)
        return np.asarray(v)
    FN={'first_dA':lambda m,w:m['A_first']-w['A_first'],'first_dAB':lambda m,w:(m['A_first']-w['A_first'])-(m['B_first']-w['B_first']) if m['B_first'] is not None and w['B_first'] is not None else None,'rr_branch':lambda m,w:m['rank_branch']/16.0,'rr_first':lambda m,w:m['rank_first']/16.0,'top1_branch':lambda m,w:float(m['rank_branch']==1),'rr_branch_improvement':lambda m,w:(w['rank_branch']-m['rank_branch'])/16.0}
    res={'reproduction':d['reproduction_base_vs_job881_max_abs_token_logprob_diff'],'W0':{kind:{k:TR.boot(pf(labels[0],kind,lambda m,w,f=f:f(w,w))) if False else None for k,f in ()} for kind in kinds},'by_state':{}}
    res['W0']={kind:{'rr_branch':TR.boot(np.mean([[W0[g][f]['rank_branch']/16.0 for f in facts] for g in kinds[kind]],axis=0)),'rr_first':TR.boot(np.mean([[W0[g][f]['rank_first']/16.0 for f in facts] for g in kinds[kind]],axis=0)),'top1_branch':TR.boot(np.mean([[float(W0[g][f]['rank_branch']==1) for f in facts] for g in kinds[kind]],axis=0))} for kind in kinds}
    for l in labels:
        res['by_state'][l]={kind:{k:TR.boot(pf(l,kind,fn)) for k,fn in FN.items()} for kind in kinds}
    res['pairs']={}
    for pt in POINTS:
        x,y,z=f'M3W1N05@c{pt}',f'M3W8N05@c{pt}',f'NATSGD@c{pt}'
        if x in labels and y in labels:
            res['pairs'][str(pt)]={f'M3w8_minus_ALLQ256':{kind:{k:TR.boot(pf(y,kind,fn)-pf(x,kind,fn)) for k,fn in FN.items()} for kind in kinds},**({'ALLQ256_minus_NAT':{kind:{k:TR.boot(pf(x,kind,fn)-pf(z,kind,fn)) for k,fn in FN.items()} for kind in kinds}} if z in labels else {})}
    (out/'e57_syn_frames_report.json').write_text(json.dumps(res,indent=1,default=float));print('wrote',out/'e57_syn_frames_report.json')
def selftest():
    assert mask_words('{X}, who may be a friend of {Y}, now lives in')==['<s>','who','may','be','a','friend','of','<s>','now','lives','in']
    assert shared(mask_words('a b c d e f g'),{6:{tuple('b c d e f g'.split())}},6)=={tuple('b c d e f g'.split())}
    print('e57_syn_frames selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    {'check':check_stage,'read':read_stage,'analyze':analyze_stage}[a.stage](a)
if __name__=='__main__':main()
