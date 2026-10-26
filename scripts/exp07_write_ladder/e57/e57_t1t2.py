"""E57 T1/T2 candidate preparation (CPU only, NO model, NO GPU, NO download; design E57 section 72-75; private, internal research only).
Input: data/news-lite/t1t2_draft_v1.json (150 items: T1 = bank question with new frames, T2 = reverse questions), bank_v1.json (frozen, sha256 9374e1f5...), the bundle of job 229 (OUT/e57_bundle/bundle_store.pkl = the 871 lead articles of the stream). Reuses e57_bank.py / e57_nominate.py UNCHANGED (bank v1 and its code stay untouched; every output goes to a NEW out dir).
  check  --draft D --bank-dir BANKDIR --out OUT      step 1-3: (1) entity-masked 6-gram overlap of the 150 frames with every sentence of the 871-article stream (+ with the original bank frame and the sibling frames of the same question); (2) the 12 T2 golds: variants / aliases counted in the stream, fixed surface (most frequent variant, as bank freeze does), c' (articles holding the gold); (3) T2 easy (<= 20) and hard (same event: frequent same-type strings + the other T2 golds of the event) candidates with c' per candidate (event articles, whole-stream articles); T1 candidates copied from bank_v1 (hash recomputed).
  verify --out OUT --nominations N.json              step 4a: the new-angle nominations per T2 item {"nominations":[{"id","nominated_from_list":[{"surface"}],"proposed_extra_needs_code_check":[str],"easy_extra":[str]}]}: extras must occur in the event lead and not be an alias of the gold (e57_nominate.check_extra); easy extras must occur anywhere in the stream; <= 4 nominated per item; NO packet evidence needed (T2 has no packet).
  freeze --out OUT --version v1                       step 4b: t1t2_<version>.json + .sha256 (never overwritten, exit 8); holds the 150 frames, the bank_v1 candidates of the T1 questions (hash checked), the T2 candidates, the overlap result and the hashes of the inputs.
Entity-masked n-gram (PROPOSED definition, not yet approved): a token is ENT if it is capitalised and not sentence-initial, or holds a digit, or is a word of the item's event name / gold / answer variants / S; two n-grams match when the sequences are equal with ENT = ENT; an n-gram counts as overlap only if it has >= k non-ENT tokens. k = 1 strict, 2 normal, 3 loose are all reported."""
import argparse,json,sys,re,hashlib,pathlib,collections,datetime
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_bank as B
import e57_nominate as NOM
NGRAM=6;KS=(1,2,3);MAXN=4;EASY_N=20
TOK=re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")
# T2 gold classes (set by hand: the extractor of e57_bank types only PLACE / PERSON / DATE / NUMBER). PERSON = person, TITLE = work title, TEAM = team; the extractor pool for TITLE / TEAM is its OTHER bucket (noisy, flagged).
T2_CLASS={'Mark Ruffalo':'PERSON','Jon Bernthal':'PERSON','Michael Mando':'PERSON','Zendaya':'PERSON','Atanu Chakraborty':'PERSON','Manny Pacquiao':'PERSON','Floyd Mayweather':'PERSON','Zohran Mamdani':'PERSON','Rashmika Mandanna':'PERSON','Spider-Man: Brand New Day':'TITLE','Hannah Montana':'TITLE','Jammu and Kashmir':'TEAM'}
# aliases that are NOT contained in the gold string (contained strings are removed from the candidates automatically by e57_bank.related)
T2_ALIAS={'Manny Pacquiao':['Emmanuel Pacquiao'],'Floyd Mayweather':['Floyd Mayweather Jr.','Floyd Mayweather Jr','Mayweather Jr.'],'Zohran Mamdani':['Zohran Kwame Mamdani'],'Spider-Man: Brand New Day':['Spider-Man Brand New Day','Brand New Day'],'Jammu and Kashmir':['Jammu & Kashmir','J&K','J & K'],'Rashmika Mandanna':['Rashmika'],'Atanu Chakraborty':['Chakraborty'],'Mark Ruffalo':['Ruffalo'],'Jon Bernthal':['Bernthal'],'Michael Mando':['Mando']}
def jhash(o):return B.jhash(o)
def sha_file(p):return hashlib.sha256(open(p,'rb').read()).hexdigest()
def kst():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
# ---------- entity-masked n-grams ----------
def sent_tokens(text):
    """-> [(token, sentence_initial)] for a text of one or more sentences"""
    out=[];init=True;last=0
    for m in TOK.finditer(text):
        gap=text[last:m.start()];last=m.end()
        if re.search(r'[.?!]\s*$',gap) or re.search(r'[.?!]\s+$',gap):init=True
        out.append((m.group(0),init));init=False
    return out
def ent_words(*strings):
    w=set()
    for s in strings:
        for t in TOK.findall(s or ''):
            w.add(t.lower());w.update(x.lower() for x in re.split(r"[-'’]",t) if x)
    return w
def mask_seq(tokens,ent):
    return ['ENT' if ((t[0].isupper() and not init) or any(c.isdigit() for c in t) or t.lower() in ent) else t.lower() for t,init in tokens]
def grams(seq,n=NGRAM):
    return {tuple(seq[i:i+n]) for i in range(len(seq)-n+1)}
def nonent(g):return sum(1 for x in g if x!='ENT')
class StreamIndex:
    """all sentences of the stream with an inverted index on lower-cased tokens (a gram can only occur in sentences holding all its non-ENT words)"""
    def __init__(self,sents):
        self.s=sents;self.toks=[sent_tokens(t) for t in sents];self.post=collections.defaultdict(set)
        for i,tk in enumerate(self.toks):
            for t,_ in tk:self.post[t.lower()].add(i)
    def overlap(self,frame,ent,kmin=1):
        """-> {k: [(gram, sentence_id)]} for k in KS: stream sentences holding an entity-masked 6-gram of the frame with >= k non-ENT tokens"""
        fs=mask_seq(sent_tokens(frame),ent);res={k:[] for k in KS}
        for g in grams(fs):
            ne=nonent(g)
            if ne<min(KS):continue
            words=[w for w in g if w!='ENT'];cand=None
            for w in sorted(set(words),key=lambda w:len(self.post.get(w,()))):
                p=self.post.get(w,set());cand=set(p) if cand is None else cand&p
                if not cand:break
            for i in sorted(cand or ()):
                if g in grams(mask_seq(self.toks[i],ent)):
                    for k in KS:
                        if ne>=k:res[k].append((' '.join(g),i))
        return res
def pair_overlap(a,b,ent):
    ga=grams(mask_seq(sent_tokens(a),ent));gb=grams(mask_seq(sent_tokens(b),ent));common=ga&gb
    return {k:sorted(' '.join(g) for g in common if nonent(g)>=k) for k in KS}
# ---------- loading ----------
def load_stream(out):
    """-> (st, sentences, sent_event, sent_article): every sentence of the lead of every independent article (the 871 articles of the stream)"""
    st,idx=B.load(out);S=[];ev=[];art=[]
    for nm,E in st.items():
        for ai,sents,cum in B.lead_stream(E):
            for t in sents:S.append(t);ev.append(nm);art.append((nm,ai))
    return st,S,ev,art
def bank_check(bankdir):
    bd=pathlib.Path(bankdir);b=json.load(open(bd/'bank_v1.json'));h=b.pop('sha256');recomputed=jhash(b);b['sha256']=h;fileh=(bd/'bank_v1.sha256').read_text().strip()
    ok=recomputed==h==fileh and h.startswith('9374e1f5');return b,dict(ok=ok,sha256=h,recomputed=recomputed,sha_file=fileh,file_bytes_sha256=sha_file(bd/'bank_v1.json'))
def count_variant(st,variants):
    """-> (articles holding any variant, sentences per variant) over the whole stream, word-boundary, case-insensitive (as bank freeze counts)"""
    rx={v:re.compile(r'(?<![\w])'+re.escape(v)+r'(?![\w])',re.I) for v in variants};per=collections.Counter();arts=set()
    for nm,E in st.items():
        for ai,sents,cum in B.lead_stream(E):
            for t in sents:
                hit=False
                for v,r in rx.items():
                    if r.search(t):per[v]+=1;hit=True
                if hit:arts.add((nm,ai))
    return len(arts),per
def alias_scan(st,gold):
    """strings of 1-4 capitalised words containing the first or last name token of the gold, with their sentence counts (for the person to see possible aliases; nothing is decided by it)"""
    toks=[t for t in TOK.findall(gold) if t[0].isupper()];keys={toks[0].lower(),toks[-1].lower()} if toks else set();c=collections.Counter()
    for nm,E in st.items():
        for ai,sents,cum in B.lead_stream(E):
            for t in sents:
                for m in B.CAPSEQ.finditer(t):
                    if ent_words(m.group(0))&keys:c[m.group(0)]+=1
    return c.most_common(10)
# ---------- check ----------
def t2_questions(draft):
    """-> one record per T2 gold (the question and the cloze form share the gold): id of the question form, event, forms"""
    by=collections.OrderedDict()
    for it in draft['items']:
        if it['kind']!='T2':continue
        sid=it['id'].replace('-cloze','')
        r=by.setdefault(sid,dict(id=sid,event=it['event'],gold=it['gold'],forms=[]));r['forms'].append(it['id'])
        assert r['gold']==it['gold'],(sid,it['gold'])
    return list(by.values())
def make_q(r):
    cls=T2_CLASS[r['gold']];var=[r['gold']]+T2_ALIAS.get(r['gold'],[])
    return dict(id=r['id'],event=r['event'],S=r['event'],type='PERSON' if cls=='PERSON' else 'OTHER',gold_class=cls,frame='{S}',canonical=r['gold'],variants=var,evidence=[],direction='A->S',author='claude-news-thread',status='draft')

NONPERSON={'new','york','delhi','city','special','anniversary','day','trophy','republic','street','square','house','states','state','kingdom','india','china','america','american','indian','venezuelan','premiere','season','episode','series','film','movie','official','officials','sports','channel','prize','peace','days','world','times','news','capital','district'}
LEAD_TITLES={'mayor','president','actor','actors','actress','actresses','dr','mr','mrs','ms','coach','minister','democratic','socialist','senator','governor','judge','chairman','director','captain','chief','prime','star','boxer','singer','pop','former','new','secretary','sen','rep','gov','justice','attorney','general','mayor-elect','mayoral','candidate','legend','veteran','skipper','duo','couple'}
def person_pool(st):
    """-> {cluster key (last token): dict(surfaces Counter, arts set, ctx_arts set)} for PERSON-typed strings of >= 2 words (possessive stripped) of the whole stream; ctx = the mention follows a title or precedes said / told / was ... (E57 bank person rule)"""
    pool=collections.defaultdict(lambda:dict(surf=collections.Counter(),arts=set(),ctx=set()))
    for nm,E in st.items():
        for ai,sents,cum in B.lead_stream(E):
            for t in sents:
                for ty,key,sf,a,b in B.type_strings(t):
                    if ty!='PERSON':continue
                    sf=re.sub(r"['’]s$","",sf).strip();w=sf.split()
                    while len(w)>2 and w[0].lower() in LEAD_TITLES:w=w[1:]
                    if len(w)==2 and w[0].lower() in LEAD_TITLES:continue
                    sf=' '.join(w)
                    if len(w)<2 or {x.lower() for x in w}&NONPERSON or re.search(r"['’]s\b",sf):continue
                    c=pool[w[-1].lower()];c['surf'][sf]+=1;c['arts'].add((nm,ai))
                    if B.PERSON_BEFORE.search(t[max(0,a-16):a]) or B.PERSON_AFTER.search(t[b:b+80]):c['ctx'].add((nm,ai))
    return pool
def person_easy(pool,q,hard,golds,n=EASY_N):
    pl=json.load(open(HERE/'e57_t1t2_easy_pools.json'));nonp=pl.get('FICTIONAL',[])+pl.get('TITLE',[])+pl.get('TEAM',[])
    """easy list of a PERSON gold (decision C): other T2 PERSON golds first, then frequent multi-word person strings of the whole stream; one entry per person (cluster by last name); not an alias / container of this gold or of a hard candidate; the mention must follow a title or precede said / told / was ... in >= 2 articles"""
    out=[];used=set();vs=q['variants']
    bad={w.lower() for v in vs for w in re.findall(r"[A-Za-z]+",v)[-1:]}|{B.norm(h['surface']).split()[-1] for h in hard if B.norm(h['surface'])}
    for g in golds:
        if g['id']!=q['id'] and g['pool_type']=='PERSON' and g['event']!=q['event'] and not any(B.related(g['gold'],v) for v in vs):
            k=g['gold'].split()[-1].lower()
            if k not in used:out.append(dict(surface=g['gold'],from_event=g['event'],source='gold of another T2 item (person)'));used.add(k)
    for k,c in sorted(pool.items(),key=lambda kv:(-len(kv[1]['arts']),kv[0])):
        if len(out)>=n:break
        if k in used or k in bad or len(c['ctx'])<2:continue
        sf=c['surf'].most_common(1)[0][0]
        if any(B.related(sf,v) for v in vs) or any(B.related(sf,g['gold']) for g in golds if g['gold_class']!='PERSON') or any(B.related(sf,x) for x in nonp):continue
        out.append(dict(surface=sf,from_event=None,source='frequent person string in the whole stream (title / said-was context in >= 2 articles, one entry per last name)',c_prime=len(c['arts'])));used.add(k)
    return out
def event_count(st,event,sf):
    rx=re.compile(r'(?<![\w])'+re.escape(sf)+r'(?![\w])',re.I);E=st.get(event)
    return 0 if E is None else sum(1 for ai,sents,cum in B.lead_stream(E) if any(rx.search(t) for t in sents))
def other_easy(st,q,golds,hard,n=EASY_N):
    """TITLE / TEAM golds: the extractor has no type for them, so the auto list is NOT used. easy = the other T2 gold of the same class, then the hand-written pool e57_t1t2_easy_pools.json of the class (kept only if it occurs in the stream, is not an alias / container of the gold and not a same-event hard candidate); easy_extra of the nominations is added by verify"""
    pools=json.load(open(HERE/'e57_t1t2_easy_pools.json'));out=[];hn={B.norm(h['surface']) for h in hard};vs=q['variants']
    for g in golds:
        if g['id']!=q['id'] and g['gold_class']==q['gold_class'] and not any(B.related(g['gold'],v) for v in vs):out.append(dict(surface=g['gold'],from_event=g['event'],source='gold of another T2 item (same class)'))
    for sf in pools.get(q['gold_class'],[]):
        if len(out)>=n:break
        if B.norm(sf) in hn or any(B.related(sf,v) for v in vs) or any(B.related(sf,e['surface']) for e in out) or stream_count(st,sf)==0 or (q['gold_class']=='TITLE' and event_count(st,q['event'],sf)>0):continue      # a title of the item's own event is a hard candidate, not an easy one
        out.append(dict(surface=sf,from_event=None,source='hand-written pool '+q['gold_class']+' (e57_t1t2_easy_pools.json), present in the stream'))
    return out
def stream_count(st,surface):
    return count_variant(st,[surface])[0]
def check(draft_path,bankdir,out,log=print,nomfile=None):
    out=pathlib.Path(out);fd=out/'e57_bank';fd.mkdir(parents=True,exist_ok=True)
    draft=json.load(open(draft_path));items=draft['items'];bank,bchk=bank_check(bankdir);log('bank_v1 hash check:',bchk['ok'],bchk['sha256'][:8])
    if not bchk['ok']:print('bank_v1 hash mismatch (exit 6)');sys.exit(6)
    st,S,ev,art=load_stream(out);narts=len({a for a in art});log('stream:',narts,'articles,',len(S),'sentences')
    ix=StreamIndex(S);BQ={q['id']:q for q in bank['questions']}
    # ---- (1) overlap ----
    res=[];t2q={r['id']:r for r in t2_questions(draft)};byq=collections.defaultdict(list)
    gk=lambda it:it['source'] if it['kind']=='T1' else it['id'].replace('-cloze','')
    for it in items:byq[gk(it)].append(it)
    for it in items:
        q=BQ.get(it['source']);gold_vars=[it['gold']]+list(it.get('variants',[]))
        ent=ent_words(it['event'],*gold_vars,q['S'] if q else '');fr=it['frame'].replace('___',' ')
        o=ix.overlap(fr,ent);r=dict(id=it['id'],kind=it['kind'],form=it['form'],event=it['event'],stream={str(k):len(v) for k,v in o.items()},stream_examples={str(k):[dict(gram=g,event=ev[i],sentence=S[i][:160]) for g,i in v[:3]] for k,v in o.items()})
        if q:r['orig']={str(k):v for k,v in pair_overlap(fr,q['frame'].replace('{S}',q['S']),ent).items()}
        r['siblings']={}
        for sb in byq[gk(it)]:
            if sb['id']!=it['id']:
                p=pair_overlap(fr,sb['frame'].replace('___',' '),ent)
                if p[1]:r['siblings'][sb['id']]={str(k):v for k,v in p.items()}
        res.append(r)
    summ={str(k):dict(stream_items_with_overlap=[r['id'] for r in res if r['stream'][str(k)]],orig_items_with_overlap=[r['id'] for r in res if r.get('orig',{}).get(str(k))],sibling_items_with_overlap=[r['id'] for r in res if any(v[str(k)] for v in r['siblings'].values())]) for k in KS}
    (out/'e57_t1t2_overlap.json').write_text(json.dumps(dict(definition=__doc__.split('Entity-masked n-gram')[1].split('\n')[0],ngram=NGRAM,ks=list(KS),n_items=len(res),summary={k:{a:len(b) for a,b in v.items()} for k,v in summ.items()},failures=summ,items=res),indent=1,ensure_ascii=False))
    log('overlap items with >=1 hit (stream/orig/sibling) k=1,2,3:',{k:[len(v['stream_items_with_overlap']),len(v['orig_items_with_overlap']),len(v['sibling_items_with_overlap'])] for k,v in summ.items()})
    # ---- (2) T2 golds ----
    Q=[make_q(r) for r in t2q.values()];gold=[]
    for q in Q:
        na,per=count_variant(st,q['variants']);ev_arts=sum(1 for ai,sents,cum in B.lead_stream(st[q['event']]) if any(re.search(r'(?<![\w])'+re.escape(v)+r'(?![\w])',t,re.I) for t in sents for v in q['variants']))
        fixed=q['canonical'];most_freq=per.most_common(1)[0][0] if per else None
        gold.append(dict(id=q['id'],event=q['event'],gold=q['canonical'],gold_class=q['gold_class'],pool_type=q['type'],variants=q['variants'],sentence_count_per_variant=dict(per),fixed_surface=fixed,most_frequent_variant=most_freq,c_prime_stream_articles=na,c_prime_event_articles=ev_arts,event_articles=len(st[q['event']]['rids']),alias_scan_top=alias_scan(st,q['canonical']),in_stream=na>0))
    (out/'e57_t1t2_t2_golds.json').write_text(json.dumps(gold,indent=1,ensure_ascii=False))
    vf=fd/'t2_validated.jsonl';vf.write_text('\n'.join(json.dumps(q,ensure_ascii=False) for q in Q)+'\n')
    # ---- (3) T2 candidates (e57_bank.distract with the pool type OTHER enabled for TITLE / TEAM golds) ----
    old=B.TYPES;B.TYPES=tuple(old)+('OTHER',)
    try:D=B.distract(out,vf,log=lambda *x:None)
    finally:B.TYPES=old
    nstream=lambda s:stream_count(st,s);pool=person_pool(st)
    for d in D:
        q=next(x for x in Q if x['id']==d['id']);d['gold_class']=q['gold_class'];d['easy_auto_unfiltered']=d['easy']
        d['easy']=person_easy(pool,q,d['hard'],gold) if q['gold_class']=='PERSON' else other_easy(st,q,gold,d['hard'])
        d.pop('easy_auto_unfiltered')
        if nomfile:d['easy_extra_checks']=add_easy_extras(st,d,q,easy_extras_of(nomfile).get(d['id'],[]))
        for lst in('easy','hard'):
            for c in d[lst]:c['c_prime_stream_articles']=nstream(c['surface'])
        d['n_easy']=len(d['easy'])
    (out/'e57_t1t2_t2_distractors.json').write_text(json.dumps(D,indent=1,ensure_ascii=False))
    # ---- T1 candidates from bank_v1 ----
    t1=[dict(id=q['id'],event=q['event'],type=q['type'],fixed_surface=q['fixed_surface'],n_easy=len(q['easy']),n_hard_all=len({B.norm(h['surface']) for h in q['hard_all']}),n_nominated=len(q['hard_nominated'])) for q in bank['questions'] if q['id'] in {it['source'] for it in items if it['kind']=='T1'}]
    miss=sorted({it['source'] for it in items if it['kind']=='T1'}-set(BQ))
    p3=summ['3'];pass3=not(p3['stream_items_with_overlap'] or p3['orig_items_with_overlap'] or p3['sibling_items_with_overlap'])
    key=dict(pass_k3_zero_overlap=pass3,created=kst(),draft_sha256=sha_file(draft_path),draft_version=draft.get('version'),bank_v1=bchk,stream_articles=narts,stream_sentences=len(S),n_items=len(items),counts=dict(collections.Counter(f"{i['kind']}-{i['form']}" for i in items)),overlap_items_with_hit={k:{a:len(b) for a,b in v.items()} for k,v in summ.items()},overlap_hit_ids={k:v['stream_items_with_overlap'] for k,v in summ.items()},t1_sources=len(t1),t1_sources_missing_in_bank=miss,t2=[dict(id=g['id'],gold=g['gold'],cls=g['gold_class'],fixed=g['fixed_surface'],c_prime_stream=g['c_prime_stream_articles'],c_prime_event=g['c_prime_event_articles'],n_easy=next(d['n_easy'] for d in D if d['id']==g['id']),n_hard=len(next(d['hard'] for d in D if d['id']==g['id']))) for g in gold])
    (out/'e57_t1t2_key_numbers.json').write_text(json.dumps(key,indent=1,ensure_ascii=False));log(json.dumps(key['t2'],ensure_ascii=False)[:1500]);return key
def easy_extras_of(nomfile):
    """-> {item id: [easy_extra strings]} from the nominations file of the news thread (keys of easy_extra_proposed may name several items: every rev-... id in the key gets the list)"""
    N=json.load(open(nomfile));EX={}
    for k,v in N.get('easy_extra_proposed',{}).items():
        for i in re.findall(r'rev-[\w-]+',k):EX[i]=list(v)
    nom=N['nominations']
    if isinstance(nom,dict):
        for k,v in nom.items():
            if v.get('easy_extra'):EX[k]=list(v['easy_extra'])
    return EX
def add_easy_extras(st,d,q,extras,cap=EASY_N):
    """verified easy extras are appended up to cap: must occur in the stream, not be an alias / container of the gold, not a same-event hard candidate, not already in the easy list. -> list of verdicts"""
    res=[];hn={B.norm(h['surface']) for h in d['hard']}
    for s_ in extras:
        has={B.norm(e['surface']) for e in d['easy']};na=stream_count(st,s_)
        if any(B.related(s_,v) for v in q['variants']):res.append(dict(extra=s_,accepted=False,reason='alias / container of the gold'))
        elif na==0:res.append(dict(extra=s_,accepted=False,reason='does not occur in the stream'))
        elif B.norm(s_) in hn:res.append(dict(extra=s_,accepted=False,reason='is a same-event hard candidate'))
        elif B.norm(s_) in has or any(B.related(s_,e['surface']) for e in d['easy']):res.append(dict(extra=s_,accepted=False,reason='already in the easy list'))
        elif len(d['easy'])>=cap:res.append(dict(extra=s_,accepted=False,reason='easy list already has %d'%cap))
        else:d['easy'].append(dict(surface=s_,from_event=None,c_prime=None,source='nominated easy extra (news thread), verified in the stream',c_prime_stream_articles=na));res.append(dict(extra=s_,accepted=True,reason='occurs in the stream',c_prime_stream_articles=na))
    d['n_easy']=len(d['easy']);return res
# ---------- verify nominations ----------
def verify(out,nomfile,log=print):
    out=pathlib.Path(out);fd=out/'e57_bank';st,idx=B.load(out);Q={q['id']:q for q in map(json.loads,[l for l in open(fd/'t2_validated.jsonl') if l.strip()])};D={d['id']:d for d in json.load(open(out/'e57_t1t2_t2_distractors.json'))}
    GOLDS=json.load(open(out/'e57_t1t2_t2_golds.json'));_pool=[]
    def POOL():
        if not _pool:_pool.append(person_pool(st))
        return _pool[0]
    EXALL=easy_extras_of(nomfile);N=json.load(open(nomfile))['nominations']
    if isinstance(N,dict):N=[dict(id=k,nominated_from_list=v.get('hard',[]),proposed_extra_needs_code_check=v.get('hard_proposed_need_stream_check',[]),easy_extra=EXALL.get(k,[])) for k,v in N.items()]
    rep=[];lines=[];prob=[]
    for n in N:
        qid=n['id']
        if qid not in Q:prob.append(f'{qid}: not a T2 gold');continue
        q=Q[qid];d=D[qid];E=st[q['event']];have={B.norm(h['surface']) for h in d['hard']};picks=[x['surface'] if isinstance(x,dict) else x for x in n.get('nominated_from_list',[])]
        moved=[x for x in picks if B.norm(x) not in have];picks=[x for x in picks if x not in moved]
        res=[NOM.check_extra(q,s,E,{}) for s in list(n.get('proposed_extra_needs_code_check',[]))+moved]
        for r in res:
            if r['accepted'] and B.norm(r['extra']) not in have:d['hard'].append(dict(surface=r['extra'],key=B.norm(r['extra']),c_prime=r['c_prime'],c_prime_win=r['c_prime_win'],source='nominated extra (news thread), verified in the stream',support=r['support'],c_prime_stream_articles=stream_count(st,r['extra'])));have.add(B.norm(r['extra']))
        hn_=[x['surface'] for x in d['hard']];d['easy']=person_easy(POOL(),q,d['hard'],GOLDS) if q['gold_class']=='PERSON' else other_easy(st,q,GOLDS,d['hard'])      # the nominated hard candidates leave the easy list, the free places are back-filled
        easy_res=add_easy_extras(st,d,q,n.get('easy_extra',[]))
        d['n_easy']=len(d['easy']);nominated=picks+[r['extra'] for r in res if r['accepted']]
        if len(nominated)>MAXN:prob.append(f'{qid}: {len(nominated)} nominated (> {MAXN})')
        lines.append(dict(id=qid,nominated=nominated));rep.append(dict(id=qid,list_picks=picks,extras=[{k:v for k,v in r.items() if k!='support'} for r in res],easy_extras=easy_res,n_nominated=len(nominated),n_easy=d['n_easy']))
    (out/'e57_t1t2_t2_distractors_ext.json').write_text(json.dumps(list(D.values()),indent=1,ensure_ascii=False));(fd/'t2_nominations_freeze.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in lines)+'\n')
    (out/'e57_t1t2_extras_check.json').write_text(json.dumps(dict(problems=prob,questions=rep,missing_nominations=sorted(set(Q)-{n['id'] for n in N})),indent=1,ensure_ascii=False));log('verify:',prob or 'no problems',{r['id']:(r['n_nominated'],r['n_easy']) for r in rep});return rep,prob
# ---------- freeze ----------
def freeze(out,version,draft_path,bankdir,log=print):
    out=pathlib.Path(out);bf=out/f't1t2_{version}.json'
    if bf.exists():print(f'{bf} exists: a frozen file is never overwritten (exit 8)');sys.exit(8)
    draft=json.load(open(draft_path));bank,bchk=bank_check(bankdir)
    if not bchk['ok']:print('bank_v1 hash mismatch (exit 6)');sys.exit(6)
    ext=out/'e57_t1t2_t2_distractors_ext.json'
    if not ext.exists():print('run verify first (exit 7)');sys.exit(7)
    D={d['id']:d for d in json.load(open(ext))};N={x['id']:x for x in map(json.loads,[l for l in open(out/'e57_bank'/'t2_nominations_freeze.jsonl') if l.strip()])}
    G={g['id']:g for g in json.load(open(out/'e57_t1t2_t2_golds.json'))};ov=json.load(open(out/'e57_t1t2_overlap.json'))
    BQ={q['id']:q for q in bank['questions']};t1={};t2={};prob=[]
    for it in draft['items']:
        if it['kind']=='T1':
            q=BQ.get(it['source'])
            if q is None:prob.append(f"{it['id']}: source {it['source']} not in bank_v1");continue
            t1[it['source']]={k:q[k] for k in('id','event','S','type','canonical','variants','fixed_surface','easy','hard_all','hard_nominated','stratum')}
    for qid,d in D.items():
        nm=[x for x in d['hard'] if B.norm(x['surface']) in {B.norm(s) for s in N.get(qid,{}).get('nominated',[])}]
        if len(nm)>MAXN or not nm:prob.append(f'{qid}: {len(nm)} nominated hard candidates (0 or > {MAXN})')
        t2[qid]=dict(id=qid,event=d['event'],gold=G[qid]['gold'],gold_class=d['gold_class'],type=d['type'],variants=G[qid]['variants'],fixed_surface=G[qid]['fixed_surface'],c_prime_stream_articles=G[qid]['c_prime_stream_articles'],easy=d['easy'],hard_all=d['hard'],hard_nominated=nm)
    if prob:print('freeze refused:',prob);sys.exit(7)
    body=dict(version=version,declaration='no model scores (log-probabilities) of any T1 / T2 item or candidate were seen before this freeze; T1 candidates are copied unchanged from bank_v1; the hash covers the 150 frames, the golds and variants, the T1 and T2 candidate lists (easy, hard pool, nominated) and the overlap result',frozen_at=kst(),draft_sha256=sha_file(draft_path),draft_version=draft.get('version'),bank_v1_sha256=bchk['sha256'],items=draft['items'],t1_candidates=t1,t2_candidates=t2,overlap_summary=ov['summary'],overlap_failures={k:v['stream_items_with_overlap'] for k,v in ov['failures'].items()},counts=dict(collections.Counter(f"{i['kind']}-{i['form']}" for i in draft['items'])),n_t1_questions=len(t1),n_t2_questions=len(t2),easy_sizes={k:len(v['easy']) for k,v in t2.items()},nominated_sizes={k:len(v['hard_nominated']) for k,v in t2.items()})
    h=jhash(body);body['sha256']=h;bf.write_text(json.dumps(body,indent=1,ensure_ascii=False));(out/f't1t2_{version}.sha256').write_text(h+'\n');log(f't1t2 {version}: {len(draft["items"])} items, sha256 {h[:16]}');return body
def _selftest():
    import tempfile
    # masking: mid-sentence capitals, numbers and entity words are ENT, the sentence-initial word is not
    ent=ent_words('Spider-Man Brand New Day','July 31')
    m=mask_seq(sent_tokens('Which day has the studio set for the premiere of Spider-Man: Brand New Day on July 31? Fans wait.'),ent)
    assert m[:1]==['which'] and m[2]=='has' and m.count('ENT')>=5 and m[-2:]==['fans','wait'],m
    assert 'ENT' not in mask_seq(sent_tokens('Then the studio set the date.'),set())[:1]
    # overlap: a stream sentence with the same non-entity words matches only when >= k non-ENT tokens
    S=['The studio set the premiere for Tuesday in London.','Nothing alike here at all today.']
    ix=StreamIndex(S);o=ix.overlap('Why did the studio set the premiere for Friday in Paris?',set())
    assert o[3] and o[1]==o[1] and any('studio set the premiere' in g for g,i in o[1]),o
    o2=ix.overlap('Completely different words appear in this frame text?',set());assert not o2[1]
    # all-ENT grams are never overlap: a frame of ENT tokens only
    assert not ix.overlap('A Foo Bar Baz Qux Quux Corge Grault',set())[1]
    p=pair_overlap('the cat sat on the mat today','the cat sat on the mat tonight',set());assert p[3],p
    # alias / hash plumbing
    assert jhash({'a':1})==jhash({'a':1}) and T2_CLASS['Zendaya']=='PERSON'
    q=make_q(dict(id='x',event='Ranji Trophy J&K',gold='Jammu and Kashmir',forms=['x#q']));assert q['type']=='OTHER' and 'J&K' in q['variants']
    d=json.load(open(DEFAULT_DRAFT)) if pathlib.Path(DEFAULT_DRAFT).exists() else None
    if d:
        t2=t2_questions(d);assert len(t2)==12 and {r['gold'] for r in t2}==set(T2_CLASS),(len(t2),sorted(r['gold'] for r in t2))
    # decision C helpers on a synthetic stream
    st_={'Ev':dict(rids=['r1','r2','r3'],hosts=['h']*3,dates=[1,2,3],cum=[[0,10,20]]*3,kept=[2,2,2],sents=[['Mayor Jane Roe said it. New York City said nothing today.','Dr Jane Roe told us. Mayor Tom Hill said no.']]*3)}
    pool=person_pool(st_);assert 'roe' in pool and 'york' not in pool and len(pool['roe']['ctx'])>=2,sorted(pool)
    golds=[dict(id='a',gold='Zed Gold',gold_class='PERSON',pool_type='PERSON',event='Ev'),dict(id='b',gold='Amy Other',gold_class='PERSON',pool_type='PERSON',event='Ev2'),dict(id='c',gold='Big Title',gold_class='TITLE',pool_type='OTHER',event='Ev3')]
    qa=dict(id='a',event='Ev',variants=['Zed Gold','Gold'],gold_class='PERSON');e=person_easy(pool,qa,[dict(surface='Tom Hill')],golds);sf=[x['surface'] for x in e];assert sf[0]=='Amy Other' and 'Jane Roe' in sf and 'Tom Hill' not in sf and 'Big Title' not in sf,sf
    assert other_easy({},dict(id='c',gold_class='TEAM',variants=['Big Title']),golds+[dict(id='d',gold='Some Team',gold_class='TEAM',pool_type='OTHER',event='Ev4')],[])[0]['surface']=='Some Team'
    d_=dict(hard=[dict(surface='Hardone Name')],easy=[]);r_=add_easy_extras(st_,d_,dict(variants=['Zed Gold']),['Jane Roe','Zed Gold','Hardone Name','Nobody Atall','Jane Roe']);assert [x['accepted'] for x in r_]==[True,False,False,False,False] and d_['n_easy']==1,r_
    import tempfile as _t;f_=pathlib.Path(_t.mkdtemp())/'n.json';f_.write_text(json.dumps({'nominations':{'rev-a':{'hard':['X']}},'easy_extra_proposed':{'titles (shared by rev-a and rev-b; need check)':['T1','T2'],'teams (rev-c)':['K']}}));ex_=easy_extras_of(f_);assert ex_=={'rev-a':['T1','T2'],'rev-b':['T1','T2'],'rev-c':['K']},ex_
    print('e57_t1t2 selftest ok')

def _selftest_e2e():
    """check -> verify -> freeze on a tiny synthetic store (monkeypatched loader / bank), so that every step of the job runs once before it is run on the real stream"""
    import tempfile
    td=pathlib.Path(tempfile.mkdtemp());sents=['Mayor Jane Roe said the Zed Gold film opened in Paris.','Dr Tom Hill told us Jane Roe will visit Paris on July 31.','Mayor Amy Other said Tom Hill was away.']
    st_={'Ev':dict(rids=['r1','r2','r3','r4'],hosts=['h']*4,dates=[1,2,3,4],cum=[[0,10,20,30]]*4,kept=[3,3,3,3],sents=[sents]*4),'Ev2':dict(rids=['s1','s2'],hosts=['h']*2,dates=[5,6],cum=[[0,10,20,30]]*2,kept=[3,3],sents=[['Mayor Amy Other said it today.','Dr Kim Park told us so.','Nothing here.']]*2)}
    me=sys.modules[__name__];old=(B.load,me.bank_check,dict(T2_CLASS));B.load=lambda out:(st_,{})
    bq=dict(id='q1',event='Ev',S='Zed Gold',type='PERSON',canonical='Jane Roe',variants=['Jane Roe'],frame='Who opened {S}?',fixed_surface='Jane Roe',easy=[dict(surface='Kim Park')],hard_all=[dict(surface='Tom Hill')],hard_nominated=[dict(surface='Tom Hill')],stratum='main')
    me.bank_check=lambda d:(dict(questions=[bq]),dict(ok=True,sha256='9374e1f5'+'0'*56,recomputed='',sha_file='',file_bytes_sha256=''))
    T2_CLASS.update({'Jane Roe':'PERSON','Amy Other':'PERSON'})
    try:
        items=[dict(kind='T1',id='q1#f1',form='question',source='q1',frame='Which person opened a big movie in a city?',gold='Jane Roe',variants=['Jane Roe'],type='PERSON',event='Ev'),dict(kind='T2',id='rev-a',form='question',source='q1',frame='Who is the mayor that said the film opened?',gold='Jane Roe',type='REVERSE',event='Ev'),dict(kind='T2',id='rev-a-cloze',form='cloze',source='q1',frame='The mayor who said so is ___',gold='Jane Roe',type='REVERSE',event='Ev'),dict(kind='T2',id='rev-b',form='question',source='q2',frame='Which mayor spoke today?',gold='Amy Other',type='REVERSE',event='Ev2')]
        dp=td/'draft.json';dp.write_text(json.dumps(dict(version='t',items=items)));nf=td/'nom.json'
        nf.write_text(json.dumps({'nominations':{'rev-a':{'hard':['Tom Hill'],'easy_extra':[]},'rev-b':{'hard':[],'hard_proposed_need_stream_check':['Kim Park']}},'easy_extra_proposed':{'persons (rev-a and rev-b)':['Kim Park','Nobody Atall']}}))
        out=td/'o';(out/'e57_bank').mkdir(parents=True)
        key=check(str(dp),str(td),out,log=lambda *x:None,nomfile=str(nf));assert 'pass_k3_zero_overlap' in key and len(key['t2'])==2,key
        rep,prob=verify(out,nf,log=lambda *x:None);assert any(x['id']=='rev-a' for x in rep),rep
        b=freeze(out,'v1',str(dp),str(td),log=lambda *x:None);assert b['n_t2_questions']==2 and (out/'t1t2_v1.sha256').exists()
        try:freeze(out,'v1',str(dp),str(td),log=lambda *x:None);assert False
        except SystemExit as e:assert e.code==8
    finally:
        B.load,me.bank_check=old[0],old[1];T2_CLASS.clear();T2_CLASS.update(old[2])
    print('e57_t1t2 e2e selftest ok')
DEFAULT_DRAFT='data/news-lite/t1t2_draft_v1.json'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',nargs='?',choices=['check','verify','freeze']);ap.add_argument('--out',default='experiments/exp07_write_ladder/e57_news/e57_t1t2');ap.add_argument('--draft',default=DEFAULT_DRAFT);ap.add_argument('--bank-dir',default='experiments/exp07_write_ladder/e57_news/e57_bank');ap.add_argument('--nominations');ap.add_argument('--version',default='v1');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();_selftest_e2e();return
    if a.cmd=='check':check(a.draft,a.bank_dir,a.out,nomfile=a.nominations)
    elif a.cmd=='verify':verify(a.out,a.nominations)
    elif a.cmd=='freeze':freeze(a.out,a.version,a.draft,a.bank_dir)
if __name__=='__main__':main()
