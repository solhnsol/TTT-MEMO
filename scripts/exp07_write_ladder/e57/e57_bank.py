"""E57 hand-written question bank: tools around the human steps (design E57 section 22; CPU only, NO model, NO GPU, NO download; input = bundle_store.pkl / bundle_index.json of e57_bundle.py (job 229), no second analysis).
Steps (E57 22.1): A packet  -> B (a person writes the questions, NOT code)  -> C validate -> D distract -> E (a person nominates hard distractors) -> F freeze. G (W0 check) is a separate GPU job.
  packet   --out OUT [--n 12]                     per event 12 independent articles by a fixed seed (57251 + md5 of the event name), their LEAD sentences numbered a01.. / s001.., NO statistics, packet_hash = sha256 of the content; events that fail the event-level ETH-v1 (index eth_v1_event_level non-empty) or have no analysis get no packet and are listed. -> OUT/e57_bank/packets/<event>.packet.json + packets_index.json. Private, internal research only (Hans 2026-10-24); copying to /mnt/project-files/data/news-lite/ is done by hand.
  validate --out OUT --bank bank.jsonl            question schema (E57 22.2) and the pre-registered thresholds of 22.3, fixed before any result: c >= 10 independent lead articles hold the answer (union of the normalized surface forms: DATE / NUMBER by the miner's normalized key (month-day, value) plus the given variants as word-boundary strings; PLACE / PERSON by the given variants), c_SA >= .5 (answer sentence or +-64 tokens holds a core word of S; sentence-only value also reported), competitor: the most frequent SAME-TYPE string inside the S window (+-64 tokens around S core words) other than the answer (a string containing / contained in an answer variant is the same value) is held by at most half of the articles that hold the answer inside that window; evidence ids exist, come from >= 2 different articles and each evidence sentence holds the answer; answer shares no word with S; no relative-time answer (yesterday, last week, weekday only); the frame (S removed) has no 5-gram in common with the lead sentences of the event; packet_hash matches the packet; an answer that is also the answer of another event's question of the same type excludes both. Rejections are listed with reasons per event (nothing is dropped silently).
  distract --out OUT --validated validated.jsonl  easy list = answers (same type, same normalized form) of the surviving questions of the OTHER events (a type with fewer than 8 easy candidates is flagged 'no rank analysis'); hard list = up to 8 most frequent same-type strings of the event's lead (answer values excluded), plus the answers of other surviving questions of the same event and type; every entry carries c' (lead articles holding it) and c'_win (inside the S window).
  freeze   --out OUT --validated validated.jsonl --distractors distractors.json [--nominations nominations.jsonl] --version v1    nominations = [{"id","nominated":[<= 4 strings from the hard list],"evidence":[ids],"discard":false}]; the bank file bank_<version>.json holds questions, easy / hard lists, the fixed surface form (the most frequent variant), author, written_at, frozen_at (KST), packet hashes and its own sha256 (bank_<version>.sha256); an existing version is never overwritten (a change = a new version).
Type rules (E57 21 / 22.9 item 5; NO dictionary is used for answer judgment): PLACE = capitalised phrase after in / at / near / from / ..., or ending in a place word (City, County, Jail, Airport, Park, Palace, Fort, Hotel, Stadium, Island, Valley, Beach, River, Lake) or a place suffix (-pur -abad -nagar -garh -ganj -ville -burg -burgh -shire -stan -polis -grad); PERSON = 2-3 capitalised words after a title or before said / told / was ..., or not PLACE and without an organisation word; DATE / NUMBER = e57_common.facts_in_sentence; the rest is OTHER."""
import argparse,json,sys,re,hashlib,random,pickle,pathlib,collections,datetime
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
SEED=57251;NPACK=12;C_MIN=10;CSA_MIN=0.5;COMP_MAX=0.5;WIN=64;EASY_MIN=8;NHARD=8;TYPES=('PLACE','DATE','PERSON','NUMBER')
PLACE_CTX=re.compile(r"(?:\b(?:in|at|near|from|across|outside|inside|around|to|into)\s)$",re.I)
PERSON_BEFORE=re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|President|Minister|Mayor|Governor|Senator|Coach|Judge|Chairman|Actor|Actress|Director|Prime Minister|Chief|Captain)\.?\s)$")
PERSON_AFTER=re.compile(r"^\s*(?:,\s*[^,]{0,60},)?\s*(?:said|says|told|added|stated|announced|confirmed|has|was|is|will|had)\b")
PLACE_WORDS={'city','county','jail','airport','park','palace','fort','hotel','stadium','island','valley','beach','river','lake'}
PLACE_SUF=('pur','abad','nagar','garh','ganj','ville','burg','burgh','shire','stan','polis','grad')
ORG_WORDS={'bank','ministry','court','university','party','council','association','company','corp','inc','ltd','department','committee','commission','group','police','government','agency','league','club','news','times','post'}
CAPSEQ=re.compile(r"\b[A-Z][A-Za-z'’\-]{2,}(?:\s+[A-Z][A-Za-z'’\-]{2,}){0,3}\b")
RELTIME=re.compile(r"^(?:yesterday|today|tomorrow|tonight|last\s+\w+|next\s+\w+|this\s+\w+|(?:mon|tues|wednes|thurs|fri|satur|sun)day)$",re.I)
STOP={'the','and','for','from','with','said','news','city','state','court','police','government','minister','president','bank','new','group','company','national','international','university','united'}
def norm(s):return ' '.join(re.findall(r"[a-z0-9]+",s.lower()))
def core(s):return {w for w in re.findall(r"[a-z]{3,}",s.lower()) if w not in STOP}
def words_of(t):return set(re.findall(r"[a-z]+",t.lower()))
def related(x,y):
    a,b=norm(x),norm(y);return bool(a and b and (a==b or a in b or b in a))
def type_strings(sent):
    """-> [(type, key, surface, start, end)]; types PLACE / PERSON / DATE / NUMBER / OTHER (names: no dictionary)"""
    out=[];fs=C.facts_in_sentence(sent);spans=[(f[3],f[4]) for f in fs if f[0]!='NAME']
    for ty,key,sf,a,b in fs:
        if ty=='DATE':out.append(('DATE',key,sf,a,b))
        elif ty=='NUM':out.append(('NUMBER',key,sf,a,b))
    for m in CAPSEQ.finditer(sent):
        a,b=m.start(),m.end();t=m.group(0);w=t.split()
        while w and (w[0] in C.STARTERS or w[0].lower() in C.MONTHNUM):a=a+len(w[0])+1;w=w[1:]
        if not w:continue
        t=' '.join(w)
        if a==0 and len(w)==1:continue
        if any(a<y and b>x for x,y in spans):continue
        low=[x.lower() for x in w];pre=sent[max(0,a-16):a];post=sent[b:b+80]
        place=bool(PLACE_CTX.search(sent[max(0,a-12):a])) or low[-1] in PLACE_WORDS or low[-1].endswith(PLACE_SUF)
        person=bool(PERSON_BEFORE.search(pre) or PERSON_AFTER.search(post)) or (2<=len(w)<=3 and not place and not set(low)&ORG_WORDS)
        ty='PLACE' if place and not PERSON_BEFORE.search(pre) else 'PERSON' if person else 'OTHER'
        out.append((ty,norm(t),t,a,b))
    return out
def ans_matcher(q):
    """-> function(sentence) -> bool: the sentence holds the answer (DATE / NUMBER by normalized key of the canonical form, plus word-boundary variants)"""
    ty=q['type'];keys=set()
    if ty in('DATE','NUMBER'):
        for s in [q['canonical']]+list(q.get('variants',[])):
            for f in C.facts_in_sentence(' '+s+' '):
                if f[0]==('DATE' if ty=='DATE' else 'NUM'):keys.add(f[1])
    vs=[v for v in [q['canonical']]+list(q.get('variants',[])) if v]
    rx=re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in vs),re.I) if vs else None
    def holds(sent):
        if rx and rx.search(sent):return True
        if keys:return any(f[0]==('DATE' if ty=='DATE' else 'NUM') and f[1] in keys for f in C.facts_in_sentence(sent))
        return False
    return holds,keys
def load(out):
    fd=pathlib.Path(out)/'e57_bundle'
    st=pickle.load(open(fd/'bundle_store.pkl','rb'))['events'];idx=json.load(open(fd/'bundle_index.json'));return st,idx
def slug(nm):return re.sub(r'[^A-Za-z0-9]+','_',nm).strip('_')[:60]
def jhash(o):return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
# ---------- A packet ----------
EXCLUDE_DEFAULT=('Kärcher','Karcher','Umar Khalid')       # agreed with the news thread 2026-10-24: mixed appliance-sale cluster; no targets (c_SA 0)
def packet(out,n=NPACK,log=print,exclude=EXCLUDE_DEFAULT):
    st,idx=load(out);pd=pathlib.Path(out)/'e57_bank'/'packets';pd.mkdir(parents=True,exist_ok=True);info=[];excl=[]
    for nm,E in st.items():
        eth=idx.get(nm,{}).get('eth_v1_event_level')
        if any(x.lower() in nm.lower() for x in exclude):excl.append(dict(event=nm,reason='excluded by default (agreed list)'));continue
        if eth:excl.append(dict(event=nm,reason='event-level ETH-v1: '+str(eth)[:120]));continue
        rng=random.Random(SEED+int(hashlib.md5(nm.encode()).hexdigest()[:8],16));order=list(range(len(E['rids'])));rng.shuffle(order);sel=sorted(order[:n],key=lambda ai:(E['dates'][ai],E['rids'][ai]))
        arts=[];k=0;m={}
        for j,ai in enumerate(sel,1):
            ss=[]
            for si in range(max(1,E['kept'][ai])):
                if si>=len(E['sents'][ai]):break
                k+=1;sid=f's{k:03d}';ss.append(dict(s=sid,text=E['sents'][ai][si]));m[sid]=[E['rids'][ai],si]
            arts.append(dict(a=f'a{j:02d}',rid=E['rids'][ai],host=E['hosts'][ai],date=E['dates'][ai],sentences=ss))
        body=dict(event=nm,seed=SEED,articles=arts,sentence_map=m);h=jhash(body)
        (pd/(slug(nm)+'.packet.json')).write_text(json.dumps(dict(note='private, internal research only; no publication or redistribution',packet_hash=h,**body),indent=1,ensure_ascii=False))
        info.append(dict(event=nm,packet_hash=h,articles=len(arts),independent_articles=len(E['rids']),short=len(arts)<n))
    (pd.parent/'packets_index.json').write_text(json.dumps(dict(packets=info,excluded=excl),indent=1));log(f'packets: {len(info)} events, excluded {len(excl)}');return info,excl
# ---------- C validate ----------
def lead_stream(E):
    for ai in range(len(E['rids'])):
        yield ai,E['sents'][ai][:max(1,E['kept'][ai])],E['cum'][ai]
def near_idx(cum,i0,n):
    lo=cum[i0]-WIN;hi=cum[i0+1]+WIN;return [j for j in range(n) if cum[j+1]>lo and cum[j]<hi]
def question_stats(q,E):
    holds,keys=ans_matcher(q);cs=core(q['S']);selfw=cs|core(q.get('event',''));arts=[];near=[];sent_only=[];comp=collections.defaultdict(set);cwin=set();vs=[q['canonical']]+list(q.get('variants',[]))
    for ai,sents,cum in lead_stream(E):
        n=len(sents);ia=[i for i,t in enumerate(sents) if holds(t)]
        if ia:
            arts.append(ai);ww=[words_of(t) for t in sents]
            if any(cs&ww[j] for i in ia for j in near_idx(cum,i,n)):near.append(ai)
            if any(cs&ww[i] for i in ia):sent_only.append(ai)
        ww=[words_of(t) for t in sents];si=[i for i in range(n) if cs&ww[i]]
        if not si:continue
        win=sorted({j for i in si for j in near_idx(cum,i,n)});hold_win=any(holds(sents[j]) for j in win)
        if hold_win:cwin.add(ai)
        for j in win:
            for ty,key,sf,a,b in type_strings(sents[j]):
                if ty!=q['type']:continue
                if any(related(sf,v) for v in vs) or (q['type'] in('DATE','NUMBER') and key in keys):continue
                if q['type'] in('PLACE','PERSON') and core(sf)&selfw:continue          # a string made of S / event-name words is the topic itself, not a competitor (fix 2026-10-26)
                comp[key if q['type'] in('DATE','NUMBER') else norm(sf)].add(ai)
    top=sorted(((k,v&cwin) for k,v in comp.items()),key=lambda kv:(-len(kv[1]),kv[0]))[:3]      # numerator and denominator on the same population: window articles that hold the answer (fix 2026-10-26)
    return dict(c=len(arts),c_SA=len(near)/max(1,len(arts)),c_SA_sentence=len(sent_only)/max(1,len(arts)),c_window=len(cwin),comp_top=[(k,len(v)) for k,v in top],comp_max=len(top[0][1]) if top else 0,answer_rids=[E['rids'][ai] for ai in arts])
def check_question(q,E,P):
    why=[]
    for f in('id','event','S','type','frame','canonical','variants','evidence','direction','author','written_at','packet_hash'):
        if f not in q or q[f] in(None,'',[]):why.append(f'missing field {f}')
    if why:return why,None
    if q['type'] not in TYPES:why.append('type not in PLACE / DATE / PERSON / NUMBER')
    if '{S}' not in q['frame']:why.append('frame has no {S}')
    if P is None:return why+['no packet for the event (excluded or unknown event)'],None
    if q['packet_hash']!=P['packet_hash']:why.append('packet_hash differs from the packet')
    sm={s['s']:(a['a'],s['text']) for a in P['articles'] for s in a['sentences']};holds,keys=ans_matcher(q)
    ev=q['evidence'];bad=[e for e in ev if e.split('-')[-1] not in sm or e.split('-')[0]!=sm.get(e.split('-')[-1],('',''))[0]]
    if bad:why.append(f'evidence ids not in the packet: {bad}')
    else:
        if len({e.split('-')[0] for e in ev})<2:why.append('evidence from fewer than 2 different articles')
        if not all(holds(sm[e.split('-')[-1]][1]) for e in ev):why.append('an evidence sentence does not hold the answer')
    vs=[q['canonical']]+list(q['variants'])
    if core(q['S']) & set().union(*[words_of(v) for v in vs]) or any(norm(v) and norm(v) in norm(q['S']) for v in vs):why.append('answer overlaps S')
    if any(RELTIME.match(v.strip()) for v in vs):why.append('relative-time answer')
    if q['type'] in('DATE','NUMBER') and not keys:why.append('no normalized key could be read from the canonical form')
    fw=C.ngr(C.words(q['frame'].replace('{S}','')),5);ng=set()
    for ai,sents,cum in lead_stream(E):
        for t in sents:ng|=C.ngr(C.words(t),5)
    if fw&ng:why.append('frame 5-gram overlap with the lead sentences')
    stt=question_stats(q,E)
    if stt['c']<C_MIN:why.append(f"c = {stt['c']} < {C_MIN}")
    if stt['c_SA']<CSA_MIN:why.append(f"c_SA = {stt['c_SA']:.2f} < {CSA_MIN}")
    if stt['comp_max']>COMP_MAX*max(1,stt['c_window']):why.append(f"same-type competitor {stt['comp_top'][0] if stt['comp_top'] else ''} in {stt['comp_max']} of {stt['c_window']} window articles (> {COMP_MAX})")
    return why,stt
def validate(out,bankfile,log=print):
    st,idx=load(out);pd=pathlib.Path(out)/'e57_bank'/'packets';Q=[json.loads(l) for l in open(bankfile) if l.strip()];res=[];ok=[]
    for q in Q:
        pf=pd/(slug(q.get('event',''))+'.packet.json');P=json.load(open(pf)) if pf.exists() else None;E=st.get(q.get('event'))
        why,stt=check_question(q,E,P) if E else (['unknown event'],None);res.append(dict(id=q.get('id'),event=q.get('event'),type=q.get('type'),reasons=why,stats=stt))
        if not why:ok.append(q)
    keyof=lambda q:(q['type'],norm(q['canonical']) if q['type'] in('PLACE','PERSON') else ''.join(sorted(ans_matcher(q)[1])))
    by=collections.defaultdict(list)
    for q in ok:by[keyof(q)].append(q)
    coll={q['id'] for v in by.values() if len({x['event'] for x in v})>1 for q in v}
    for r in res:
        if r['id'] in coll:r['reasons'].append('same answer is the answer of another event (same type)')
    ok=[q for q in ok if q['id'] not in coll];fd=pathlib.Path(out)/'e57_bank';fd.mkdir(exist_ok=True)
    (fd/'validated.jsonl').write_text('\n'.join(json.dumps(q,ensure_ascii=False) for q in ok)+('\n' if ok else ''));(fd/'validation.json').write_text(json.dumps(dict(thresholds=dict(c_min=C_MIN,c_sa_min=CSA_MIN,comp_max=COMP_MAX,win=WIN),results=res),indent=1,default=str))
    L=['# E57 question validation (counts and values only)','',f'{len(Q)} questions, {len(ok)} survive. Thresholds fixed before any result: c >= {C_MIN} lead articles, c_SA >= {CSA_MIN}, same-type competitor <= {COMP_MAX} of the window articles holding the answer.','','| id | event | type | c | c_SA (sentence only) | window articles | top competitor | verdict |','|---|---|---|---|---|---|---|---|']
    for r in res:
        s=r['stats'];L.append(f"| {r['id']} | {r['event']} | {r['type']} | {s['c'] if s else ''} | {('%.2f (%.2f)'%(s['c_SA'],s['c_SA_sentence'])) if s else ''} | {s['c_window'] if s else ''} | {s['comp_top'][0] if s and s['comp_top'] else ''} | {'ok' if not r['reasons'] else '; '.join(r['reasons'])} |")
    ev=collections.Counter(r['event'] for r in res if not r['reasons']);L+=['','survivors per event: '+', '.join(f'{k}: {v}' for k,v in ev.items())]
    (fd/'validation_report.md').write_text('\n'.join(L)+'\n');log('\n'.join(L)[:4000]);return ok,res
# ---------- D distract ----------
def easy_form(q):
    if q['type']=='DATE':
        k=sorted(ans_matcher(q)[1])[0];mo,d=k.split('-');import calendar;return f"{calendar.month_name[int(mo)]} {int(d)}"
    return q['canonical']
def distract(out,validated,log=print):
    st,idx=load(out);Q=[json.loads(l) for l in open(validated) if l.strip()];res=[];bytype=collections.defaultdict(list)
    for q in Q:bytype[q['type']].append(q)
    for q in Q:
        E=st[q['event']];holds,keys=ans_matcher(q);cs=core(q['S']);vs=[q['canonical']]+list(q['variants']);lead=collections.defaultdict(set);win=collections.defaultdict(set);surf={}
        for ai,sents,cum in lead_stream(E):
            n=len(sents);ww=[words_of(t) for t in sents];sw={j for i in range(n) if cs&ww[i] for j in near_idx(cum,i,n)}
            for j,t in enumerate(sents):
                for ty,key,sf,a,b in type_strings(t):
                    if ty!=q['type']:continue
                    k=key if q['type'] in('DATE','NUMBER') else norm(sf)
                    if any(related(sf,v) for v in vs) or (q['type'] in('DATE','NUMBER') and key in keys):continue
                    lead[k].add(ai);surf.setdefault(k,sf)
                    if j in sw:win[k].add(ai)
        hard=[dict(surface=surf[k],key=k,c_prime=len(v),c_prime_win=len(win.get(k,())),source='frequent in the event lead') for k,v in sorted(lead.items(),key=lambda kv:(-len(kv[1]),kv[0]))[:NHARD]]
        have={h['key'] for h in hard}
        for o in bytype[q['type']]:
            if o['event']==q['event'] and o['id']!=q['id'] and not related(o['canonical'],q['canonical']):
                k=norm(o['canonical']) if q['type'] in('PLACE','PERSON') else easy_form(o)
                hard.append(dict(surface=easy_form(o),key=k,c_prime=None,c_prime_win=None,source=f"answer of question {o['id']} of the same event"))
        easy=[]
        for o in bytype[q['type']]:
            if o['event']!=q['event'] and not related(o['canonical'],q['canonical']):
                f=easy_form(o);cnt=sum(1 for ai,sents,cum in lead_stream(E) if any(related(f,sf) for t in sents for ty,key,sf,a,b in type_strings(t) if ty==q['type']))
                easy.append(dict(surface=f,from_event=o['event'],c_prime=cnt))
        res.append(dict(id=q['id'],event=q['event'],type=q['type'],easy=easy,hard=hard,rank_analysis=len(easy)>=EASY_MIN))
    fd=pathlib.Path(out)/'e57_bank';(fd/'distractors.json').write_text(json.dumps(res,indent=1,ensure_ascii=False))
    nt=collections.Counter(q['type'] for q in Q);log('distractors:',len(res),'questions;',{t:sum(1 for r in res if r['type']==t and r['rank_analysis']) for t in nt},'with >= 8 easy candidates');return res
# ---------- F freeze ----------
def freeze(out,validated,distractors,nominations,version,log=print):
    fd=pathlib.Path(out)/'e57_bank';bf=fd/f'bank_{version}.json'
    if bf.exists():print(f'{bf} exists: a frozen bank is never overwritten, use a new version (exit 8)');sys.exit(8)
    st,idx=load(out);Q={json.loads(l)['id']:json.loads(l) for l in open(validated) if l.strip()};D={d['id']:d for d in json.load(open(distractors))}
    N={}
    if nominations and pathlib.Path(nominations).exists():N={x['id']:x for x in map(json.loads,[l for l in open(nominations) if l.strip()])}
    pd=fd/'packets';items=[];prob=[]
    for qid,q in Q.items():
        d=D[qid];nm=N.get(qid);hard=d['hard']
        if nm:
            keys={norm(h['surface']) for h in hard};bad=[x for x in nm.get('nominated',[]) if norm(x) not in keys]
            if bad or len(nm.get('nominated',[]))>4:prob.append(f'{qid}: nominated strings not in the hard list {bad} or more than 4');continue
            P=json.load(open(pd/(slug(q['event'])+'.packet.json')));sids={s['s'] for a in P['articles'] for s in a['sentences']}
            if any(e.split('-')[-1] not in sids for e in nm.get('evidence',[])):prob.append(f'{qid}: nomination evidence ids not in the packet');continue
            if nm.get('discard'):continue
            nom=[h for h in hard if norm(h['surface']) in {norm(x) for x in nm['nominated']}]
        else:nom=[]
        holds,keys=ans_matcher(q);E=st[q['event']];cnt=collections.Counter()
        vs=[q['canonical']]+list(q['variants'])
        for ai,sents,cum in lead_stream(E):
            for t in sents:
                for v in vs:
                    if re.search(r'(?<![\w])'+re.escape(v)+r'(?![\w])',t,re.I):cnt[v]+=1
        fixed=cnt.most_common(1)[0][0] if cnt else q['canonical']
        items.append(dict(q,fixed_surface=fixed,easy=d['easy'],rank_analysis=d['rank_analysis'],hard_all=hard,hard_nominated=nom,nomination=nm))
    if prob:print('freeze refused:',prob);sys.exit(7)
    tz=datetime.timezone(datetime.timedelta(hours=9));body=dict(version=version,frozen_at=datetime.datetime.now(tz).isoformat(timespec='seconds'),n_questions=len(items),questions=items,authors=sorted({q['author'] for q in items}),packet_hashes={q['event']:q['packet_hash'] for q in items})
    h=jhash(body);body['sha256']=h;bf.write_text(json.dumps(body,indent=1,ensure_ascii=False));(fd/f'bank_{version}.sha256').write_text(h+'\n');log(f'bank {version}: {len(items)} questions, sha256 {h[:16]}');return body
def _selftest():
    import tempfile,e57_bundle as B,e57_namedevents3 as N3
    def after(td):
        a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=5,min_domains=5,min_c=3,workers=1,seed=5,rerun=True,n_extra=4,drop=(),per_event=8,excerpt_chars=300,keyword_events=None)
        N3.DROP=();B.run(a);info,excl=packet(td);assert info
        nm=info[0]['event'];P=json.load(open(td/'e57_bank'/'packets'/(slug(nm)+'.packet.json')));assert len(P['articles'])<=NPACK and P['packet_hash']==info[0]['packet_hash'] and 'sentence_map' in P
        st,_=load(td);E=st[nm];qs=[]
        # find a (NAME in S, NUM/DATE answer) pair in one packet sentence
        for A in P['articles']:
            for s in A['sentences']:
                fs=C.facts_in_sentence(s['text']);nm_=[f for f in fs if f[0]=='NAME'];nu=[f for f in fs if f[0]=='NUM']
                if nm_ and nu:qs.append((A['a'],s,nm_[0],nu[0]))
        assert qs,'synthetic packet has no S / A sentence'
        a1,s1,S,Af=qs[0];others=[(A['a'],s) for A in P['articles'] for s in A['sentences'] if Af[2] in s['text'] and A['a']!=a1]
        q=dict(id='t1',event=nm,S=S[2],type='NUMBER',frame='Question: how much money was reported together with {S}?\nAnswer:',canonical=Af[2],variants=[Af[2]],evidence=[f"{a1}-{s1['s']}",f"{others[0][0]}-{others[0][1]['s']}"] if others else [],direction='S->A',author='selftest',written_at='2026-10-24T12:00:00+09:00',packet_hash=P['packet_hash'],status='draft')
        bad=dict(q,id='t2',packet_hash='x',evidence=[f"{a1}-s999"]);bank=td/'bank.jsonl';bank.write_text(json.dumps(q)+'\n'+json.dumps(bad)+'\n')
        ok,res=validate(td,bank,log=lambda *x:None);byid={r['id']:r for r in res};assert any('packet_hash' in x for x in byid['t2']['reasons']) and any('evidence ids' in x for x in byid['t2']['reasons']),byid['t2']
        assert byid['t1']['stats'] and byid['t1']['stats']['c']>=1 and (td/'e57_bank'/'validation_report.md').exists(),byid['t1']
        # force the survivor path with relaxed thresholds
        global C_MIN
        old=C_MIN;C_MIN=1
        try:ok,res=validate(td,bank,log=lambda *x:None)
        finally:C_MIN=old
        okids={x['id'] for x in ok};print('selftest t1 reasons:',byid['t1']['reasons'])
        if 't1' in okids:
            d=distract(td,td/'e57_bank'/'validated.jsonl',log=lambda *x:None);assert d and d[0]['hard'] is not None
            b=freeze(td,td/'e57_bank'/'validated.jsonl',td/'e57_bank'/'distractors.json',None,'v1',log=lambda *x:None);assert b['n_questions']==1 and (td/'e57_bank'/'bank_v1.sha256').exists()
            try:freeze(td,td/'e57_bank'/'validated.jsonl',td/'e57_bank'/'distractors.json',None,'v1',log=lambda *x:None);assert False
            except SystemExit as e:assert e.code==8
        # type rules without a dictionary
        ts={(t,s) for t,k,s,a,b in type_strings('He lives in Springfield. Dr Ivan Petrov said so at Central Park and in Rajpur.')};assert ('PLACE','Springfield') in ts and ('PERSON','Ivan Petrov') in ts and ('PLACE','Rajpur') in ts and ('PLACE','Central Park') in ts,ts
        print('e57_bank selftest ok:',nm,'t1 survived' if 't1' in okids else 't1 rejected (reasons above)')
    N3._selftest(after)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',nargs='?',choices=['packet','validate','distract','freeze']);ap.add_argument('--out',default='e57_news');ap.add_argument('--n',type=int,default=NPACK);ap.add_argument('--bank');ap.add_argument('--validated');ap.add_argument('--distractors');ap.add_argument('--nominations');ap.add_argument('--version',default='v1');ap.add_argument('--exclude',default=','.join(EXCLUDE_DEFAULT),help='comma list of event-name substrings left out of the packets');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    out=pathlib.Path(a.out)
    if a.cmd=='packet':packet(out,a.n,exclude=tuple(x for x in a.exclude.split(',') if x))
    elif a.cmd=='validate':validate(out,a.bank)
    elif a.cmd=='distract':distract(out,a.validated or out/'e57_bank'/'validated.jsonl')
    elif a.cmd=='freeze':freeze(out,a.validated or out/'e57_bank'/'validated.jsonl',a.distractors or out/'e57_bank'/'distractors.json',a.nominations,a.version)
if __name__=='__main__':main()
