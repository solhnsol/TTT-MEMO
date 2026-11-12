"""E57-lite freeze (CPU only, no model, no GPU, NO download, no new clustering; job 223; E57 section 16.1 / 16.10 requests 1-2). Fixes, BEFORE any result of this job:
 1) the event list: the 12 fixed regex events of job 220 / 221 without Nipah virus and IDFC First Bank fraud (10 events) + the next --n-extra (default 10) clusters of the job-219 list (e57_biglist.json: members >= 50, core >= .5, largest component >= .5 already; here: not recurring-type / product-type suspects (keyword share >= 0.3), not overlapping with an event already taken (>= 20 % of the members shared or >= 2 of the 3 top title tokens equal), in the order of members) -> about 16 events. The ethics filter is applied per target only (rules of job 222: b / c / d in any window; address, health, victim words, private-individual proxy of S in >= 20 % of the windows; family words reject only together with the private-individual proxy).
 2) the frozen bank of the LEAD variant (first sentences up to 350 tokens, c >= 3 independent articles): per target the question for S, the candidate strings and the S' for the cross null.
   - question (fixed frame per answer type, S inserted): 'Question: In the news about this event, which <name | number or amount | date> is mentioned together with <S>?\\nAnswer:'. The frame without S is checked for 5-gram overlap with the lead sentences of the event (target counted as overlapping if any; reported, the frame is NOT changed here).
   - candidates: strings of the same type with a lead exposure c' >= 1 in the same event (c' = independent articles whose lead contains the string), excluding the answer, names containing / contained in S or in the answer, and strings that appear together with S in a lead sentence of >= 2 independent articles (a competing true pair; with one article the candidate stays and carries cooc_S = 1). Fill order: strict (|c' - c| <= 1), then middle (c' >= 2 and |c' - c| <= 2), then relaxed (any), by |c' - c| then a seeded hash, at most 16 per target; each candidate carries its tier. G = number of candidates; main = G >= 8 (otherwise supplementary only). Counts of the full tiers are stored (n_strict / n_middle / n_relaxed) and whether each tier alone has >= 8.
   - cross null: up to 4 seeded random S' per target = subjects of OTHER targets of the same event, unrelated to S (no name containment) and never in a lead sentence together with the answer A.
 3) the strata table: targets per event and per c stratum 3-4 / 5-9 / 10-19 / 20+ (all targets, main targets with G >= 8, targets with 4 S', targets per answer type, tier sizes).
Outputs <out>/e57_lite/: lite_events.json, lite_bank.json, lite_manifest.json (sha256 of both + rules), e57_lite_report.md. The bank holds no article text (article ids, sentence indices, strings). The step-by-step article lead texts for the stream are not frozen here (section 16.10 request 3)."""
import argparse,json,sys,random,pickle,pathlib,time,collections,re,hashlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_events as EV
import e57_namedevents3 as N3
import e57_namedevents2 as N2
CAPN=(10,15,20);FREEZE_SEED=57231;G_MIN=8;NCAND=16;NSP=4;STRATA=(('3-4',3,4),('5-9',5,9),('10-19',10,19),('20+',20,10**9))
NOUN={'NAME':'name','NUM':'number or amount','DATE':'date'}
FRAME='Question: In the news about this event, which {noun} is mentioned together with {S}?\nAnswer:'
def hsh(*x):return hashlib.md5(('|'.join(map(str,(FREEZE_SEED,)+x))).encode()).hexdigest()
def stratum(c):
    for nm,lo,hi in STRATA:
        if lo<=c<=hi:return nm
def related(x,y):return x==y or x in y or y in x
def event_bank(ei,name,r,D,id0):
    """bank entries of one event (lead variant); r = filtered analysis result with detail"""
    V=N.variant(r,'lead');kept=V['kept'];arts_of=collections.defaultdict(set);surf={};P=collections.defaultdict(set);lead_sents=[]
    for ai in range(len(r['arts'])):
        for si in range(kept[ai]):
            fs=r['facts'][ai][si];lead_sents.append(r['sents'][ai][si])
            for ty,key,sf in fs:arts_of[(ty,key)].add(ai);surf.setdefault((ty,key),sf)
            for n_ in(f for f in fs if f[0]=='NAME'):
                for f in fs:
                    if (f[0],f[1])!=(n_[0],n_[1]):P[(n_[1],f[0],f[1])].add(ai)
    ng=set()
    for s_ in lead_sents:ng|=C.ngr(C.words(s_),5)
    tg=V['targets'];subj_surf={}
    for t in tg:
        for ai,sidx in t['occ'].items():
            for si in sidx:
                if si<kept[ai]:
                    for f in r['facts'][ai][si]:
                        if f[0]=='NAME' and f[1]==t['S']:subj_surf.setdefault(t['S'],f[2])
    subjects=sorted({t['S'] for t in tg});bank=[]
    for k,t in enumerate(tg):
        S=t['S'];ty=t['ty'];A=t['key'];c=t['c_v'];tid=id0+k
        Ss=subj_surf.get(S,S);q=FRAME.format(noun=NOUN.get(ty,'string'),S=Ss);fw=C.ngr(C.words(FRAME.format(noun=NOUN.get(ty,'string'),S='')),5)
        cands=[]
        for (ty2,key2),aset in arts_of.items():
            if ty2!=ty or key2==A or len(P.get((S,ty,key2),()))>=2:continue
            if ty=='NAME' and (related(key2,S) or related(key2,A)):continue
            cands.append(dict(key=key2,surface=surf[(ty2,key2)],c_prime=len(aset),cooc_S=len(P.get((S,ty,key2),()))))
        cands.sort(key=lambda x:(abs(x['c_prime']-c),hsh(tid,x['key'])))
        strict=[x for x in cands if abs(x['c_prime']-c)<=1];mid=[x for x in cands if x['c_prime']>=2 and abs(x['c_prime']-c)<=2]
        filled=[];seen=set()
        for tier,lst in(('strict',strict),('middle',mid),('relaxed',cands)):
            for x in lst:
                if x['key'] in seen or len(filled)>=NCAND:continue
                seen.add(x['key']);filled.append(dict(x,tier=tier))
        elig=[u for u in subjects if not related(u,S) and (u,ty,A) not in P]
        rng=random.Random(int(hsh(tid,'sp')[:12],16));sp=rng.sample(elig,min(NSP,len(elig)))
        occ=[dict(rid=r['rids'][ai],sent_idx=[si for si in sidx if si<kept[ai]]) for ai,sidx in sorted(t['occ'].items())]
        bank.append(dict(id=tid,event=ei,event_name=name,type=ty,subject_key=S,subject_surface=Ss,answer_key=A,answer_surface=t['surface'],c_lead=c,f_forms=t['f'],stratum=stratum(c),question=q,frame_overlap=bool(fw&ng),occurrences=occ,
            n_strict=len(strict),n_middle=len(mid),n_relaxed=len(cands),strict_ge8=len(strict)>=G_MIN,middle_ge8=len(mid)>=G_MIN,relaxed_ge8=len(cands)>=G_MIN,candidates=filled,G=len(filled),main=len(filled)>=G_MIN,
            cross_null=[dict(S=u,surface=subj_surf.get(u,u),question=FRAME.format(noun=NOUN.get(ty,'string'),S=subj_surf.get(u,u))) for u in sp]))
    return bank
def subset_view(r,keep):
    """analysis result restricted to the articles `keep` (old indices), re-indexed; targets keep the mined set, their occurrences are restricted"""
    new={o:n for n,o in enumerate(keep)};r2=dict(r)
    for k in('arts','sents','facts','rids'):r2[k]=[r[k][o] for o in keep]
    r2['targets']=[dict(t,occ={new[ai]:v for ai,v in t['occ'].items() if ai in new}) for t in r['targets']]
    return r2
def cap_select(r,pub_date,N_):
    """-> (main rule: articles with the most lead targets first, ties by publication date; date rule: the first N_ by date); old indices"""
    V=N.variant(r,'lead');score=collections.Counter()
    for t in V['targets']:
        for ai in t['occ_tok']:score[ai]+=1
    n=len(r['arts']);dt=lambda ai:(pub_date[r['rids'][ai]],r['rids'][ai])
    main=sorted(sorted(range(n),key=dt),key=lambda ai:-score[ai])[:N_];date=sorted(range(n),key=dt)[:N_]
    return sorted(main),sorted(date)
def cap_tables(names,res,D,pub_date):
    """for every N in CAPN and both rules: per event frozen selection and the bank statistics"""
    out={};sel={}
    for N_ in CAPN:
        for rule in('main','date'):
            per=[];allb=[];tok_tot=0;nid=0
            for ei,(nm,r) in enumerate(zip(names,res)):
                if not r['ok']:continue
                m,d=cap_select(r,pub_date,N_);keep=m if rule=='main' else d;r2=subset_view(r,keep);b=event_bank(ei,nm,r2,D,nid);nid+=len(b);allb+=b
                V=N.variant(r2,'lead');tok=sum(V['tok'].values());tok_tot+=tok
                per.append(dict(event=ei,name=nm,articles=len(keep),tokens=tok,targets=len(b),main=sum(x['main'] for x in b),sp4=sum(len(x['cross_null'])==NSP for x in b)))
                sel[f'{N_}|{rule}|{nm}']=[r['rids'][ai] for ai in keep]
            out[f'{N_}|{rule}']=dict(per=per,targets=len(allb),strata={s[0]:sum(x['stratum']==s[0] for x in allb) for s in STRATA},main=sum(x['main'] for x in allb),sp4=sum(len(x['cross_null'])==NSP for x in allb),tokens=tok_tot,
                tok_med=sorted(p['tokens'] for p in per)[len(per)//2] if per else 0,tok_max=max((p['tokens'] for p in per),default=0),over8k=sum(p['tokens']>8192 for p in per),events=len(per))
    return out,sel
def art_counts(r):
    """lead variant: per article the number of distinct binding sentences (a sentence of the lead that holds S and A of a frozen target) and the number of distinct targets; per (article, target) the number of binding sentences; example sentences"""
    V=N.variant(r,'lead');n=len(r['arts']);ss_=[set() for _ in range(n)];tg=[0]*n;rep=[];ex=[]
    for t in V['targets']:
        for ai,occ in t['occ'].items():
            ss=[si for si in occ if si<V['kept'][ai]]
            if ss:tg[ai]+=1;ss_[ai]|=set(ss);rep.append(len(ss));ex.append(dict(S=t['S'],ty=t['ty'],A=t['key'],rid=r['rids'][ai],n_sent=len(ss),sentence=r['sents'][ai][ss[0]]))
    return dict(n=n,sent=[len(x) for x in ss_],targ=tg,rep=rep,examples=ex)
def dist(v):
    n=len(v);d={k:sum(1 for x in v if x==k)/n for k in(0,1,2,3)} if n else {};d['4+']=sum(1 for x in v if x>=4)/n if n else None
    return dict(n=n,dist=d,mean=sum(v)/n if n else None,median=sorted(v)[n//2] if n else None)
def binding_tables(names,res,pub_date,seed):
    """-> dict label -> per event stats + total; labels: 'all', 'N10', 'N20' (main rule selection, articles with the most lead targets first); plus 10 seeded examples"""
    out={};exs=[]
    for lab_,N_ in(('all',None),('N10',10),('N20',20)):
        per=[];A_s=[];A_t=[];A_r=[]
        for ei,(nm,r) in enumerate(zip(names,res)):
            if not r['ok']:continue
            r2=r if N_ is None else subset_view(r,cap_select(r,pub_date,N_)[0]);c=art_counts(r2)
            if lab_=='all':exs+=[dict(e,event=nm) for e in c['examples']]
            rep=c['rep'];per.append(dict(event=nm,articles=c['n'],sent=dist(c['sent']),targ=dist(c['targ']),zero=sum(1 for x in c['targ'] if x==0)/max(1,c['n']),rep_pairs=len(rep),rep_ge2=sum(1 for x in rep if x>=2)/max(1,len(rep))))
            A_s+=c['sent'];A_t+=c['targ'];A_r+=rep
        out[lab_]=dict(per=per,total=dict(articles=len(A_s),sent=dist(A_s),targ=dist(A_t),zero=sum(1 for x in A_t if x==0)/max(1,len(A_t)),rep_pairs=len(A_r),rep_ge2=sum(1 for x in A_r if x>=2)/max(1,len(A_r))))
    random.Random(seed).shuffle(exs);return out,exs[:10]
def membership(X,names,res,D,out):
    """(1) articles and events: partition check of the clusters, overlap of the title matches per event pair, other-event mentions in the lead; 50 seeded articles for reading"""
    import itertools
    cl=pickle.load(open(out/'e57_bigclusters'/'clusters_big.pkl','rb'))['clusters'];cnt=collections.Counter(i for c in cl for i in c['members'])
    part=dict(clusters=len(cl),articles_in_clusters=len(cnt),in_more_than_one=sum(1 for v in cnt.values() if v>1))
    m={k:set(v) for k,v in X['matched'].items()};pairs=[]
    for a_,b_ in itertools.combinations(m,2):
        o=len(m[a_]&m[b_])
        if o:pairs.append(dict(a=a_,b=b_,overlap=o,sizes=[len(m[a_]),len(m[b_])],share_of_smaller=o/min(len(m[a_]),len(m[b_]))))
    rx={k:[re.compile(x,re.I) for x in v] for k,v in N2.EVENTS.items() if k in names}
    mixed={};sample=[]
    for nm,r in zip(names,res):
        if not r['ok']:continue
        lt=EV.lead_texts(r);oth=[sum(1 for k,g in rx.items() if k!=nm and all(x.search(t) for x in g)) for t in lt]
        mixed[nm]=dict(articles=len(lt),with_other_event=sum(1 for x in oth if x>0),share=sum(1 for x in oth if x>0)/max(1,len(lt)))
        for ai in range(len(lt)):sample.append(dict(event=nm,rid=r['rids'][ai],title=D['title'][D['_ridx'][r['rids'][ai]]],lead=lt[ai][:700]))
    random.Random(FREEZE_SEED+7).shuffle(sample);return dict(partition=part,pairs=sorted(pairs,key=lambda x:-x['overlap']),mixed=mixed),sample[:50]
WEEKDAYS={'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'}
PLACE_CTX=re.compile(r"(?:\b(?:in|at|near|from|across|outside|inside|around|to|into|of)\s)$",re.I)
PERSON_BEFORE=re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|President|Minister|Mayor|Governor|Senator|Coach|Judge|Chairman|Actor|Actress|Director|Prime Minister|Chief|Captain)\.?\s)$")
PERSON_AFTER=re.compile(r"^\s*(?:,\s*[^,]{0,60},)?\s*(?:said|says|told|added|stated|announced|confirmed|has|was|is|will|had)\b")
def attributes(names,res,topn=5):
    """per event: the most frequent attribute strings of the LEAD (first sentences, ~350 tokens) per class and the share of the event's independent articles whose lead holds them. Classes: place / person are PROXIES from the local context of a capitalised phrase (place: preceded by in / at / near / from ...; person: preceded by a title or followed by said / told / was ...; majority of its labelled occurrences decides; unlabelled phrases stay 'other name'); date and number as mined by e57_common"""
    out={}
    for nm,r in zip(names,res):
        if not r['ok']:continue
        V=N.variant(r,'lead');n=len(r['arts']);arts=collections.defaultdict(set);surf={};votes=collections.defaultdict(collections.Counter)
        for ai in range(n):
            for si in range(V['kept'][ai]):
                sent=r['sents'][ai][si]
                fs_=C.facts_in_sentence(sent);spans=[(x[3],x[4]) for x in fs_ if x[0]=='NAME']
                for m_ in re.finditer(r'\b[A-Z][a-z]{3,}\b',sent):       # single capitalised words (places like Mumbai): ignored by the target miner, added here for the attribute table only
                    w_=m_.group(0)
                    if m_.start()==0 or re.search(r'[.!?"]\s$',sent[max(0,m_.start()-2):m_.start()]) or w_.lower() in C.MONTHNUM or w_ in C.STARTERS or w_ in WEEKDAYS or any(a0<=m_.start()<b0 for a0,b0 in spans):continue
                    fs_.append(('NAME',w_.lower(),w_,m_.start(),m_.end()))
                for ty,key,sf,a_,b_ in fs_:
                    arts[(ty,key)].add(ai);surf.setdefault((ty,key),sf)
                    if ty=='NAME':
                        if PLACE_CTX.search(sent[max(0,a_-12):a_]):votes[key]['place']+=1
                        if PERSON_BEFORE.search(sent[max(0,a_-16):a_]) or PERSON_AFTER.search(sent[b_:b_+80]):votes[key]['person']+=1
        cls=collections.defaultdict(list)
        for (ty,key),aset in arts.items():
            if ty=='NAME':
                v=votes.get(key,{});c='place' if v.get('place',0)>v.get('person',0) else 'person' if v.get('person',0)>v.get('place',0) else 'other name'
            else:c={'DATE':'date','NUM':'number'}.get(ty,ty)
            cls[c].append((len(aset),surf[(ty,key)],key))
        out[nm]={'articles':n}
        for c,lst in cls.items():
            lst.sort(key=lambda x:(-x[0],x[2]));out[nm][c]=[dict(surface=sf,key=k,articles=cnt,share=cnt/max(1,n)) for cnt,sf,k in lst[:topn]]
    return out
SUP_STOP={'the','and','for','from','with','said','news','city','state','court','police','government','minister','president','bank','new','group','company','national','international','university','united'}
WIN=64;CHUNK=8192;CKPT=(0.25,0.5,0.75);BANDS=(('<10',0,9),('10-19',10,19),('20-39',20,39),('40-79',40,79),('80+',80,10**9))
def core_tokens(S):return {w for w in re.findall(r"[a-z]{3,}",S.lower()) if w not in SUP_STOP}
def words_of(t):return set(re.findall(r"[a-z]+",t.lower()))
def place_stats(names,res,bank,attr,rows,seed):
    """supervisor check (E57 section 18.9) a / b / c / d. PLACE answer A of an event = the top-1 'place' string of attributes() (share s = share of the event's independent articles whose lead holds it), runner-up = top-2. c_SA: for every lead article that holds A, the S of the frozen targets of the event (subject_key) whose core token (word >= 3 letters, not in SUP_STOP) lies in a sentence holding A or in a sentence whose token range is within +-64 tokens of it (lead sentences only); c_SA = share of A-articles with >= 1 such S; strata of the number of such S per A-article 0 / 1-3 / 4+. Near-duplicate share = (articles before - after the Jaccard 0.5 removal) / before."""
    ev={};allA=[];cand=[]
    for ei,(nm,r) in enumerate(zip(names,res)):
        if not r['ok']:continue
        pl=attr.get(nm,{}).get('place',[]);o=dict(event=ei,name=nm,articles=len(r['arts']),n_before=rows[ei]['n_before'],n_dedup=rows[ei]['n_dedup'],dup_share=1-rows[ei]['n_dedup']/max(1,rows[ei]['n_before']))
        if not pl:o.update(A=None);ev[nm]=o;continue
        V=N.variant(r,'lead');A=pl[0];A2=pl[1] if len(pl)>1 else None
        rxA=re.compile(r'\b'+re.escape(A['surface'])+r'\b');rx2=re.compile(r'\b'+re.escape(A2['surface'])+r'\b') if A2 else None
        Sk={b['subject_key']:(core_tokens(b['subject_key']),b['subject_surface']) for b in bank if b['event']==ei};Sk={k:v for k,v in Sk.items() if v[0]}
        nA=0;hit=0;hit_in=0;nS=[];conf=0
        for ai in range(len(r['arts'])):
            kept=V['kept'][ai];sents=r['sents'][ai][:kept];asi=[si for si,t in enumerate(sents) if rxA.search(t)]
            if not asi:continue
            nA+=1;ww=[words_of(t) for t in sents];cum=r['arts'][ai]['cum'];near=set();inn=set();first=None
            for si in asi:
                lo=cum[si]-WIN;hi=cum[si+1]+WIN
                for sj in range(kept):
                    if cum[sj+1]>lo and cum[sj]<hi:
                        for k,(ct,sf) in Sk.items():
                            if ct&ww[sj]:
                                near.add(k)
                                if first is None:first=(k,si)
                for k,(ct,sf) in Sk.items():
                    if ct&ww[si]:inn.add(k)
            c2=bool(rx2 and any(rx2.search(t) for t in sents));conf+=c2;hit+=bool(near);hit_in+=bool(inn);nS.append(len(near))
            cand.append((ei,nm,ai,first[0] if first else None,Sk[first[0]][1] if first else '-',A['surface'],asi[0],c2))
        o.update(A=A['surface'],s=A['share'],A2=A2['surface'] if A2 else None,s2=A2['share'] if A2 else None,gap=A['share']-(A2['share'] if A2 else 0),A_articles=nA,c_SA=hit/max(1,nA),c_SA_in_sentence=hit_in/max(1,nA),
            strata={k:sum(1 for x in nS if (x==0 if k=='0' else 1<=x<=3 if k=='1-3' else x>=4)) for k in('0','1-3','4+')},both_places=conf/max(1,nA),n_S=len(Sk));ev[nm]=o;allA.append((nA,hit,hit_in,nS,conf))
    nA=sum(x[0] for x in allA);tot=dict(A_articles=nA,c_SA=sum(x[1] for x in allA)/max(1,nA),c_SA_in_sentence=sum(x[2] for x in allA)/max(1,nA),both_places=sum(x[4] for x in allA)/max(1,nA),
        strata={k:sum(1 for x in allA for y in x[3] if (y==0 if k=='0' else 1<=y<=3 if k=='1-3' else y>=4)) for k in('0','1-3','4+')},
        dup_share=1-sum(o['n_dedup'] for o in ev.values())/max(1,sum(o['n_before'] for o in ev.values())),events_with_place=len(allA),events=len(ev))
    random.Random(seed+99).shuffle(cand);return ev,tot,cand[:20]
def stream_stats(bank,res,pub_date,names,K):
    """supervisor check (e) / (f) for the first K events: items, c distribution, candidates, one-read chunks; combined stream (all lead articles of the K events ordered by publication date, rid, event) with the first-mention position of every target and the checkpoint tables 25 / 50 / 75 %"""
    b_=[b for b in bank if b['event']<K];ok=[ei for ei in range(min(K,len(names))) if res[ei]['ok']];st_=[]
    for ei in ok:
        r=res[ei];V=N.variant(r,'lead');tok={r['rids'][ai]:V['tok'][ai] for ai in range(len(r['arts']))}
        for ai,rid in enumerate(r['rids']):st_.append((pub_date[rid],rid,ei,tok[rid]))
    st_.sort();pos={(e,rid):i for i,(d,rid,e,t) in enumerate(st_)};tokens=sum(x[3] for x in st_);N_=len(st_)
    o=dict(K=K,events=len(ok),items=len(b_),main=sum(b['main'] for b in b_),strata={s[0]:sum(b['stratum']==s[0] for b in b_) for s in STRATA},cands_total=sum(b['G'] for b in b_),G_median=sorted(b['G'] for b in b_)[len(b_)//2] if b_ else None,
        stream_articles=N_,stream_tokens=tokens,chunks_all=-(-tokens//CHUNK),chunk=CHUNK)
    fm=[];ck={q:dict(unseen=0,seen_bands={x[0]:0 for x in BANDS}) for q in CKPT}
    for b in b_:
        ps=sorted(pos[(b['event'],oc['rid'])] for oc in b['occurrences']);fm.append(dict(id=b['id'],event=b['event'],first_pos=ps[0],first_date=str(st_[ps[0]][0]),positions=ps))
        for q in CKPT:
            cut=int(-(-q*N_//1))
            n_seen=sum(1 for x in ps if x<cut)
            if n_seen==0:ck[q]['unseen']+=1
            else:
                for nm_,lo,hi in BANDS:
                    if lo<=n_seen<=hi:ck[q]['seen_bands'][nm_]+=1
    o['checkpoints']={str(q):v for q,v in ck.items()};o['first_pos_quartiles']=[sorted(x['first_pos'] for x in fm)[int(len(fm)*q)] for q in(0.25,0.5,0.75)] if fm else None
    return o,[dict(pos=i,date=str(d),rid=rid,event=e,tokens=t) for i,(d,rid,e,t) in enumerate(st_)],fm
def supervisor(names,res,bank,attr,rows,pub_date,seed,X,fd):
    ev,tot,cand=place_stats(names,res,bank,attr,rows,seed);Ks={};stream=None;fms={}
    for K in(16,20):
        o,st_,fm=stream_stats(bank,res,pub_date,names,K);Ks[K]=o
        if K==20:stream=st_;fms=fm
    smp=[];ex=[]
    for ei,nm,ai,Sk,Ssf,A,si,c2 in cand:
        r=res[ei];smp.append(dict(event=nm,rid=r['rids'][ai],S_phrase=Ssf,answer=A,sent_idx=si,conflict_2nd_place_in_lead=bool(c2)))
        ex.append(dict(smp[-1],evidence_sentence=r['sents'][ai][si]))
    (fd/'lite_place.json').write_text(json.dumps(dict(rules='place answer = top-1 place string of attributes(); c_SA window +-%d tokens; see place_stats() in e57_lite.py'%WIN,events=ev,total=tot,sample20=smp),indent=1,default=str))
    (fd/'lite_place_sample.json').write_text(json.dumps(dict(note='publisher sentences: stays on the server',sample20=ex),indent=1,default=str))
    (fd/'lite_stream_columns.json').write_text(json.dumps(dict(note='combined lead stream of the first 20 events ordered by (date, rid, event); first_pos / positions per target index into stream; no article text',stream=stream,first_mention=fms,K=Ks),indent=1,default=str))
    return dict(place=ev,total=tot,sample=smp,K=Ks)
def run(a):
    out=pathlib.Path(a.out);fd=out/'e57_lite';fd.mkdir(exist_ok=True);rep=fd/'e57_lite_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    a.drop=N3.DROP;X=N3.analyze_all(a);D,names,res,meta,rows,pub_date=[X[k] for k in('D','names','res','meta','rows','pub_date')];t0=X['t0']
    bank=[];events=[];nid=0
    for ei,(nm,r,row) in enumerate(zip(names,res,rows)):
        b=event_bank(ei,nm,r,D,nid) if r['ok'] else [];nid+=len(b);bank+=b
        arts=[] if not r['ok'] else sorted(((pub_date[x],x) for x in r['rids']))
        events.append(dict(event=ei,name=nm,source=row['source'],matched_titles=row['titles'],independent_articles=row['n_dedup'] if r['ok'] else 0,ok=r['ok'],first=row['first'],last=row['last'],
            targets_before_ethics=row.get('targets_f2_before'),rejected_by_ethics=row.get('rejected_targets'),bank_targets=len(b),articles=[dict(rid=x,date=d) for d,x in arts]))
    bt,exs=binding_tables(names,res,pub_date,a.seed);mem,samp=membership(X,names,res,D,out)
    (fd/'lite_binding_examples.json').write_text(json.dumps(dict(note='article sentences (publisher text): stays on the server',examples=exs,sample50_for_reading=samp),indent=1,default=str))
    attr=attributes(names,res);sup=supervisor(names,res,bank,attr,rows,pub_date,a.seed,X,fd);(fd/'lite_attributes.json').write_text(json.dumps(attr,indent=1))
    caps,sel=cap_tables(names,res,D,pub_date);(fd/'lite_caps.json').write_text(json.dumps(dict(rule='main = articles with the most lead targets first, ties by publication date then id; date = first N by publication date',caps=caps,selection=sel),indent=1))
    ej=json.dumps(dict(seed=FREEZE_SEED,rules=dict(g_min=G_MIN,ncand=NCAND,nsp=NSP,frame=FRAME,dropped=list(N3.DROP),n_extra=a.n_extra),events=events),indent=1,default=str)
    bj=json.dumps(bank,indent=1,default=str);(fd/'lite_events.json').write_text(ej);(fd/'lite_bank.json').write_text(bj)
    man=dict(files={'lite_caps.json':hashlib.sha256((fd/'lite_caps.json').read_bytes()).hexdigest(),'lite_events.json':hashlib.sha256(ej.encode()).hexdigest(),'lite_bank.json':hashlib.sha256(bj.encode()).hexdigest()},n_events=len(events),n_events_with_bank=sum(bool(e['bank_targets']) for e in events),n_targets=len(bank),seed=FREEZE_SEED)
    (fd/'lite_manifest.json').write_text(json.dumps(man,indent=1));txt=report(events,bank,X,a,time.time()-t0,caps,bt,mem,exs,samp,attr,sup);rep.write_text(txt);print(txt[:9000])
def report(events,bank,X,a,sec,caps=None,bt=None,mem=None,exs=None,samp=None,attr=None,sup=None):
    f=EV.fmt;L=['# E57-lite freeze (CPU, regex proxy)','',f"{len(events)} events ({sum(e['source']=='regex' for e in events)} fixed regex events + {sum(e['source']!='regex' for e in events)} clusters of the job-219 list), {len(bank)} frozen targets (lead variant, c >= 3, after the target-level ethics filter). Candidates / questions / cross null as defined in the docstring of e57_lite.py; the pool uses LEAD exposures (job 220 / 221 used whole-article exposures for the pool tables, so the numbers differ). Run {sec:.0f} s.",'',
       '## events','','| # | event | source | matched titles | independent articles | first | last | targets f>=2 before ethics | rejected | frozen targets | main (G >= 8) |','|---|---|---|---|---|---|---|---|---|---|---|']
    for e in events:
        bb=[b for b in bank if b['event']==e['event']];L.append(f"| {e['event']} | {e['name']} | {e['source']} | {e['matched_titles']} | {e['independent_articles']}{'' if e['ok'] else ' (below the analysis minimum)'} | {e['first']} | {e['last']} | {e['targets_before_ethics']} | {e['rejected_by_ethics']} | {len(bb)} | {sum(b['main'] for b in bb)} |")
    st=[s[0] for s in STRATA]
    def tab(title,key,rows_):
        o=['',f'## {title}','','| '+key+' | '+' | '.join(st)+' | all |','|---|'+'---|'*(len(st)+1)]
        for nm,sel in rows_:
            cs=[sum(1 for b in sel if b['stratum']==s) for s in st];o.append(f"| {nm} | "+' | '.join(map(str,cs))+f' | {len(sel)} |')
        return o
    L+=tab('targets per c stratum (all frozen targets)','event',[(e['name'],[b for b in bank if b['event']==e['event']]) for e in events]+[('all',bank)])
    L+=tab('main targets (G >= 8 candidates after the fill order strict > middle > relaxed)','event',[(e['name'],[b for b in bank if b['event']==e['event'] and b['main']]) for e in events]+[('all',[b for b in bank if b['main']])])
    L+=tab('per answer type (all frozen targets)','type',[(ty,[b for b in bank if b['type']==ty]) for ty in sorted({b['type'] for b in bank})])
    L+=tab('per answer type (main targets)','type',[(ty,[b for b in bank if b['type']==ty and b['main']]) for ty in sorted({b['type'] for b in bank})])
    L+=['','## tier sizes (a tier alone has >= 8 candidates)','','| stratum | targets | strict >= 8 | middle >= 8 | relaxed >= 8 | median G | cross null with 4 S\' | frame 5-gram overlap |','|---|---|---|---|---|---|---|---|']
    import statistics
    for s in st+['all']:
        sel=[b for b in bank if s=='all' or b['stratum']==s]
        if sel:L.append(f"| {s} | {len(sel)} | {sum(b['strict_ge8'] for b in sel)} | {sum(b['middle_ge8'] for b in sel)} | {sum(b['relaxed_ge8'] for b in sel)} | {statistics.median(b['G'] for b in sel):.0f} | {sum(len(b['cross_null'])==NSP for b in sel)} | {sum(b['frame_overlap'] for b in sel)} |")
        else:L.append(f'| {s} | 0 | | | | | | |')
    cn=collections.Counter(len(b['cross_null']) for b in bank);L+=['',f"- cross-null S' available per target (count of S' -> targets): {dict(sorted(cn.items()))}",f"- ethics: {sum(e['rejected_by_ethics'] or 0 for e in events)} targets rejected of {sum(e['targets_before_ethics'] or 0 for e in events)}; reasons per event in e57_namedevents3-style counts are in the rejects json of this run (kept by job 222, not written here)."]
    if caps:
        L+=['',f"## article cap per event N = {', '.join(map(str,CAPN))} (selection frozen in lite_caps.json)",'','Rule main: articles ordered by the number of lead targets they contain (most first), ties by publication date; rule date: the first N articles by publication date. Targets are re-counted on the selected articles only (c >= 3 inside the selection; the mined target set is the one of all articles). Round tokens = lead tokens (+2 per article) of the selected articles of all events; per event tokens are compared with 8192.','',
           '| N | rule | events | targets c >= 3 | '+' | '.join('c '+x for x in st)+' | main (G >= 8) | with 4 S\' | round tokens (all events) | tokens per event median / max | events over 8192 |','|---|---|---|---|'+'---|'*len(st)+'---|---|---|---|---|']
        for N_ in CAPN:
            for rule in('main','date'):
                c=caps[f'{N_}|{rule}'];L.append(f"| {N_} | {rule} | {c['events']} | {c['targets']} | "+' | '.join(str(c['strata'][x]) for x in st)+f" | {c['main']} ({c['main']/max(1,c['targets']):.2f}) | {c['sp4']} ({c['sp4']/max(1,c['targets']):.2f}) | {c['tokens']} | {c['tok_med']} / {c['tok_max']} | {c['over8k']} |")
        for N_ in CAPN:
            L+=['',f'### N = {N_}, rule main: per event','','| event | articles | tokens | targets | main | with 4 S\' |','|---|---|---|---|---|---|']
            L+=[f"| {p['name']} | {p['articles']} | {p['tokens']} | {p['targets']} | {p['main']} | {p['sp4']} |" for p in caps[f'{N_}|main']['per']]
    if mem:
        P=mem['partition'];L+=['','## do articles belong to one event only? (1)','',f"- clusters of the title stage (job 215, no size limit): {P['clusters']} clusters, {P['articles_in_clusters']} articles in clusters, **{P['in_more_than_one']} in more than one cluster** (a partition, one cluster per article, when 0). The added clusters are such clusters; the 10 regex events are NOT clusters: they are title matches, so one article can match two events.",'','### overlap of the title matches per event pair (pairs with overlap > 0)','','| event A | event B | titles in both | size A / size B | share of the smaller |','|---|---|---|---|---|']
        L+=[f"| {x['a']} | {x['b']} | {x['overlap']} | {x['sizes'][0]} / {x['sizes'][1]} | {f(x['share_of_smaller'])} |" for x in mem['pairs']] or ['| none | | | | |']
        L+=['','### lead mentions another regex event (proxy for mixed stories: the lead text matches ALL regex groups of another event of the fixed regex events)','','| event | articles | with another event in the lead | share |','|---|---|---|---|']
        L+=[f"| {k} | {v['articles']} | {v['with_other_event']} | {f(v['share'])} |" for k,v in mem['mixed'].items()]
        L+=['',"- 50 seeded articles for READING (title + the first 700 characters of the lead) are in lite_binding_examples.json on the server (publisher text, not in this report); the manual judgement 'does the lead mix different events' is NOT made by this job. Their titles:"]+[f"  - {x['event']}: {x['title'][:100]}" for x in samp]
    if bt:
        L+=['','## binding sentences in the lead of an article (2)','','Binding sentence = a sentence of the lead (about 350 tokens) that holds S and A of a frozen target (c >= 3). Per article: distinct binding sentences, and distinct targets present. Shares of articles with 0 / 1 / 2 / 3 / 4+.','']
        for lab_,title in(('all','all independent articles of the events'),('N10','N = 10 per event (main rule: articles with the most lead targets first)'),('N20','N = 20 per event (main rule)')):
            T=bt[lab_];L+=[f'### {title}','','| event | articles | binding sentences: 0 / 1 / 2 / 3 / 4+ | mean | median | distinct targets: 0 / 1 / 2 / 3 / 4+ | mean | median | articles without target | (article, target) pairs | pairs with >= 2 sentences |','|---|---|---|---|---|---|---|---|---|---|---|']
            dd=lambda x:' / '.join(f(x['dist'][k]) for k in(0,1,2,3,'4+'))
            for p_ in T['per']+[dict(event='ALL EVENTS',**T['total'])]:
                L.append(f"| {p_['event']} | {p_['articles']} | {dd(p_['sent'])} | {f(p_['sent']['mean'])} | {p_['sent']['median']} | {dd(p_['targ'])} | {f(p_['targ']['mean'])} | {p_['targ']['median']} | {f(p_['zero'])} | {p_['rep_pairs']} | {f(p_['rep_ge2'])} |")
            L.append('')
        L+=['Pairs with >= 2 sentences = the same target repeated inside one article lead (possible within-article shadowing). Ten seeded examples (S, A, event; the sentences are in lite_binding_examples.json on the server):','']+[f"- {e['event']} | S = {e['S']} | A = {e['A']} ({e['ty']}) | {e['n_sent']} binding sentence(s) in the article" for e in exs]
    if attr:
        L+=['','## event attribute candidates (lead of the event articles)','',"Per event and class the 5 most frequent strings and the share of the event's independent articles whose lead (first sentences, ~350 tokens) holds the string. place / person are PROXIES from the local context of a capitalised phrase (see attributes() in e57_lite.py), 'other name' = organisations and unlabelled phrases. Single capitalised words (e.g. a city name) are included here although the target miner ignores them. Gap = share of the 1st minus the 2nd string (how clearly one value stands out). Strings are mined, not checked by hand.",'']
        for nm,o in attr.items():
            L+=[f"### {nm} ({o['articles']} independent articles)",'','| class | strings (share of articles) | top-1 share | gap 1st - 2nd |','|---|---|---|---|']
            for c in('place','person','other name','date','number'):
                lst=o.get(c,[])
                if lst:L.append(f"| {c} | "+'; '.join(f"{x['surface']} ({x['share']:.2f})" for x in lst)+f" | {lst[0]['share']:.2f} | {lst[0]['share']-(lst[1]['share'] if len(lst)>1 else 0):.2f} |")
                else:L.append(f'| {c} | none | | |')
            L.append('')
    if sup:
        P=sup['place'];T=sup['total']
        L+=['','## supervisor check (E57 section 18.9)','',"Rules fixed before any result: PLACE answer = top-1 place string of the lead (proxy of attributes()), s = its share of the event's independent articles, runner-up = top-2; c_SA window +-%d tokens around every lead sentence holding the place; S = subjects of the frozen targets of the event, matched by core tokens (words >= 3 letters, small stop list); family-word and ethics filters as in the freeze; no PERSON attribute, no K >= 30, no extended candidate set."%WIN,'',
           '### (a) PLACE answer per event, (b) c_SA, (c) near-duplicates','','| event | articles | near-dup share (before -> after) | PLACE answer A | s | 2nd place | s2 | gap | A-articles | c_SA | c_SA (A sentence only) | strata of S near A: 0 / 1-3 / 4+ | A-articles with 2nd place too |','|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for nm,o in P.items():
            if o.get('A') is None:L.append(f"| {nm} | {o['articles']} | {o['dup_share']:.2f} ({o['n_before']} -> {o['n_dedup']}) | none | | | | | | | | | |");continue
            L.append(f"| {nm} | {o['articles']} | {o['dup_share']:.2f} ({o['n_before']} -> {o['n_dedup']}) | {o['A']} | {o['s']:.2f} | {o['A2'] or '-'} | {(o['s2'] or 0):.2f} | {o['gap']:.2f} | {o['A_articles']} | {o['c_SA']:.2f} | {o['c_SA_in_sentence']:.2f} | {o['strata']['0']} / {o['strata']['1-3']} / {o['strata']['4+']} | {o['both_places']:.2f} |")
        L+=['',f"- overall: {T['events_with_place']} of {T['events']} analysed events have a place; A-articles {T['A_articles']}; c_SA {T['c_SA']:.2f} (A sentence only {T['c_SA_in_sentence']:.2f}); strata 0 / 1-3 / 4+ = {T['strata']['0']} / {T['strata']['1-3']} / {T['strata']['4+']} articles; near-duplicate share {T['dup_share']:.2f}; A-articles with the 2nd place in the lead too {T['both_places']:.2f}."]
        L+=['','### (d) 20 seeded PLACE extractions to check by hand','','The evidence sentences are publisher text and stay in lite_place_sample.json on the server; here article id and sentence index.','','| # | event | S phrase near A | answer A | article id | sentence | 2nd place also in lead (conflict) |','|---|---|---|---|---|---|---|']
        L+=[f"| {i+1} | {x['event']} | {x['S_phrase']} | {x['answer']} | {x['rid']} | {x['sent_idx']} | {'yes' if x['conflict_2nd_place_in_lead'] else 'no'} |" for i,x in enumerate(sup['sample'])] or ['| none | | | | | |']
        L+=['','### (e) items and one-read chunks for K = 16 and K = 20 events (first K of the list)','','| K | events | items (targets c >= 3) | main (G >= 8) | c 3-4 / 5-9 / 10-19 / 20+ | candidates total | median G | stream articles | stream lead tokens | one-pass chunks of %d |'%CHUNK,'|---|---|---|---|---|---|---|---|---|---|']
        for K,o in sup['K'].items():L.append(f"| {K} | {o['events']} | {o['items']} | {o['main']} | "+' / '.join(str(o['strata'][s]) for s in st)+f" | {o['cands_total']} | {o['G_median']} | {o['stream_articles']} | {o['stream_tokens']} | {o['chunks_all']} |")
        L+=['','### (f) stream columns for the event-label permutation null and the \'before first article\' stratum','','lite_stream_columns.json: every lead article of the first 20 events in (date, id, event) order with date, event label and token count (the label permutation shuffles event labels among articles, dates stay); per target the positions of its articles in this stream and the date of the first one. Checkpoints = 25 / 50 / 75 % of the stream articles: targets with NO article yet (before the first article) and, for the others, the count of articles so far in bands.','','| K | checkpoint | before first article | seen with < 10 | 10-19 | 20-39 | 40-79 | 80+ |','|---|---|---|---|---|---|---|---|']
        for K,o in sup['K'].items():
            for q,v in o['checkpoints'].items():L.append(f"| {K} | {float(q):.0%} | {v['unseen']} | "+' | '.join(str(v['seen_bands'][x[0]]) for x in BANDS)+' |')
    L+=['','Titles of the added clusters are shown only to identify them; no article text is written.']
    return '\n'.join(L)+'\n'
def _selftest():
    def after(td):
        a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=5,min_domains=5,min_c=3,workers=1,seed=5,rerun=True,n_extra=4,drop=())
        N3.DROP=();run(a);fd=td/'e57_lite';B=json.load(open(fd/'lite_bank.json'));E=json.load(open(fd/'lite_events.json'));M=json.load(open(fd/'lite_manifest.json'))
        assert B and M['n_targets']==len(B) and (fd/'e57_lite_report.md').exists() and (fd/'lite_caps.json').exists() and 'article cap per event' in (fd/'e57_lite_report.md').read_text() and 'lite_caps.json' in M['files'] and (fd/'lite_binding_examples.json').exists() and 'binding sentences in the lead' in (fd/'e57_lite_report.md').read_text() and 'one event only' in (fd/'e57_lite_report.md').read_text() and 'attribute candidates' in (fd/'e57_lite_report.md').read_text() and (fd/'lite_attributes.json').exists() and (fd/'lite_place.json').exists() and (fd/'lite_stream_columns.json').exists() and 'supervisor check' in (fd/'e57_lite_report.md').read_text()
        for b in B:
            assert b['question'].startswith('Question:') and b['subject_surface'] in b['question'] and all(c['key']!=b['answer_key'] for c in b['candidates']) and b['stratum'] in('3-4','5-9','10-19','20+')
            assert all(u['S']!=b['subject_key'] for u in b['cross_null']) and b['G']==len(b['candidates']) and b['main']==(b['G']>=G_MIN)
        assert hashlib.sha256((fd/'lite_bank.json').read_bytes()).hexdigest()==M['files']['lite_bank.json']
        print('e57_lite selftest ok:',len(E['events']),'events,',len(B),'targets, main',sum(b['main'] for b in B))
    N3._selftest(after)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-articles',type=int,default=5);ap.add_argument('--min-domains',type=int,default=5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--n-extra',type=int,default=10);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
