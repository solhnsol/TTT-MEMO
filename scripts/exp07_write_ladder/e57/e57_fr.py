"""E57 section 8 Federal Register CPU measurement (no GPU, no model): group proposed rules, final rules and corrections by docket id / RIN (union of both keys) and count the facts that are repeated ACROSS the documents of a group.
Sources: --source api (federalregister.gov API v1: types RULE and PRORULE incl. corrections, publication date from --date-from; raw text of the documents of groups with >= 2 documents) | --source milo (HF milo-data/federal-register-final-rules-2026, final rules only: a CPU stand-in that exercises the grouping and counting code; groups there are final rules sharing a docket / RIN).
Fact proxy (no LLM): money, percent, plain numbers, dates, capitalised names (e57_common.facts_in_sentence) + CFR citations + acronyms of 3-6 capitals (generic ones removed). A fact counts as repeated when it occurs in sentences of >= 2 different documents of the group, in >= 3 sentences in total, with >= 2 distinct slot-masked 5-gram forms (f >= 2).
Output: <out>/e57_fr_report.md/.json"""
import argparse,json,os,re,sys,time,collections,pathlib,urllib.request,urllib.parse,random
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
T=C.T
CFR=re.compile(r'\b\d{1,2}\s?CFR\s?(?:part|parts|§|§§)?\s?\d+(?:\.\d+)?(?:\([a-z0-9]+\))*',re.I)
ACR=re.compile(r'\b[A-Z]{3,6}\b')
GENERIC=set('THE AND FOR NOT ALL ANY USC CFR FR DATE FEDERAL REGISTER FOR NOTICE RULE FINAL PROPOSED ACTION SUMMARY AGENCY DOCKET NO ET AL ACT RIN PART SEC SECTION YES'.split())
API='https://www.federalregister.gov/api/v1/documents.json'
FIELDS=['document_number','type','title','publication_date','docket_ids','regulation_id_numbers','raw_text_url','correction_of','agency_names']
def api_list(date_from,date_to,types=('RULE','PRORULE')):
    docs=[];import datetime
    d0=datetime.date.fromisoformat(date_from);d1=datetime.date.fromisoformat(date_to)
    cur=datetime.date(d0.year,d0.month,1)
    while cur<=d1:
        nxt=datetime.date(cur.year+(cur.month==12),cur.month%12+1,1);lo=max(cur,d0);hi=min(nxt-datetime.timedelta(days=1),d1)
        for ty in types:
            page=1
            while True:
                q=[('per_page','1000'),('page',str(page)),('conditions[type][]',ty),('conditions[publication_date][gte]',lo.isoformat()),('conditions[publication_date][lte]',hi.isoformat())]+[('fields[]',f) for f in FIELDS]
                url=API+'?'+urllib.parse.urlencode(q);r=json.load(urllib.request.urlopen(url,timeout=120));docs+=r.get('results',[])
                print(f'  api {lo} {ty} page {page}: {len(r.get("results",[]))} (total so far {len(docs)})',flush=True)
                if page>=r.get('total_pages',1):break
                page+=1;time.sleep(0.2)
        cur=nxt
    return docs
def api_text(d):
    for k in range(4):
        try:return urllib.request.urlopen(d['raw_text_url'],timeout=120).read().decode('utf8','ignore')
        except Exception as e:time.sleep(2*(k+1))
    return ''
def groups_of(docs):
    par={};
    def find(x):
        par.setdefault(x,x)
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    for d in docs:
        ks=['D:'+x for x in (d.get('docket_ids') or [])]+['R:'+x for x in (d.get('regulation_id_numbers') or d.get('rins') or [])]
        d['keys']=ks
        for k in ks:find(k)
        for a,b in zip(ks,ks[1:]):par[find(a)]=find(b)
    g=collections.defaultdict(list)
    for d in docs:
        if d['keys']:g[find(d['keys'][0])].append(d)
    return list(g.values())
def clean(text):
    t=re.sub(r'<[^>]+>',' ',text);return t
def sentences(text):
    out=[]
    for para in re.split(r'\n\s*\n|\n(?=[A-Z0-9])',clean(text)):
        para=re.sub(r'\s+',' ',para).strip()
        if len(para)<30:continue
        out+=T.split_sents(para)
    return out
def facts(s):
    fs=[(ty,key,sf) for ty,key,sf,_,_ in C.facts_in_sentence(s)]
    fs+=[('CFR',re.sub(r'\s+','',m.group(0).lower()),m.group(0)) for m in CFR.finditer(s)]
    fs+=[('ACR',m.group(0),m.group(0)) for m in ACR.finditer(s) if m.group(0) not in GENERIC]
    return fs
def analyze_group(g,enc,max_tokens=8000):
    docs=[d for d in g if d.get('text')];info=dict(n_docs=len(g),n_with_text=len(docs),types=dict(collections.Counter(d['type'] for d in g)))
    if len(docs)<1:return dict(info,facts=[],within=0,tokens=0)
    per=[];tok=0
    for d in docs:
        ss=sentences(d['text'])[:3000];n=sum(len(x.ids) for x in enc.encode_batch(ss,add_special_tokens=False)) if ss else 0;tok+=n
        fa=collections.defaultdict(list)
        for si,s in enumerate(ss):
            for ty,key,sf in set(facts(s)):fa[ty,key].append((s,sf))
        per.append(dict(id=d['document_number'],type=d['type'],ss=ss,tokens=n,facts=fa))
    allk=collections.defaultdict(dict)
    for pi,p in enumerate(per):
        for k,v in p['facts'].items():allk[k][pi]=v
    rep=[]
    for k,byd in allk.items():
        tot=sum(len(v) for v in byd.values())
        if len(byd)>=2 and tot>=3:
            items=[(s,[sf]) for v in byd.values() for s,sf in v]
            f=C.n_forms(items)
            if f>=2:rep.append(dict(ty=k[0],key=k[1],n_docs=len(byd),n_sent=tot,f=f))
    within=sum(1 for p in per for k,v in p['facts'].items() if len(v)>=3 and C.n_forms([(s,[sf]) for s,sf in v])>=2)
    return dict(info,facts=rep,within=within,tokens=tok,doc_tokens=[p['tokens'] for p in per],has_prorule_and_rule=bool({'PRORULE','RULE'}<={d['type'] for d in docs}))
def run(a):
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=out/'docs.jsonl'
    if a.source=='milo':
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        p=hf_hub_download('milo-data/federal-register-final-rules-2026','data.parquet',repo_type='dataset',local_dir=str(out/'milo'));t=pq.read_table(p,columns=['document_number','title','full_text','publication_date','docket_ids','rins','word_count'])
        docs=[dict(document_number=r['document_number'],type='RULE',title=r['title'],publication_date=r['publication_date'],docket_ids=list(r['docket_ids'] or []),regulation_id_numbers=list(r['rins'] or []),text=r['full_text']) for r in t.to_pylist()]
    else:
        if cache.exists() and not a.rerun:docs=[json.loads(l) for l in open(cache)]
        else:
            docs=api_list(a.date_from,a.date_to);g=groups_of(docs);need={d['document_number'] for gg in g if len(gg)>=2 for d in gg};print(f'{len(docs)} documents, {len(g)} keyed groups, {sum(1 for gg in g if len(gg)>=2)} with >= 2 documents; fetching text of {len(need)}',flush=True)
            for i,d in enumerate(docs):
                if d['document_number'] in need:d['text']=api_text(d);time.sleep(0.1)
                if i%200==0:print(f'  text {i}/{len(docs)}',flush=True)
            with open(cache,'w') as f:
                for d in docs:f.write(json.dumps(d)+'\n')
    G=groups_of(docs);big=[g for g in G if len(g)>a.max_group];multi=[g for g in G if 2<=len(g)<=a.max_group];print(f'{len(big)} groups with more than {a.max_group} documents (key chains) are excluded',flush=True);print(f'{len(docs)} documents; keyed groups {len(G)}; multi-document groups {len(multi)}',flush=True)
    enc=T.load_enc(os.environ.get('E57_TOKENIZER'));res=[analyze_group(g,enc) for g in multi]
    tcount=collections.Counter(); 
    for g in multi:tcount[tuple(sorted(collections.Counter(d['type'] for d in g).items()))]+=1
    ok=[r for r in res if r['facts']]
    rows=sorted(ok,key=lambda r:-len(r['facts'])/max(r['tokens'],1))
    n=0;toks=0;used=0
    for r in rows:
        if n>=60:break
        n+=len(r['facts']);toks+=r['tokens'];used+=1
    M=dict(n_excluded_chain_groups=len(big),excluded_chain_group_sizes=sorted(len(g) for g in big),n_documents=len(docs),n_keyed_groups=len(G),n_multi_groups=len(multi),group_type_mix=[(list(k),v) for k,v in tcount.most_common(12)],
      groups_with_prorule_and_rule=sum(1 for r in res if r.get('has_prorule_and_rule')),groups_with_repeated_fact=len(ok),
      repeated_facts_total=sum(len(r['facts']) for r in res),repeated_per_multi_group=sum(len(r['facts']) for r in res)/max(1,len(multi)),
      within_doc_repeated_facts_per_group=sum(r['within'] for r in res)/max(1,len(multi)),
      facts_by_type=dict(collections.Counter(f['ty'] for r in res for f in r['facts'])),
      docs_per_multi_group=sum(r['n_docs'] for r in res)/max(1,len(multi)),median_group_tokens=sorted(r['tokens'] for r in res)[len(res)//2] if res else None,
      greedy_60_facts=dict(groups=used,facts=n,tokens=toks,note='groups taken in order of repeated facts per token until 60 facts; tokens = text of those groups once (one round)'),
      facts_in_groups_with_prorule_and_rule=sum(len(r['facts']) for r in res if r.get('has_prorule_and_rule')),
      power_if_n=C.power_table(max(1,n)))
    json.dump(M,open(out/'e57_fr_report.json','w'),indent=1,default=str)
    L=['# E57 section 8 Federal Register CPU measurement (regex proxy, source '+a.source+')','',f"documents {M['n_documents']}; excluded key-chain groups (> {a.max_group} documents): {M['excluded_chain_group_sizes']}; groups by docket / RIN {M['n_keyed_groups']}; groups with >= 2 documents {M['n_multi_groups']} (mean {M['docs_per_multi_group']:.2f} documents); document-type mixes (most common): {M['group_type_mix']}",
       f"groups containing a proposed rule AND a final rule: {M['groups_with_prorule_and_rule']}",'',
       f"facts repeated across >= 2 documents of a group (>= 3 sentences, f >= 2): {M['repeated_facts_total']} in {M['groups_with_repeated_fact']} groups = {M['repeated_per_multi_group']:.2f} per multi-document group; by type {M['facts_by_type']}; in groups with proposed+final: {M['facts_in_groups_with_prorule_and_rule']}",
       f"baseline (design 8.1 style): facts with >= 3 sentences INSIDE one document and f >= 2: {M['within_doc_repeated_facts_per_group']:.2f} per multi-document group",
       f"greedy bank of 60 repeated facts: {M['greedy_60_facts']}",'','power for n = that bank size (DEFF 1 / 1.5 / 2):','']
    for x in M['power_if_n']:L.append(f"- DEFF {x['deff']}: n_eff {x['n_eff']:.1f}, theta {x['theta']:.3f}, power @0.03 {x['power_0.03']:.2f}, @0.10 {x['power_0.1']:.2f}, @0.15 {x['power_0.15']:.2f}")
    L+=['','Reading: the criterion of E57 section 3 M2/M6 applied to this corpus is n >= 60 and power >= 0.5 at 0.10. The regex proxy overcounts acronyms and capitalised names and undercounts paraphrased facts; the LLM extraction of the final bank may differ in both directions. Federal Register documents are US-government works (no copyright), so the texts themselves can be published.']
    open(out/'e57_fr_report.md','w').write('\n'.join(L)+'\n');print('\n'.join(L))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['api','milo'],default='api');ap.add_argument('--out',default='e57_fr');ap.add_argument('--date-from',default='2026-01-01');ap.add_argument('--date-to',default='2026-10-19');ap.add_argument('--max-group',type=int,default=40);ap.add_argument('--rerun',action='store_true');run(ap.parse_args())
