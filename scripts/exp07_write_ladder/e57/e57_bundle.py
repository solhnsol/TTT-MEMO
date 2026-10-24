"""E57 reading bundle for hand-written questions (CPU only, NO model, NO GPU, NO download; the news thread idea of Hans 2026-10-24: famous events, questions written per article set by Claude, not by code).
Builds from the same analysis as job 223 (e57_namedevents3.analyze_all: one article per hostname, near-duplicate removal Jaccard 0.5): for every event (the 223 list incl. --n-extra clusters of the job-219 list, plus optional hand-chosen keyword events --keyword-events file.json {name: [regex, ...]}, a title matches when ALL regexes match):
 1) reading_bundle.json (SERVER ONLY, publisher text, not redistributable): up to --per-event (default 40) independent articles chosen by a seeded shuffle (all if fewer; events with fewer than 30 are flagged): event, article id, hostname, date, title, excerpt = the first sentences of the lead up to --excerpt-chars (default 700) characters.
 2) bundle_facts.json (strings and counts only, no article text): per event the top proper nouns (2+ capitalised words), single capitalised words with their place / person class proxy, dates and numbers, each with the number of independent articles that hold it anywhere and in the lead (350 tokens).
 3) bundle_index.json (no text): per event articles before / after the near-duplicate removal, selected article ids, hostnames, dates.
 3b) news-lite/<event>.json: one file per event with the selected excerpts (the copy for /mnt/project-files/data/news-lite/, private internal research only, Hans 2026-10-24; copied there by hand, not by this job).
 4) bundle_store.pkl (SERVER ONLY): all independent articles of every event with their sentences, token offsets, lead length, ids, dates: the input of the verification helper e57_verify.py (how many independent articles hold a hand-written answer string, the share where it shares a sentence / +-64 tokens with the S keyword).
 e57_bundle_report.md: counts only (events, articles, selected, flags, top-10 strings per class).
Rules fixed before any result: seed 57241; selection = seeded shuffle, no quality filter; excerpt = whole sentences up to the character limit (at least the first sentence). No ethics filter is applied here (hand-written questions are checked by a human)."""
import argparse,json,sys,random,pickle,pathlib,time,collections,re,hashlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
import e57_events as EV
import e57_namedevents3 as N3
import e57_lite as LT
SEED=57241
def excerpt(sents,kept,lim):
    out=[];n=0
    for si in range(max(1,kept)):
        if si>=len(sents):break
        if out and n+len(sents[si])>lim:break
        out.append(sents[si]);n+=len(sents[si])+1
    return ' '.join(out)[:max(lim,len(out[0]) if out else 0)]
def fact_tables(r,topn=25):
    V=N.variant(r,'lead');kept=V['kept'];whole=collections.defaultdict(set);lead=collections.defaultdict(set);surf={}
    for ai in range(len(r['arts'])):
        for si,fs in enumerate(r['facts'][ai]):
            for ty,key,sf in fs:
                whole[(ty,key)].add(ai);surf.setdefault((ty,key),sf)
                if si<kept[ai]:lead[(ty,key)].add(ai)
    out={}
    for ty,lab in(('NAME','proper nouns (2+ capitalised words)'),('DATE','dates'),('NUM','numbers and amounts')):
        lst=sorted(((len(v),surf[k],len(lead.get(k,()))) for k,v in whole.items() if k[0]==ty),key=lambda x:(-x[0],x[1]))[:topn]
        out[ty]=dict(label=lab,items=[dict(surface=s,articles=n,lead_articles=l) for n,s,l in lst])
    return out
def run(a):
    out=pathlib.Path(a.out);fd=out/'e57_bundle';fd.mkdir(exist_ok=True);rep=fd/'e57_bundle_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; RERUN=1 to redo):',rep);sys.exit(8)
    a.drop=N3.DROP;X=N3.analyze_all(a);D,names,res,rows,pub_date=[X[k] for k in('D','names','res','rows','pub_date')]
    attr=LT.attributes(names,res,topn=15);bundle=[];facts={};index={};store={};flags=[]
    for ei,(nm,r,row) in enumerate(zip(names,res,rows)):
        if not r['ok']:index[nm]=dict(ok=False,n_before=row['n_before'],n_dedup=row['n_dedup']);continue
        n=len(r['arts']);V=N.variant(r,'lead');kept=V['kept'];order=list(range(n));random.Random(SEED+ei).shuffle(order);sel=sorted(order[:a.per_event],key=lambda ai:(pub_date[r['rids'][ai]],r['rids'][ai]))
        if n<30:flags.append(f'{nm}: only {n} independent articles')
        for ai in sel:
            rid=r['rids'][ai];bundle.append(dict(event=nm,rid=rid,host=r['arts'][ai]['host'],date=str(pub_date[rid]),title=D['title'][D['_ridx'][rid]],excerpt=excerpt(r['sents'][ai],kept[ai],a.excerpt_chars)))
        ft=fact_tables(r);cl=attr.get(nm,{});ft['PLACE (proxy)']=dict(label='single capitalised words / names by context proxy',items=[dict(surface=x['surface'],articles=None,lead_articles=round(x['share']*n),share_of_lead=x['share']) for c in('place','person','other name') for x in cl.get(c,[])[:8]])
        facts[nm]=dict(independent_articles=n,tables=ft)
        index[nm]=dict(ok=True,n_before=row['n_before'],n_dedup=row['n_dedup'],selected=[dict(rid=r['rids'][ai],host=r['arts'][ai]['host'],date=str(pub_date[r['rids'][ai]])) for ai in sel],first=str(row['first']),last=str(row['last']))
        store[nm]=dict(rids=r['rids'],hosts=[x['host'] for x in r['arts']],dates=[str(pub_date[x]) for x in r['rids']],sents=r['sents'],cum=[x['cum'] for x in r['arts']],kept=[kept[ai] for ai in range(n)],n_before=row['n_before'],n_dedup=row['n_dedup'])
    (fd/'reading_bundle.json').write_text(json.dumps(dict(note='publisher text: stays on the server',seed=SEED,per_event=a.per_event,excerpt_chars=a.excerpt_chars,articles=bundle),indent=1,ensure_ascii=False))
    nl=fd/'news-lite';nl.mkdir(exist_ok=True)
    for nm in index:
        arts=[x for x in bundle if x['event']==nm]
        if arts:(nl/(re.sub(r'[^A-Za-z0-9]+','_',nm).strip('_')[:60]+'.json')).write_text(json.dumps(dict(note='private, internal research only; no publication or redistribution',event=nm,independent_articles=index[nm]['n_dedup'],articles=arts),indent=1,ensure_ascii=False))
    (fd/'bundle_facts.json').write_text(json.dumps(facts,indent=1,ensure_ascii=False));(fd/'bundle_index.json').write_text(json.dumps(index,indent=1,default=str))
    pickle.dump(dict(seed=SEED,events=store),open(fd/'bundle_store.pkl','wb'))
    L=['# E57 reading bundle (counts only; article text stays on the server)','',f'{len([v for v in index.values() if v["ok"]])} events analysed of {len(index)}; {len(bundle)} excerpts selected (<= {a.per_event} per event, seed {SEED}).','','| event | articles before near-dup removal | independent articles | selected | first | last |','|---|---|---|---|---|---|']
    L+=[f"| {k} | {v['n_before']} | {v['n_dedup']} | {len(v.get('selected',[]))} | {v.get('first','')} | {v.get('last','')} |" for k,v in index.items()]
    if flags:L+=['','Flags (fewer than 30 independent articles):']+[f'- {x}' for x in flags]
    for nm,f in facts.items():
        L+=['',f"### {nm} ({f['independent_articles']} independent articles): most frequent strings (articles holding it anywhere / in the lead)",'']
        for ty,t in f['tables'].items():L.append(f"- {t['label']}: "+'; '.join(f"{x['surface']} ({x['articles'] if x['articles'] is not None else '-'}/{x['lead_articles']})" for x in t['items'][:10]))
    rep.write_text('\n'.join(L)+'\n');print('\n'.join(L)[:5000])
def _selftest():
    import e57_verify as VF
    def after(td):
        kw=td/'kw.json';kw.write_text(json.dumps({'Zebra by keyword':['zebra','festival']}))
        a=argparse.Namespace(out=str(td),dup_jac=0.5,min_articles=5,min_domains=5,min_c=3,workers=1,seed=5,rerun=True,n_extra=4,drop=(),per_event=8,excerpt_chars=300,keyword_events=str(kw))
        N3.DROP=();run(a);fd=td/'e57_bundle';B=json.load(open(fd/'reading_bundle.json'));I=json.load(open(fd/'bundle_index.json'));F=json.load(open(fd/'bundle_facts.json'))
        assert list((fd/'news-lite').glob('*.json')) and B['articles'] and all(set(x)>={'event','rid','host','date','title','excerpt'} and x['excerpt'] for x in B['articles']) and 'Zebra by keyword' in I and F and (fd/'bundle_store.pkl').exists()
        assert max(sum(1 for x in B['articles'] if x['event']==k) for k in I)<=8
        ev=[k for k,v in I.items() if v['ok']][0];print('bundle events',len(I),'excerpts',len(B['articles']))
        VF._selftest_on(fd,ev)
    N3._selftest(after)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='e57_news');ap.add_argument('--raw',default='news_raw');ap.add_argument('--dup-jac',type=float,default=0.5);ap.add_argument('--min-articles',type=int,default=5);ap.add_argument('--min-domains',type=int,default=5);ap.add_argument('--min-c',type=int,default=3)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seed',type=int,default=57);ap.add_argument('--n-extra',type=int,default=20);ap.add_argument('--per-event',type=int,default=40);ap.add_argument('--excerpt-chars',type=int,default=700);ap.add_argument('--keyword-events',default=None);ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:_selftest();return
    run(a)
if __name__=='__main__':main()
